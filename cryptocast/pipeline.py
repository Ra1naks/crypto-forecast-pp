"""Оркестрация этапов и команда run-all."""
from __future__ import annotations

import json
import logging
import shutil
import traceback
from collections.abc import Callable

import pandas as pd

from cryptocast.config import Config
from cryptocast.features import run_features
from cryptocast.ingest.sources import ingest_all
from cryptocast.interpret import run_interpret
from cryptocast.leakage import run_leakage_report
from cryptocast.ml.experiments import (Dataset, ensure_ml_schema, export_experiments, load_tuned,
                                        run_evaluation, run_feature_analysis, run_tuning)
from cryptocast.ml.registry import (ensure_registry_schema, export_forecasts, export_registry, production_row,
                                     rollback, run_predict, train_and_promote)
from cryptocast.monitoring import ensure_monitoring_schema, run_monitoring
from cryptocast.quality import run_quality
from cryptocast.staging import run_staging
from cryptocast.storage import Store, write_parquet
from cryptocast.utils import new_run_id, utcnow

log = logging.getLogger(__name__)


def prepare_schema(store: Store) -> None:
    ensure_ml_schema(store)
    ensure_registry_schema(store)
    ensure_monitoring_schema(store)


def step(store: Store, run_id: str, name: str, func: Callable[[], object]) -> object:
    started = utcnow()
    log.info("=== Этап: %s", name)
    try:
        result = func()
    except Exception:
        store.con.execute("INSERT INTO ops.pipeline_runs VALUES (?,?,?,?,?,?)", [run_id, name, "failed", started, utcnow(), traceback.format_exc()[-1500:]])
        raise
    details = json.dumps(result, ensure_ascii=False, default=str)[:1500] if isinstance(result, dict) else ""
    store.con.execute("INSERT INTO ops.pipeline_runs VALUES (?,?,?,?,?,?)", [run_id, name, "success", started, utcnow(), details])
    return result


def export_ops(store: Store) -> None:
    cfg = store.cfg
    write_parquet(cfg, "mart", "load_log", store.df("SELECT * FROM ops.load_log ORDER BY started_at"))
    write_parquet(cfg, "mart", "dq_results", store.df("SELECT * FROM ops.dq_results ORDER BY checked_at"))
    write_parquet(cfg, "mart", "pipeline_runs", store.df("SELECT * FROM ops.pipeline_runs ORDER BY started_at"))


def write_status(store: Store, run_id: str, status: str, details: dict) -> None:
    cfg = store.cfg
    last_candle = store.scalar("SELECT max(open_time) FROM raw.klines")
    payload = {"run_id": run_id, "status": status, "finished_at": utcnow().isoformat(), "last_candle": str(last_candle), **details}
    (cfg.lake_dir / "mart").mkdir(parents=True, exist_ok=True)
    (cfg.lake_dir / "mart" / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def needs_models(store: Store) -> bool:
    return any(production_row(store, t) is None for t in store.cfg.targets)


def full_model_cycle(store: Store, run_id: str, dataset: Dataset, *, tune: bool, analysis: bool) -> dict:
    """Подбор (при необходимости), сравнение, абляция, регистрация, объяснения, проверка утечки."""
    output: dict[str, object] = {}
    dataset.freeze(store.cfg)
    if tune or not any(load_tuned(store.cfg).get(t) for t in store.cfg.targets):
        output["tuning"] = step(store, run_id, "tune", lambda: run_tuning(store, dataset))
    evaluation = step(store, run_id, "evaluate", lambda: run_evaluation(store, dataset))
    output["promotion"] = step(store, run_id, "train_and_promote", lambda: train_and_promote(store, dataset, evaluation))  # type: ignore[arg-type]
    if analysis:
        step(store, run_id, "feature_analysis", lambda: run_feature_analysis(store, dataset))
        output["interpret"] = step(store, run_id, "interpret", lambda: run_interpret(store, dataset))
        output["leakage"] = step(store, run_id, "leakage", lambda: run_leakage_report(store, dataset))
    export_experiments(store)
    export_registry(store)
    return output


def run_all(cfg: Config, *, offline: bool = False, force_train: bool = False, tune: bool = False, analysis: bool = True,
            gdelt_max_requests: int | None = None) -> int:
    """Полный цикл: сбор → качество → витрина → модели → прогноз → мониторинг.

    Критичный сбой качества блокирует обновление витрины и завершается кодом 2,
    при этом дашборд продолжает показывать последнюю успешную витрину.
    """
    run_id = new_run_id()
    with Store(cfg) as store:
        prepare_schema(store)
        if not offline:
            step(store, run_id, "ingest", lambda: ingest_all(store, run_id, gdelt_max_requests=gdelt_max_requests))
        ok = step(store, run_id, "quality", lambda: run_quality(store, run_id))
        if not ok:
            export_ops(store)
            write_status(store, run_id, "blocked_by_quality", {"message": "витрина не обновлена: критичный сбой качества"})
            log.error("Обновление витрины заблокировано контролем качества.")
            return 2
        step(store, run_id, "staging", lambda: run_staging(store))
        step(store, run_id, "features", lambda: run_features(store))
        dataset = Dataset.load_current(store)
        cycle: dict[str, object] = {}
        if force_train or needs_models(store):
            cycle = full_model_cycle(store, run_id, dataset, tune=tune, analysis=analysis)
        step(store, run_id, "predict", lambda: run_predict(store, dataset).shape)
        decisions = step(store, run_id, "monitor", lambda: run_monitoring(store, dataset, run_id))
        retrained = {}
        for target, decision in (decisions or {}).items():  # type: ignore[union-attr]
            if decision.get("rollback"):
                rollback(store, target, "автоматический откат: live MAE новой версии хуже теневой")
                retrained[target] = "rollback"
        if any(d.get("retrain_now") for d in (decisions or {}).values()):  # type: ignore[union-attr]
            log.warning("Сработало правило переобучения: %s", decisions)
            retrained["cycle"] = full_model_cycle(store, run_id, dataset, tune=False, analysis=False)
            step(store, run_id, "predict_after_retrain", lambda: run_predict(store, dataset).shape)
        export_registry(store)
        export_experiments(store)
        export_forecasts(store)
        export_ops(store)
        write_status(store, run_id, "ok", {"data_version": dataset.version, "last_feature_date": str(dataset.frame["date"].max().date()), "retrain": retrained, "decisions": decisions})
        return 0


def make_snapshot(cfg: Config) -> dict:
    """Копирует витрину, БД и артефакты для офлайн-показа на защите."""
    destination = cfg.snapshot_dir / "offline_demo"
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "data").mkdir(parents=True)
    shutil.copy2(cfg.db_path, destination / "data" / cfg.db_path.name)
    shutil.copytree(cfg.lake_dir, destination / "data" / cfg.lake_dir.name)
    shutil.copytree(cfg.artifacts_dir, destination / "artifacts")
    info = {"created_at": utcnow().isoformat(), "last_candle": None}
    status_path = cfg.lake_dir / "mart" / "status.json"
    if status_path.exists():
        info["last_candle"] = json.loads(status_path.read_text(encoding="utf-8")).get("last_candle")
    (destination / "SNAPSHOT.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


__all__ = ["run_all", "make_snapshot", "pd"]
