"""Реестр версий моделей, правила вывода в эксплуатацию и отката, суточный прогноз."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cryptocast.config import Config
from cryptocast.ml.core import FittedPipeline, Spec
from cryptocast.ml.experiments import Dataset
from cryptocast.ml.metrics import compute_metrics
from cryptocast.storage import Store, write_parquet
from cryptocast.utils import utcnow

log = logging.getLogger(__name__)

REGISTRY_DDL = """
CREATE SCHEMA IF NOT EXISTS ml;
CREATE TABLE IF NOT EXISTS ml.registry (
    version_id VARCHAR PRIMARY KEY, target VARCHAR, model_name VARCHAR, stage VARCHAR,
    created_at TIMESTAMP, promoted_at TIMESTAMP, retired_at TIMESTAMP, trained_until DATE,
    feature_list VARCHAR, params VARCHAR, metrics VARCHAR, data_version VARCHAR,
    experiment_id VARCHAR, artifact_path VARCHAR, decision_note VARCHAR
);
CREATE TABLE IF NOT EXISTS ml.forecasts (
    origin_date DATE, target_date DATE, symbol VARCHAR, target VARCHAR, version_id VARCHAR, role VARCHAR,
    source VARCHAR, y_pred DOUBLE, y_lower DOUBLE, y_upper DOUBLE, created_at TIMESTAMP,
    PRIMARY KEY (origin_date, symbol, target, version_id, role)
);
CREATE TABLE IF NOT EXISTS ml.registry_events (
    event_at TIMESTAMP, target VARCHAR, version_id VARCHAR, event VARCHAR, details VARCHAR
);
"""


def ensure_registry_schema(store: Store) -> None:
    store.con.execute(REGISTRY_DDL)


def _record_event(store: Store, target: str, version_id: str, event: str, details: str) -> None:
    store.con.execute("INSERT INTO ml.registry_events VALUES (?,?,?,?,?)", [utcnow(), target, version_id, event, details])
    log.info("Реестр [%s] %s: %s (%s)", target, version_id, event, details)


def production_row(store: Store, target: str) -> pd.Series | None:
    frame = store.df("SELECT * FROM ml.registry WHERE target = ? AND stage = 'production'", [target])
    return None if frame.empty else frame.iloc[0]


def load_bundle(row: pd.Series) -> dict:
    return joblib.load(Path(row["artifact_path"]) / "bundle.joblib")


def _top_features(pipeline: FittedPipeline, train: pd.DataFrame, target: str, cfg: Config) -> list[str]:
    """Ключевые признаки для мониторинга: gain у бустинга, иначе все признаки модели."""
    top = cfg["monitoring"]["top_features"]
    model = pipeline.model
    if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        importance = pd.Series(model.model.feature_importances_, index=model.model.feature_name_)
        return [f for f in importance.sort_values(ascending=False).index if f != "symbol"][:top]
    return list(pipeline.features)[:top]


def register_candidate(
    store: Store, dataset: Dataset, spec: Spec, experiment_id: str, cv_preds: pd.DataFrame,
    holdout_preds: pd.DataFrame, cv_metrics: dict, holdout_metrics: dict,
) -> str:
    """Обучает итоговую модель на всех доступных данных и записывает кандидата в реестр."""
    cfg = store.cfg
    target_col = f"target_{spec.target}"
    train = dataset.frame[dataset.frame[target_col].notna()]
    pipeline = FittedPipeline(spec, cfg).fit(train, dataset.news)
    version_id = f"{spec.target}-{spec.model}-{utcnow():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:4]}"
    folder = cfg.artifacts_dir / "registry" / version_id
    folder.mkdir(parents=True, exist_ok=True)
    residuals = cv_preds.assign(resid=cv_preds["y_true"] - cv_preds["y_pred"])
    low, high = cfg["decision"]["interval_quantiles"]
    grouped = residuals.groupby("symbol")["resid"]
    quantiles = {s: (float(grouped.get_group(s).quantile(low)), float(grouped.get_group(s).quantile(high))) for s in grouped.groups}
    top = _top_features(pipeline, train, spec.target, cfg)
    reference = train[train["date"] > train["date"].max() - pd.Timedelta(days=cfg["monitoring"]["reference_days"])]
    reference_columns = [c for c in top if c in reference.columns]
    reference[["date", "symbol"] + reference_columns].to_parquet(folder / "reference.parquet", index=False)
    bundle = {"pipeline": pipeline, "spec": spec, "residual_quantiles": quantiles, "top_features": reference_columns,
              "cv_metrics": cv_metrics, "holdout_metrics": holdout_metrics}
    joblib.dump(bundle, folder / "bundle.joblib")
    metrics = {"cv": cv_metrics, "holdout": holdout_metrics}
    (folder / "meta.json").write_text(json.dumps({"spec": spec.__dict__, "metrics": metrics, "data_version": dataset.version,
                                                  "features": pipeline.features}, ensure_ascii=False, indent=1), encoding="utf-8")
    store.con.execute(
        "INSERT INTO ml.registry VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [version_id, spec.target, spec.model, "candidate", utcnow(), None, None, pd.Timestamp(train["date"].max()).date(),
         json.dumps(pipeline.features), json.dumps(spec.params), json.dumps(metrics), dataset.version, experiment_id, str(folder), ""],
    )
    _record_event(store, spec.target, version_id, "registered", f"кандидат из эксперимента {experiment_id}")
    return version_id


def _insert_forecasts(store: Store, rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    store.con.register("_fc", rows)
    try:
        store.con.execute(
            """INSERT INTO ml.forecasts SELECT origin_date, target_date, symbol, target, version_id, role, source,
                      y_pred, y_lower, y_upper, created_at FROM _fc ON CONFLICT DO NOTHING"""
        )
    finally:
        store.con.unregister("_fc")


def _holdout_rows(holdout_preds: pd.DataFrame, spec: Spec, version_id: str, quantiles: dict) -> pd.DataFrame:
    rows = holdout_preds.copy()
    rows["origin_date"] = pd.to_datetime(rows["date"]).dt.date
    rows["target_date"] = (pd.to_datetime(rows["date"]) + pd.Timedelta(days=1)).dt.date
    rows["target"] = spec.target
    rows["version_id"] = version_id
    rows["role"] = "production"
    rows["source"] = "holdout"
    rows["y_lower"] = rows["y_pred"] + rows["symbol"].map(lambda s: quantiles[s][0])
    rows["y_upper"] = rows["y_pred"] + rows["symbol"].map(lambda s: quantiles[s][1])
    rows["created_at"] = utcnow()
    return rows


def promote(store: Store, version_id: str, holdout_preds: pd.DataFrame, cv_by_model: pd.DataFrame | None = None) -> dict:
    """Решает, вывести ли кандидата в эксплуатацию.

    Правило: кандидат должен быть не хуже текущей production-версии на общем отрезке
    (её реальные прогнозы против прогнозов кандидата, обученного до этого отрезка) с учётом
    min_relative_improvement; без production-версии — не хуже наивного прогноза × max_worse_than_naive.
    """
    cfg = store.cfg
    row = store.df("SELECT * FROM ml.registry WHERE version_id = ?", [version_id]).iloc[0]
    target = row["target"]
    metric = cfg["models"]["selection_metric"][target]
    bundle = load_bundle(row)
    cand = compute_metrics(target, holdout_preds["y_true"], holdout_preds["y_pred"])[metric]
    current = production_row(store, target)
    decision: dict[str, object] = {"version_id": version_id, "target": target, "metric": metric, "candidate_value": cand}
    promoted = False
    if current is None:
        limit = cfg["registry"]["max_worse_than_naive"]
        naive_value = None
        if cv_by_model is not None:
            naive = cv_by_model[(cv_by_model["model"] == "naive") & (cv_by_model["target"] == target) & (cv_by_model["symbol"] == "ALL")]
            naive_value = float(naive.iloc[0][metric]) if not naive.empty else None
        promoted = naive_value is None or float(bundle["cv_metrics"][metric]) <= naive_value * limit
        decision.update(rule="первичный вывод: CV не хуже наивной × порог", naive_cv=naive_value, candidate_cv=float(bundle["cv_metrics"][metric]))
    else:
        champion = store.df(
            "SELECT origin_date, symbol, y_pred AS champ FROM ml.forecasts WHERE version_id = ? AND role = 'production'",
            [current["version_id"]],
        )
        champion["origin_date"] = pd.to_datetime(champion["origin_date"])
        merged = holdout_preds.assign(origin_date=pd.to_datetime(holdout_preds["date"])).merge(champion, on=["origin_date", "symbol"])
        if len(merged) >= 20:
            champ_value = compute_metrics(target, merged["y_true"], merged["champ"])[metric]
            cand_value = compute_metrics(target, merged["y_true"], merged["y_pred"])[metric]
            promoted = cand_value <= champ_value * (1 - cfg["registry"]["min_relative_improvement"])
            decision.update(rule="кандидат против текущей версии на общем отрезке", n=len(merged), champion_value=champ_value, candidate_value=cand_value)
        else:
            champ_cv = float(json.loads(current["metrics"])["cv"][metric])
            promoted = float(bundle["cv_metrics"][metric]) <= champ_cv
            decision.update(rule="нет общего отрезка: сравнение по CV", champion_value=champ_cv, candidate_value=float(bundle["cv_metrics"][metric]))
    decision["promoted"] = bool(promoted)
    note = json.dumps(decision, ensure_ascii=False, default=float)
    if promoted:
        if current is not None:
            store.con.execute("UPDATE ml.registry SET stage = 'archived', retired_at = ? WHERE version_id = ?", [utcnow(), current["version_id"]])
        store.con.execute("UPDATE ml.registry SET stage = 'production', promoted_at = ?, decision_note = ? WHERE version_id = ?", [utcnow(), note, version_id])
        spec = bundle["spec"]
        _insert_forecasts(store, _holdout_rows(holdout_preds, spec, version_id, bundle["residual_quantiles"]))
        _record_event(store, target, version_id, "promoted", note)
    else:
        store.con.execute("UPDATE ml.registry SET stage = 'rejected', decision_note = ? WHERE version_id = ?", [note, version_id])
        _record_event(store, target, version_id, "rejected", note)
    return decision


def rollback(store: Store, target: str, reason: str = "ручной откат") -> str | None:
    """Возвращает предыдущую архивную версию в production."""
    previous = store.df(
        "SELECT * FROM ml.registry WHERE target = ? AND stage = 'archived' ORDER BY retired_at DESC LIMIT 1", [target]
    )
    if previous.empty:
        log.warning("Нет версии для отката по %s", target)
        return None
    current = production_row(store, target)
    version_id = previous.iloc[0]["version_id"]
    if current is not None:
        store.con.execute("UPDATE ml.registry SET stage = 'archived', retired_at = ? WHERE version_id = ?", [utcnow(), current["version_id"]])
    store.con.execute("UPDATE ml.registry SET stage = 'production', promoted_at = ? WHERE version_id = ?", [utcnow(), version_id])
    _record_event(store, target, version_id, "rollback", reason)
    return version_id


def export_registry(store: Store) -> None:
    cfg = store.cfg
    write_parquet(cfg, "mart", "registry", store.df("SELECT * FROM ml.registry ORDER BY created_at"))
    write_parquet(cfg, "mart", "registry_events", store.df("SELECT * FROM ml.registry_events ORDER BY event_at"))


def run_predict(store: Store, dataset: Dataset) -> pd.DataFrame:
    """Строит прогнозы на D+1 по всем инструментам и пересчитывает ретроспективу прогнозов."""
    cfg = store.cfg
    latest = dataset.frame["date"].max()
    rows_frame = dataset.frame[dataset.frame["date"] == latest]
    outputs: list[pd.DataFrame] = []
    for target in cfg.targets:
        production = production_row(store, target)
        if production is None:
            log.warning("Нет production-версии для %s", target)
            continue
        versions = [("production", production)]
        archived = store.df("SELECT * FROM ml.registry WHERE target = ? AND stage = 'archived' ORDER BY retired_at DESC LIMIT 1", [target])
        if not archived.empty:
            versions.append(("shadow", archived.iloc[0]))
        for role, row in versions:
            bundle = load_bundle(row)
            pipeline: FittedPipeline = bundle["pipeline"]
            preds = pipeline.predict(rows_frame, dataset.news)
            quantiles = bundle["residual_quantiles"]
            result = pd.DataFrame({
                "origin_date": rows_frame["date"].dt.date.values, "target_date": (rows_frame["date"] + pd.Timedelta(days=1)).dt.date.values,
                "symbol": rows_frame["symbol"].values, "target": target, "version_id": row["version_id"], "role": role,
                "source": "live", "y_pred": preds, "created_at": utcnow(),
            })
            result["y_lower"] = preds + result["symbol"].map(lambda s: quantiles[s][0])
            result["y_upper"] = preds + result["symbol"].map(lambda s: quantiles[s][1])
            outputs.append(result)
    if outputs:
        _insert_forecasts(store, pd.concat(outputs, ignore_index=True))
    return export_forecasts(store)


def export_forecasts(store: Store) -> pd.DataFrame:
    """Факт и прогнозы в едином виде, с понятными пользователю величинами и уровнем риска."""
    cfg = store.cfg
    forecasts = store.df("SELECT * FROM ml.forecasts ORDER BY target_date, symbol")
    if forecasts.empty:
        return forecasts
    daily = store.df("SELECT symbol, date, quote_volume, rv FROM staging.daily")
    daily["date"] = pd.to_datetime(daily["date"])
    forecasts["target_date"] = pd.to_datetime(forecasts["target_date"])
    forecasts["origin_date"] = pd.to_datetime(forecasts["origin_date"])
    merged = forecasts.merge(daily.rename(columns={"date": "target_date"}), on=["symbol", "target_date"], how="left")
    merged["actual_log"] = np.where(merged["target"] == "log_qv", np.log(merged["quote_volume"]), np.log(merged["rv"]))
    merged["forecast_value"] = np.exp(merged["y_pred"])
    merged["forecast_low"] = np.exp(merged["y_lower"])
    merged["forecast_high"] = np.exp(merged["y_upper"])
    write_parquet(cfg, "mart", "forecasts", merged)

    latest_origin = merged[merged["source"] == "live"]["origin_date"].max()
    decision = _decision_table(cfg, merged, daily, latest_origin)
    write_parquet(cfg, "mart", "decision", decision)
    return merged


def _decision_table(cfg: Config, forecasts: pd.DataFrame, daily: pd.DataFrame, latest_origin: pd.Timestamp) -> pd.DataFrame:
    """Сводка на завтра: условная ёмкость исполнения и уровень риска (не торговый совет)."""
    params = cfg["decision"]
    if pd.isna(latest_origin):
        return pd.DataFrame()
    rows = []
    live = forecasts[(forecasts["origin_date"] == latest_origin) & (forecasts["role"] == "production") & (forecasts["source"] == "live")]
    for symbol in cfg.symbols:
        volume = live[(live["symbol"] == symbol) & (live["target"] == "log_qv")]
        rv = live[(live["symbol"] == symbol) & (live["target"] == "log_rv")]
        if volume.empty or rv.empty:
            continue
        history = daily[daily["symbol"] == symbol].sort_values("date").tail(params["risk_lookback_days"])["rv"].dropna()
        predicted_rv = float(rv.iloc[0]["forecast_value"])
        percentile = float((history < predicted_rv).mean())
        level = "высокий" if percentile >= params["risk_percentiles"]["high"] else "повышенный" if percentile >= params["risk_percentiles"]["elevated"] else "обычный"
        rows.append({
            "symbol": symbol, "origin_date": latest_origin, "target_date": volume.iloc[0]["target_date"],
            "forecast_volume_usd": float(volume.iloc[0]["forecast_value"]),
            "volume_low_usd": float(volume.iloc[0]["forecast_low"]), "volume_high_usd": float(volume.iloc[0]["forecast_high"]),
            "capacity_usd": float(volume.iloc[0]["forecast_low"]) * params["participation_rate"],
            "participation_rate": params["participation_rate"], "forecast_rv_pct": predicted_rv * 100,
            "rv_low_pct": float(rv.iloc[0]["forecast_low"]) * 100, "rv_high_pct": float(rv.iloc[0]["forecast_high"]) * 100,
            "rv_percentile_365d": percentile, "risk_level": level,
            "volume_version": volume.iloc[0]["version_id"], "rv_version": rv.iloc[0]["version_id"],
        })
    return pd.DataFrame(rows)


def train_and_promote(store: Store, dataset: Dataset, evaluation: dict) -> dict[str, dict]:
    """Выбирает лучшую модель по CV, регистрирует и применяет правило вывода в эксплуатацию."""
    cfg = store.cfg
    metrics: pd.DataFrame = evaluation["metrics"]
    predictions: pd.DataFrame = evaluation["predictions"]
    results: dict[str, dict] = {}
    for target in cfg.targets:
        selection = cfg["models"]["selection_metric"][target]
        board = metrics[(metrics["target"] == target) & (metrics["symbol"] == "ALL") & (metrics["split"] == "cv") & (~metrics["model"].isin(["naive", "naive_ma7"]))]
        best_model = board.sort_values(selection).iloc[0]["model"]
        spec: Spec = evaluation["specs"][(target, best_model)]
        selected = predictions[(predictions["target"] == target) & (predictions["model"] == best_model)]
        cv_preds = selected[selected["split"] == "cv"]
        holdout_preds = selected[selected["split"] == "holdout"]
        cv_metrics = compute_metrics(target, cv_preds["y_true"], cv_preds["y_pred"])
        holdout_metrics = compute_metrics(target, holdout_preds["y_true"], holdout_preds["y_pred"])
        version_id = register_candidate(
            store, dataset, spec, evaluation["experiments"][target][best_model],
            cv_preds, holdout_preds, cv_metrics, holdout_metrics,
        )
        results[target] = promote(store, version_id, holdout_preds, metrics[metrics["split"] == "cv"])
    return results

