"""Интерфейс командной строки: python -m cryptocast <команда>."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Annotated

import typer

from cryptocast.config import PROJECT_ROOT, load_config
from cryptocast.utils import new_run_id, setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help="cryptocast: прогноз объёма и волатильности цифровых активов.")


@app.callback()
def main() -> None:
    setup_logging()


def _store():
    from cryptocast.pipeline import prepare_schema
    from cryptocast.storage import Store

    store = Store(load_config())
    prepare_schema(store)
    return store


def _dataset(store):
    from cryptocast.ml.experiments import Dataset

    return Dataset.load_current(store)


def _print(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=1, default=str, skipkeys=True))


@app.command("run-all")
def run_all_command(
    offline: Annotated[bool, typer.Option(help="Не обращаться к источникам.")] = False,
    force_train: Annotated[bool, typer.Option(help="Обучить модели заново.")] = False,
    tune: Annotated[bool, typer.Option(help="Запустить Optuna заново.")] = False,
    no_analysis: Annotated[bool, typer.Option(help="Пропустить абляцию, объяснения и проверку утечки.")] = False,
) -> None:
    """Полный цикл: сбор, качество, витрина, модели, прогноз, мониторинг."""
    from cryptocast.pipeline import run_all

    code = run_all(load_config(), offline=offline, force_train=force_train, tune=tune, analysis=not no_analysis)
    raise typer.Exit(code)


@app.command()
def ingest(sources: Annotated[str, typer.Option(help="klines,fng,rss,gdelt или all")] = "all") -> None:
    """Инкрементальный сбор источников."""
    from cryptocast.ingest.sources import ingest_all
    from cryptocast.pipeline import export_ops

    with _store() as store:
        selected = None if sources == "all" else set(sources.split(","))
        _print(ingest_all(store, new_run_id(), sources=selected))
        export_ops(store)


@app.command("backfill-gdelt")
def backfill_gdelt(requests: int = 300) -> None:
    """Догружает историю новостей GDELT (долго: у API жёсткий лимит частоты)."""
    from cryptocast.ingest.sources import export_gdelt_archive, ingest_all

    with _store() as store:
        _print(ingest_all(store, new_run_id(), sources={"gdelt"}, gdelt_max_requests=requests))
        _print({"archive_rows": export_gdelt_archive(store)})


@app.command()
def quality() -> None:
    """Контроль качества; при критичном сбое код возврата 2."""
    from cryptocast.quality import run_quality

    with _store() as store:
        ok = run_quality(store, new_run_id())
    raise typer.Exit(0 if ok else 2)


@app.command()
def staging() -> None:
    """Обновляет staging."""
    from cryptocast.staging import run_staging

    with _store() as store:
        _print(run_staging(store))


@app.command()
def features() -> None:
    """Обновляет витрину признаков."""
    from cryptocast.features import run_features

    with _store() as store:
        _print(run_features(store))


@app.command()
def tune() -> None:
    """Подбор гиперпараметров Optuna (ограниченный бюджет, без отложенной выборки)."""
    from cryptocast.ml.experiments import export_experiments, run_tuning

    with _store() as store:
        dataset = _dataset(store)
        dataset.freeze(store.cfg)
        _print(run_tuning(store, dataset))
        export_experiments(store)


@app.command()
def train(analysis: bool = False) -> None:
    """Сравнение моделей, регистрация кандидата, правило вывода в эксплуатацию."""
    from cryptocast.pipeline import full_model_cycle

    with _store() as store:
        _print(full_model_cycle(store, new_run_id(), _dataset(store), tune=False, analysis=analysis))


@app.command()
def analyze() -> None:
    """Абляция по группам признаков, отбор, кодирование инструмента."""
    from cryptocast.ml.experiments import run_feature_analysis

    with _store() as store:
        typer.echo(run_feature_analysis(store, _dataset(store)).to_string())


@app.command()
def interpret() -> None:
    """Важность признаков, SHAP, PDP/ICE, сравнение периодов."""
    from cryptocast.interpret import run_interpret

    with _store() as store:
        result = run_interpret(store, _dataset(store))
    _print({"permutation_shap_spearman": result["permutation_shap_spearman"]})


@app.command("check-leakage")
def check_leakage(no_shuffle: bool = False) -> None:
    """Проверки отсутствия утечки: усечение, метки времени, перемешанный таргет."""
    from cryptocast.leakage import run_leakage_report

    with _store() as store:
        report = run_leakage_report(store, _dataset(store), shuffled=not no_shuffle)
    _print(report)
    raise typer.Exit(0 if report["passed"] else 1)


@app.command()
def predict() -> None:
    """Прогнозы на следующие сутки."""
    from cryptocast.ml.registry import run_predict

    with _store() as store:
        typer.echo(run_predict(store, _dataset(store)).tail(10).to_string())


@app.command()
def monitor() -> None:
    """Дрейф, качество на фактах, решение о переобучении."""
    from cryptocast.monitoring import run_monitoring

    with _store() as store:
        _print(run_monitoring(store, _dataset(store), new_run_id()))


@app.command()
def rollback(target: str, reason: str = "ручной откат") -> None:
    """Возвращает предыдущую версию модели (log_qv или log_rv)."""
    from cryptocast.ml.registry import export_registry
    from cryptocast.ml.registry import rollback as do_rollback

    with _store() as store:
        typer.echo(do_rollback(store, target, reason))
        export_registry(store)


@app.command("simulate-drift")
def simulate_drift(days: int = 45, volume_factor: float = 3.0, return_factor: float = 2.2) -> None:
    """Реакция системы на ухудшение на копии данных, явно помеченной как симуляция."""
    from cryptocast.monitoring import simulate_drift as do_simulate

    with _store() as store:
        _print(do_simulate(store, days=days, volume_factor=volume_factor, return_factor=return_factor))


@app.command()
def reproduce(experiment_id: Annotated[str, typer.Option("--experiment-id")]) -> None:
    """Повторяет эксперимент из журнала и сравнивает метрики."""
    from cryptocast.ml.experiments import reproduce_experiment

    with _store() as store:
        result = reproduce_experiment(store, experiment_id)
    _print(result)
    raise typer.Exit(0 if result["passed"] else 1)


@app.command("list-experiments")
def list_experiments() -> None:
    """Журнал экспериментов."""
    with _store() as store:
        typer.echo(store.df(
            "SELECT experiment_id, kind, target, model_name, data_version, git_commit FROM ml.experiments ORDER BY created_at DESC"
        ).to_string())


@app.command()
def snapshot() -> None:
    """Снимок данных и моделей для офлайн-показа на защите."""
    from cryptocast.pipeline import make_snapshot

    _print(make_snapshot(load_config()))


@app.command()
def dashboard(port: int = 8501, offline: bool = False) -> None:
    """Запускает дашборд; --offline показывает снимок и не зависит от сети."""
    env = os.environ.copy()
    if offline:
        env["CRYPTOCAST_DATA_DIR"] = str(PROJECT_ROOT / "snapshot" / "offline_demo" / "data")
        env["CRYPTOCAST_ARTIFACTS_DIR"] = str(PROJECT_ROOT / "snapshot" / "offline_demo" / "artifacts")
    app_path = PROJECT_ROOT / "cryptocast" / "dashboard" / "app.py"
    raise typer.Exit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        env=env, cwd=PROJECT_ROOT,
    ))


@app.command()
def schedule() -> None:
    """Запускает run-all ежедневно по расписанию из config.yaml (APScheduler)."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from cryptocast.pipeline import run_all

    cfg = load_config()
    hour, minute = cfg["schedule"]["daily_time_utc"].split(":")
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(lambda: run_all(load_config()), "cron", hour=int(hour), minute=int(minute), max_instances=1, coalesce=True)
    typer.echo(f"Расписание: ежедневно в {hour}:{minute} UTC. Ctrl+C для остановки.")
    scheduler.start()


if __name__ == "__main__":
    app()
