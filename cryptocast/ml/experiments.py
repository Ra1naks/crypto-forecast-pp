"""Журнал экспериментов, версии данных, подбор Optuna, анализ признаков, воспроизведение."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import optuna
import pandas as pd

from cryptocast.config import Config
from cryptocast.features import data_version
from cryptocast.ml.core import Fold, Spec, evaluate_spec, run_walk_forward, split_holdout, walk_forward_folds, FittedPipeline
from cryptocast.ml.metrics import add_improvement, compute_metrics, metrics_frame
from cryptocast.storage import Store, write_parquet
from cryptocast.utils import git_state, new_run_id, utcnow

log = logging.getLogger(__name__)

EXPERIMENT_DDL = """
CREATE SCHEMA IF NOT EXISTS ml;
CREATE TABLE IF NOT EXISTS ml.experiments (
    experiment_id VARCHAR PRIMARY KEY, created_at TIMESTAMP, kind VARCHAR, target VARCHAR,
    model_name VARCHAR, spec VARCHAR, config_hash VARCHAR, data_version VARCHAR, seed INTEGER,
    git_commit VARCHAR, git_dirty BOOLEAN, cv_metrics VARCHAR, holdout_metrics VARCHAR, note VARCHAR
);
"""


class Dataset:
    """Зафиксированная версия данных для обучения: витрина и очищенные новости."""

    def __init__(self, frame: pd.DataFrame, news: pd.DataFrame, version: str):
        self.frame = frame
        self.news = news
        self.version = version

    @classmethod
    def load_current(cls, store: Store) -> "Dataset":
        frame = store.df("SELECT * FROM mart.features ORDER BY symbol, date")
        frame["date"] = pd.to_datetime(frame["date"])
        news = store.df("SELECT * FROM staging.news")
        if not news.empty:
            news["date"] = pd.to_datetime(news["date"])
        return cls(frame, news, data_version(frame, news))

    def freeze(self, cfg: Config) -> Path:
        folder = cfg.artifacts_dir / "datasets" / self.version
        if not (folder / "features.parquet").exists():
            folder.mkdir(parents=True, exist_ok=True)
            self.frame.to_parquet(folder / "features.parquet", index=False)
            self.news.to_parquet(folder / "news.parquet", index=False)
        return folder

    @classmethod
    def load_frozen(cls, cfg: Config, version: str) -> "Dataset":
        folder = cfg.artifacts_dir / "datasets" / version
        if not folder.exists():
            raise FileNotFoundError(f"Версия данных {version} не найдена в {folder}")
        frame = pd.read_parquet(folder / "features.parquet")
        news = pd.read_parquet(folder / "news.parquet")
        return cls(frame, news, data_version(frame, news))


def dev_holdout(dataset: Dataset, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[Fold]]:
    """Режет витрину на разработку и отложенную выборку (90 дней, в подборе не участвует)."""
    params = cfg["validation"]
    frame = dataset.frame[dataset.frame["target_log_qv"].notna() | dataset.frame["target_log_rv"].notna()]
    warmup_end = frame["date"].min() + pd.Timedelta(days=params["warmup_days"])
    frame = frame[frame["date"] >= warmup_end]
    dev_end, holdout_start = split_holdout(frame["date"], params["holdout_days"], params["gap_days"])
    dev = frame[frame["date"] <= dev_end].copy()
    holdout = frame[frame["date"] >= holdout_start].copy()
    return dev, holdout, walk_forward_folds(dev["date"], cfg)


def log_experiment(
    store: Store, kind: str, spec: Spec, version: str, cv: dict, holdout: dict | None, note: str = ""
) -> str:
    experiment_id = f"exp-{new_run_id()}"
    commit, dirty = git_state()
    store.con.execute(
        "INSERT INTO ml.experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [experiment_id, utcnow(), kind, spec.target, spec.model, json.dumps(asdict(spec), ensure_ascii=False),
         store.cfg.config_hash(), version, spec.seed, commit, dirty, json.dumps(cv),
         json.dumps(holdout) if holdout else None, note],
    )
    return experiment_id


def model_params(cfg: Config, model: str, tuned: dict | None = None) -> dict:
    if model in ("naive", "naive_ma7"):
        return {}
    params = dict(cfg["models"][model])
    return {**params, **(tuned or {})}


def spec_for(cfg: Config, target: str, model: str, tuned: dict | None = None, **kwargs) -> Spec:
    return Spec(target=target, model=model, params=model_params(cfg, model, tuned), seed=cfg.seed, **kwargs)


# ---------------------------------- Подбор гиперпараметров -----------------------------
def suggest(trial: optuna.Trial, model: str) -> dict:
    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        }
    if model == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model == "mlp":
        width = trial.suggest_categorical("width", [32, 64, 128])
        depth = trial.suggest_int("depth", 1, 3)
        return {
            "hidden_layer_sizes": [width] * depth,
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
        }
    raise ValueError(model)


def run_tuning(store: Store, dataset: Dataset) -> dict[str, dict[str, dict]]:
    """Байесовский подбор (бюджет: trials и timeout). Отложенная выборка сюда не попадает."""
    cfg = store.cfg
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev, _, folds = dev_holdout(dataset, cfg)
    tuning_folds = folds[-cfg["tuning"]["folds_last_n"]:]
    best: dict[str, dict[str, dict]] = {t: {} for t in cfg.targets}
    trials_all: list[pd.DataFrame] = []
    for target in cfg.targets:
        metric = cfg["models"]["selection_metric"][target]
        for model, budget in cfg["tuning"]["budgets"].items():
            def objective(trial: optuna.Trial) -> float:
                spec = spec_for(cfg, target, model, suggest(trial, model))
                _, metrics = evaluate_spec(spec, dev, dataset.news, tuning_folds, cfg)
                return metrics[metric]

            sampler = optuna.samplers.TPESampler(seed=cfg.seed)
            study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"{target}-{model}")
            study.optimize(objective, n_trials=budget["n_trials"], timeout=budget["timeout_seconds"], n_jobs=1)
            params = dict(study.best_trial.user_attrs) or dict(study.best_params)
            # Восстанавливаем формат параметров MLP
            if model == "mlp":
                params = {"hidden_layer_sizes": [params.pop("width")] * params.pop("depth"), **params}
            best[target][model] = params
            spec = spec_for(cfg, target, model, params)
            cv_metrics = {metric: float(study.best_value), "n_trials": len(study.trials), "folds_used": len(tuning_folds)}
            log_experiment(store, "tuning", spec, dataset.version, cv_metrics, None,
                           note=f"Optuna TPE, бюджет {budget}, фактически trials={len(study.trials)}")
            trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            trials.insert(0, "target", target)
            trials.insert(1, "model", model)
            trials_all.append(trials)
            log.info("Подбор %s/%s: %s=%.5f, trials=%s", target, model, metric, study.best_value, len(study.trials))
    out = cfg.artifacts_dir / "tuned_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, ensure_ascii=False, indent=1), encoding="utf-8")
    write_parquet(cfg, "mart", "optuna_trials", pd.concat(trials_all, ignore_index=True).astype({"value": float}))
    return best


def load_tuned(cfg: Config) -> dict[str, dict[str, dict]]:
    path = cfg.artifacts_dir / "tuned_params.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {t: {} for t in cfg.targets}


# ---------------------------------- Единое сравнение моделей ---------------------------
MODEL_ORDER = ["naive", "naive_ma7", "elasticnet", "lightgbm", "mlp"]


def run_evaluation(store: Store, dataset: Dataset) -> dict[str, dict[str, object]]:
    """Walk-forward и отложенная выборка для всех моделей; всё журналируется."""
    cfg = store.cfg
    tuned = load_tuned(cfg)
    dev, holdout, folds = dev_holdout(dataset, cfg)
    log.info("Разработка: %s..%s, фолдов: %s, отложенная: %s..%s", dev["date"].min().date(),
             dev["date"].max().date(), len(folds), holdout["date"].min().date(), holdout["date"].max().date())
    oos_all, holdout_all, specs = [], [], {}
    experiments: dict[str, dict[str, str]] = {}
    for target in cfg.targets:
        experiments[target] = {}
        for model in MODEL_ORDER:
            spec = spec_for(cfg, target, model, tuned.get(target, {}).get(model))
            specs[(target, model)] = spec
            oos, cv_metrics = evaluate_spec(spec, dev, dataset.news, folds, cfg)
            target_col = f"target_{target}"
            holdout_data = holdout[holdout[target_col].notna()]
            pipeline = FittedPipeline(spec, cfg).fit(dev[dev[target_col].notna()], dataset.news)
            preds = pipeline.predict(holdout_data, dataset.news)
            hold_metrics = compute_metrics(target, holdout_data[target_col], preds)
            hold = pd.DataFrame({"symbol": holdout_data["symbol"].values, "date": holdout_data["date"].values,
                                 "y_true": holdout_data[target_col].values, "y_pred": preds, "fold": -1})
            exp_id = log_experiment(store, "evaluation", spec, dataset.version, cv_metrics, hold_metrics)
            experiments[target][model] = exp_id
            oos_all.append(oos.assign(model=model, target=target, split="cv"))
            holdout_all.append(hold.assign(model=model, target=target, split="holdout"))
            log.info("%s/%s: CV %s", target, model, {k: round(v, 4) for k, v in cv_metrics.items() if k in ("mae", "rmse", "qlike")})
    oos_frame = pd.concat(oos_all, ignore_index=True)
    hold_frame = pd.concat(holdout_all, ignore_index=True)
    predictions = pd.concat([oos_frame, hold_frame], ignore_index=True)
    write_parquet(cfg, "mart", "predictions", predictions)
    metrics = pd.concat(
        [add_improvement(metrics_frame(part[part["target"] == t], t)).assign(split=split)
         for split, part in (("cv", oos_frame), ("holdout", hold_frame)) for t in cfg.targets],
        ignore_index=True,
    )
    write_parquet(cfg, "mart", "metrics", metrics)
    store.replace_table("mart.metrics", metrics)
    write_parquet(cfg, "mart", "folds", pd.DataFrame([{"fold": f.index, "train_end": f.train_end, "test_start": f.test_start, "test_end": f.test_end} for f in folds]))
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "evaluation_experiments.json").write_text(json.dumps(experiments, indent=1), encoding="utf-8")
    return {"specs": specs, "metrics": metrics, "predictions": predictions, "experiments": experiments}


# ---------------------------------- Абляция, отбор, кодирование --------------------------
def run_feature_analysis(store: Store, dataset: Dataset) -> pd.DataFrame:
    """Ablation по группам, отбор top-k и сравнение кодирования инструмента (один walk-forward)."""
    cfg = store.cfg
    tuned = load_tuned(cfg)
    dev, _, folds = dev_holdout(dataset, cfg)
    folds = folds[-cfg["validation"]["analysis_folds"]:]
    rows: list[dict] = []
    variants: list[tuple[str, str, dict]] = [
        ("ablation", "только рынок и календарь", {"feature_set": ["market", "calendar"], "use_svd": False}),
        ("ablation", "+ индекс страха", {"feature_set": ["market", "calendar", "fng"], "use_svd": False}),
        ("ablation", "+ текст: счётчики и тональность", {"feature_set": ["market", "calendar", "fng", "text"], "use_svd": False}),
        ("ablation", "+ текст: TF-IDF+SVD", {"feature_set": ["market", "calendar", "fng", "text"], "use_svd": True}),
        ("selection", "все признаки", {}),
        ("selection", "top-40 по gain (train)", {"top_k": 40}),
        ("selection", "top-20 по gain (train)", {"top_k": 20}),
        ("encoding", "LightGBM: нативная категория", {"encoding": "native"}),
        ("encoding", "LightGBM: one-hot", {"encoding": "onehot"}),
        ("encoding", "LightGBM: без инструмента", {"encoding": "none"}),
    ]
    for target in cfg.targets:
        metric = cfg["models"]["selection_metric"][target]
        for section, name, overrides in variants:
            spec = spec_for(cfg, target, "lightgbm", tuned.get(target, {}).get("lightgbm"), **overrides)
            oos, metrics = evaluate_spec(spec, dev, dataset.news, folds, cfg)
            rows.append({"section": section, "variant": name, "target": target, "model": "lightgbm",
                         "metric": metric, "value": metrics[metric], "mae": metrics["mae"], "rmse": metrics["rmse"]})
            log.info("%s / %s / %s: %s=%.5f", section, target, name, metric, metrics[metric])
        # One-hot — стандарт для линейной модели: сравниваем с HAR без инструмента.
        for name, drop in (("ElasticNet: one-hot инструмента", False), ("ElasticNet: без инструмента", True)):
            spec = spec_for(cfg, target, "elasticnet", tuned.get(target, {}).get("elasticnet"))
            local_cfg = _symbols_override(cfg) if drop else cfg
            oos, metrics = evaluate_spec(spec, dev, dataset.news, folds, local_cfg)
            rows.append({"section": "encoding", "variant": name, "target": target, "model": "elasticnet",
                         "metric": metric, "value": metrics[metric], "mae": metrics["mae"], "rmse": metrics["rmse"]})
    frame = pd.DataFrame(rows)
    write_parquet(cfg, "mart", "feature_analysis", frame)
    return frame


def _symbols_override(cfg: Config) -> Config:
    """Конфигурация, где инструмент не кодируется (one-hot вырождается в один столбец-нуль)."""
    raw = json.loads(json.dumps(cfg.raw))
    raw["universe"]["symbols"] = ["__none__"]
    return Config(raw, cfg.root, cfg.data_dir, cfg.artifacts_dir, cfg.snapshot_dir)


# ---------------------------------- Воспроизведение -------------------------------------
def reproduce_experiment(store: Store, experiment_id: str, tolerance: float = 1e-6) -> dict:
    """Повторяет эксперимент по журналу на замороженной версии данных и сравнивает метрики."""
    cfg = store.cfg
    row = store.df("SELECT * FROM ml.experiments WHERE experiment_id = ?", [experiment_id])
    if row.empty:
        raise KeyError(f"Эксперимент {experiment_id} не найден")
    record = row.iloc[0]
    spec = Spec(**json.loads(record["spec"]))
    dataset = Dataset.load_frozen(cfg, record["data_version"])
    version_ok = dataset.version == record["data_version"]
    dev, holdout, folds = dev_holdout(dataset, cfg)
    target_col = f"target_{spec.target}"
    logged_cv = json.loads(record["cv_metrics"])
    if record["kind"] == "tuning":
        tuning_folds = folds[-cfg["tuning"]["folds_last_n"]:]
        _, cv_metrics = evaluate_spec(spec, dev, dataset.news, tuning_folds, cfg)
        metric = cfg["models"]["selection_metric"][spec.target]
        new = {metric: cv_metrics[metric]}
        compare = {metric: (float(logged_cv[metric]), float(cv_metrics[metric]))}
    else:
        oos = run_walk_forward(spec, dev, dataset.news, folds, cfg)
        new_cv = compute_metrics(spec.target, oos["y_true"], oos["y_pred"])
        pipeline = FittedPipeline(spec, cfg).fit(dev[dev[target_col].notna()], dataset.news)
        hold = holdout[holdout[target_col].notna()]
        new_hold = compute_metrics(spec.target, hold[target_col], pipeline.predict(hold, dataset.news))
        logged_hold = json.loads(record["holdout_metrics"]) if record["holdout_metrics"] else {}
        compare = {f"cv.{k}": (float(logged_cv[k]), float(new_cv[k])) for k in new_cv if k in logged_cv and k != "n"}
        compare.update({f"holdout.{k}": (float(logged_hold[k]), float(new_hold[k])) for k in new_hold if k in logged_hold and k != "n"})
        new = {"cv": new_cv, "holdout": new_hold}
    diffs = {k: abs(a - b) for k, (a, b) in compare.items()}
    passed = bool(version_ok and max(diffs.values(), default=0.0) <= tolerance)
    result = {
        "experiment_id": experiment_id, "kind": record["kind"], "target": spec.target, "model": spec.model,
        "data_version": record["data_version"], "data_version_match": version_ok, "git_commit": record["git_commit"],
        "seed": int(record["seed"]), "max_abs_diff": max(diffs.values(), default=0.0), "tolerance": tolerance,
        "passed": passed, "compared": {k: {"logged": a, "reproduced": b} for k, (a, b) in compare.items()}, "new": new,
    }
    out = cfg.artifacts_dir / "reproductions"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{experiment_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def export_experiments(store: Store) -> None:
    frame = store.df("SELECT * FROM ml.experiments ORDER BY created_at")
    write_parquet(store.cfg, "mart", "experiments", frame)


def ensure_ml_schema(store: Store) -> None:
    store.con.execute(EXPERIMENT_DDL)

