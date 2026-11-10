"""Интерпретируемость: permutation и SHAP, вклады для произвольной даты, PDP/ICE, сравнение периодов."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap

from cryptocast.config import Config
from cryptocast.ml.core import FittedPipeline, Fold, LightGBMModel, Spec
from cryptocast.ml.experiments import Dataset, dev_holdout, load_tuned, spec_for
from cryptocast.ml.metrics import compute_metrics
from cryptocast.storage import Store, write_parquet

log = logging.getLogger(__name__)


def _shap_for(pipeline: FittedPipeline, frame: pd.DataFrame, news: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float, pd.DataFrame]:
    """SHAP-значения LightGBM (сумма вкладов + базовое значение = прогноз)."""
    model: LightGBMModel = pipeline.model  # type: ignore[assignment]
    prepared = pipeline.prepare(frame.reset_index(drop=True), news)
    matrix = model._matrix(prepared)
    explainer = shap.TreeExplainer(model.model.booster_)
    values = explainer.shap_values(matrix)
    base = float(np.ravel(explainer.expected_value)[0])
    return prepared, np.asarray(values), base, matrix


def _shap_table(prepared: pd.DataFrame, matrix: pd.DataFrame, values: np.ndarray, base: float,
                target: str, split: str, fold: int) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=[f"shap__{c}" for c in matrix.columns])
    numeric = matrix.drop(columns=["symbol"], errors="ignore")
    for column in numeric.columns:
        frame[f"value__{column}"] = numeric[column].to_numpy(dtype=float)
    frame["symbol"] = prepared["symbol"].values
    frame["date"] = prepared["date"].values
    frame["target"] = target
    frame["base_value"] = base
    frame["prediction"] = base + values.sum(axis=1)
    frame["y_true"] = prepared[f"target_{target}"].values
    frame["split"] = split
    frame["fold"] = fold
    return frame


def _permutation(pipeline: FittedPipeline, test: pd.DataFrame, news: pd.DataFrame, target: str, rng: np.random.Generator, repeats: int = 2) -> pd.DataFrame:
    prepared = pipeline.prepare(test.reset_index(drop=True), news)
    base = pipeline.model.predict(prepared)  # type: ignore[union-attr]
    y = prepared[f"target_{target}"].to_numpy(dtype=float)
    baseline = float(np.mean(np.abs(base - y)))
    rows = []
    for feature in pipeline.features:
        losses = []
        for _ in range(repeats):
            shuffled = prepared.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            losses.append(float(np.mean(np.abs(pipeline.model.predict(shuffled) - y))))  # type: ignore[union-attr]
        rows.append({"feature": feature, "increase_mae": float(np.mean(losses) - baseline)})
    return pd.DataFrame(rows)


def _pdp_ice(pipeline: FittedPipeline, frame: pd.DataFrame, news: pd.DataFrame, features: list[str], cfg: Config, target: str) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    prepared = pipeline.prepare(frame.reset_index(drop=True), news)
    sample_index = rng.choice(len(prepared), size=min(120, len(prepared)), replace=False)
    sample = prepared.iloc[sample_index].reset_index(drop=True)
    rows = []
    for feature in features:
        values = prepared[feature].dropna().to_numpy()
        grid = np.unique(np.quantile(values, np.linspace(0.02, 0.98, 15)))
        curves = np.empty((len(grid), len(sample)))
        for i, value in enumerate(grid):
            changed = sample.copy()
            changed[feature] = value
            curves[i] = pipeline.model.predict(changed)  # type: ignore[union-attr]
        mean_curve = curves.mean(axis=1)
        for i, value in enumerate(grid):
            rows.append({"target": target, "feature": feature, "grid_value": float(value), "kind": "pdp", "ice_id": -1, "prediction": float(mean_curve[i])})
        for j in range(min(30, len(sample))):
            for i, value in enumerate(grid):
                rows.append({"target": target, "feature": feature, "grid_value": float(value), "kind": "ice", "ice_id": j, "prediction": float(curves[i, j])})
    return pd.DataFrame(rows)


def _fit_fold_pipelines(spec: Spec, dev: pd.DataFrame, news: pd.DataFrame, folds: list[Fold], cfg: Config) -> list[FittedPipeline]:
    target_col = f"target_{spec.target}"
    data = dev[dev[target_col].notna()]
    return [FittedPipeline(spec, cfg).fit(data[data["date"] <= fold.train_end], news) for fold in folds]


def run_interpret(store: Store, dataset: Dataset) -> dict:
    """Рассчитывает и сохраняет все объяснения для дашборда (работает офлайн)."""
    cfg = store.cfg
    tuned = load_tuned(cfg)
    dev, holdout, folds = dev_holdout(dataset, cfg)
    rng = np.random.default_rng(cfg.seed)
    shap_all, perm_all, pdp_all, importance_periods, explainers = [], [], [], [], {}
    for target in cfg.targets:
        spec = spec_for(cfg, target, "lightgbm", tuned.get(target, {}).get("lightgbm"))
        target_col = f"target_{target}"
        data = dev[dev[target_col].notna()]
        hold = holdout[holdout[target_col].notna()]
        fold_pipelines = _fit_fold_pipelines(spec, dev, dataset.news, folds, cfg)
        for fold, pipeline in zip(folds, fold_pipelines):
            test = data[(data["date"] >= fold.test_start) & (data["date"] <= fold.test_end)]
            prepared, values, base, matrix = _shap_for(pipeline, test, dataset.news)
            shap_all.append(_shap_table(prepared, matrix, values, base, target, "cv", fold.index))
        recent = folds[-4:]
        for fold, pipeline in zip(recent, fold_pipelines[-4:]):
            test = data[(data["date"] >= fold.test_start) & (data["date"] <= fold.test_end)]
            perm_all.append(_permutation(pipeline, test, dataset.news, target, rng).assign(target=target, fold=fold.index))
        holdout_pipeline = FittedPipeline(spec, cfg).fit(data, dataset.news)
        prepared, values, base, matrix = _shap_for(holdout_pipeline, hold, dataset.news)
        shap_all.append(_shap_table(prepared, matrix, values, base, target, "holdout", -1))
        permutation_rank = (pd.concat([p for p in perm_all if p["target"].iloc[0] == target]).groupby("feature")["increase_mae"].mean().sort_values(ascending=False))
        top_features = [f for f in permutation_rank.index[:8]]
        pdp_all.append(_pdp_ice(holdout_pipeline, data[data["date"] > data["date"].max() - pd.Timedelta(days=365)], dataset.news, top_features, cfg, target))
        explainers[target] = holdout_pipeline

    shap_frame = pd.concat(shap_all, ignore_index=True)
    # Для компактности и единообразия: длинных строк нет, только широкая таблица на цель.
    for target in cfg.targets:
        part = shap_frame[shap_frame["target"] == target].dropna(axis=1, how="all")
        write_parquet(cfg, "mart", f"shap_{target}", part)
        shap_cols = [c for c in part.columns if c.startswith("shap__")]
        shares = part[shap_cols].abs()
        shares.columns = [c.replace("shap__", "") for c in shap_cols]
        by_year = shares.assign(period=pd.to_datetime(part["date"]).dt.year.astype(str)).groupby("period").mean()
        normalized = by_year.div(by_year.sum(axis=1), axis=0)
        long = normalized.reset_index().melt(id_vars="period", var_name="feature", value_name="share").assign(target=target)
        importance_periods.append(long)
    importance_frame = pd.concat(importance_periods, ignore_index=True)
    write_parquet(cfg, "mart", "importance_periods", importance_frame)

    perm = pd.concat(perm_all, ignore_index=True).groupby(["target", "feature"], as_index=False)["increase_mae"].mean()
    shap_global = []
    for target in cfg.targets:
        part = shap_frame[shap_frame["target"] == target]
        cols = [c for c in part.columns if c.startswith("shap__")]
        shap_global.append(pd.DataFrame({"target": target, "feature": [c.replace("shap__", "") for c in cols], "mean_abs_shap": part[cols].abs().mean().values}))
    joined = perm.merge(pd.concat(shap_global, ignore_index=True), on=["target", "feature"], how="outer").fillna(0.0)
    write_parquet(cfg, "mart", "importance_global", joined)
    write_parquet(cfg, "mart", "pdp_ice", pd.concat(pdp_all, ignore_index=True))

    # Согласованность структуры важности между соседними годами.
    corr_rows = []
    for target, part in importance_frame.groupby("target"):
        pivot = part.pivot_table(index="feature", columns="period", values="share", aggfunc="first")
        periods = sorted(pivot.columns)
        for a, b in zip(periods, periods[1:]):
            corr_rows.append({"target": target, "period_a": a, "period_b": b, "spearman": float(pivot[a].corr(pivot[b], method="spearman"))})
    write_parquet(cfg, "mart", "importance_stability", pd.DataFrame(corr_rows))
    agreement = {
        t: float(joined[joined["target"] == t][["increase_mae", "mean_abs_shap"]].corr(method="spearman").iloc[0, 1]) for t in cfg.targets
    }
    return {"permutation_shap_spearman": agreement, "explainers": explainers}


def check_prediction_sanity(frame: pd.DataFrame) -> float:
    return float(compute_metrics("log_qv", frame["y_true"], frame["prediction"])["mae"])
