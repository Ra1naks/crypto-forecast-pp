"""Метрики, мониторинг, правило переобучения, реестр и воспроизводимость."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cryptocast.leakage import features_from_raw
from cryptocast.ml.core import FittedPipeline, Spec
from cryptocast.ml.experiments import Dataset, dev_holdout, log_experiment, reproduce_experiment, run_evaluation
from cryptocast.ml.metrics import compute_metrics, qlike
from cryptocast.ml.registry import production_row, promote, register_candidate, rollback
from cryptocast.monitoring import _degrade_hourly, compute_drift, evaluate_rule, psi, rolling_quality
from tests.conftest import make_fng, make_klines, make_news


def test_qlike_is_zero_when_perfect():
    x = np.array([0.02, 0.05])
    assert abs(qlike(x, x)) < 1e-12


def test_qlike_penalizes_underestimate_more():
    y = np.array([0.04])
    assert qlike(y, y / 2) > qlike(y, y * 2)


def test_metrics_translate_to_percent():
    m = compute_metrics("log_qv", np.array([0.0]), np.array([np.log(1.2)]))
    assert np.isclose(m["mean_pct_error_volume"], 20.0)
    r = compute_metrics("log_rv", np.log([0.03]), np.log([0.04]))
    assert np.isclose(r["mae"], 1.0)  # один процентный пункт


def test_psi_identical_is_small_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.05
    assert psi(a, rng.normal(loc=2, size=5000)) > 0.25


def _quality_table(values):
    return pd.DataFrame({"target_date": pd.date_range("2024-01-01", periods=len(values)), "rolling_mae": values})


def test_retrain_rule_needs_consecutive_days(cfg):
    rule = evaluate_rule(_quality_table([0.1, 0.5, 0.1, 0.5, 0.5]), validation_mae=0.2, psi_share=0.0, cfg=cfg)
    assert rule["mae_rule"] is False
    rule = evaluate_rule(_quality_table([0.1, 0.5, 0.5, 0.5]), validation_mae=0.2, psi_share=0.0, cfg=cfg)
    assert rule["mae_rule"] is True and rule["retrain"] is True


def test_retrain_rule_psi(cfg):
    rule = evaluate_rule(_quality_table([0.1] * 5), validation_mae=0.2, psi_share=0.31, cfg=cfg)
    assert rule["psi_rule"] is True and rule["mae_rule"] is False


def test_rolling_quality(cfg):
    rows = pd.DataFrame({"target_date": pd.date_range("2024-01-01", periods=20), "symbol": "BTCUSDT", "actual_log": 1.0, "y_pred": 1.5})
    quality = rolling_quality(rows, cfg)
    assert quality["rolling_mae"].iloc[-1] == 0.5 and quality["rolling_mae"].iloc[0] != quality["rolling_mae"].iloc[0]


def test_degrade_only_after_start(cfg):
    klines = make_klines(cfg.symbols, days=20)
    start = pd.Timestamp("2024-01-10")
    changed = _degrade_hourly(klines, start, volume_factor=3.0, return_factor=2.0)
    before = changed[changed["open_time"] < start].sort_values(["symbol", "open_time"]).reset_index(drop=True)
    source = klines[klines["open_time"] < start].sort_values(["symbol", "open_time"]).reset_index(drop=True)
    assert np.allclose(before["quote_volume"], source["quote_volume"])
    assert (changed["high"] >= changed["low"]).all()


def test_compute_drift_flags(cfg):
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"f": rng.normal(size=2000), "g": rng.normal(size=2000)})
    cur = pd.DataFrame({"f": rng.normal(loc=3, size=200), "g": rng.normal(size=200)})
    drift = compute_drift(ref, cur, ["f", "g"], cfg).set_index("feature")
    assert drift.loc["f", "drifted"] and not drift.loc["g", "drifted"]


def _dataset(cfg) -> Dataset:
    klines, fng, news = make_klines(cfg.symbols), make_fng(), make_news()
    frame = features_from_raw(klines, fng, news, cfg)
    return Dataset(frame, news, "testversion")


def test_promotion_and_rollback(store):
    cfg = store.cfg
    data = _dataset(cfg)
    dev, holdout, folds = dev_holdout(data, cfg)
    spec = Spec(target="log_qv", model="lightgbm", params=cfg["models"]["lightgbm"], seed=cfg.seed)
    pipeline = FittedPipeline(spec, cfg).fit(dev[dev["target_log_qv"].notna()], data.news)
    hold = holdout[holdout["target_log_qv"].notna()]
    preds = pipeline.predict(hold, data.news)
    preds_frame = pd.DataFrame({"symbol": hold["symbol"].values, "date": hold["date"].values, "y_true": hold["target_log_qv"].values, "y_pred": preds, "fold": -1})
    metrics = compute_metrics("log_qv", preds_frame["y_true"], preds_frame["y_pred"])
    first = register_candidate(store, data, spec, "exp1", preds_frame.assign(symbol=preds_frame["symbol"]), preds_frame, metrics, metrics)
    result = promote(store, first, preds_frame)
    assert result["promoted"] is True
    assert production_row(store, "log_qv")["version_id"] == first

    # Более сильный кандидат занимает место текущей версии.
    better = preds_frame.assign(y_pred=preds_frame["y_true"] + 0.01)
    second = register_candidate(store, data, spec, "exp2", better, better, metrics, compute_metrics("log_qv", better["y_true"], better["y_pred"]))
    assert promote(store, second, better)["promoted"] is True
    assert production_row(store, "log_qv")["version_id"] == second

    # Более слабый кандидат отклоняется.
    worse = preds_frame.assign(y_pred=preds_frame["y_true"] + 2.0)
    third = register_candidate(store, data, spec, "exp3", worse, worse, metrics, compute_metrics("log_qv", worse["y_true"], worse["y_pred"]))
    assert promote(store, third, worse)["promoted"] is False

    # Откат возвращает предыдущую версию.
    assert rollback(store, "log_qv") == first
    assert production_row(store, "log_qv")["version_id"] == first


def test_reproduce_experiment_matches(store):
    cfg = store.cfg
    data = _dataset(cfg)
    data.version = "fixedv"
    frozen = cfg.artifacts_dir / "datasets" / data.version
    frozen.mkdir(parents=True)
    data.frame.to_parquet(frozen / "features.parquet", index=False)
    data.news.to_parquet(frozen / "news.parquet", index=False)
    reloaded = Dataset.load_frozen(cfg, "fixedv")
    version = reloaded.version
    (cfg.artifacts_dir / "datasets" / version).mkdir(exist_ok=True)
    reloaded.frame.to_parquet(cfg.artifacts_dir / "datasets" / version / "features.parquet", index=False)
    reloaded.news.to_parquet(cfg.artifacts_dir / "datasets" / version / "news.parquet", index=False)
    dev, holdout, folds = dev_holdout(reloaded, cfg)
    from cryptocast.ml.core import evaluate_spec

    spec = Spec(target="log_rv", model="lightgbm", params=cfg["models"]["lightgbm"], seed=cfg.seed)
    _, cv = evaluate_spec(spec, dev, reloaded.news, folds, cfg)
    target_col = "target_log_rv"
    pipeline = FittedPipeline(spec, cfg).fit(dev[dev[target_col].notna()], reloaded.news)
    hold = holdout[holdout[target_col].notna()]
    holdout_metrics = compute_metrics("log_rv", hold[target_col], pipeline.predict(hold, reloaded.news))
    exp = log_experiment(store, "evaluation", spec, version, cv, holdout_metrics)
    result = reproduce_experiment(store, exp)
    assert result["passed"], result


def test_evaluation_writes_results(store):
    cfg = store.cfg
    cfg.raw["models"]["mlp"]["max_iter"] = 5
    data = _dataset(cfg)
    result = run_evaluation(store, data)
    frame = result["metrics"]
    assert set(frame["split"]) == {"cv", "holdout"}
    assert {"naive", "elasticnet", "lightgbm", "mlp"} <= set(frame["model"])
    assert "qlike" in frame.columns
    assert (frame["symbol"] == "ALL").any()
    assert json.loads(store.scalar("SELECT cv_metrics FROM ml.experiments LIMIT 1"))
