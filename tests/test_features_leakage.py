"""Единство кода признаков и доказательства отсутствия утечки."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cryptocast.features import TARGET_COLUMNS, build_feature_frame, feature_columns
from cryptocast.leakage import features_from_raw, truncation_diff
from cryptocast.ml import core
from cryptocast.ml.core import FittedPipeline, Spec, walk_forward_folds
from cryptocast.staging import build_daily, build_hourly
from tests.conftest import make_fng, make_klines, make_news


def _inputs(cfg):
    return make_klines(cfg.symbols), make_fng(), make_news()


def test_truncation_invariance(cfg):
    klines, fng, news = _inputs(cfg)
    for cutoff in ("2024-02-10", "2024-03-15", "2024-04-30"):
        result = truncation_diff(klines, fng, news, cfg, pd.Timestamp(cutoff))
        assert result["rows"] > 0
        assert result["max_abs_diff"] < 1e-9, result
        assert result["nan_mismatches"] == 0, result
        assert result["last_day_target_is_nan"]


def test_truncation_detects_a_leak(cfg, monkeypatch):
    """Контроль самого теста: признак со сглаживанием по будущему обязан быть пойман."""
    klines, fng, news = _inputs(cfg)
    original = build_feature_frame

    def leaky(daily, fng_, news_, cfg_, include_targets=True):
        frame = original(daily, fng_, news_, cfg_, include_targets)
        frame["mkt_future"] = frame.groupby("symbol")["mkt_log_qv"].transform(lambda s: s.shift(-1).fillna(0))
        return frame

    monkeypatch.setattr("cryptocast.leakage.build_feature_frame", leaky)
    result = truncation_diff(klines, fng, news, cfg, pd.Timestamp("2024-03-15"))
    assert result["max_abs_diff"] > 1e-3 or result["nan_mismatches"] > 0


def test_targets_are_next_day(cfg):
    klines, fng, news = _inputs(cfg)
    frame = features_from_raw(klines, fng, news, cfg)
    btc = frame[frame["symbol"] == "BTCUSDT"].sort_values("date").reset_index(drop=True)
    assert np.allclose(btc["target_log_qv"].iloc[:-1].to_numpy(), btc["mkt_log_qv"].iloc[1:].to_numpy())
    assert np.isnan(btc["target_log_qv"].iloc[-1])


def test_features_do_not_contain_targets(cfg):
    klines, fng, news = _inputs(cfg)
    frame = features_from_raw(klines, fng, news, cfg)
    assert not set(TARGET_COLUMNS) & set(feature_columns(frame))


def test_training_and_inference_use_same_code(cfg):
    klines, fng, news = _inputs(cfg)
    hourly, _ = build_hourly(klines, cfg)
    daily = build_daily(hourly, cfg)
    train_frame = build_feature_frame(daily, fng, news, cfg, include_targets=True)
    infer_frame = build_feature_frame(daily, fng, news, cfg, include_targets=False)
    columns = feature_columns(train_frame)
    pd.testing.assert_frame_equal(train_frame[["symbol", "date"] + columns], infer_frame[["symbol", "date"] + columns])


def test_inference_on_truncated_equals_training_row(cfg):
    klines, fng, news = _inputs(cfg)
    full = features_from_raw(klines, fng, news, cfg)
    cutoff = pd.Timestamp("2024-04-10")
    partial = features_from_raw(klines[klines["open_time"] < cutoff + pd.Timedelta(days=1)], fng[fng["date"] <= cutoff], news[news["date"] <= cutoff], cfg)
    columns = feature_columns(full)
    a = full[full["date"] == cutoff].set_index("symbol")[columns].sort_index()
    b = partial[partial["date"] == cutoff].set_index("symbol")[columns].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False)


def _panel(cfg):
    klines, fng, news = _inputs(cfg)
    frame = features_from_raw(klines, fng, news, cfg)
    return frame[frame["target_log_qv"].notna()].reset_index(drop=True), news


def test_folds_are_ordered_and_have_gap(cfg):
    frame, _ = _panel(cfg)
    folds = walk_forward_folds(frame["date"], cfg)
    assert len(folds) >= 2
    for fold in folds:
        assert fold.test_start > fold.train_end + pd.Timedelta(days=cfg["validation"]["gap_days"])


def test_transforms_fit_only_on_train(cfg, monkeypatch):
    frame, news = _panel(cfg)
    folds = walk_forward_folds(frame["date"], cfg)
    seen: list[pd.Timestamp] = []
    original = core.TextSVD.fit

    def spy(self, data):
        seen.append(pd.Timestamp(data["date"].max()))
        return original(self, data)

    monkeypatch.setattr(core.TextSVD, "fit", spy)
    spec = Spec(target="log_qv", model="lightgbm", params=cfg["models"]["lightgbm"], seed=cfg.seed)
    for fold in folds:
        train = frame[frame["date"] <= fold.train_end]
        FittedPipeline(spec, cfg).fit(train, news)
        assert seen[-1] <= fold.train_end


def test_scaler_ignores_test_rows(cfg):
    frame, news = _panel(cfg)
    fold = walk_forward_folds(frame["date"], cfg)[0]
    train = frame[frame["date"] <= fold.train_end]
    test = frame[frame["date"] >= fold.test_start].copy()
    spec = Spec(target="log_qv", model="elasticnet", params={"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 2000}, seed=cfg.seed)
    pipeline = FittedPipeline(spec, cfg).fit(train, news)
    base = pipeline.predict(test, news)
    outlier = test.copy()
    outlier.iloc[0, outlier.columns.get_loc("mkt_log_qv")] = 1e6
    changed = pipeline.predict(outlier, news)
    assert np.allclose(base[1:], changed[1:])


def test_lightgbm_is_deterministic(cfg):
    frame, news = _panel(cfg)
    spec = Spec(target="log_qv", model="lightgbm", params=cfg["models"]["lightgbm"], seed=7)
    test_rows = frame.tail(20)
    a = FittedPipeline(spec, cfg).fit(frame, news).predict(test_rows, news)
    b = FittedPipeline(spec, cfg).fit(frame, news).predict(test_rows, news)
    assert np.array_equal(a, b)


def test_mlp_is_deterministic(cfg):
    frame, news = _panel(cfg)
    params = {**cfg["models"]["mlp"], "hidden_layer_sizes": [16]}
    spec = Spec(target="log_rv", model="mlp", params=params, seed=7)
    test_rows = frame.tail(20)
    a = FittedPipeline(spec, cfg).fit(frame, news).predict(test_rows, news)
    b = FittedPipeline(spec, cfg).fit(frame, news).predict(test_rows, news)
    assert np.array_equal(a, b)


def test_categorical_encodings(cfg):
    frame, news = _panel(cfg)
    for encoding in ("native", "onehot", "none"):
        spec = Spec(target="log_qv", model="lightgbm", params=cfg["models"]["lightgbm"], encoding=encoding, seed=1)
        preds = FittedPipeline(spec, cfg).fit(frame, news).predict(frame.tail(5), news)
        assert len(preds) == 5 and np.isfinite(preds).all()
