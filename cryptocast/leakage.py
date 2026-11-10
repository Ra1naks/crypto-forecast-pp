"""Доказательства отсутствия утечки: усечение данных, метки времени, перемешанный таргет."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from cryptocast.config import Config
from cryptocast.features import KEY_COLUMNS, TARGET_COLUMNS, build_feature_frame, feature_columns
from cryptocast.ml.core import Spec, evaluate_spec
from cryptocast.ml.experiments import Dataset, dev_holdout, load_tuned, spec_for
from cryptocast.staging import build_daily, build_hourly
from cryptocast.storage import Store

log = logging.getLogger(__name__)


def features_from_raw(klines: pd.DataFrame, fng: pd.DataFrame, news: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Полный путь от сырых свечей до витрины (тот же код, что в конвейере)."""
    hourly, _ = build_hourly(klines, cfg)
    return build_feature_frame(build_daily(hourly, cfg), fng, news, cfg)


def truncation_diff(klines: pd.DataFrame, fng: pd.DataFrame, news: pd.DataFrame, cfg: Config, cutoff: pd.Timestamp,
                    full: pd.DataFrame | None = None) -> dict:
    """Признаки на дату t не меняются, если удалить все данные после t."""
    cutoff = pd.Timestamp(cutoff).normalize()
    end = cutoff + pd.Timedelta(days=1)
    full_frame = full if full is not None else features_from_raw(klines, fng, news, cfg)
    cut_news = news[news["date"] <= cutoff] if not news.empty else news
    cut = features_from_raw(klines[klines["open_time"] < end], fng[fng["date"] <= cutoff], cut_news, cfg)
    columns = feature_columns(full_frame)
    left = full_frame[full_frame["date"] <= cutoff].set_index(KEY_COLUMNS)[columns].sort_index()
    right = cut.set_index(KEY_COLUMNS)[columns].sort_index()
    left, right = left.align(right, join="inner", axis=0)
    diff = (left - right).abs().to_numpy(dtype=float)
    both_nan = left.isna().to_numpy() & right.isna().to_numpy()
    mismatch_nan = left.isna().to_numpy() ^ right.isna().to_numpy()
    diff[both_nan] = 0.0
    max_diff = float(np.nanmax(diff)) if diff.size else 0.0
    last_targets_nan = cut[cut["date"] == cutoff][TARGET_COLUMNS].isna().all().all()
    return {"cutoff": str(cutoff.date()), "rows": len(left), "max_abs_diff": max_diff, "nan_mismatches": int(mismatch_nan.sum()), "last_day_target_is_nan": bool(last_targets_nan)}


def timestamp_checks(store: Store) -> list[dict]:
    checks = []
    alignment = store.df(
        """WITH d AS (SELECT symbol, date, lead(date) OVER w AS next_date,
                       lead(ln(quote_volume)) OVER w AS next_lqv FROM staging.daily WINDOW w AS (PARTITION BY symbol ORDER BY date))
           SELECT count(*) AS n, sum(CASE WHEN abs(f.target_log_qv - d.next_lqv) > 1e-9 THEN 1 ELSE 0 END) AS bad
           FROM mart.features f JOIN d ON f.symbol = d.symbol AND f.date = d.date
           WHERE f.target_log_qv IS NOT NULL AND d.next_date = d.date + INTERVAL 1 DAY"""
    ).iloc[0]
    checks.append({"check": "target_is_next_day", "passed": int(alignment["bad"]) == 0, "value": int(alignment["bad"]), "details": f"строк: {int(alignment['n'])}"})
    news_max = store.scalar("SELECT count(*) FROM staging.news WHERE date > seen_at OR date + INTERVAL 1 DAY <= seen_at")
    checks.append({"check": "news_date_contains_seen_at", "passed": int(news_max) == 0, "value": int(news_max), "details": "дата статьи — сутки публикации"})
    last_day_candles = store.scalar(
        """SELECT count(*) FROM mart.features f JOIN staging.hourly h ON f.symbol = h.symbol AND h.date = f.date
           WHERE h.open_time >= f.date + INTERVAL 1 DAY"""
    )
    checks.append({"check": "candles_within_feature_date", "passed": int(last_day_candles) == 0, "value": int(last_day_candles), "details": "свечи агрегируются только внутри суток признака"})
    incomplete = store.scalar(
        """SELECT count(*) FROM staging.daily d WHERE d.date + INTERVAL 23 HOUR >
           (SELECT max(open_time) FROM staging.hourly h WHERE h.symbol = d.symbol)"""
    )
    checks.append({"check": "no_incomplete_day_in_mart", "passed": int(incomplete) == 0, "value": int(incomplete), "details": "незавершённый день не попадает в витрину"})
    return checks


def shuffled_target_test(store: Store, dataset: Dataset, tuned_params: dict | None = None) -> dict:
    """После перемешивания таргета модель не должна иметь предсказательной силы (R²≈0)."""
    cfg = store.cfg
    dev, _, folds = dev_holdout(dataset, cfg)
    tuned = tuned_params if tuned_params is not None else load_tuned(cfg)
    rng = np.random.default_rng(cfg.seed)
    result = {}
    for target in cfg.targets:
        col = f"target_{target}"
        data = dev[dev[col].notna()].copy()
        spec: Spec = spec_for(cfg, target, "lightgbm", tuned.get(target, {}).get("lightgbm"))
        real_oos, real_metrics = evaluate_spec(spec, data, dataset.news, folds, cfg)
        # Глобальная перестановка делает таргет независимым от признаков (и от инструмента, и от даты).
        shuffled = data.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        shuf_oos, shuf_metrics = evaluate_spec(spec, shuffled, dataset.news, folds, cfg)
        r2 = 1 - np.mean((shuf_oos["y_true"] - shuf_oos["y_pred"]) ** 2) / np.var(shuf_oos["y_true"])
        result[target] = {
            "mae_log_real": real_metrics["mae_log"], "mae_log_shuffled": shuf_metrics["mae_log"],
            "r2_shuffled": float(r2), "passed": bool(r2 < 0.05 and shuf_metrics["mae_log"] > real_metrics["mae_log"]),
        }
    return result


def run_leakage_report(store: Store, dataset: Dataset, shuffled: bool = True, n_cutoffs: int = 6) -> dict:
    """Отчёт о проверках; результат доступен на дашборде и в файле."""
    cfg = store.cfg
    klines = store.df("SELECT * FROM raw.klines ORDER BY symbol, open_time")
    news = store.df("SELECT * FROM staging.news")
    if not news.empty:
        news["date"] = pd.to_datetime(news["date"])
    fng = store.df("SELECT * FROM staging.fng")
    fng["date"] = pd.to_datetime(fng["date"])
    full = features_from_raw(klines, fng, news, cfg)
    dates = pd.Series(sorted(full["date"].unique()))
    picks = dates.iloc[np.linspace(60, len(dates) - 2, n_cutoffs).astype(int)]
    truncation = [truncation_diff(klines, fng, news, cfg, d, full) for d in picks]
    report = {
        "truncation": truncation,
        "truncation_passed": all(t["max_abs_diff"] < 1e-9 and t["nan_mismatches"] == 0 and t["last_day_target_is_nan"] for t in truncation),
        "timestamps": timestamp_checks(store),
    }
    if shuffled:
        report["shuffled_target"] = shuffled_target_test(store, dataset)
    report["passed"] = report["truncation_passed"] and all(c["passed"] for c in report["timestamps"]) and all(v["passed"] for v in report.get("shuffled_target", {}).values())
    path = cfg.lake_dir / "mart" / "leakage_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return report

