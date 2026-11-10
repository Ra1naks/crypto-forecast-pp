"""Метрики качества. Таргеты моделируются в логарифмах, RV оценивается в исходном масштабе."""
from __future__ import annotations

import numpy as np
import pandas as pd


def qlike(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    """QLIKE на дисперсиях: штрафует недооценку риска сильнее, чем переоценку."""
    ratio = (rv_true**2) / np.maximum(rv_pred**2, 1e-12)
    return float(np.mean(ratio - np.log(np.maximum(ratio, 1e-12)) - 1))


def compute_metrics(target: str, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """Единый набор метрик; mae/rmse для log_qv — в логарифмах, для log_rv — в п.п. волатильности."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    errors = y_pred_log - y_true_log
    result = {
        "n": int(len(errors)),
        "mae_log": float(np.mean(np.abs(errors))),
        "rmse_log": float(np.sqrt(np.mean(errors**2))),
        "bias_log": float(np.mean(errors)),
    }
    if target == "log_qv":
        relative = np.abs(np.expm1(errors))
        result.update(
            mae=result["mae_log"],
            rmse=result["rmse_log"],
            mean_pct_error_volume=float(relative.mean() * 100),
            median_pct_error_volume=float(np.median(relative) * 100),
        )
    else:
        rv_true, rv_pred = np.exp(y_true_log), np.exp(y_pred_log)
        result.update(
            mae=float(np.mean(np.abs(rv_pred - rv_true)) * 100),
            rmse=float(np.sqrt(np.mean((rv_pred - rv_true) ** 2)) * 100),
            qlike=qlike(rv_true, rv_pred),
            mean_pct_error_vol=float(np.mean(np.abs(rv_pred / rv_true - 1)) * 100),
        )
    return result


def metrics_frame(oos: pd.DataFrame, target: str) -> pd.DataFrame:
    """Метрики по моделям в целом и по инструментам (symbol='ALL' для всех сразу)."""
    rows: list[dict] = []
    for (model, symbol), part in pd.concat([oos, oos.assign(symbol="ALL")]).groupby(["model", "symbol"], sort=True):
        rows.append({"model": model, "symbol": symbol, "target": target, **compute_metrics(target, part["y_true"], part["y_pred"])})
    return pd.DataFrame(rows)


def add_improvement(frame: pd.DataFrame, baseline: str = "naive") -> pd.DataFrame:
    """Улучшение относительно наивного прогноза в % (положительное значит лучше)."""
    data = frame.copy()
    for metric in ("mae", "rmse", "qlike"):
        if metric not in data:
            continue
        base = data[data["model"] == baseline].set_index(["target", "symbol"])[metric]
        keys = pd.MultiIndex.from_frame(data[["target", "symbol"]])
        data[f"{metric}_improvement_pct"] = (1 - data[metric].values / base.reindex(keys).values) * 100
    return data
