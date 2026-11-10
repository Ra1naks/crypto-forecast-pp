"""Мониторинг: дрейф (PSI, KS), качество на фактах, правило переобучения, симуляция деградации."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from cryptocast.config import Config
from cryptocast.features import build_feature_frame
from cryptocast.ml.core import FittedPipeline
from cryptocast.ml.experiments import Dataset
from cryptocast.ml.metrics import compute_metrics
from cryptocast.ml.registry import load_bundle, production_row
from cryptocast.staging import build_daily, build_hourly
from cryptocast.storage import Store, write_parquet
from cryptocast.utils import new_run_id, utcnow

log = logging.getLogger(__name__)

MONITORING_DDL = """
CREATE TABLE IF NOT EXISTS ops.drift_metrics (
    run_id VARCHAR, target VARCHAR, version_id VARCHAR, feature VARCHAR, psi DOUBLE, ks_stat DOUBLE, ks_pvalue DOUBLE,
    drifted BOOLEAN, reference_from DATE, current_from DATE, computed_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ops.retrain_decisions (
    run_id VARCHAR, target VARCHAR, version_id VARCHAR, decided_at TIMESTAMP, retrain BOOLEAN, on_cooldown BOOLEAN,
    mae_rule BOOLEAN, psi_rule BOOLEAN, rolling_mae DOUBLE, threshold DOUBLE, psi_share DOUBLE, reasons VARCHAR
);
"""


def ensure_monitoring_schema(store: Store) -> None:
    store.con.execute(MONITORING_DDL)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index по квантильным корзинам опорного периода."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < bins or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share, cur_share = np.clip(ref_share, 1e-4, None), np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], cfg: Config) -> pd.DataFrame:
    params = cfg["monitoring"]
    rows = []
    for feature in features:
        ref, cur = reference[feature].to_numpy(dtype=float), current[feature].to_numpy(dtype=float)
        value = psi(ref, cur, params["psi_bins"])
        ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
        ks = stats.ks_2samp(ref_ok, cur_ok) if len(ref_ok) and len(cur_ok) else None
        rows.append({
            "feature": feature, "psi": value,
            "ks_stat": float(ks.statistic) if ks else float("nan"),
            "ks_pvalue": float(ks.pvalue) if ks else float("nan"),
            "drifted": bool(value > params["psi_threshold"]),
        })
    return pd.DataFrame(rows)


def rolling_quality(actuals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Абсолютная ошибка лог-таргета по дням и скользящая MAE."""
    window = cfg["monitoring"]["retrain"]["rolling_window_days"]
    data = actuals.dropna(subset=["actual_log", "y_pred"]).copy()
    data["abs_error"] = (data["y_pred"] - data["actual_log"]).abs()
    daily = data.groupby("target_date", as_index=False)["abs_error"].mean().rename(columns={"abs_error": "mae_day"}).sort_values("target_date")
    daily["rolling_mae"] = daily["mae_day"].rolling(window, min_periods=window).mean()
    return daily


def evaluate_rule(quality: pd.DataFrame, validation_mae: float, psi_share: float, cfg: Config) -> dict:
    rule = cfg["monitoring"]["retrain"]
    threshold = validation_mae * rule["mae_ratio_threshold"]
    breaches = (quality["rolling_mae"] > threshold).astype(int)
    streak_ok = len(breaches) >= rule["consecutive_days"] and bool(breaches.tail(rule["consecutive_days"]).sum() == rule["consecutive_days"])
    psi_rule = psi_share >= rule["psi_share_threshold"]
    reasons = []
    last = float(quality["rolling_mae"].dropna().iloc[-1]) if quality["rolling_mae"].notna().any() else float("nan")
    if streak_ok:
        reasons.append(f"скользящая MAE {last:.4f} выше порога {threshold:.4f} {rule['consecutive_days']} дня подряд")
    if psi_rule:
        reasons.append(f"PSI > {cfg['monitoring']['psi_threshold']} у {psi_share:.0%} ключевых признаков")
    return {"mae_rule": streak_ok, "psi_rule": psi_rule, "rolling_mae": last, "threshold": threshold, "retrain": bool(streak_ok or psi_rule), "reasons": "; ".join(reasons)}


def run_monitoring(store: Store, dataset: Dataset, run_id: str) -> dict[str, dict]:
    """Считает дрейф и качество, применяет правило переобучения и правило автоотката."""
    cfg = store.cfg
    params = cfg["monitoring"]
    forecasts = store.df("SELECT * FROM ml.forecasts")
    if forecasts.empty:
        return {}
    daily = store.df("SELECT symbol, date, quote_volume, rv FROM staging.daily")
    daily["date"] = pd.to_datetime(daily["date"])
    forecasts["target_date"] = pd.to_datetime(forecasts["target_date"])
    decisions: dict[str, dict] = {}
    quality_frames, drift_frames = [], []
    for target in cfg.targets:
        row = production_row(store, target)
        if row is None:
            continue
        bundle = load_bundle(row)
        version = row["version_id"]
        reference = pd.read_parquet(f"{row['artifact_path']}/reference.parquet")
        cutoff = dataset.frame["date"].max() - pd.Timedelta(days=params["current_days"])
        current = dataset.frame[dataset.frame["date"] > cutoff]
        drift = compute_drift(reference, current, bundle["top_features"], cfg)
        drift_table = drift.assign(run_id=run_id, target=target, version_id=version, reference_from=pd.Timestamp(reference["date"].min()).date(),
                                   current_from=(cutoff + pd.Timedelta(days=1)).date(), computed_at=utcnow())
        drift_table = drift_table[["run_id", "target", "version_id", "feature", "psi", "ks_stat", "ks_pvalue", "drifted", "reference_from", "current_from", "computed_at"]]
        store.con.register("_d", drift_table)
        store.con.execute("INSERT INTO ops.drift_metrics SELECT * FROM _d")
        store.con.unregister("_d")
        drift_frames.append(drift.assign(target=target))
        psi_share = float(drift["drifted"].mean())

        production = forecasts[(forecasts["target"] == target) & (forecasts["version_id"] == version) & (forecasts["role"] == "production")]
        actual = daily.assign(actual_log=np.log(daily["quote_volume"] if target == "log_qv" else daily["rv"])).rename(columns={"date": "target_date"})
        joined = production.merge(actual[["symbol", "target_date", "actual_log"]], on=["symbol", "target_date"], how="left")
        quality = rolling_quality(joined, cfg)
        validation_mae = float(json.loads(row["metrics"])["cv"]["mae_log"])
        rule = evaluate_rule(quality, validation_mae, psi_share, cfg)
        quality_frames.append(quality.assign(target=target, version_id=version, validation_mae=validation_mae, threshold=rule["threshold"]))

        last_promoted = pd.Timestamp(row["promoted_at"])
        cooldown = (pd.Timestamp(utcnow()) - last_promoted) < pd.Timedelta(days=params["retrain"]["cooldown_days"])
        should_retrain = rule["retrain"] and not cooldown
        store.con.execute(
            "INSERT INTO ops.retrain_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [run_id, target, version, utcnow(), should_retrain, bool(cooldown), rule["mae_rule"], rule["psi_rule"], rule["rolling_mae"], rule["threshold"], psi_share, rule["reasons"]],
        )
        decisions[target] = {**rule, "on_cooldown": bool(cooldown), "retrain_now": should_retrain, "psi_share": psi_share}
        # Правило автоотката: новая версия на живых фактах заметно хуже теневой предыдущей.
        rollback_rule = params["rollback"]
        shadow = forecasts[(forecasts["target"] == target) & (forecasts["role"] == "shadow") & (forecasts["source"] == "live")]
        if not shadow.empty:
            joined_shadow = shadow.merge(actual[["symbol", "target_date", "actual_log"]], on=["symbol", "target_date"], how="left").dropna(subset=["actual_log"])
            live = joined[(joined["source"] == "live") & joined["actual_log"].notna()]
            common = live.merge(joined_shadow[["symbol", "target_date", "y_pred"]], on=["symbol", "target_date"], suffixes=("", "_shadow"))
            if common["target_date"].nunique() >= rollback_rule["min_live_days"]:
                live_mae = (common["y_pred"] - common["actual_log"]).abs().mean()
                shadow_mae = (common["y_pred_shadow"] - common["actual_log"]).abs().mean()
                decisions[target]["rollback"] = bool(live_mae > shadow_mae * rollback_rule["live_vs_shadow_ratio"])
    if quality_frames:
        write_parquet(cfg, "mart", "quality_daily", pd.concat(quality_frames, ignore_index=True))
    if drift_frames:
        write_parquet(cfg, "mart", "drift_latest", pd.concat(drift_frames, ignore_index=True))
    write_parquet(cfg, "mart", "drift_history", store.df("SELECT * FROM ops.drift_metrics"))
    write_parquet(cfg, "mart", "retrain_decisions", store.df("SELECT * FROM ops.retrain_decisions ORDER BY decided_at"))
    return decisions


# ---------------------------------- Симуляция деградации ----------------------------------
SIMULATION_NOTE = "СИМУЛЯЦИЯ: копия реальных данных с искусственным ухудшением. Не является фактом рынка."


def _degrade_hourly(klines: pd.DataFrame, start: pd.Timestamp, volume_factor: float, return_factor: float) -> pd.DataFrame:
    data = klines.sort_values(["symbol", "open_time"]).copy()
    mask = data["open_time"] >= start
    data.loc[mask, ["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]] = (
        data.loc[mask, ["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]].astype(float) * volume_factor
    )
    data["trades"] = data["trades"].astype("int64")
    pieces = []
    for symbol, part in data.groupby("symbol", sort=False):
        part = part.copy()
        logret = np.log(part["close"]).diff().fillna(0.0)
        boosted = logret.where(part["open_time"] < start, logret * return_factor)
        new_close = np.exp(np.log(part["close"].iloc[0]) + boosted.cumsum())
        ratio = new_close / part["close"]
        for column in ("open", "high", "low", "close"):
            part[column] = part[column] * ratio
        part[["high", "low"]] = np.column_stack([part[["open", "high", "low", "close"]].max(axis=1), part[["open", "high", "low", "close"]].min(axis=1)])
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def simulate_drift(store: Store, days: int = 45, volume_factor: float = 3.0, return_factor: float = 2.2) -> dict:
    """Показывает реакцию системы на ухудшение на КОПИИ данных; рабочее хранилище не меняется."""
    cfg = store.cfg
    rid = f"sim-{new_run_id()}"
    workspace = cfg.data_dir / "simulation" / rid
    workspace.mkdir(parents=True, exist_ok=True)
    klines = store.df("SELECT * FROM raw.klines ORDER BY symbol, open_time")
    news = store.df("SELECT * FROM staging.news")
    if not news.empty:
        news["date"] = pd.to_datetime(news["date"])
    fng = store.df("SELECT * FROM staging.fng")
    fng["date"] = pd.to_datetime(fng["date"])
    start = (klines["open_time"].max() - pd.Timedelta(days=days)).floor("D")
    klines.to_parquet(workspace / "SIMULATION_klines_original.parquet", index=False)
    degraded = _degrade_hourly(klines, start, volume_factor, return_factor)
    degraded.to_parquet(workspace / "SIMULATION_klines_degraded.parquet", index=False)
    hourly, _ = build_hourly(degraded, cfg)
    daily = build_daily(hourly, cfg)
    sim_frame = build_feature_frame(daily, fng, news, cfg)
    sim_daily = daily[["symbol", "date", "quote_volume", "rv"]].rename(columns={"date": "target_date"})
    quality_rows, trigger_rows = [], []
    summary: dict[str, dict] = {"simulation_id": rid, "note": SIMULATION_NOTE, "start": str(start.date()),
                                "volume_factor": volume_factor, "return_factor": return_factor, "targets": {}}
    params = cfg["monitoring"]
    for target in cfg.targets:
        row = production_row(store, target)
        if row is None:
            continue
        bundle = load_bundle(row)
        pipeline: FittedPipeline = bundle["pipeline"]
        version = row["version_id"]
        validation_mae = float(json.loads(row["metrics"])["cv"]["mae_log"])
        reference = pd.read_parquet(f"{row['artifact_path']}/reference.parquet")
        window = sim_frame[sim_frame["date"] >= start - pd.Timedelta(days=40)].copy()
        preds = pipeline.predict(window, news)
        result = pd.DataFrame({"symbol": window["symbol"].values, "origin_date": window["date"].values,
                               "target_date": window["date"].values + pd.Timedelta(days=1), "y_pred": preds, "target": target})
        result["target_date"] = pd.to_datetime(result["target_date"])
        source = sim_daily.assign(actual_log=np.log(sim_daily["quote_volume"] if target == "log_qv" else sim_daily["rv"]))
        joined = result.merge(source[["symbol", "target_date", "actual_log"]], on=["symbol", "target_date"], how="left")
        quality = rolling_quality(joined, cfg)
        rule_cfg = params["retrain"]
        threshold = validation_mae * rule_cfg["mae_ratio_threshold"]
        quality["threshold"] = threshold
        quality["breach"] = quality["rolling_mae"] > threshold
        quality["streak"] = quality["breach"].groupby((~quality["breach"]).cumsum()).cumsum()
        fired = quality[quality["streak"] >= rule_cfg["consecutive_days"]]
        trigger_date = pd.Timestamp(fired.iloc[0]["target_date"]) if not fired.empty else None
        cutoff = sim_frame["date"].max() - pd.Timedelta(days=params["current_days"])
        drift = compute_drift(reference, sim_frame[sim_frame["date"] > cutoff], bundle["top_features"], cfg)
        psi_share = float(drift["drifted"].mean())
        # Реакция: кандидат обучается на копии до момента срабатывания и сравнивается с текущей версией.
        decision = {"retrain_triggered": trigger_date is not None}
        retrained_preds = None
        if trigger_date is not None:
            train = sim_frame[(sim_frame["date"] < trigger_date) & sim_frame[f"target_{target}"].notna()]
            candidate = FittedPipeline(bundle["spec"], cfg).fit(train, news)
            after = sim_frame[(sim_frame["date"] >= trigger_date) & sim_frame[f"target_{target}"].notna()]
            if not after.empty:
                retrained_preds = pd.DataFrame({"symbol": after["symbol"].values, "origin_date": after["date"].values,
                                                "target_date": after["date"].values + pd.Timedelta(days=1), "y_pred": candidate.predict(after, news)})
                retrained_preds["target_date"] = pd.to_datetime(retrained_preds["target_date"])
                scored = retrained_preds.merge(source[["symbol", "target_date", "actual_log"]], on=["symbol", "target_date"]).dropna()
                old_after = joined.merge(scored[["symbol", "target_date"]], on=["symbol", "target_date"])
                candidate_metrics = compute_metrics(target, scored["actual_log"], scored["y_pred"])
                champion_metrics = compute_metrics(target, old_after["actual_log"], old_after["y_pred"])
                promoted = candidate_metrics["mae_log"] <= champion_metrics["mae_log"]
                decision.update(candidate_mae_log=candidate_metrics["mae_log"], champion_mae_log=champion_metrics["mae_log"], candidate_promoted_in_simulation=bool(promoted))
                retrained_daily = rolling_quality(retrained_preds.merge(source[["symbol", "target_date", "actual_log"]], on=["symbol", "target_date"]), cfg)
                quality = quality.merge(retrained_daily[["target_date", "mae_day", "rolling_mae"]].rename(columns={"mae_day": "mae_day_retrained", "rolling_mae": "rolling_mae_retrained"}), on="target_date", how="left")
        quality = quality.assign(target=target, simulation_id=rid, is_simulation=True, regime_start=start)
        quality_rows.append(quality)
        trigger_rows.append(drift.assign(target=target, simulation_id=rid, is_simulation=True))
        summary["targets"][target] = {"version": version, "validation_mae_log": validation_mae, "threshold": threshold,
                                      "first_trigger_date": str(trigger_date.date()) if trigger_date is not None else None,
                                      "max_rolling_mae": float(quality["rolling_mae"].max()), "psi_share": psi_share, **decision}
    write_parquet(cfg, "mart", "simulation_quality", pd.concat(quality_rows, ignore_index=True))
    write_parquet(cfg, "mart", "simulation_drift", pd.concat(trigger_rows, ignore_index=True))
    (workspace / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (cfg.lake_dir / "mart" / "simulation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for original in workspace.glob("SIMULATION_klines_original.parquet"):
        original.unlink()  # копия исходных свечей больше не нужна
    log.info("Симуляция %s завершена: %s", rid, summary["targets"])
    return summary
