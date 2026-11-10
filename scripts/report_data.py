"""Сбор фактов и графиков из реальных результатов конвейера для отчёта и презентации.

Ничего не вычисляется «вручную»: все числа читаются из data/lake/mart, DuckDB и artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cryptocast.config import load_config  # noqa: E402

CFG = load_config()
MART = CFG.lake_dir / "mart"
FIG = ROOT / "docs" / "figures"
FACT, FORECAST, ALT, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8981"
MODEL_COLORS = {"naive": "#8a8981", "naive_ma7": "#c3c2b7", "elasticnet": "#1baf7a", "lightgbm": "#2a78d6", "mlp": "#eb6834"}
MODEL_NAMES = {"naive": "Наивный (вчера)", "naive_ma7": "Среднее 7 дней", "elasticnet": "ElasticNet (HAR)", "lightgbm": "LightGBM", "mlp": "MLP"}
TARGET_NAMES = {"log_qv": "объём торгов", "log_rv": "волатильность"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe6", "axes.edgecolor": "#b5b4ad", "figure.dpi": 150})


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(MART / f"{name}.parquet")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect() -> dict:
    facts: dict = {}
    con = duckdb.connect(str(CFG.db_path), read_only=True)
    q = lambda sql: con.execute(sql).df()  # noqa: E731
    facts["counts"] = {
        "klines": int(q("SELECT count(*) n FROM raw.klines")["n"][0]),
        "klines_first": str(q("SELECT min(open_time) v FROM raw.klines")["v"][0]),
        "klines_last": str(q("SELECT max(open_time) v FROM raw.klines")["v"][0]),
        "fng": int(q("SELECT count(*) n FROM raw.fng")["n"][0]),
        "news_raw": int(q("SELECT count(*) n FROM raw.news")["n"][0]),
        "news_by_source": q("SELECT split_part(source, ':', 1) s, count(*) n FROM raw.news GROUP BY 1 ORDER BY 2 DESC").set_index("s")["n"].to_dict(),
        "news_clean": int(q("SELECT count(*) n FROM staging.news")["n"][0]),
        "news_first": str(q("SELECT min(date) v FROM staging.news")["v"][0])[:10],
        "daily": int(q("SELECT count(*) n FROM staging.daily")["n"][0]),
        "daily_invalid": int(q("SELECT count(*) n FROM staging.daily WHERE NOT is_valid")["n"][0]),
        "daily_scaled": int(q("SELECT count(*) n FROM staging.daily WHERE is_scaled")["n"][0]),
        "rejected": int(q("SELECT count(*) n FROM staging.hourly_rejected")["n"][0]),
        "features": int(q("SELECT count(*) n FROM information_schema.columns WHERE table_schema='mart' AND table_name='features' AND (column_name LIKE 'mkt_%' OR column_name LIKE 'cal_%' OR column_name LIKE 'fng_%' OR column_name LIKE 'txt_%')")["n"][0]),
        "feature_rows": int(q("SELECT count(*) n FROM mart.features")["n"][0]),
        "gaps": q("SELECT scope, value, details FROM ops.dq_results WHERE check_name='klines_gap_ratio' AND checked_at=(SELECT max(checked_at) FROM ops.dq_results)").to_dict("records"),
    }
    runs = q("SELECT run_id, min(started_at) s FROM ops.load_log GROUP BY 1 ORDER BY 2")
    facts["runs"] = []
    for run_id in runs["run_id"]:
        log = q(f"SELECT source, status, received, inserted, duplicates, error FROM ops.load_log WHERE run_id='{run_id}'")
        facts["runs"].append({"run_id": run_id, "inserted": int(log["inserted"].sum()), "received": int(log["received"].sum()),
                              "statuses": log["status"].value_counts().to_dict(), "log": log.to_dict("records")})
    facts["pipeline_steps"] = q("SELECT run_id, step, status, started_at, finished_at FROM ops.pipeline_runs ORDER BY started_at").assign(
        seconds=lambda d: (pd.to_datetime(d["finished_at"]) - pd.to_datetime(d["started_at"])).dt.total_seconds()).to_dict("records")
    facts["registry"] = q("SELECT version_id, target, model_name, stage, trained_until, data_version FROM ml.registry ORDER BY created_at").to_dict("records")
    facts["events"] = q("SELECT event_at, target, version_id, event FROM ml.registry_events ORDER BY event_at").to_dict("records")
    facts["experiments_count"] = int(q("SELECT count(*) n FROM ml.experiments")["n"][0])
    facts["experiments_kinds"] = q("SELECT kind, count(*) n FROM ml.experiments GROUP BY 1").set_index("kind")["n"].to_dict()
    facts["dq_last"] = q("SELECT check_name, scope, severity, passed, value, threshold FROM ops.dq_results WHERE checked_at=(SELECT max(checked_at) FROM ops.dq_results)").to_dict("records")
    facts["decisions_last"] = q("SELECT * FROM ops.retrain_decisions ORDER BY decided_at DESC LIMIT 2").to_dict("records")
    con.close()

    facts["metrics"] = read("metrics")
    facts["analysis"] = read("feature_analysis")
    facts["importance"] = read("importance_global")
    facts["stability"] = read("importance_stability")
    facts["decision"] = read("decision")
    facts["drift"] = read("drift_latest")
    facts["folds"] = read("folds")
    facts["status"] = read_json(MART / "status.json")
    facts["leakage"] = read_json(MART / "leakage_report.json")
    facts["simulation"] = read_json(MART / "simulation_summary.json")
    facts["tuned"] = read_json(CFG.artifacts_dir / "tuned_params.json")
    facts["trials"] = read("optuna_trials")
    reproductions = sorted((CFG.artifacts_dir / "reproductions").glob("*.json"))
    facts["reproductions"] = [read_json(p) for p in reproductions]
    verification = ROOT / "docs" / "verification.json"
    facts["verification"] = read_json(verification)
    return facts


def metric_value(m: pd.DataFrame, split: str, target: str, model: str, column: str, symbol: str = "ALL") -> float:
    row = m[(m["split"] == split) & (m["target"] == target) & (m["model"] == model) & (m["symbol"] == symbol)]
    return float(row.iloc[0][column]) if not row.empty else float("nan")


def make_figures(facts: dict) -> dict[str, Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    predictions = read("predictions")
    registry = pd.DataFrame(facts["registry"])

    # 1. Факт и прогноз на отложенной выборке
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    for ax, target in zip(axes, ("log_qv", "log_rv")):
        prod = registry[(registry["target"] == target) & (registry["stage"] == "production")]
        model = prod.iloc[0]["model_name"] if not prod.empty else "lightgbm"
        part = predictions[(predictions["target"] == target) & (predictions["model"] == model) & (predictions["symbol"] == "BTCUSDT") & (predictions["split"] == "holdout")].sort_values("date")
        naive = predictions[(predictions["target"] == target) & (predictions["model"] == "naive") & (predictions["symbol"] == "BTCUSDT") & (predictions["split"] == "holdout")].sort_values("date")
        scale = 1e-6 if target == "log_qv" else 100
        x = pd.to_datetime(part["date"]) + pd.Timedelta(days=1)
        ax.plot(x, np.exp(part["y_true"]) * scale, color=FACT, lw=2, label="Факт")
        ax.plot(x, np.exp(part["y_pred"]) * scale, color=FORECAST, lw=2, label=f"Прогноз {MODEL_NAMES[model]}")
        ax.plot(pd.to_datetime(naive["date"]) + pd.Timedelta(days=1), np.exp(naive["y_pred"]) * scale, color=MUTED, lw=1.2, ls="--", label="Наивный")
        ax.set_ylabel("млн USDT" if target == "log_qv" else "% за сутки")
        ax.set_title(f"BTC/USDT, {TARGET_NAMES[target]}: отложенная выборка", loc="left", fontsize=11)
        ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    out["fact_forecast"] = FIG / "fact_forecast.png"
    fig.savefig(out["fact_forecast"])
    plt.close(fig)

    # 2. Улучшение к наивному по инструментам (отложенная выборка)
    m = facts["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (target, col, label) in zip(axes, (("log_qv", "mae_improvement_pct", "MAE"), ("log_rv", "qlike_improvement_pct", "QLIKE"))):
        symbols = [s for s in CFG.symbols]
        width = 0.26
        for i, model in enumerate(("elasticnet", "lightgbm", "mlp")):
            vals = [metric_value(m, "holdout", target, model, col, s) for s in symbols]
            ax.bar(np.arange(len(symbols)) + (i - 1) * width, vals, width=width * 0.92, color=MODEL_COLORS[model], label=MODEL_NAMES[model])
        ax.axhline(0, color="#b5b4ad", lw=1)
        ax.set_xticks(np.arange(len(symbols)), [s.replace("USDT", "") for s in symbols])
        ax.set_title(f"{TARGET_NAMES[target].capitalize()}: улучшение {label} к наивному, %", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out["improvement"] = FIG / "improvement.png"
    fig.savefig(out["improvement"])
    plt.close(fig)

    # 3. Абляция
    a = facts["analysis"]
    ab = a[a["section"] == "ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, target in zip(axes, ("log_qv", "log_rv")):
        part = ab[ab["target"] == target]
        ax.barh(part["variant"], part["value"], color=FACT)
        lo, hi = part["value"].min(), part["value"].max()
        ax.set_xlim(lo - (hi - lo) * 3 - 1e-4, hi + (hi - lo) * 1.5 + 1e-4)
        for y, v in enumerate(part["value"]):
            ax.text(v, y, f" {v:.4f}", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{TARGET_NAMES[target].capitalize()}: {part['metric'].iloc[0].upper()}, меньше — лучше", loc="left", fontsize=9)
    fig.tight_layout()
    out["ablation"] = FIG / "ablation.png"
    fig.savefig(out["ablation"])
    plt.close(fig)

    # 4. Важность: SHAP и permutation
    imp = facts["importance"]
    from cryptocast.dashboard_names import human  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, target in zip(axes, ("log_qv", "log_rv")):
        part = imp[imp["target"] == target].sort_values("mean_abs_shap", ascending=False).head(10).iloc[::-1]
        y = np.arange(len(part))
        ax.barh(y + 0.2, part["mean_abs_shap"] / part["mean_abs_shap"].max(), height=0.38, color=FACT, label="SHAP")
        ax.barh(y - 0.2, part["increase_mae"].clip(lower=0) / max(part["increase_mae"].max(), 1e-12), height=0.38, color=FORECAST, label="Permutation")
        ax.set_yticks(y, [human(f) for f in part["feature"]], fontsize=8)
        ax.set_title(f"{TARGET_NAMES[target].capitalize()}: важность (доля от максимума)", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out["importance"] = FIG / "importance.png"
    fig.savefig(out["importance"])
    plt.close(fig)

    # 5. Мониторинг: качество на фактах
    quality = read("quality_daily")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, target in zip(axes, ("log_qv", "log_rv")):
        part = quality[quality["target"] == target]
        x = pd.to_datetime(part["target_date"])
        ax.plot(x, part["rolling_mae"], color=FACT, lw=2, label="Скользящая MAE, 14 дн.")
        ax.plot(x, part["threshold"], color=FORECAST, lw=1.5, ls="--", label="Порог 1,2 × валидационная")
        ax.plot(x, part["validation_mae"], color=MUTED, lw=1.2, ls=":", label="Валидационная MAE")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{TARGET_NAMES[target].capitalize()}: ошибка на поступающих фактах", loc="left", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    out["monitoring"] = FIG / "monitoring.png"
    fig.savefig(out["monitoring"])
    plt.close(fig)

    # 6. Симуляция деградации
    sim_path = MART / "simulation_quality.parquet"
    if sim_path.exists():
        sim = pd.read_parquet(sim_path)
        summary = facts["simulation"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, target in zip(axes, ("log_qv", "log_rv")):
            part = sim[sim["target"] == target]
            x = pd.to_datetime(part["target_date"])
            ax.plot(x, part["rolling_mae"], color=FORECAST, lw=2, label="Текущая модель")
            if "rolling_mae_retrained" in part:
                ax.plot(x, part["rolling_mae_retrained"], color=ALT, lw=2, label="После переобучения")
            ax.plot(x, part["threshold"], color=MUTED, lw=1.5, ls="--", label="Порог")
            ax.axvline(pd.Timestamp(summary["start"]), color=MUTED, ls=":", lw=1)
            trig = summary["targets"].get(target, {}).get("first_trigger_date")
            if trig:
                ax.axvline(pd.Timestamp(trig), color="#e34948", lw=1)
            ax.set_title(f"СИМУЛЯЦИЯ. {TARGET_NAMES[target].capitalize()}", loc="left", fontsize=10)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.set_ylim(bottom=0)
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        out["simulation"] = FIG / "simulation.png"
        fig.savefig(out["simulation"])
        plt.close(fig)
    return out
