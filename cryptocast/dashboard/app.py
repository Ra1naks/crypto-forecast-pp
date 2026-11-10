"""Дашборд cryptocast. Каждый элемент отвечает на вопрос из docs/dashboard_questions.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cryptocast.config import load_config  # noqa: E402
from streamlit_autorefresh import st_autorefresh  # noqa: E402

FACT, FORECAST, BAND, ALT, MUTED = "#2a78d6", "#eb6834", "rgba(235,104,52,0.15)", "#1baf7a", "#8a8981"
MODEL_COLORS = {"naive": "#8a8981", "naive_ma7": "#c3c2b7", "elasticnet": "#1baf7a", "lightgbm": "#2a78d6", "mlp": "#eb6834"}
MODEL_NAMES = {"naive": "Наивный (вчера)", "naive_ma7": "Среднее за 7 дней", "elasticnet": "ElasticNet (HAR)", "lightgbm": "LightGBM", "mlp": "MLP"}
TARGET_NAMES = {"log_qv": "Объём торгов", "log_rv": "Волатильность"}
FEATURE_NAMES = {
    "mkt_log_qv": "объём сегодня (лог)", "mkt_log_rv": "волатильность сегодня (лог)",
    "mkt_log_qv_ma7": "средний объём 7д (лог)", "mkt_log_qv_ma30": "средний объём 30д (лог)",
    "mkt_log_rv_ma7": "средняя волатильность 7д (лог)", "mkt_log_rv_ma30": "средняя волатильность 30д (лог)",
    "mkt_log_qv_vs_ma30": "объём относительно 30д", "mkt_log_rv_vs_ma30": "волатильность относительно 30д",
    "mkt_log_qv_vs_ma7": "объём относительно 7д", "mkt_log_rv_vs_ma7": "волатильность относительно 7д",
    "mkt_log_trades": "число сделок (лог)", "mkt_taker_buy_ratio": "доля покупок тейкеров",
    "mkt_abs_return_1d": "модуль доходности за день", "mkt_range_log": "дневной диапазон (лог)",
    "mkt_parkinson": "волатильность Паркинсона", "mkt_max_abs_ret": "макс. часовая доходность",
    "mkt_rs_pos_share": "доля положительной вариации", "mkt_volume_hhi": "концентрация объёма в часах",
    "mkt_last4h_share": "доля объёма за последние 4 ч", "mkt_btc_log_qv": "объём BTC (лог)",
    "mkt_btc_log_rv": "волатильность BTC (лог)", "mkt_btc_return_1d": "доходность BTC",
    "mkt_cross_rv_mean": "средняя волатильность рынка", "mkt_cross_qv_mean": "средний объём рынка",
    "mkt_rv_vs_cross": "волатильность относительно рынка", "cal_is_weekend": "прогноз на выходной день",
    "cal_dow": "день недели прогноза", "fng_value": "индекс страха и жадности", "fng_change_7d": "индекс: изменение 7д",
    "fng_ma7": "индекс: среднее 7д", "fng_z30": "индекс: отклонение от 30д", "txt_sent_mean": "тональность новостей",
    "txt_log_n_articles": "число новостей (лог)", "txt_topic_hack_share": "доля новостей о взломах",
    "txt_topic_regulation_share": "доля новостей о регулировании", "txt_topic_etf_share": "доля новостей об ETF",
    "txt_mention_share": "доля упоминаний инструмента", "txt_available": "есть текстовые данные",
    "symbol": "инструмент", "cal_dow_sin": "день недели прогноза (sin)", "cal_dow_cos": "день недели прогноза (cos)",
    "cal_month_sin": "месяц (sin)", "cal_month_cos": "месяц (cos)", "cal_is_month_end": "конец месяца",
    "mkt_return_1d": "доходность за день", "mkt_return_3d": "доходность за 3 дня", "mkt_return_7d": "доходность за 7 дней",
    "mkt_return_30d": "доходность за 30 дней", "mkt_log_trades_ma7": "число сделок, среднее 7д (лог)",
    "mkt_taker_ratio_ma7": "доля покупок тейкеров, среднее 7д", "mkt_n_anomalies": "часов с аномальным объёмом",
    "txt_neg_share": "доля негативных заголовков", "txt_sent_ma7": "тональность, среднее 7д",
    "txt_mention_sentiment": "тональность упоминаний инструмента", "txt_sent_std": "разброс тональности",
}


def human(feature: str) -> str:
    if feature in FEATURE_NAMES:
        return FEATURE_NAMES[feature]
    for suffix, label in (("_lag", "лаг"), ("_ma", "среднее"), ("_std", "ст. откл.")):
        if suffix in feature:
            base, _, n = feature.rpartition(suffix)
            if n.isdigit() and base in FEATURE_NAMES:
                return f"{FEATURE_NAMES[base]} ({label} {n}д)"
    return feature


st.set_page_config(page_title="cryptocast", page_icon="📈", layout="wide")
CFG = load_config()
MART = CFG.lake_dir / "mart"
st_autorefresh(interval=CFG["dashboard"]["autorefresh_seconds"] * 1000, key="autorefresh")


@st.cache_data(ttl=CFG["dashboard"]["autorefresh_seconds"], show_spinner=False)
def _read(name: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(MART / f"{name}.parquet")


def load(name: str) -> pd.DataFrame:
    path = MART / f"{name}.parquet"
    return _read(name, path.stat().st_mtime) if path.exists() else pd.DataFrame()


def load_json(name: str) -> dict:
    path = MART / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def money(value: float) -> str:
    if value >= 1e9:
        return f"${value / 1e9:,.2f} млрд"
    if value >= 1e6:
        return f"${value / 1e6:,.1f} млн"
    return f"${value:,.0f}"


def style_axes(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height, paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10), legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor="#d8d7d1")
    fig.update_yaxes(gridcolor="#ecebe6", zeroline=False)
    return fig


def draw(fig: go.Figure) -> None:
    st.plotly_chart(fig, config={"displaylogo": False}, width="stretch")


status = load_json("status")
decision = load("decision")
st.title("cryptocast: объём и волатильность на завтра")
st.caption("Шаг прогноза: сутки UTC. Инструменты к USDT на Binance. Информационный инструмент для контроля риска, не торговый совет.")

if not status:
    st.error("Витрина ещё не построена. Выполните: python -m cryptocast run-all")
    st.stop()

with st.sidebar:
    st.header("Фильтры")
    symbol = st.selectbox("Инструмент", CFG.symbols, format_func=lambda s: s.replace("USDT", "/USDT"))
    target = st.radio("Показатель", CFG.targets, format_func=lambda t: TARGET_NAMES[t])
    last_date = pd.Timestamp(status.get("last_feature_date", pd.Timestamp.utcnow().date()))
    preset = st.radio("Период", ["90 дней", "180 дней", "1 год", "Весь бэктест"], index=1)
    span = {"90 дней": 90, "180 дней": 180, "1 год": 365, "Весь бэктест": 4000}[preset]
    period_start = last_date - pd.Timedelta(days=span)
    st.divider()
    st.caption(f"Последняя свеча: {status.get('last_candle')} UTC")
    st.caption(f"Последний запуск: {status.get('finished_at', '')[:19]} UTC")
    st.caption(f"Автообновление: {CFG['dashboard']['autorefresh_seconds']} с")

if status.get("status") == "blocked_by_quality":
    st.error("Контроль качества выявил критичный сбой. Витрина не обновлена, показаны последние проверенные данные.")

tab_decision, tab_series, tab_quality, tab_explain, tab_monitor, tab_exp, tab_data = st.tabs(
    ["Решение на завтра", "Факт и прогноз", "Качество моделей", "Объяснение", "Мониторинг", "Эксперименты", "Данные"]
)

# ------------------------------------- Решение на завтра --------------------------------
with tab_decision:
    if decision.empty:
        st.warning("Прогнозов ещё нет.")
    else:
        row = decision[decision["symbol"] == symbol].iloc[0]
        target_day = pd.Timestamp(row["target_date"]).date()
        st.subheader(f"{symbol.replace('USDT', '/USDT')}: прогноз на {target_day} UTC")
        if target_day <= pd.Timestamp.utcnow().date():
            st.caption("Прогноз построен по последним закрытым суткам; целевые сутки уже идут. При запуске по расписанию в 00:20 UTC это прогноз на завтра.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ожидаемый объём", money(row["forecast_volume_usd"]), help="Прогноз суточного объёма в USDT")
        c1.caption(f"Диапазон 80%: {money(row['volume_low_usd'])} – {money(row['volume_high_usd'])}")
        c2.metric(f"Условная ёмкость исполнения ({row['participation_rate']:.0%} объёма)", money(row["capacity_usd"]),
                  help="Нижняя граница прогноза объёма × допустимая доля участия из config.yaml")
        c2.caption("Рассчитана от нижней границы диапазона: консервативная оценка.")
        c3.metric("Ожидаемая волатильность за сутки", f"{row['forecast_rv_pct']:.2f}%")
        c3.caption(f"Диапазон 80%: {row['rv_low_pct']:.2f}% – {row['rv_high_pct']:.2f}%")
        level = row["risk_level"]
        text = f"Уровень риска: **{level}**. Прогноз волатильности выше {row['rv_percentile_365d']:.0%} значений последнего года."
        {"высокий": st.error, "повышенный": st.warning, "обычный": st.success}[level](text)
        st.markdown("##### Сравнение инструментов на завтра")
        table = pd.DataFrame({
            "Инструмент": decision["symbol"],
            "Объём, прогноз": decision["forecast_volume_usd"].map(money),
            "Ёмкость исполнения": decision["capacity_usd"].map(money),
            "Волатильность, %": decision["forecast_rv_pct"].round(2),
            "Перцентиль за год": (decision["rv_percentile_365d"] * 100).round(0).astype(int).astype(str) + "%",
            "Риск": decision["risk_level"],
        })
        st.dataframe(table, hide_index=True, width="stretch")
        st.info("Показатели помогают оценить масштаб операций и риск, но не являются рекомендацией покупать или продавать.")

# ------------------------------------- Факт и прогноз ------------------------------------
forecasts = load("forecasts")
predictions = load("predictions")
registry = load("registry")
with tab_series:
    current = registry[(registry["target"] == target) & (registry["stage"] == "production")] if not registry.empty else pd.DataFrame()
    if current.empty:
        st.warning("Нет production-версии модели.")
    else:
        model_name = current.iloc[0]["model_name"]
        st.subheader(f"{TARGET_NAMES[target]}: факт и прогноз, модель {MODEL_NAMES.get(model_name, model_name)}")
        multiplier = 100.0 if target == "log_rv" else 1 / 1e6
        unit = "% за сутки" if target == "log_rv" else "млн USDT"
        hist = predictions[(predictions["target"] == target) & (predictions["model"] == model_name) & (predictions["symbol"] == symbol)].copy()
        hist["date"] = pd.to_datetime(hist["date"]) + pd.Timedelta(days=1)
        hist = hist[hist["date"] >= period_start].sort_values("date")
        hist["fact"] = np.exp(hist["y_true"]) * multiplier
        hist["forecast"] = np.exp(hist["y_pred"]) * multiplier
        if not hist.empty:
            hist = hist.set_index("date").reindex(pd.date_range(hist["date"].min(), hist["date"].max(), freq="D")).rename_axis("date").reset_index()
        live = forecasts[(forecasts["target"] == target) & (forecasts["symbol"] == symbol) & (forecasts["role"] == "production") & (forecasts["source"] == "live")].sort_values("target_date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["fact"], name="Факт", mode="lines", line=dict(color=FACT, width=2), connectgaps=False))
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["forecast"], name="Прогноз (вне обучения)", mode="lines", line=dict(color=FORECAST, width=2), connectgaps=False))
        if not live.empty:
            fig.add_trace(go.Scatter(x=live["target_date"], y=live["forecast_value"] * multiplier, name="Прогноз production", mode="markers", marker=dict(size=10, color=FORECAST, symbol="diamond", line=dict(color="#fcfcfb", width=2))))
        holdout_start = hist[hist["split"] == "holdout"]["date"].min()
        if pd.notna(holdout_start):
            fig.add_vline(x=holdout_start, line=dict(color=MUTED, dash="dot", width=1))
            fig.add_annotation(x=holdout_start, y=1.0, yref="paper", text="отложенная выборка", showarrow=False, xanchor="left", font=dict(color="#52514e", size=11))
        fig.update_yaxes(title=unit)
        draw(style_axes(fig))
        if not hist.empty:
            recent = hist.dropna(subset=["fact", "forecast"]).tail(14)
            if target == "log_qv":
                err = (recent["forecast"] / recent["fact"] - 1).abs().mean() * 100
                st.metric("Средняя ошибка за последние 14 дней", f"{err:.1f}% от объёма")
            else:
                err = (recent["forecast"] - recent["fact"]).abs().mean()
                st.metric("Средняя ошибка за последние 14 дней", f"{err:.2f} п.п. волатильности")

# ------------------------------------- Качество моделей ----------------------------------
metrics = load("metrics")
with tab_quality:
    st.subheader("Насколько можно доверять прогнозу?")
    if metrics.empty:
        st.warning("Метрики ещё не рассчитаны.")
    else:
        split = st.radio("Период оценки", ["cv", "holdout"], horizontal=True, format_func=lambda x: {"cv": "Walk-forward (разработка)", "holdout": "Отложенная выборка, 90 дней"}[x])
        scope = st.radio("Разрез", ["Выбранный инструмент", "Все инструменты вместе"], horizontal=True)
        sym = symbol if scope == "Выбранный инструмент" else "ALL"
        board = metrics[(metrics["target"] == target) & (metrics["split"] == split) & (metrics["symbol"] == sym)].copy()
        board = board.sort_values("mae")
        names = board["model"].map(MODEL_NAMES)
        if target == "log_qv":
            table = pd.DataFrame({"Модель": names, "Средняя ошибка, % от объёма": board["mean_pct_error_volume"].round(1), "Медианная ошибка, %": board["median_pct_error_volume"].round(1),
                                  "MAE (лог)": board["mae"].round(4), "RMSE (лог)": board["rmse"].round(4), "Улучшение MAE к наивному, %": board["mae_improvement_pct"].round(1)})
            improvement_col, metric_label = "mae_improvement_pct", "MAE"
        else:
            table = pd.DataFrame({"Модель": names, "MAE, п.п. волатильности": board["mae"].round(3), "RMSE, п.п.": board["rmse"].round(3), "QLIKE": board["qlike"].round(4),
                                  "Улучшение MAE к наивному, %": board["mae_improvement_pct"].round(1), "Улучшение QLIKE, %": board["qlike_improvement_pct"].round(1)})
            improvement_col, metric_label = "qlike_improvement_pct", "QLIKE"
        st.dataframe(table, hide_index=True, width="stretch")
        all_symbols = metrics[(metrics["target"] == target) & (metrics["split"] == split) & (metrics["model"] != "naive") & (metrics["symbol"] != "ALL")]
        fig = go.Figure()
        for model in ("elasticnet", "lightgbm", "mlp"):
            part = all_symbols[all_symbols["model"] == model].sort_values("symbol")
            fig.add_trace(go.Bar(x=part["symbol"].str.replace("USDT", ""), y=part[improvement_col], name=MODEL_NAMES[model], marker=dict(color=MODEL_COLORS[model], line=dict(color="#fcfcfb", width=2), cornerradius=4)))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title=f"Улучшение {metric_label} к наивному, %", zeroline=True, zerolinecolor="#b5b4ad")
        draw(style_axes(fig, f"По инструментам: выигрыш над наивным прогнозом ({metric_label})", 340))
        best = board[~board["model"].isin(["naive", "naive_ma7"])].sort_values("mae").head(1)
        if not best.empty:
            value = float(best.iloc[0]["mae_improvement_pct"])
            direction = "лучше" if value > 0 else "хуже"
            st.markdown(f"Лучшая модель здесь: **{MODEL_NAMES[best.iloc[0]['model']]}**. По MAE она на **{abs(value):.1f}%** {direction} наивного прогноза.")

# ------------------------------------- Объяснение ----------------------------------------
with tab_explain:
    st.subheader("Почему модель дала такой прогноз?")
    shap_frame = load(f"shap_{target}")
    if shap_frame.empty:
        st.warning("Объяснения ещё не рассчитаны: python -m cryptocast interpret")
    else:
        shap_frame["date"] = pd.to_datetime(shap_frame["date"])
        production = registry[(registry["target"] == target) & (registry["stage"] == "production")]
        if not production.empty and production.iloc[0]["model_name"] != "lightgbm":
            st.info(f"В эксплуатации для этого показателя модель {MODEL_NAMES[production.iloc[0]['model_name']]}; ниже объяснения LightGBM, обученного тем же walk-forward. Для объяснения нужна модель деревьев.")
        col_a, col_b = st.columns(2)
        sym_choice = col_a.selectbox("Инструмент для объяснения", CFG.symbols, index=CFG.symbols.index(symbol), key="explain_symbol")
        available = sorted(shap_frame[shap_frame["symbol"] == sym_choice]["date"].dt.date.unique())
        chosen = col_b.selectbox("Дата прогноза (конец суток)", available, index=len(available) - 1, key="explain_date")
        row = shap_frame[(shap_frame["symbol"] == sym_choice) & (shap_frame["date"].dt.date == chosen)].iloc[0]
        contrib = pd.DataFrame({"feature": [c.replace("shap__", "") for c in shap_frame.columns if c.startswith("shap__")],
                                "shap": [row[c] for c in shap_frame.columns if c.startswith("shap__")]}).dropna()
        contrib["value"] = contrib["feature"].map(lambda f: row.get(f"value__{f}", np.nan))
        contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index)
        top = contrib.head(12).iloc[::-1]
        scale = "лог. объёма" if target == "log_qv" else "лог. волатильности"
        base, pred, true_value = float(row["base_value"]), float(row["prediction"]), float(row["y_true"])
        def natural(value: float) -> str:
            return money(float(np.exp(value))) if target == "log_qv" else f"{np.exp(value) * 100:.2f}%"
        m1, m2, m3 = st.columns(3)
        m1.metric("Среднее модели (база)", natural(base), help=f"в логарифмах: {base:.3f}")
        m2.metric("Прогноз", natural(pred), help=f"база + сумма вкладов всех признаков; в логарифмах {pred:.3f}")
        m3.metric("Факт на следующий день", natural(true_value) if np.isfinite(true_value) else "ещё нет", help=f"в логарифмах: {true_value:.3f}")
        st.caption("Вклад +0.10 в логарифме означает примерно +10,5% к прогнозу, вклад −0.10 — примерно −9,5%.")
        fig = go.Figure(go.Bar(y=top["feature"].map(human), x=top["shap"], orientation="h",
                               marker=dict(color=[FACT if v > 0 else FORECAST for v in top["shap"]], cornerradius=4),
                               customdata=top["value"], hovertemplate="%{y}: вклад %{x:+.3f}, значение %{customdata:.4f}<extra></extra>"))
        fig.update_xaxes(title=f"Вклад в прогноз ({scale}); синий повышает, оранжевый понижает", zeroline=True, zerolinecolor="#b5b4ad")
        fig.update_layout(hovermode="closest")
        draw(style_axes(fig, "12 признаков с наибольшим вкладом", 440))
        with st.expander("Все признаки: значение и вклад"):
            st.dataframe(contrib.assign(Признак=contrib["feature"].map(human)).rename(columns={"value": "Значение", "shap": "Вклад"})[["Признак", "Значение", "Вклад"]], hide_index=True, width="stretch")

        st.markdown("##### Что важно для модели в целом: два независимых способа")
        global_importance = load("importance_global")
        part = global_importance[global_importance["target"] == target].copy()
        part["shap_rank_value"] = part["mean_abs_shap"] / part["mean_abs_shap"].max()
        part["perm_rank_value"] = part["increase_mae"].clip(lower=0) / part["increase_mae"].max()
        top_features = part.sort_values("mean_abs_shap", ascending=False).head(12).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top_features["feature"].map(human), x=top_features["shap_rank_value"], name="SHAP (доля от максимума)", orientation="h", marker=dict(color=FACT, line=dict(color="#fcfcfb", width=2), cornerradius=4)))
        fig.add_trace(go.Bar(y=top_features["feature"].map(human), x=top_features["perm_rank_value"], name="Permutation (доля от максимума)", orientation="h", marker=dict(color=FORECAST, line=dict(color="#fcfcfb", width=2), cornerradius=4)))
        fig.update_layout(barmode="group", hovermode="closest")
        draw(style_axes(fig, "Важность признаков: SHAP и перестановка", 460))

        st.markdown("##### Как прогноз зависит от отдельного признака (PDP и ICE)")
        pdp = load("pdp_ice")
        pdp_part = pdp[pdp["target"] == target]
        feature = st.selectbox("Признак", sorted(pdp_part["feature"].unique()), format_func=human)
        curve = pdp_part[pdp_part["feature"] == feature]
        fig = go.Figure()
        for ice_id, ice in curve[curve["kind"] == "ice"].groupby("ice_id"):
            fig.add_trace(go.Scatter(x=ice["grid_value"], y=ice["prediction"], mode="lines", line=dict(color="rgba(138,137,129,0.25)", width=1), showlegend=False, hoverinfo="skip"))
        mean_curve = curve[curve["kind"] == "pdp"]
        fig.add_trace(go.Scatter(x=mean_curve["grid_value"], y=mean_curve["prediction"], mode="lines+markers", name="Среднее (PDP)", line=dict(color=FACT, width=3), marker=dict(size=8)))
        fig.update_xaxes(title=human(feature))
        fig.update_yaxes(title="Прогноз (лог)")
        fig.update_layout(hovermode="closest")
        draw(style_axes(fig, "Серые линии: отдельные наблюдения (ICE)", 360))

        st.markdown("##### Меняется ли структура важности со временем?")
        periods = load("importance_periods")
        period_part = periods[periods["target"] == target]
        pivot = period_part.pivot_table(index="feature", columns="period", values="share", aggfunc="first")
        top_periods = pivot.mean(axis=1).sort_values(ascending=False).head(12).index
        heat = pivot.loc[top_periods].iloc[::-1]
        fig = go.Figure(go.Heatmap(z=heat.values, x=list(heat.columns), y=[human(f) for f in heat.index], colorscale=[[0, "#f4f8fd"], [1, FACT]], colorbar=dict(title="Доля вклада"),
                                   hovertemplate="%{y}, %{x}: %{z:.1%}<extra></extra>"))
        fig.update_layout(hovermode="closest")
        draw(style_axes(fig, "Доля SHAP-вклада признака по годам", 420))
        stability = load("importance_stability")
        stable = stability[stability["target"] == target].rename(columns={"period_a": "Период A", "period_b": "Период B", "spearman": "Корреляция Спирмена"})
        st.dataframe(stable[["Период A", "Период B", "Корреляция Спирмена"]].round(3), hide_index=True)

# ------------------------------------- Мониторинг ---------------------------------------
with tab_monitor:
    st.subheader("Не ухудшилась ли модель?")
    quality = load("quality_daily")
    decisions = load("retrain_decisions")
    if quality.empty:
        st.warning("Данных мониторинга ещё нет.")
    else:
        q = quality[quality["target"] == target].copy()
        q["target_date"] = pd.to_datetime(q["target_date"])
        last_row = q.dropna(subset=["rolling_mae"]).tail(1)
        c1, c2, c3 = st.columns(3)
        if not last_row.empty:
            c1.metric("Скользящая MAE (14 дн.)", f"{float(last_row['rolling_mae'].iloc[0]):.4f}")
            c2.metric("Порог переобучения", f"{float(last_row['threshold'].iloc[0]):.4f}", help="1.2 × валидационная MAE (лог)")
        target_decisions = decisions[decisions["target"] == target].sort_values("decided_at")
        if not target_decisions.empty:
            last = target_decisions.iloc[-1]
            c3.metric("Доля признаков с дрейфом (PSI>0.25)", f"{float(last['psi_share']):.0%}")
            if last["retrain"]:
                st.error(f"Правило переобучения сработало: {last['reasons']}")
            elif last["on_cooldown"] and last["reasons"]:
                st.warning(f"Условие выполнено, но идёт период ожидания после обучения: {last['reasons']}")
            else:
                st.success("Переобучение не требуется: пороги не превышены.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q["target_date"], y=q["rolling_mae"], name="Скользящая MAE (14 дн.)", mode="lines", line=dict(color=FACT, width=2)))
        fig.add_trace(go.Scatter(x=q["target_date"], y=q["threshold"], name="Порог переобучения", mode="lines", line=dict(color=FORECAST, width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=q["target_date"], y=q["validation_mae"], name="Валидационная MAE", mode="lines", line=dict(color=MUTED, width=2, dash="dot")))
        fig.update_yaxes(title="MAE (лог)", rangemode="tozero")
        draw(style_axes(fig, "Ошибка на поступающих фактах относительно порога", 340))

        st.markdown("##### Изменились ли входные данные?")
        drift = load("drift_latest")
        drifted = drift[drift["target"] == target].sort_values("psi", ascending=False)
        table = pd.DataFrame({"Признак": drifted["feature"].map(human), "PSI": drifted["psi"].round(3), "KS": drifted["ks_stat"].round(3), "p-value (KS)": drifted["ks_pvalue"].map(lambda v: f"{v:.2e}"), "Дрейф": drifted["drifted"].map({True: "Да", False: "Нет"})})
        st.dataframe(table, hide_index=True, width="stretch", height=300)

        st.markdown("##### Какая версия модели работает и что с ней происходило?")
        reg = registry[registry["target"] == target].sort_values("created_at", ascending=False)
        reg_table = pd.DataFrame({"Версия": reg["version_id"], "Модель": reg["model_name"].map(MODEL_NAMES), "Статус": reg["stage"], "Обучена до": reg["trained_until"], "Данные (хеш)": reg["data_version"], "Создана": reg["created_at"]})
        st.dataframe(reg_table, hide_index=True, width="stretch")
        events = load("registry_events")
        if not events.empty:
            ev = events[events["target"] == target].sort_values("event_at", ascending=False).head(8)
            st.dataframe(ev[["event_at", "event", "version_id"]].rename(columns={"event_at": "Время", "event": "Событие", "version_id": "Версия"}), hide_index=True, width="stretch")

    st.markdown("##### Как система реагирует на ухудшение? (симуляция)")
    sim_summary = load_json("simulation_summary")
    if sim_summary:
        st.warning(sim_summary["note"])
        sim_quality = load("simulation_quality")
        s = sim_quality[sim_quality["target"] == target].copy()
        s["target_date"] = pd.to_datetime(s["target_date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["target_date"], y=s["rolling_mae"], name="Текущая модель", mode="lines", line=dict(color=FORECAST, width=2)))
        if "rolling_mae_retrained" in s:
            fig.add_trace(go.Scatter(x=s["target_date"], y=s["rolling_mae_retrained"], name="После переобучения", mode="lines", line=dict(color=ALT, width=2)))
        fig.add_trace(go.Scatter(x=s["target_date"], y=s["threshold"], name="Порог", mode="lines", line=dict(color=MUTED, width=2, dash="dash")))
        fig.add_vline(x=pd.Timestamp(sim_summary["start"]), line=dict(color=MUTED, dash="dot"))
        fig.update_yaxes(title="MAE (лог)", rangemode="tozero")
        draw(style_axes(fig, "СИМУЛЯЦИЯ: реакция на искусственный сдвиг", 340))
        info = sim_summary["targets"].get(target, {})
        st.json({k: info[k] for k in ("first_trigger_date", "psi_share", "retrain_triggered", "candidate_promoted_in_simulation") if k in info})
    else:
        st.caption("Симуляция ещё не запускалась: python -m cryptocast simulate-drift")

# ------------------------------------- Эксперименты -------------------------------------
with tab_exp:
    st.subheader("Какая конфигурация показала лучшее качество?")
    experiments = load("experiments")
    if experiments.empty:
        st.warning("Журнал экспериментов пуст.")
    else:
        part = experiments[experiments["target"] == target].copy()
        def metric_of(raw: str, key: str) -> float:
            if raw is None or raw != raw:
                return np.nan
            return float(json.loads(raw).get(key, np.nan))
        part["mae_cv"] = part["cv_metrics"].map(lambda v: metric_of(v, "mae"))
        part["qlike_cv"] = part["cv_metrics"].map(lambda v: metric_of(v, "qlike"))
        part["mae_holdout"] = part["holdout_metrics"].map(lambda v: metric_of(v, "mae"))
        part["qlike_holdout"] = part["holdout_metrics"].map(lambda v: metric_of(v, "qlike"))
        kinds = st.multiselect("Тип записи", sorted(part["kind"].unique()), default=["evaluation"])
        models = st.multiselect("Модели", sorted(part["model_name"].unique()), default=sorted(part["model_name"].unique()))
        view = part[part["kind"].isin(kinds) & part["model_name"].isin(models)].sort_values("created_at", ascending=False)
        columns = ["experiment_id", "kind", "model_name", "mae_cv", "mae_holdout"] + (["qlike_cv", "qlike_holdout"] if target == "log_rv" else []) + ["data_version", "seed", "git_commit", "git_dirty", "created_at"]
        table_view = view[columns].copy()
        table_view["git_commit"] = table_view["git_commit"].str[:8]
        table_view = table_view.rename(columns={"experiment_id": "ID эксперимента", "kind": "Тип", "model_name": "Модель", "mae_cv": "MAE walk-forward",
                                                "mae_holdout": "MAE отложенная", "qlike_cv": "QLIKE walk-forward", "qlike_holdout": "QLIKE отложенная",
                                                "data_version": "Версия данных", "seed": "Seed", "git_commit": "Коммит", "git_dirty": "Незакоммиченные правки", "created_at": "Создан"})
        st.dataframe(table_view.round(4), hide_index=True, width="stretch", height=280)
        st.caption("Повторить любой эксперимент: python -m cryptocast reproduce --experiment-id <id>")
        shown = view.dropna(subset=["mae_holdout"]).drop_duplicates("model_name")
        if not shown.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=shown["model_name"].map(MODEL_NAMES), y=shown["mae_cv"], name="Walk-forward", marker=dict(color=FACT, line=dict(color="#fcfcfb", width=2), cornerradius=4)))
            fig.add_trace(go.Bar(x=shown["model_name"].map(MODEL_NAMES), y=shown["mae_holdout"], name="Отложенная выборка", marker=dict(color=FORECAST, line=dict(color="#fcfcfb", width=2), cornerradius=4)))
            fig.update_layout(barmode="group", hovermode="closest")
            fig.update_yaxes(title="MAE")
            draw(style_axes(fig, "Последний запуск: качество при разработке и на отложенной выборке", 320))
        trials = load("optuna_trials")
        if not trials.empty:
            opt_model = st.selectbox("Подбор гиперпараметров: модель", sorted(trials["model"].unique()))
            t = trials[(trials["target"] == target) & (trials["model"] == opt_model)].sort_values("number")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t["number"], y=t["value"], mode="markers", name="Попытка", marker=dict(size=9, color=FACT, line=dict(color="#fcfcfb", width=1))))
            fig.add_trace(go.Scatter(x=t["number"], y=t["value"].cummin(), mode="lines", name="Лучшее значение", line=dict(color=FORECAST, width=2)))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title="Номер попытки")
            draw(style_axes(fig, "Optuna: сходимость подбора", 300))

# ------------------------------------- Данные --------------------------------------------
with tab_data:
    st.subheader("Свежие ли и корректные ли данные?")
    dq = load("dq_results")
    log_table = load("load_log")
    if not dq.empty:
        latest_run = dq[dq["checked_at"] == dq["checked_at"].max()]
        critical = latest_run[(~latest_run["passed"]) & (latest_run["severity"] == "critical")]
        if critical.empty:
            st.success(f"Контроль качества: критичных сбоев нет (проверок: {len(latest_run)}).")
        else:
            st.error(f"Критичные сбои: {len(critical)}. Обновление витрины заблокировано.")
        st.dataframe(latest_run[["check_name", "scope", "severity", "passed", "value", "threshold", "details"]].rename(columns={"check_name": "Проверка", "scope": "Объект", "severity": "Важность", "passed": "Пройдена", "value": "Значение", "threshold": "Порог", "details": "Детали"}), hide_index=True, width="stretch", height=260)
    if not log_table.empty:
        st.markdown("##### Журнал последних загрузок")
        st.dataframe(log_table.sort_values("started_at", ascending=False).head(30)[["run_id", "source", "status", "received", "inserted", "duplicates", "watermark_after", "error"]].rename(columns={"source": "Источник", "status": "Статус", "received": "Получено", "inserted": "Вставлено", "duplicates": "Дубликатов", "watermark_after": "Watermark", "error": "Сообщение"}), hide_index=True, width="stretch", height=300)
    leakage = load_json("leakage_report")
    if leakage:
        st.markdown("##### Проверка отсутствия утечки")
        (st.success if leakage.get("passed") else st.error)("Все проверки утечки пройдены." if leakage.get("passed") else "Проверки утечки не пройдены!")
        truncation = pd.DataFrame(leakage.get("truncation", []))
        if not truncation.empty:
            st.dataframe(truncation.rename(columns={"cutoff": "Дата усечения", "rows": "Строк", "max_abs_diff": "Макс. разница", "nan_mismatches": "Расхождений NaN", "last_day_target_is_nan": "Таргет последнего дня пуст"}), hide_index=True)
        shuffled = leakage.get("shuffled_target", {})
        if shuffled:
            st.dataframe(pd.DataFrame(shuffled).T.reset_index().rename(columns={"index": "Показатель", "mae_log_real": "MAE (реальный таргет)", "mae_log_shuffled": "MAE (перемешанный)", "r2_shuffled": "R² (перемешанный)", "passed": "Пройдена"}), hide_index=True)
