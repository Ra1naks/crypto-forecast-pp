"""Единый код признаков для обучения и инференса.

Договор о времени: строка (symbol, date) содержит только информацию, известную в конце
суток date (UTC), а таргеты относятся к date + 1. Календарные признаки описывают дату
прогноза date + 1: она заранее известна и утечкой не является.

Таргеты:
    log_qv — логарифм суммарного quote volume за сутки date + 1;
    log_rv — логарифм реализованной волатильности (корень из суммы квадратов часовых
             лог-доходностей) за сутки date + 1. Метрики считаются в исходном масштабе RV.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from cryptocast.config import Config
from cryptocast.storage import Store, write_parquet

KEY_COLUMNS = ["symbol", "date"]
TARGET_COLUMNS = ["target_log_qv", "target_log_rv"]
GROUP_PREFIXES = {"market": "mkt_", "calendar": "cal_", "fng": "fng_", "text": "txt_"}
# Столбец с прошлым значением таргета: наивный прогноз.
NAIVE_COLUMN = {"log_qv": "mkt_log_qv", "log_rv": "mkt_log_rv"}
HAR_COLUMNS = {
    "log_qv": ["mkt_log_qv", "mkt_log_qv_ma7", "mkt_log_qv_ma30"],
    "log_rv": ["mkt_log_rv", "mkt_log_rv_ma7", "mkt_log_rv_ma30"],
}


def _market_features(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    lags = cfg["features"]["lags"]
    windows = cfg["features"]["windows"]
    frame = daily.sort_values(KEY_COLUMNS).copy()
    frame["mkt_log_qv"] = np.log(frame["quote_volume"])
    frame["mkt_log_rv"] = np.log(frame["rv"].replace(0, np.nan))
    frame["mkt_log_trades"] = np.log(frame["trades"].replace(0, np.nan)).where(frame["is_valid"])
    frame["mkt_taker_buy_ratio"] = (frame["taker_buy_quote"] / frame["quote_volume_raw"]).where(frame["is_valid"])
    frame["mkt_return_1d"] = np.log(frame["close"] / frame.groupby("symbol")["close"].shift(1))
    frame["mkt_abs_return_1d"] = frame["mkt_return_1d"].abs()
    frame["mkt_range_log"] = np.log(frame["high"] / frame["low"]).where(frame["is_valid"])
    frame["mkt_parkinson"] = frame["mkt_range_log"] / np.sqrt(4 * np.log(2))
    frame["mkt_max_abs_ret"] = frame["max_abs_ret"].where(frame["is_valid"])
    frame["mkt_rs_pos_share"] = (frame["rs_pos"] ** 2 / frame["rv"].replace(0, np.nan) ** 2)
    frame["mkt_volume_hhi"] = frame["volume_hhi"].where(frame["is_valid"])
    frame["mkt_last4h_share"] = (frame["qv_last4h"] / frame["quote_volume_raw"]).where(frame["is_valid"])
    frame["mkt_n_anomalies"] = frame["n_volume_anomaly"].astype(float)
    frame["mkt_is_scaled"] = frame["is_scaled"].astype(float)
    group = frame.groupby("symbol", sort=False)
    new_columns: dict[str, pd.Series] = {}
    for base in ("mkt_log_qv", "mkt_log_rv"):
        for lag in lags:
            new_columns[f"{base}_lag{lag}"] = group[base].shift(lag)
        for window in windows:
            min_periods = max(2, window // 2)
            rolling = group[base].rolling(window, min_periods=min_periods)
            new_columns[f"{base}_ma{window}"] = rolling.mean().reset_index(level=0, drop=True)
            new_columns[f"{base}_std{window}"] = rolling.std().reset_index(level=0, drop=True)
    for window in (3, 7, 30):
        new_columns[f"mkt_return_{window}d"] = group["close"].transform(lambda s: np.log(s / s.shift(window)))
    new_columns["mkt_log_trades_ma7"] = group["mkt_log_trades"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    new_columns["mkt_taker_ratio_ma7"] = group["mkt_taker_buy_ratio"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    frame = pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1)
    frame["mkt_log_qv_vs_ma30"] = frame["mkt_log_qv"] - frame["mkt_log_qv_ma30"]
    frame["mkt_log_rv_vs_ma30"] = frame["mkt_log_rv"] - frame["mkt_log_rv_ma30"]
    frame["mkt_log_qv_vs_ma7"] = frame["mkt_log_qv"] - frame["mkt_log_qv_ma7"]
    frame["mkt_log_rv_vs_ma7"] = frame["mkt_log_rv"] - frame["mkt_log_rv_ma7"]
    frame["mkt_days_since_invalid"] = group["is_valid"].transform(
        lambda s: (s.astype(int).groupby((~s).cumsum()).cumcount()).astype(float)
    )
    # Общерыночный контекст: суточные показатели BTC и среднего по инструментам.
    btc = frame.loc[frame["symbol"] == "BTCUSDT", ["date", "mkt_log_qv", "mkt_log_rv", "mkt_return_1d"]]
    btc = btc.rename(columns={"mkt_log_qv": "mkt_btc_log_qv", "mkt_log_rv": "mkt_btc_log_rv", "mkt_return_1d": "mkt_btc_return_1d"})
    frame = frame.merge(btc, on="date", how="left")
    frame["mkt_cross_rv_mean"] = frame.groupby("date")["mkt_log_rv"].transform("mean")
    frame["mkt_cross_qv_mean"] = frame.groupby("date")["mkt_log_qv"].transform("mean")
    frame["mkt_rv_vs_cross"] = frame["mkt_log_rv"] - frame["mkt_cross_rv_mean"]
    return frame


def _calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    target_date = frame["date"] + pd.Timedelta(days=1)
    frame["cal_dow"] = target_date.dt.day_of_week.astype(float)
    frame["cal_is_weekend"] = (target_date.dt.day_of_week >= 5).astype(float)
    frame["cal_dow_sin"] = np.sin(2 * np.pi * target_date.dt.day_of_week / 7)
    frame["cal_dow_cos"] = np.cos(2 * np.pi * target_date.dt.day_of_week / 7)
    frame["cal_month_sin"] = np.sin(2 * np.pi * (target_date.dt.month - 1) / 12)
    frame["cal_month_cos"] = np.cos(2 * np.pi * (target_date.dt.month - 1) / 12)
    frame["cal_is_month_end"] = target_date.dt.is_month_end.astype(float)
    frame["cal_is_quarter_end"] = target_date.dt.is_quarter_end.astype(float)
    return frame


def _fng_features(fng: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Индекс на дату D публикуется в начале суток D, поэтому в конце D он известен."""
    calendar = pd.DataFrame({"date": pd.date_range(dates.min(), dates.max(), freq="D")})
    if fng.empty:
        calendar["fng_available"] = 0.0
        return calendar
    data = fng.rename(columns={"fng_value": "value"})[["date", "value"]].sort_values("date")
    merged = pd.merge_asof(calendar, data, on="date", direction="backward", tolerance=pd.Timedelta(days=2))
    merged["fng_available"] = merged["value"].notna().astype(float)
    merged["fng_value"] = merged["value"]
    merged["fng_change_1d"] = merged["value"].diff(1)
    merged["fng_change_7d"] = merged["value"].diff(7)
    merged["fng_ma7"] = merged["value"].rolling(7, min_periods=3).mean()
    merged["fng_z30"] = (merged["value"] - merged["value"].rolling(30, min_periods=10).mean()) / merged["value"].rolling(30, min_periods=10).std()
    merged["fng_is_extreme"] = ((merged["value"] <= 25) | (merged["value"] >= 75)).astype(float).where(merged["value"].notna())
    return merged.drop(columns=["value"])


def _text_features(news: pd.DataFrame, panel: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Суточные текстовые признаки без подбора на данных: только счётчики и тональность.

    Объём новостей в истории — выборка API с лимитом на окно, поэтому абсолютные счётчики
    между источниками несопоставимы; используются доли и флаг наличия текста.
    """
    columns = pd.MultiIndex.from_frame(panel[KEY_COLUMNS]).unique()
    out = pd.DataFrame(index=columns).reset_index()
    if news.empty:
        out["txt_available"] = 0.0
        return out
    topic_columns = [c for c in news.columns if c.startswith("topic_")]
    agg_spec: dict = {"sentiment": ["mean", "std", "count"]}
    agg_spec.update({c: "mean" for c in topic_columns})
    daily = news.groupby("date").agg(agg_spec)
    daily.columns = ["sent_mean", "sent_std", "n_articles"] + [f"txt_{c}_share" for c in topic_columns]
    daily["sent_neg_share"] = news.assign(neg=(news["sentiment"] < -0.05).astype(float)).groupby("date")["neg"].mean()
    daily = daily.reset_index()
    full = pd.DataFrame({"date": pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")})
    daily = full.merge(daily, on="date", how="left")
    daily["txt_available"] = (daily["n_articles"].fillna(0) > 0).astype(float)
    # Счётчик берётся по потокам с постоянной историей: RSS накапливается только с начала сбора.
    history_news = news[news["origin"] != "rss"] if "origin" in news else news
    history_count = history_news.groupby("date").size().reindex(daily["date"]).fillna(0).to_numpy()
    daily["txt_log_n_articles"] = np.log1p(history_count)
    daily["txt_sent_mean"] = daily["sent_mean"]
    daily["txt_sent_std"] = daily["sent_std"]
    daily["txt_neg_share"] = daily["sent_neg_share"]
    daily["txt_sent_ma3"] = daily["sent_mean"].rolling(3, min_periods=1).mean()
    daily["txt_sent_ma7"] = daily["sent_mean"].rolling(7, min_periods=1).mean()
    for topic in ("topic_hack_share", "topic_regulation_share", "topic_etf_share"):
        column = f"txt_{topic}"
        if column in daily:
            daily[column.replace("_share", "_ma7")] = daily[column].rolling(7, min_periods=1).mean()
    daily = daily.drop(columns=["sent_mean", "sent_std", "n_articles", "sent_neg_share"])
    out = out.merge(daily, on="date", how="left")

    # Упоминания инструмента: своя доля упоминаний и тональность таких заголовков.
    mention_rows = []
    for symbol in symbols:
        mention_column = f"mention_{symbol}"
        if mention_column not in news:
            continue
        grouped = news.assign(flag=news[mention_column]).groupby("date")
        row = pd.DataFrame(
            {
                "date": grouped.size().index,
                "symbol": symbol,
                "txt_mention_share": grouped["flag"].mean().values,
                "txt_mention_sentiment": news[news[mention_column] == 1].groupby("date")["sentiment"].mean().reindex(grouped.size().index).values,
            }
        )
        mention_rows.append(row)
    mentions = pd.concat(mention_rows, ignore_index=True)
    out = out.merge(mentions, on=KEY_COLUMNS, how="left")
    out["txt_mention_share_ma7"] = out.sort_values(KEY_COLUMNS).groupby("symbol")["txt_mention_share"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return out


def build_feature_frame(
    daily: pd.DataFrame, fng: pd.DataFrame, news: pd.DataFrame, cfg: Config, include_targets: bool = True
) -> pd.DataFrame:
    """Единая функция расчёта признаков (обучение и инференс используют только её)."""
    frame = _market_features(daily, cfg)
    frame = _calendar_features(frame)
    frame = frame.merge(_fng_features(fng, frame["date"]), on="date", how="left")
    frame = frame.merge(_text_features(news, frame[KEY_COLUMNS], cfg.symbols), on=KEY_COLUMNS, how="left")
    if include_targets:
        group = frame.groupby("symbol", sort=False)
        same_next_day = group["date"].shift(-1) == frame["date"] + pd.Timedelta(days=1)
        frame["target_log_qv"] = group["mkt_log_qv"].shift(-1).where(same_next_day)
        frame["target_log_rv"] = group["mkt_log_rv"].shift(-1).where(same_next_day)
    else:
        frame["target_log_qv"] = np.nan
        frame["target_log_rv"] = np.nan
    feature_columns = [c for c in frame.columns if c.startswith(tuple(GROUP_PREFIXES.values()))]
    result = frame[KEY_COLUMNS + feature_columns + TARGET_COLUMNS].copy()
    result[feature_columns] = result[feature_columns].astype(float)
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame, groups: list[str] | None = None) -> list[str]:
    prefixes = tuple(GROUP_PREFIXES[g] for g in (groups or list(GROUP_PREFIXES)))
    return [c for c in frame.columns if c.startswith(prefixes)]


def data_version(features: pd.DataFrame, news: pd.DataFrame) -> str:
    """Хеш содержимого витрины и новостей — идентификатор версии данных."""
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(features.sort_values(KEY_COLUMNS), index=False).values.tobytes())
    if not news.empty:
        text_part = news.sort_values("url_hash")[["url_hash", "date", "text"]]
        hasher.update(pd.util.hash_pandas_object(text_part, index=False).values.tobytes())
    return hasher.hexdigest()[:16]


def run_features(store: Store) -> dict[str, object]:
    cfg = store.cfg
    daily = store.df("SELECT * FROM staging.daily ORDER BY symbol, date")
    daily["date"] = pd.to_datetime(daily["date"])
    fng = store.df("SELECT * FROM staging.fng")
    fng["date"] = pd.to_datetime(fng["date"])
    news = store.df("SELECT * FROM staging.news")
    if not news.empty:
        news["date"] = pd.to_datetime(news["date"])
    frame = build_feature_frame(daily, fng, news, cfg)
    store.replace_table("mart.features", frame)
    write_parquet(cfg, "mart", "features", frame)
    fact = daily[["symbol", "date", "quote_volume", "rv", "close", "is_valid", "n_candles"]]
    write_parquet(cfg, "mart", "daily_fact", fact)
    return {"rows": len(frame), "features": len(feature_columns(frame)), "last_date": str(frame["date"].max().date())}
