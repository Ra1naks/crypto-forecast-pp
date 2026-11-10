"""Слой staging: единый вид данных, пропуски, аномалии, суточные агрегаты и чистка новостей.

Правила:
* Некорректные свечи (high<low, отрицательный объём) в расчёты не попадают.
* Всплески объёма и резкие движения цены НЕ удаляются: именно их система должна учитывать.
  Они только помечаются флагом, робастность моделей обеспечивается логарифмами и
  клиппингом, обученным на train каждого фолда.
* Сутки считаются полными, когда есть свечи и данные за последний час уже закрыты.
  При 22-23 свечах объём и сумма квадратов доходностей масштабируются на 24/n;
  при меньшем числе день невалиден, таргет и признаки этого дня не строятся.
* Текущий незавершённый день не входит в витрину.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from cryptocast.config import Config
from cryptocast.storage import Store, write_parquet

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_TEXT_RE = re.compile(r"[^a-z0-9$%\s'-]+")
SPACES_RE = re.compile(r"\s+")

CRYPTO_LEXICON = {
    "hack": -2.0, "hacked": -2.5, "exploit": -2.0, "exploited": -2.5, "scam": -2.5, "rug": -2.0,
    "crash": -2.5, "plunge": -2.2, "plunges": -2.2, "dump": -1.5, "selloff": -2.0, "liquidated": -1.8,
    "bearish": -1.8, "lawsuit": -1.5, "ban": -1.5, "fraud": -2.8, "insolvent": -2.8, "bankrupt": -2.8,
    "rally": 2.0, "surge": 1.8, "surges": 1.8, "soars": 2.0, "bullish": 1.8, "breakout": 1.2,
    "adoption": 1.2, "approved": 1.5, "approval": 1.5, "record": 0.8, "rebound": 1.4,
}
TOPICS = {
    "hack": r"\b(?:hack\w*|exploit\w*|breach\w*|stolen|drain\w*|phish\w*|attack\w*)\b",
    "regulation": r"\b(?:sec|cftc|regulat\w*|lawsuit\w*|ban|bans|banned|mica|congress|senate|legislat\w*|compliance|sanction\w*)\b",
    "etf": r"\b(?:etf|etfs|exchange[- ]traded)\b",
}
MENTIONS = {
    "BTCUSDT": r"\b(?:bitcoin|btc)\b",
    "ETHUSDT": r"\b(?:ethereum|ether|eth)\b",
    "SOLUSDT": r"\b(?:solana|sol)\b",
    "BNBUSDT": r"\b(?:bnb|binance coin)\b",
    "XRPUSDT": r"\b(?:xrp|ripple)\b",
}


def build_hourly(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Чистит часовые свечи и считает доходности; возвращает (хорошие, отклонённые)."""
    data = frame.sort_values(["symbol", "open_time"]).drop_duplicates(["symbol", "open_time"], keep="first")
    bad_mask = (
        (data["high"] < data["low"]) | (data["volume"] < 0) | (data["quote_volume"] < 0)
        | (data["close"] <= 0) | (data["open"] <= 0) | data[["open", "high", "low", "close"]].isna().any(axis=1)
    )
    rejected = data[bad_mask].assign(reject_reason="ohlcv_invalid")
    data = data[~bad_mask].copy()
    grouped = data.groupby("symbol", sort=False)
    data["log_return"] = np.log(data["close"] / grouped["close"].shift(1))
    data["log_quote_volume"] = np.log1p(data["quote_volume"])
    params = cfg["staging"]
    # Базовый уровень — только прошлое (скользящие 30 суток), иначе флаг зависел бы от будущего.
    past = grouped["log_quote_volume"].shift(1)
    median = past.groupby(data["symbol"]).transform(lambda s: s.rolling(720, min_periods=168).median())
    deviation = (past - median).abs()
    mad = deviation.groupby(data["symbol"]).transform(lambda s: s.rolling(720, min_periods=168).median()).replace(0, np.nan)
    data["is_volume_anomaly"] = ((data["log_quote_volume"] - median).abs() / (1.4826 * mad)) > params["anomaly_robust_z"]
    data["is_price_jump"] = data["log_return"].abs() > params["price_jump_abs_log_return"]
    data["date"] = data["open_time"].dt.floor("D")
    return data, rejected


def _herfindahl(series: pd.Series) -> float:
    total = series.sum()
    return float(((series / total) ** 2).sum()) if total > 0 else np.nan


def build_daily(hourly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Суточные агрегаты по завершённым UTC-суткам."""
    min_candles = cfg["staging"]["min_candles_per_day"]
    last_open = hourly.groupby("symbol")["open_time"].transform("max")
    complete = hourly["date"] + pd.Timedelta(hours=23) <= last_open
    data = hourly[complete].copy()
    data["r2"] = data["log_return"] ** 2
    data["r2_pos"] = data["r2"].where(data["log_return"] > 0, 0.0)
    data["r2_neg"] = data["r2"].where(data["log_return"] < 0, 0.0)
    data["abs_ret"] = data["log_return"].abs()
    data["hour"] = data["open_time"].dt.hour
    group = data.groupby(["symbol", "date"], sort=True)
    daily = group.agg(
        n_candles=("close", "size"), open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), quote_volume_raw=("quote_volume", "sum"), trades=("trades", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"), sum_r2=("r2", "sum"), sum_r2_pos=("r2_pos", "sum"),
        sum_r2_neg=("r2_neg", "sum"), max_abs_ret=("abs_ret", "max"),
        n_volume_anomaly=("is_volume_anomaly", "sum"), n_price_jump=("is_price_jump", "sum"),
    ).reset_index()
    daily["volume_hhi"] = group["quote_volume"].apply(_herfindahl).values
    last_hours = data[data["hour"] >= 20].groupby(["symbol", "date"])["quote_volume"].sum()
    daily["qv_last4h"] = last_hours.reindex(pd.MultiIndex.from_frame(daily[["symbol", "date"]])).values
    scale = 24.0 / daily["n_candles"]
    daily["is_valid"] = daily["n_candles"] >= min_candles
    daily["is_scaled"] = daily["is_valid"] & (daily["n_candles"] < 24)
    # Первая свеча всей истории не имеет доходности: сутки валидны, но RV занижена на 1/24 — несущественно.
    daily["quote_volume"] = (daily["quote_volume_raw"] * scale).where(daily["is_valid"])
    daily["rv"] = np.sqrt(daily["sum_r2"] * scale).where(daily["is_valid"])
    daily["rs_pos"] = np.sqrt(daily["sum_r2_pos"] * scale).where(daily["is_valid"])
    daily["rs_neg"] = np.sqrt(daily["sum_r2_neg"] * scale).where(daily["is_valid"])
    return daily


def clean_text(text: str) -> str:
    lowered = URL_RE.sub(" ", (text or "").lower())
    return SPACES_RE.sub(" ", NON_TEXT_RE.sub(" ", lowered)).strip()


def build_news(raw_news: pd.DataFrame) -> pd.DataFrame:
    """Очистка и дедупликация заголовков, тональность, темы и упоминания инструментов."""
    if raw_news.empty:
        return pd.DataFrame()
    news = raw_news.copy()
    news = news[news["language"].isin(["english", ""])]
    news["text"] = (news["title"].fillna("") + " " + news["summary"].fillna("")).map(clean_text)
    news["title_clean"] = news["title"].fillna("").map(clean_text)
    news = news[news["title_clean"].str.len() >= 15]
    # Перепечатки: одинаковый заголовок считаем одной новостью, оставляем самую раннюю.
    news = news.sort_values(["seen_at", "url_hash"]).drop_duplicates("title_clean", keep="first")
    news["date"] = news["seen_at"].dt.floor("D")
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(CRYPTO_LEXICON)
    news["sentiment"] = news["title_clean"].map(lambda t: analyzer.polarity_scores(t)["compound"])
    for topic, pattern in TOPICS.items():
        news[f"topic_{topic}"] = news["title_clean"].str.contains(pattern, regex=True).astype(int)
    for symbol, pattern in MENTIONS.items():
        news[f"mention_{symbol}"] = news["title_clean"].str.contains(pattern, regex=True).astype(int)
    news["origin"] = news["source"].str.split(":").str[0]
    keep = ["url_hash", "seen_at", "date", "domain", "origin", "title_clean", "text", "sentiment"]
    keep += [c for c in news.columns if c.startswith(("topic_", "mention_"))]
    return news[keep].reset_index(drop=True)


def run_staging(store: Store) -> dict[str, int]:
    cfg = store.cfg
    hourly, rejected = build_hourly(store.df("SELECT * FROM raw.klines ORDER BY symbol, open_time"), cfg)
    daily = build_daily(hourly, cfg)
    news = build_news(store.df("SELECT * FROM raw.news"))
    fng = store.df("SELECT date, value AS fng_value, classification FROM raw.fng ORDER BY date")
    fng["date"] = pd.to_datetime(fng["date"])
    hourly_keep = hourly.drop(columns=["ingested_at", "close_time"], errors="ignore")
    store.replace_table("staging.hourly", hourly_keep)
    store.replace_table("staging.hourly_rejected", rejected)
    store.replace_table("staging.daily", daily)
    store.replace_table("staging.news", news)
    store.replace_table("staging.fng", fng)
    write_parquet(cfg, "staging", "hourly", hourly_keep)
    write_parquet(cfg, "staging", "daily", daily)
    write_parquet(cfg, "staging", "news", news)
    write_parquet(cfg, "staging", "fng", fng)
    write_parquet(cfg, "raw", "klines", store.df("SELECT * EXCLUDE (ingested_at) FROM raw.klines ORDER BY symbol, open_time"))
    write_parquet(cfg, "raw", "fng", store.df("SELECT * EXCLUDE (ingested_at) FROM raw.fng ORDER BY date"))
    write_parquet(cfg, "raw", "news", store.df("SELECT * EXCLUDE (ingested_at) FROM raw.news ORDER BY url_hash"))
    return {"hourly": len(hourly_keep), "rejected": len(rejected), "daily": len(daily), "news": len(news), "fng": len(fng)}
