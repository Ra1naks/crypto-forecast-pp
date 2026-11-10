"""Контроль качества и слой staging."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cryptocast.quality import run_quality
from cryptocast.staging import build_daily, build_hourly, build_news
from tests.conftest import make_klines

NOW = dt.datetime(2024, 5, 20, 1)


def _load(store, frame):
    store.upsert_ignore("raw.klines", frame)


def test_quality_passes_on_clean_data(store):
    _load(store, make_klines(store.cfg.symbols))
    assert run_quality(store, "r", now=NOW) is True


def test_high_below_low_is_critical(store):
    frame = make_klines(store.cfg.symbols)
    frame.loc[10, "high"] = frame.loc[10, "low"] - 1
    _load(store, frame)
    assert run_quality(store, "r", now=NOW) is False
    failed = store.df("SELECT check_name FROM ops.dq_results WHERE NOT passed AND severity='critical'")
    assert "klines_high_ge_low" in set(failed["check_name"])


def test_negative_volume_is_critical(store):
    frame = make_klines(store.cfg.symbols)
    frame.loc[5, "volume"] = -1
    _load(store, frame)
    assert run_quality(store, "r", now=NOW) is False


def test_stale_data_is_critical(store):
    _load(store, make_klines(store.cfg.symbols))
    assert run_quality(store, "r", now=NOW + dt.timedelta(days=10)) is False
    failed = store.df("SELECT check_name FROM ops.dq_results WHERE NOT passed AND severity='critical'")
    assert "klines_freshness_hours" in set(failed["check_name"])


def test_large_gap_is_critical(store):
    frame = make_klines(store.cfg.symbols)
    _load(store, frame[~((frame["symbol"] == "BTCUSDT") & (frame["open_time"].between("2024-02-01", "2024-02-20")))])
    assert run_quality(store, "r", now=NOW) is False


def test_daily_rules(cfg):
    frame = make_klines(cfg.symbols, days=10)
    frame = frame[~((frame["open_time"] >= "2024-01-03 00:00") & (frame["open_time"] < "2024-01-03 05:00"))]
    drop_two = (frame["open_time"] >= "2024-01-05 03:00") & (frame["open_time"] < "2024-01-05 05:00") & (frame["symbol"] == "ETHUSDT")
    frame = frame[~drop_two]
    hourly, rejected = build_hourly(frame, cfg)
    daily = build_daily(hourly, cfg)
    assert rejected.empty
    invalid = daily[daily["date"] == pd.Timestamp("2024-01-03")]
    assert (~invalid["is_valid"]).all() and invalid["rv"].isna().all()
    scaled = daily[(daily["date"] == pd.Timestamp("2024-01-05")) & (daily["symbol"] == "ETHUSDT")].iloc[0]
    assert scaled["is_scaled"]
    assert np.isclose(scaled["quote_volume"], scaled["quote_volume_raw"] * 24 / 22)


def test_rv_formula(cfg):
    frame = make_klines(cfg.symbols, days=3)
    hourly, _ = build_hourly(frame, cfg)
    daily = build_daily(hourly, cfg)
    btc_day = hourly[(hourly["symbol"] == "BTCUSDT") & (hourly["date"] == pd.Timestamp("2024-01-02"))]
    expected = np.sqrt((btc_day["log_return"] ** 2).sum())
    actual = daily[(daily["symbol"] == "BTCUSDT") & (daily["date"] == pd.Timestamp("2024-01-02"))].iloc[0]["rv"]
    assert np.isclose(actual, expected)


def test_incomplete_last_day_excluded(cfg):
    frame = make_klines(cfg.symbols, days=5)
    cut = frame[frame["open_time"] < "2024-01-05 10:00"]
    hourly, _ = build_hourly(cut, cfg)
    daily = build_daily(hourly, cfg)
    assert daily["date"].max() == pd.Timestamp("2024-01-04")


def test_invalid_rows_rejected(cfg):
    frame = make_klines(cfg.symbols, days=3)
    frame.loc[3, "high"] = frame.loc[3, "low"] - 1
    hourly, rejected = build_hourly(frame, cfg)
    assert len(rejected) == 1 and len(hourly) == len(frame) - 1


def test_news_dedup_and_topics():
    frame = pd.DataFrame({
        "url_hash": ["a", "b", "c"], "title": ["Bitcoin ETF approved by SEC today", "Bitcoin ETF approved by SEC today", "Ethereum hack drains funds from bridge"],
        "summary": ["", "", ""], "domain": ["a", "b", "c"], "language": ["english"] * 3,
        "seen_at": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-02 12:00"]),
        "source": ["gdelt:bitcoin", "rss:x", "rss:x"], "ingested_at": pd.Timestamp("2024-01-03"),
    })
    news = build_news(frame)
    assert len(news) == 2
    etf = news[news["title_clean"].str.contains("etf")].iloc[0]
    assert etf["topic_etf"] == 1 and etf["topic_regulation"] == 1 and etf["mention_BTCUSDT"] == 1
    assert etf["date"] == pd.Timestamp("2024-01-01")
    hack = news[news["title_clean"].str.contains("hack")].iloc[0]
    assert hack["topic_hack"] == 1 and hack["sentiment"] < 0
