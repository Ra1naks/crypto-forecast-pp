from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from cryptocast.config import Config, load_config
from cryptocast.pipeline import prepare_schema
from cryptocast.storage import Store


@pytest.fixture
def cfg(tmp_path) -> Config:
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["universe"]["symbols"] = ["BTCUSDT", "ETHUSDT"]
    raw["universe"]["start_date"] = "2024-01-01"
    raw["sources"]["klines"]["pause_seconds"] = 0
    raw["sources"]["gdelt"]["pause_seconds"] = 0
    raw["sources"]["gdelt"]["retry_pause_seconds"] = 0
    raw["sources"]["gdelt"]["archive_file"] = "missing_archive.parquet"
    raw["validation"].update({"holdout_days": 15, "initial_train_days": 60, "test_days": 15, "gap_days": 2, "warmup_days": 5})
    raw["models"]["lightgbm"].update({"n_estimators": 20, "num_threads": 1})
    raw["models"]["mlp"].update({"max_iter": 30})
    return Config(raw, base.root, tmp_path / "data", tmp_path / "artifacts", tmp_path / "snap")


@pytest.fixture
def store(cfg):
    with Store(cfg) as st:
        prepare_schema(st)
        yield st


def make_klines(symbols: list[str], start: str = "2024-01-01", days: int = 140, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, symbol in enumerate(symbols):
        hours = pd.date_range(start, periods=days * 24, freq="h")
        returns = rng.normal(0, 0.01, len(hours))
        close = 100 * (i + 1) * np.exp(np.cumsum(returns))
        openp = np.r_[close[0], close[:-1]]
        high = np.maximum(openp, close) * 1.002
        low = np.minimum(openp, close) * 0.998
        volume = np.exp(rng.normal(5, 0.5, len(hours)))
        quote = volume * close
        frames.append(pd.DataFrame({
            "symbol": symbol, "open_time": hours, "open": openp, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": quote, "trades": rng.integers(100, 1000, len(hours)),
            "taker_buy_base": volume * 0.5, "taker_buy_quote": quote * 0.5,
            "close_time": hours + pd.Timedelta(minutes=59, seconds=59, milliseconds=999), "source": "synthetic",
            "ingested_at": pd.Timestamp("2024-06-01"),
        }))
    return pd.concat(frames, ignore_index=True)


def make_fng(days: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2023-12-01", periods=days, freq="D")
    return pd.DataFrame({"date": dates, "fng_value": (40 + 20 * np.sin(np.arange(days) / 9)).astype(int), "classification": "Neutral"})


def make_news(days: int = 140) -> pd.DataFrame:
    rows = []
    words = ["bitcoin rally etf approved", "ethereum hack stolen funds", "sec lawsuit regulation crypto", "market rebound adoption grows", "btc crash selloff fear"]
    for day in range(days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for j in range(6):
            text = f"{words[(day + j) % 5]} story {day % 7}"
            rows.append({"url_hash": f"h{day}-{j}", "seen_at": date + pd.Timedelta(hours=j), "date": date, "domain": "x.test", "origin": "gdelt",
                         "title_clean": text, "text": text, "sentiment": float((j - 3) / 3), "topic_hack": int("hack" in text), "topic_regulation": int("sec" in text),
                         "topic_etf": int("etf" in text), "mention_BTCUSDT": int("bitcoin" in text or "btc" in text), "mention_ETHUSDT": int("ethereum" in text)})
    return pd.DataFrame(rows)
