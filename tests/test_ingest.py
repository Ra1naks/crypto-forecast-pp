"""Инкрементальность, идемпотентность, устойчивость и журнал загрузок."""
from __future__ import annotations

import datetime as dt

import pytest

from cryptocast.ingest import sources
from cryptocast.ingest.http_client import HttpClient, SourceUnavailable
from cryptocast.ingest.sources import collect_fng, collect_klines, normalize_url, parse_klines, run_with_log, url_hash


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text="", content=b""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self.content = content

    def json(self):
        return self._payload


def kline_row(open_ms: int):
    return [open_ms, "100.0", "110.0", "90.0", "105.0", "10.0", open_ms + 3_599_999, "1000.0", 50, "5.0", "500.0", "0"]


class FakeKlinesClient:
    """Ведёт себя как биржа: отдаёт свечи начиная со startTime; последняя свеча может быть незакрытой."""

    def __init__(self, end: dt.datetime):
        self.end_ms = int(end.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        hour = 3_600_000
        start = params["startTime"] - params["startTime"] % hour
        limit = params["limit"]
        rows = [kline_row(start + i * hour) for i in range(limit) if start + i * hour <= self.end_ms]
        return FakeResponse(payload=rows)


NOW = dt.datetime(2024, 1, 3, 5, 30)


def test_only_closed_candles(store):
    client = FakeKlinesClient(NOW)
    collect_klines(store, client, "BTCUSDT", now=NOW)
    # Свеча 05:00-06:00 ещё не закрыта на момент 05:30 и не сохраняется.
    assert store.scalar("SELECT max(open_time) FROM raw.klines") == dt.datetime(2024, 1, 3, 4)


def test_incremental_loading(store):
    client = FakeKlinesClient(NOW)
    first = collect_klines(store, client, "BTCUSDT", now=NOW)
    assert first.inserted == 53
    next_time = NOW + dt.timedelta(hours=2)
    client.end_ms = int(next_time.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    client.calls.clear()
    second = collect_klines(store, client, "BTCUSDT", now=next_time)
    assert second.inserted == 2
    assert client.calls[0]["startTime"] == int(dt.datetime(2024, 1, 3, 5, tzinfo=dt.timezone.utc).timestamp() * 1000)


def test_idempotent_two_runs_same_state(store):
    client = FakeKlinesClient(NOW)
    collect_klines(store, client, "BTCUSDT", now=NOW)
    state_one = store.state_hash()
    client.calls.clear()
    result = collect_klines(store, client, "BTCUSDT", now=NOW)
    assert store.state_hash() == state_one
    assert result.inserted == 0


def test_upsert_ignores_existing_keys(store):
    import pandas as pd

    frame = parse_klines("BTCUSDT", [kline_row(1704067200000)], now_ms=1804067200000, source="x")
    assert store.upsert_ignore("raw.klines", frame) == 1
    assert store.upsert_ignore("raw.klines", frame) == 0
    assert isinstance(frame, pd.DataFrame)


def test_retry_after_is_respected():
    sleeps: list[float] = []
    responses = [FakeResponse(429, headers={"Retry-After": "7"}, text="slow"), FakeResponse(200, payload={"ok": 1})]

    class Session:
        headers: dict = {}

        def get(self, *a, **k):
            return responses.pop(0)

    client = HttpClient(max_retries=2, sleep=sleeps.append, session=Session())
    assert client.get("http://x").json() == {"ok": 1}
    assert sleeps == [7.0]


def test_backoff_then_failure_raises():
    sleeps: list[float] = []

    class Session:
        headers: dict = {}

        def get(self, *a, **k):
            return FakeResponse(503, text="down")

    client = HttpClient(max_retries=2, backoff_base=1, sleep=sleeps.append, session=Session())
    with pytest.raises(SourceUnavailable):
        client.get("http://x")
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_failure_of_one_source_is_logged_not_raised(store):
    def boom():
        raise SourceUnavailable("источник недоступен")

    assert run_with_log(store, "run1", "fake", boom) == "failed"
    row = store.df("SELECT * FROM ops.load_log WHERE run_id='run1'").iloc[0]
    assert row["status"] == "failed" and "недоступен" in row["error"]
    assert row["inserted"] == 0


def test_load_log_records_counts(store):
    client = FakeKlinesClient(NOW)
    run_with_log(store, "r2", "klines:BTCUSDT", lambda: collect_klines(store, client, "BTCUSDT", now=NOW))
    row = store.df("SELECT * FROM ops.load_log WHERE run_id='r2'").iloc[0]
    assert row["received"] == 53 and row["inserted"] == 53 and row["duplicates"] == 0
    assert row["watermark_after"] is not None


def test_fng_incremental(store):
    class Fng:
        def __init__(self):
            self.limits = []

        def get(self, url, params=None, timeout=None):
            self.limits.append(params["limit"])
            ts = int(dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc).timestamp())
            return FakeResponse(payload={"data": [{"value": "50", "value_classification": "Neutral", "timestamp": str(ts)}]})

    client = Fng()
    collect_fng(store, client, now=dt.datetime(2024, 1, 2, 12))
    collect_fng(store, client, now=dt.datetime(2024, 1, 5, 12))
    assert client.limits == [0, 5]
    assert store.scalar("SELECT count(*) FROM raw.fng") == 1


def test_news_hash_stable_and_no_authors():
    assert url_hash("https://A.com/x/?utm_source=z#frag") == url_hash("https://a.com/x")
    assert normalize_url("https://a.com/x/") == "https://a.com/x"
    entry = {"title": "Bitcoin rises", "link": "https://a.com/p", "summary": "<p>Text by John Smith here</p>",
             "published_parsed": (2024, 1, 1, 0, 0, 0, 0, 1, 0), "author": "John Smith"}
    frame = sources.parse_rss_entries([entry], "feed")
    assert "author" not in frame.columns
    assert "John" not in frame.iloc[0]["summary"]


def test_gdelt_two_way_watermark(store):
    class Gdelt:
        def __init__(self):
            self.windows = []

        def get(self, url, params=None, timeout=None):
            self.windows.append((params["startdatetime"], params["enddatetime"]))
            import json as _json

            article = {"url": f"https://a.com/{params['startdatetime']}", "title": "Bitcoin title long enough", "seendate": params["startdatetime"][:8] + "T010000Z", "domain": "a.com", "language": "English"}
            return FakeResponse(text=_json.dumps({"articles": [article]}))

    client = Gdelt()
    now = dt.datetime(2024, 2, 1, 12)
    sources.collect_gdelt(store, client, "bitcoin", now=now, max_requests=2)
    assert store.get_watermark("gdelt:bitcoin:head") == "2024-02-01T12:00:00"
    assert store.get_watermark("gdelt:bitcoin:tail") == "2024-01-18T12:00:00"
    client.windows.clear()
    later = now + dt.timedelta(days=1)
    sources.collect_gdelt(store, client, "bitcoin", now=later, max_requests=1)
    assert client.windows[0][0] == "20240130120000"
