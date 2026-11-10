"""Коллекторы источников. Каждый возвращает SourceResult и сам двигает свой watermark."""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import html
import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import pandas as pd

from cryptocast.ingest.http_client import HttpClient, SourceUnavailable
from cryptocast.storage import Store
from cryptocast.utils import utcnow

log = logging.getLogger(__name__)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
BYLINE_RE = re.compile(r"\b(?:by|author:?)\s+[A-ZА-Я][\w'.-]+(?:\s+[A-ZА-Я][\w'.-]+){0,2}", re.UNICODE)
TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref", "source")


@dataclass
class SourceResult:
    source: str
    received: int = 0
    inserted: int = 0
    watermark_before: str | None = None
    watermark_after: str | None = None
    note: str | None = None
    partial_error: str | None = None


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith(TRACKING_PARAMS)]
    )
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


def clean_summary(text: str | None) -> str:
    """Текст без HTML и подписей авторов: персональные данные в проекте не храним."""
    if not text:
        return ""
    plain = html.unescape(TAG_RE.sub(" ", text))
    return SPACE_RE.sub(" ", BYLINE_RE.sub(" ", plain)).strip()[:1000]


# --------------------------------- Биржевые свечи ---------------------------------
KLINE_COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def parse_klines(symbol: str, rows: list[list], now_ms: int, source: str) -> pd.DataFrame:
    """Разбирает ответ биржи и оставляет только закрытые свечи."""
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        return frame
    frame = frame[frame["close_time_ms"].astype("int64") < now_ms].copy()
    if frame.empty:
        return frame
    for column in ("open", "high", "low", "close", "volume", "quote_volume",
                   "taker_buy_base", "taker_buy_quote"):
        frame[column] = frame[column].astype(float)
    frame["trades"] = frame["trades"].astype("int64")
    frame["open_time"] = pd.to_datetime(frame["open_time_ms"].astype("int64"), unit="ms")
    frame["close_time"] = pd.to_datetime(frame["close_time_ms"].astype("int64"), unit="ms")
    frame["symbol"] = symbol
    frame["source"] = source
    frame["ingested_at"] = utcnow()
    return frame[["symbol", "open_time", "open", "high", "low", "close", "volume", "quote_volume",
                  "trades", "taker_buy_base", "taker_buy_quote", "close_time", "source", "ingested_at"]]


def _get_klines_page(client: HttpClient, base_urls: list[str], params: dict):
    """Перебирает зеркала биржи; ошибка только если недоступны все."""
    errors = []
    for base in base_urls:
        try:
            return client.get(f"{base}/api/v3/klines", params=params).json(), base
        except SourceUnavailable as exc:
            errors.append(f"{base}: {exc}")
    raise SourceUnavailable("; ".join(errors))


def collect_klines(store: Store, client: HttpClient, symbol: str, now: dt.datetime | None = None) -> SourceResult:
    cfg = store.cfg["sources"]["klines"]
    source = f"klines:{symbol}"
    result = SourceResult(source=source, watermark_before=store.get_watermark(source))
    now = now or utcnow()
    now_ms = int(now.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    hour_ms = 3_600_000
    if result.watermark_before:
        start_ms = int(pd.Timestamp(result.watermark_before).tz_localize("UTC").timestamp() * 1000) + hour_ms
    else:
        start_ms = int(pd.Timestamp(store.cfg["universe"]["start_date"]).tz_localize("UTC").timestamp() * 1000)
    while start_ms + hour_ms <= now_ms:
        params = {"symbol": symbol, "interval": cfg["interval"], "startTime": start_ms, "limit": cfg["page_limit"]}
        try:
            rows, _base = _get_klines_page(client, cfg["base_urls"], params)
        except SourceUnavailable as exc:
            if result.received:  # уже сохранённые страницы сохраняются; watermark не теряется
                result.partial_error = str(exc)
                break
            raise
        if not rows:
            break
        frame = parse_klines(symbol, rows, now_ms, source="binance")
        result.received += len(rows) if not frame.empty else 0
        if frame.empty:
            break
        result.inserted += store.upsert_ignore("raw.klines", frame)
        watermark = frame["open_time"].max()
        store.set_watermark(source, watermark.isoformat())
        result.watermark_after = watermark.isoformat()
        start_ms = int(rows[-1][0]) + hour_ms
        if len(rows) < cfg["page_limit"]:
            break
        time.sleep(cfg["pause_seconds"])
    result.watermark_after = result.watermark_after or result.watermark_before
    return result


# --------------------------------- Индекс страха и жадности ------------------------
def collect_fng(store: Store, client: HttpClient, now: dt.datetime | None = None) -> SourceResult:
    cfg = store.cfg["sources"]["fng"]
    source = "fng"
    result = SourceResult(source=source, watermark_before=store.get_watermark(source))
    now = now or utcnow()
    if result.watermark_before:
        limit = (now.date() - dt.date.fromisoformat(result.watermark_before)).days + 2
    else:
        limit = 0  # вся история
    payload = client.get(cfg["url"], params={"limit": limit, "format": "json"}).json()
    rows = payload.get("data", [])
    frame = pd.DataFrame(
        {
            "date": [dt.datetime.fromtimestamp(int(r["timestamp"]), dt.timezone.utc).date() for r in rows],
            "value": [int(r["value"]) for r in rows],
            "classification": [r["value_classification"] for r in rows],
        }
    )
    result.received = len(frame)
    if not frame.empty:
        frame["ingested_at"] = utcnow()
        frame = frame.drop_duplicates("date")
        frame["date"] = pd.to_datetime(frame["date"])
        result.inserted = store.upsert_ignore("raw.fng", frame)
        newest = store.scalar("SELECT max(date) FROM raw.fng")
        store.set_watermark(source, newest.isoformat())
        result.watermark_after = newest.isoformat()
    else:
        result.watermark_after = result.watermark_before
    return result


# --------------------------------- Новости: GDELT ----------------------------------
def _gdelt_time(value: dt.datetime) -> str:
    return value.strftime("%Y%m%d%H%M%S")


def parse_gdelt(articles: Iterable[dict], source: str = "gdelt") -> pd.DataFrame:
    rows = []
    for article in articles:
        url, title = article.get("url"), (article.get("title") or "").strip()
        if not url or not title:
            continue
        rows.append(
            {
                "url_hash": url_hash(url),
                "url": normalize_url(url),
                "title": clean_summary(title),
                "summary": "",
                "domain": article.get("domain", ""),
                "seen_at": pd.to_datetime(article["seendate"], format="%Y%m%dT%H%M%SZ"),
                "source": source,
                "language": (article.get("language") or "").lower(),
                "ingested_at": utcnow(),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates("url_hash") if not frame.empty else frame


def bootstrap_gdelt_archive(store: Store, result: SourceResult) -> None:
    """Если БД пуста, поднимает историю из архива, ранее собранного этим же коллектором."""
    archive = store.cfg.root / store.cfg["sources"]["gdelt"]["archive_file"]
    if result.watermark_before or not archive.exists():
        return
    frame = pd.read_parquet(archive)
    frame["ingested_at"] = utcnow()
    columns = ["url_hash", "url", "title", "summary", "domain", "seen_at", "source", "language", "ingested_at"]
    result.received += len(frame)
    result.inserted += store.upsert_ignore("raw.news", frame[columns])
    watermark = frame.attrs.get("watermark") or pd.Timestamp(frame["seen_at"].max()).normalize().isoformat()
    store.set_watermark("gdelt", watermark)
    result.watermark_before = None
    result.note = f"история поднята из архива {archive.name}"


def collect_gdelt(store: Store, client: HttpClient, now: dt.datetime | None = None,
                  max_requests: int | None = None) -> SourceResult:
    """Загружает окна новостей по возрастанию дат; watermark — конец последнего успешного окна."""
    cfg = store.cfg["sources"]["gdelt"]
    source = "gdelt"
    result = SourceResult(source=source, watermark_before=store.get_watermark(source))
    bootstrap_gdelt_archive(store, result)
    now = now or utcnow()
    current = store.get_watermark(source)
    start = (
        dt.datetime.fromisoformat(current) - dt.timedelta(days=cfg["overlap_days"])
        if current
        else dt.datetime.fromisoformat(store.cfg["universe"]["start_date"])
    )
    window = dt.timedelta(days=cfg["window_days"])
    budget = cfg["max_requests_per_run"] if max_requests is None else max_requests
    done = 0
    while start < now and done < budget:
        end = min(start + window, now)
        params = {
            "query": cfg["query"], "mode": "ArtList", "format": "json",
            "maxrecords": cfg["maxrecords"], "sort": cfg["sort"],
            "startdatetime": _gdelt_time(start), "enddatetime": _gdelt_time(end),
        }
        try:
            response = client.get(cfg["url"], params=params, timeout=cfg["timeout_seconds"])
            try:
                articles = json.loads(response.text, strict=False).get("articles", [])
            except json.JSONDecodeError as exc:
                raise SourceUnavailable(f"битый JSON GDELT: {exc}") from exc
        except SourceUnavailable as exc:
            result.partial_error = f"окно {start:%Y-%m-%d}: {exc}"
            break
        frame = parse_gdelt(articles)
        result.received += len(frame)
        result.inserted += store.upsert_ignore("raw.news", frame)
        done += 1
        if end >= now:
            # Хвост (незавершённая неделя) перечитывается за счёт overlap_days.
            store.set_watermark(source, end.replace(minute=0, second=0, microsecond=0).isoformat())
        else:
            store.set_watermark(source, end.isoformat())
        start = end
        if start < now and done < budget:
            time.sleep(cfg["pause_seconds"])
    result.watermark_after = store.get_watermark(source)
    if start < now and result.partial_error is None:
        result.note = f"исчерпан лимит запросов; история догрузится в следующих запусках (до {start:%Y-%m-%d})"
    return result


# --------------------------------- Новости: RSS -------------------------------------
def parse_rss_entries(entries: Iterable, feed_name: str) -> pd.DataFrame:
    rows = []
    for entry in entries:
        link, title = entry.get("link"), (entry.get("title") or "").strip()
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if not link or not title or not parsed:
            continue
        rows.append(
            {
                "url_hash": url_hash(link),
                "url": normalize_url(link),
                "title": clean_summary(title),
                "summary": clean_summary(entry.get("summary")),
                "domain": urlsplit(link).netloc.lower(),
                # Поля автора (author, dc:creator) намеренно не читаем.
                "seen_at": dt.datetime.fromtimestamp(calendar.timegm(parsed), dt.timezone.utc).replace(tzinfo=None),
                "source": f"rss:{feed_name}",
                "language": "english",
                "ingested_at": utcnow(),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates("url_hash") if not frame.empty else frame


def collect_rss_feed(store: Store, client: HttpClient, feed_name: str, url: str) -> SourceResult:
    source = f"rss:{feed_name}"
    result = SourceResult(source=source, watermark_before=store.get_watermark(source))
    response = client.get(url, timeout=store.cfg["sources"]["rss"]["timeout_seconds"])
    frame = parse_rss_entries(feedparser.parse(response.content).entries, feed_name)
    result.received = len(frame)
    if frame.empty:
        result.watermark_after = result.watermark_before
        return result
    result.inserted = store.upsert_ignore("raw.news", frame)
    newest = frame["seen_at"].max().isoformat()
    if result.watermark_before is None or newest > result.watermark_before:
        store.set_watermark(source, newest)
    result.watermark_after = store.get_watermark(source)
    return result


# --------------------------------- Журналируемый запуск ------------------------------
def run_with_log(store: Store, run_id: str, source: str, func: Callable[[], SourceResult]) -> str:
    """Выполняет сбор одного источника; любая ошибка фиксируется и не роняет запуск."""
    started = utcnow()
    before = store.get_watermark(source)
    try:
        result = func()
        status = "partial" if result.partial_error else "success"
        error = result.partial_error or result.note
        received, inserted = result.received, result.inserted
        after = result.watermark_after
    except Exception as exc:  # noqa: BLE001 — намеренная граница отказа источника
        log.exception("Источник %s завершился с ошибкой", source)
        status, error, received, inserted = "failed", f"{type(exc).__name__}: {exc}", 0, 0
        after = store.get_watermark(source)
    store.con.execute(
        "INSERT INTO ops.load_log VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [run_id, source, started, utcnow(), status, received, inserted,
         max(received - inserted, 0), before, after, error],
    )
    log.info("%s: %s получено=%s вставлено=%s %s", source, status, received, inserted, error or "")
    return status


def ingest_all(store: Store, run_id: str, client: HttpClient | None = None,
               sources: set[str] | None = None, gdelt_max_requests: int | None = None) -> dict[str, str]:
    cfg = store.cfg
    http = cfg["http"]
    client = client or HttpClient(
        timeout=http["timeout_seconds"], max_retries=http["max_retries"],
        backoff_base=http["backoff_base_seconds"], backoff_max=http["backoff_max_seconds"],
        user_agent=http["user_agent"],
    )
    enabled = lambda name: cfg["sources"][name]["enabled"] and (sources is None or name in sources)  # noqa: E731
    statuses: dict[str, str] = {}
    if enabled("klines"):
        for symbol in cfg.symbols:
            statuses[f"klines:{symbol}"] = run_with_log(
                store, run_id, f"klines:{symbol}", lambda s=symbol: collect_klines(store, client, s)
            )
    if enabled("fng"):
        statuses["fng"] = run_with_log(store, run_id, "fng", lambda: collect_fng(store, client))
    if enabled("rss"):
        for name, url in cfg["sources"]["rss"]["feeds"].items():
            statuses[f"rss:{name}"] = run_with_log(
                store, run_id, f"rss:{name}", lambda n=name, u=url: collect_rss_feed(store, client, n, u)
            )
    if enabled("gdelt"):
        statuses["gdelt"] = run_with_log(
            store, run_id, "gdelt", lambda: collect_gdelt(store, client, max_requests=gdelt_max_requests)
        )
    return statuses
