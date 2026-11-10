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
from pathlib import Path
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
        result.received += len(frame)
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


def _archive_paths(store: Store) -> tuple[Path, Path]:
    archive = store.cfg.root / store.cfg["sources"]["gdelt"]["archive_file"]
    return archive, archive.with_suffix(".state.json")


def bootstrap_gdelt_archive(store: Store, result: SourceResult) -> None:
    """Если состояния ещё нет, поднимает историю из архива, собранного этим же коллектором."""
    archive, state_file = _archive_paths(store)
    if store.get_watermark(f"{result.source}:head") or not archive.exists() or not state_file.exists():
        return
    frame = pd.read_parquet(archive)
    frame = frame[frame["source"] == result.source].copy()
    frame["ingested_at"] = utcnow()
    columns = ["url_hash", "url", "title", "summary", "domain", "seen_at", "source", "language", "ingested_at"]
    result.received += len(frame)
    result.inserted += store.upsert_ignore("raw.news", frame[columns])
    state = json.loads(state_file.read_text(encoding="utf-8")).get(result.source, {})
    for edge in ("head", "tail"):
        if state.get(edge):
            store.set_watermark(f"{result.source}:{edge}", state[edge])
    result.note = f"история поднята из архива {archive.name}"


def export_gdelt_archive(store: Store) -> int:
    """Сохраняет новости GDELT и границы загрузки для быстрого развёртывания с нуля."""
    archive, state_file = _archive_paths(store)
    archive.parent.mkdir(parents=True, exist_ok=True)
    frame = store.df("SELECT * EXCLUDE (ingested_at) FROM raw.news WHERE source LIKE 'gdelt%' ORDER BY url_hash")
    frame.to_parquet(archive, index=False)
    state: dict[str, dict[str, str | None]] = {}
    for key in store.cfg["sources"]["gdelt"]["queries"]:
        source = f"gdelt:{key}"
        state[source] = {edge: store.get_watermark(f"{source}:{edge}") for edge in ("head", "tail")}
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(frame)


def _fetch_gdelt_window(store: Store, client: HttpClient, source: str, query: str,
                        start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    cfg = store.cfg["sources"]["gdelt"]
    params = {
        "query": query, "mode": "ArtList", "format": "json", "maxrecords": cfg["maxrecords"],
        "sort": cfg["sort"], "startdatetime": _gdelt_time(start), "enddatetime": _gdelt_time(end),
    }
    response = client.get(cfg["url"], params=params, timeout=cfg["timeout_seconds"])
    try:
        articles = json.loads(response.text, strict=False).get("articles", [])
    except json.JSONDecodeError as exc:
        raise SourceUnavailable(f"битый JSON GDELT: {exc}") from exc
    return parse_gdelt(articles, source=source)


def collect_gdelt(store: Store, client: HttpClient, key: str, now: dt.datetime | None = None,
                  max_requests: int | None = None) -> SourceResult:
    """Двусторонняя загрузка окон: сначала свежие данные вперёд, затем история назад.

    head — конец последнего загруженного свежего окна, tail — начало самого старого.
    Обе границы двигаются только после успешной записи окна, поэтому обрыв ничего не теряет.
    """
    cfg = store.cfg["sources"]["gdelt"]
    source = f"gdelt:{key}"
    head_key, tail_key = f"{source}:head", f"{source}:tail"
    result = SourceResult(source=source)
    bootstrap_gdelt_archive(store, result)
    result.watermark_before = f"{store.get_watermark(head_key)}|{store.get_watermark(tail_key)}"
    now = (now or utcnow()).replace(minute=0, second=0, microsecond=0)
    window = dt.timedelta(days=cfg["window_days"])
    history_start = dt.datetime.fromisoformat(store.cfg["universe"]["start_date"])
    budget = cfg["max_requests_per_run"] if max_requests is None else max_requests
    done, failures = 0, 0

    def plan() -> tuple[dt.datetime, dt.datetime, str] | None:
        head, tail = store.get_watermark(head_key), store.get_watermark(tail_key)
        if head is None:
            return now - window, now, "first"
        head_dt = dt.datetime.fromisoformat(head)
        if head_dt < now:
            start = head_dt - dt.timedelta(days=cfg["overlap_days"])
            return start, min(start + window, now), "forward"
        tail_dt = dt.datetime.fromisoformat(tail)
        if tail_dt > history_start:
            return max(tail_dt - window, history_start), tail_dt, "backward"
        return None

    while done < budget and (step := plan()) is not None:
        start, end, direction = step
        try:
            frame = _fetch_gdelt_window(store, client, source, cfg["queries"][key], start, end)
        except SourceUnavailable as exc:
            failures += 1
            done += 1
            if failures >= cfg["max_consecutive_failures"]:
                result.partial_error = f"окно {start:%Y-%m-%d}: {str(exc)[:200]}"
                break
            time.sleep(cfg["retry_pause_seconds"])
            continue
        failures = 0
        done += 1
        result.received += len(frame)
        result.inserted += store.upsert_ignore("raw.news", frame)
        if direction in ("first", "forward"):
            store.set_watermark(head_key, end.isoformat())
        if direction in ("first", "backward"):
            store.set_watermark(tail_key, start.isoformat())
        if done < budget:
            time.sleep(cfg["pause_seconds"])
    result.watermark_after = f"{store.get_watermark(head_key)}|{store.get_watermark(tail_key)}"
    return result


# --------------------------------- Новости: Hacker News -----------------------------
def parse_hn(hits: Iterable[dict], key: str) -> pd.DataFrame:
    """Заголовки Hacker News. Поле автора намеренно не читается."""
    rows = []
    for hit in hits:
        title = (hit.get("title") or "").strip()
        object_id = hit.get("objectID")
        if not title or not object_id:
            continue
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        rows.append(
            {
                "url_hash": url_hash(url),
                "url": normalize_url(url),
                "title": clean_summary(title),
                "summary": "",
                "domain": urlsplit(url).netloc.lower(),
                "seen_at": dt.datetime.fromtimestamp(int(hit["created_at_i"]), dt.timezone.utc).replace(tzinfo=None),
                "source": f"hn:{key}",
                "language": "english",
                "ingested_at": utcnow(),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates("url_hash") if not frame.empty else frame


def collect_hn(store: Store, client: HttpClient, key: str, now: dt.datetime | None = None) -> SourceResult:
    """Загружает окна по возрастанию дат; watermark — конец последнего успешного окна."""
    cfg = store.cfg["sources"]["hn"]
    source = f"hn:{key}"
    result = SourceResult(source=source, watermark_before=store.get_watermark(source))
    now = (now or utcnow()).replace(minute=0, second=0, microsecond=0)
    window = dt.timedelta(days=cfg["window_days"])
    if result.watermark_before:
        start = dt.datetime.fromisoformat(result.watermark_before) - dt.timedelta(days=cfg["overlap_days"])
    else:
        start = dt.datetime.fromisoformat(store.cfg["universe"]["start_date"])
    epoch = lambda value: int(value.replace(tzinfo=dt.timezone.utc).timestamp())  # noqa: E731
    while start < now:
        end = min(start + window, now)
        params = {"query": key, "tags": "story", "hitsPerPage": cfg["hits_per_page"],
                  "numericFilters": f"created_at_i>={epoch(start)},created_at_i<{epoch(end)}"}
        try:
            hits = client.get(cfg["url"], params=params).json().get("hits", [])
        except SourceUnavailable as exc:
            result.partial_error = f"окно {start:%Y-%m-%d}: {exc}"
            break
        frame = parse_hn(hits, key)
        result.received += len(frame)
        result.inserted += store.upsert_ignore("raw.news", frame)
        store.set_watermark(source, end.isoformat())
        start = end
        if start < now:
            time.sleep(cfg["pause_seconds"])
    result.watermark_after = store.get_watermark(source)
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
    if enabled("hn"):
        for key in cfg["sources"]["hn"]["queries"]:
            statuses[f"hn:{key}"] = run_with_log(store, run_id, f"hn:{key}", lambda k=key: collect_hn(store, client, k))
    if enabled("gdelt"):
        gdelt = cfg["sources"]["gdelt"]
        gdelt_client = HttpClient(
            timeout=gdelt["timeout_seconds"], max_retries=1, backoff_base=gdelt["retry_pause_seconds"],
            backoff_max=http["backoff_max_seconds"], user_agent=http["user_agent"],
        )
        for key in gdelt["queries"]:
            statuses[f"gdelt:{key}"] = run_with_log(
                store, run_id, f"gdelt:{key}",
                lambda k=key: collect_gdelt(store, gdelt_client, k, max_requests=gdelt_max_requests),
            )
    return statuses
