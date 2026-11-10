"""Контроль качества исходных данных. Критичный сбой блокирует обновление витрины."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from cryptocast.storage import Store
from cryptocast.utils import utcnow

log = logging.getLogger(__name__)
CRITICAL, WARNING = "critical", "warning"


@dataclass
class Check:
    check_name: str
    scope: str
    severity: str
    passed: bool
    value: float
    threshold: float
    details: str = ""


def _expected_hours(first: pd.Timestamp, last: pd.Timestamp) -> int:
    return int((last - first) / pd.Timedelta(hours=1)) + 1


def check_klines(store: Store, now: dt.datetime) -> list[Check]:
    q = store.cfg["quality"]
    checks: list[Check] = []
    start = pd.Timestamp(store.cfg["universe"]["start_date"])
    for symbol in store.cfg.symbols:
        stats = store.df(
            """SELECT count(*) AS n, min(open_time) AS first, max(open_time) AS last,
                      sum(CASE WHEN high < low THEN 1 ELSE 0 END) AS bad_hl,
                      sum(CASE WHEN high < greatest(open, close) OR low > least(open, close)
                               THEN 1 ELSE 0 END) AS bad_ohlc,
                      sum(CASE WHEN volume < 0 OR quote_volume < 0 THEN 1 ELSE 0 END) AS bad_vol
               FROM raw.klines WHERE symbol = ?""",
            [symbol],
        ).iloc[0]
        if int(stats["n"]) == 0:
            checks.append(Check("klines_present", symbol, CRITICAL, False, 0, 1, "нет свечей"))
            continue
        first, last = pd.Timestamp(stats["first"]), pd.Timestamp(stats["last"])
        expected = _expected_hours(max(first, start), last)
        missing = max(expected - int(stats["n"]), 0)
        ratio = missing / expected
        checks.append(Check("klines_gap_ratio", symbol, CRITICAL, ratio <= q["max_gap_ratio_critical"],
                            ratio, q["max_gap_ratio_critical"], f"пропущено {missing} из {expected} часов"))
        recent_from = last - pd.Timedelta(days=7)
        recent_n = store.scalar(
            "SELECT count(*) FROM raw.klines WHERE symbol = ? AND open_time > ?", [symbol, recent_from]
        )
        recent_expected = _expected_hours(recent_from + pd.Timedelta(hours=1), last)
        recent_missing = max(recent_expected - int(recent_n), 0)
        checks.append(Check("klines_recent_gaps", symbol, WARNING, recent_missing <= q["max_gap_hours_recent"],
                            recent_missing, q["max_gap_hours_recent"], "пропуски за последние 7 суток"))
        duplicates = store.scalar(
            """SELECT count(*) FROM (SELECT open_time FROM raw.klines WHERE symbol = ?
               GROUP BY open_time HAVING count(*) > 1)""",
            [symbol],
        )
        checks.append(Check("klines_duplicates", symbol, CRITICAL, int(duplicates) == 0, duplicates, 0))
        checks.append(Check("klines_high_ge_low", symbol, CRITICAL, int(stats["bad_hl"]) == 0, stats["bad_hl"], 0))
        checks.append(Check("klines_ohlc_consistent", symbol, CRITICAL, int(stats["bad_ohlc"]) == 0, stats["bad_ohlc"], 0))
        checks.append(Check("klines_volume_nonneg", symbol, CRITICAL, int(stats["bad_vol"]) == 0, stats["bad_vol"], 0))
        age_hours = (pd.Timestamp(now) - last - pd.Timedelta(hours=1)) / pd.Timedelta(hours=1)
        checks.append(Check("klines_freshness_hours", symbol, CRITICAL,
                            age_hours <= q["freshness_klines_hours_critical"], age_hours,
                            q["freshness_klines_hours_critical"], f"последняя свеча {last}"))
    return checks


def check_auxiliary(store: Store, now: dt.datetime) -> list[Check]:
    q = store.cfg["quality"]
    checks: list[Check] = []
    last_fng = store.scalar("SELECT max(date) FROM raw.fng")
    fng_age = (now.date() - last_fng).days if last_fng else 9999.0
    checks.append(Check("fng_freshness_days", "fng", WARNING, fng_age <= q["freshness_fng_days_warn"],
                        fng_age, q["freshness_fng_days_warn"]))
    last_news = store.scalar("SELECT max(seen_at) FROM raw.news")
    news_age = (pd.Timestamp(now) - pd.Timestamp(last_news)) / pd.Timedelta(days=1) if last_news else 9999.0
    checks.append(Check("news_freshness_days", "news", WARNING, news_age <= q["freshness_news_days_warn"],
                        news_age, q["freshness_news_days_warn"]))
    bad_title = store.scalar("SELECT count(*) FROM raw.news WHERE title IS NULL OR length(trim(title)) = 0")
    checks.append(Check("news_title_present", "news", WARNING, int(bad_title) == 0, bad_title, 0))
    return checks


def run_quality(store: Store, run_id: str, now: dt.datetime | None = None) -> bool:
    """Записывает проверки в ops.dq_results; возвращает True, если нет критичных сбоев."""
    now = now or utcnow()
    checks = check_klines(store, now) + check_auxiliary(store, now)
    frame = pd.DataFrame([c.__dict__ for c in checks])
    frame.insert(0, "run_id", run_id)
    frame["value"] = frame["value"].astype(float)
    frame["threshold"] = frame["threshold"].astype(float)
    frame["checked_at"] = now
    store.con.register("_dq", frame)
    store.con.execute("INSERT INTO ops.dq_results SELECT * FROM _dq")
    store.con.unregister("_dq")
    failed = frame[(~frame["passed"]) & (frame["severity"] == CRITICAL)]
    warned = frame[(~frame["passed"]) & (frame["severity"] == WARNING)]
    log.info("Контроль качества: проверок=%s, критичных сбоев=%s, предупреждений=%s",
             len(frame), len(failed), len(warned))
    for row in failed.itertuples():
        log.error("КРИТИЧНО %s [%s]: %s порог %s", row.check_name, row.scope, row.value, row.threshold)
    return failed.empty
