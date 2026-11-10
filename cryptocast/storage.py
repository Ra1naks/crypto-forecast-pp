"""DuckDB (транзакционное состояние) + Parquet (слои raw/staging/mart для чтения дашбордом).

DuckDB допускает одного пишущего процесса, поэтому дашборд читает только Parquet:
файлы заменяются атомарно и не конфликтуют с идущим конвейером.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import duckdb
import pandas as pd

from cryptocast.config import Config

DDL = """
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS raw.klines (
    symbol VARCHAR NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume DOUBLE, quote_volume DOUBLE, trades BIGINT,
    taker_buy_base DOUBLE, taker_buy_quote DOUBLE,
    close_time TIMESTAMP, source VARCHAR, ingested_at TIMESTAMP,
    PRIMARY KEY (symbol, open_time)
);
CREATE TABLE IF NOT EXISTS raw.fng (
    date DATE PRIMARY KEY, value INTEGER, classification VARCHAR, ingested_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS raw.news (
    url_hash VARCHAR PRIMARY KEY,
    url VARCHAR, title VARCHAR, summary VARCHAR, domain VARCHAR,
    seen_at TIMESTAMP, source VARCHAR, language VARCHAR, ingested_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ops.watermarks (
    source VARCHAR PRIMARY KEY, value VARCHAR, updated_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ops.load_log (
    run_id VARCHAR, source VARCHAR, started_at TIMESTAMP, finished_at TIMESTAMP,
    status VARCHAR, received BIGINT, inserted BIGINT, duplicates BIGINT,
    watermark_before VARCHAR, watermark_after VARCHAR, error VARCHAR
);
CREATE TABLE IF NOT EXISTS ops.dq_results (
    run_id VARCHAR, check_name VARCHAR, scope VARCHAR, severity VARCHAR,
    passed BOOLEAN, value DOUBLE, threshold DOUBLE, details VARCHAR, checked_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id VARCHAR, step VARCHAR, status VARCHAR, started_at TIMESTAMP,
    finished_at TIMESTAMP, details VARCHAR
);
"""

# Таблицы состояния данных для проверки идемпотентности.
STATE_TABLES = {
    "raw.klines": ["symbol", "open_time"],
    "raw.fng": ["date"],
    "raw.news": ["url_hash"],
}
EXCLUDED_FROM_HASH = {"ingested_at"}


class Store:
    def __init__(self, cfg: Config, read_only: bool = False):
        self.cfg = cfg
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.lake_dir.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(cfg.db_path), read_only=read_only)
        if not read_only:
            self.con.execute(DDL)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def df(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def scalar(self, sql: str, params: list | None = None):
        row = self.con.execute(sql, params or []).fetchone()
        return row[0] if row else None

    def get_watermark(self, source: str) -> str | None:
        return self.scalar("SELECT value FROM ops.watermarks WHERE source = ?", [source])

    def set_watermark(self, source: str, value: str) -> None:
        self.con.execute(
            """INSERT INTO ops.watermarks VALUES (?, ?, now()::TIMESTAMP)
               ON CONFLICT (source) DO UPDATE SET value = excluded.value,
               updated_at = excluded.updated_at""",
            [source, value],
        )

    def upsert_ignore(self, table: str, frame: pd.DataFrame) -> int:
        """Вставляет строки с новым ключом и возвращает их число (ON CONFLICT DO NOTHING)."""
        if frame.empty:
            return 0
        before = self.scalar(f"SELECT count(*) FROM {table}")
        columns = ", ".join(frame.columns)
        self.con.register("_incoming", frame)
        try:
            self.con.execute(
                f"INSERT INTO {table} ({columns}) SELECT {columns} FROM _incoming "
                "ON CONFLICT DO NOTHING"
            )
        finally:
            self.con.unregister("_incoming")
        return int(self.scalar(f"SELECT count(*) FROM {table}") - before)

    def replace_table(self, table: str, frame: pd.DataFrame) -> None:
        self.con.register("_frame", frame)
        try:
            self.con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _frame")
        finally:
            self.con.unregister("_frame")

    def state_hash(self) -> dict[str, str]:
        """Содержательные хеши таблиц исходных данных (без служебного времени вставки)."""
        result = {}
        for table, keys in STATE_TABLES.items():
            info = self.con.execute(f"PRAGMA table_info('{table}')").df()
            columns = [c for c in info["name"] if c not in EXCLUDED_FROM_HASH]
            frame = self.df(f"SELECT {', '.join(columns)} FROM {table} ORDER BY {', '.join(keys)}")
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=False).values.tobytes()
            ).hexdigest()
            result[table] = f"{len(frame)}:{digest[:16]}"
        return result


def _atomic_replace(tmp: Path, dest: Path, attempts: int = 20) -> None:
    for attempt in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:  # Windows: файл кратко занят читателем
            if attempt == attempts - 1:
                raise
            time.sleep(0.25)


def write_parquet(cfg: Config, layer: str, name: str, frame: pd.DataFrame) -> Path:
    folder = cfg.lake_dir / layer
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{name}.parquet"
    tmp = folder / f".{name}.{os.getpid()}.tmp"
    frame.to_parquet(tmp, index=False)
    _atomic_replace(tmp, dest)
    return dest


def read_parquet(cfg: Config, layer: str, name: str) -> pd.DataFrame:
    path = cfg.lake_dir / layer / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def export_table(store: Store, layer: str, table: str, name: str | None = None) -> None:
    write_parquet(store.cfg, layer, name or table.split(".")[-1], store.df(f"SELECT * FROM {table}"))
