"""Общие утилиты."""
from __future__ import annotations

import datetime as dt
import logging
import os
import subprocess
import sys
import uuid

from cryptocast.config import PROJECT_ROOT


def setup_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=os.getenv("CRYPTOCAST_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def utcnow() -> dt.datetime:
    """Текущее время UTC без tzinfo: в хранилище все метки времени наивные UTC."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def new_run_id() -> str:
    return utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def git_state() -> tuple[str, bool]:
    """Текущий коммит и признак незакоммиченных изменений."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            ).strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
