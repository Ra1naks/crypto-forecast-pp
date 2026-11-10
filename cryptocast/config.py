"""Загрузка конфигурации: config.yaml + переменные окружения (.env)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path
    data_dir: Path
    artifacts_dir: Path
    snapshot_dir: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.raw["paths"]["db_file"]

    @property
    def lake_dir(self) -> Path:
        return self.data_dir / self.raw["paths"]["lake_dir"]

    @property
    def symbols(self) -> list[str]:
        return list(self.raw["universe"]["symbols"])

    @property
    def targets(self) -> list[str]:
        return list(self.raw["models"]["targets"])

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    def config_hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def with_dirs(self, data_dir: Path, artifacts_dir: Path) -> "Config":
        """Копия конфигурации с другими каталогами (тесты, симуляция)."""
        return Config(self.raw, self.root, Path(data_dir), Path(artifacts_dir), self.snapshot_dir)


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = _resolve(PROJECT_ROOT, path or os.getenv("CRYPTOCAST_CONFIG", "config.yaml"))
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    data_dir = _resolve(PROJECT_ROOT, os.getenv("CRYPTOCAST_DATA_DIR", raw["paths"]["data_dir"]))
    artifacts_dir = _resolve(
        PROJECT_ROOT, os.getenv("CRYPTOCAST_ARTIFACTS_DIR", raw["paths"]["artifacts_dir"])
    )
    snapshot_dir = _resolve(PROJECT_ROOT, raw["paths"]["snapshot_dir"])
    return Config(raw, PROJECT_ROOT, data_dir, artifacts_dir, snapshot_dir)
