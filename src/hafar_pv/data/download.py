"""Dataset discovery and download helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable

from ..config import AppSettings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata describing an available dataset."""

    name: str
    url: str
    license: str
    note: str
    fetcher: Callable[[Path], None]


class DatasetRegistry:
    """Simple registry for dataset download routines."""

    def __init__(self) -> None:
        self._datasets: Dict[str, DatasetInfo] = {}

    def register(self, info: DatasetInfo) -> None:
        logger.debug("Registering dataset %s", info.name)
        if info.name in self._datasets:
            msg = f"Dataset '{info.name}' already registered"
            raise ValueError(msg)
        self._datasets[info.name] = info

    def list(self) -> Iterable[DatasetInfo]:
        return self._datasets.values()

    def get(self, name: str) -> DatasetInfo:
        return self._datasets[name]


def download_datasets(registry: DatasetRegistry | None = None, settings: AppSettings | None = None) -> None:
    """Iterate over the registry and trigger dataset downloads.

    Each dataset is responsible for handling its own idempotency and extraction.
    """

    settings = settings or get_settings()
    registry = registry or DatasetRegistry()

    target_root = settings.data_root / "external"
    target_root.mkdir(parents=True, exist_ok=True)

    if not any(True for _ in registry.list()):
        logger.warning("No datasets registered; skipping download")
        return

    for dataset in registry.list():
        dataset_dir = target_root / dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching dataset %s from %s", dataset.name, dataset.url)
        try:
            dataset.fetcher(dataset_dir)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to fetch %s: %s", dataset.name, exc)
            raise
