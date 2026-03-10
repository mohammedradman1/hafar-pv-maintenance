"""CLI entry point to download and preprocess datasets."""

from __future__ import annotations

import logging

from hafar_pv.config import get_settings
from hafar_pv.data import DatasetRegistry, download_datasets
from hafar_pv.utils import configure_logging


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    settings = get_settings()
    logger.info("Using data root: %s", settings.data_root)

    registry = DatasetRegistry()
    # TODO: register dataset fetchers once implemented.
    download_datasets(registry, settings)


if __name__ == "__main__":
    main()
