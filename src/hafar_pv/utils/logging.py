"""Logging helpers."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO, force: bool = False) -> None:
    """Configure application-wide logging with Rich handler."""

    if logging.getLogger().handlers and not force:
        return

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
