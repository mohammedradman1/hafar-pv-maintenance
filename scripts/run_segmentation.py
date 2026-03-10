"""Train segmentation models from the command line."""

from __future__ import annotations

import logging

from hafar_pv.segmentation import SegmentationConfig, SegmentationTrainer
from hafar_pv.utils import configure_logging


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Segmentation training stub")

    # TODO: load datasets via datamodule abstraction
    trainer = SegmentationTrainer(SegmentationConfig())
    logger.info("Instantiate trainer: %s", trainer)


if __name__ == "__main__":
    main()
