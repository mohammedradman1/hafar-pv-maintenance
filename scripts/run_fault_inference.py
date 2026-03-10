"""Run fault detection inference on prepared crops."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from hafar_pv.faults import FaultDetector
from hafar_pv.utils import configure_logging


def main(crops_dir: Path) -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    detector = FaultDetector(model_name="efficientnet_v2_s", num_classes=4)
    logger.info("Loaded detector: %s", detector.model_name)

    placeholders = [torch.rand(3, 224, 224) for _ in range(2)]
    scores = detector(placeholders)
    for idx, score in enumerate(scores):
        logger.info("Panel %d scores: %s", idx, score)


if __name__ == "__main__":
    main(Path("data/processed/panel_crops"))
