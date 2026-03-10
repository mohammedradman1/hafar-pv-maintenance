"""Glue code to wire segmentation and fault detection together."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from ..data.preprocess import build_panel_crops, normalize_image
from ..faults import FaultDetector
from ..segmentation import Segmenter


@dataclasses.dataclass
class PipelineResult:
    """Outputs for a single source image."""

    panels: list[torch.Tensor]
    fault_scores: list[torch.Tensor]


@dataclasses.dataclass
class InferencePipeline:
    """Chain segmentation and fault detection with simple orchestration."""

    segmenter: Segmenter
    detector: FaultDetector

    def run(self, frame: np.ndarray, masks: Iterable[np.ndarray]) -> PipelineResult:
        normalized = normalize_image(frame)
        crops = build_panel_crops(normalized, masks)
        crops_tensor = [torch.from_numpy(crop).permute(2, 0, 1).float() for crop in crops]
        fault_scores = self.detector(crops_tensor)
        return PipelineResult(panels=crops_tensor, fault_scores=fault_scores)

    def run_on_path(self, image_path: Path) -> PipelineResult:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to load image from {image_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masks = self.segmenter([torch.from_numpy(frame).permute(2, 0, 1).float()])
        np_masks = [mask.numpy() for mask in masks]
        return self.run(frame, np_masks)
