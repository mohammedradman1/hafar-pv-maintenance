"""Segmentation inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .model_zoo import build_segmentation_model


@dataclass
class Segmenter:
    """Load a segmentation checkpoint and run predictions."""

    model_name: str
    checkpoint_path: Path | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model = build_segmentation_model(self.model_name)
        if self.checkpoint_path and self.checkpoint_path.exists():
            state = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, images: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for image in images:
            image = image.to(self.device)
            logits = self.model(image.unsqueeze(0))
            outputs.append(torch.sigmoid(logits).squeeze(0).cpu())
        return outputs
