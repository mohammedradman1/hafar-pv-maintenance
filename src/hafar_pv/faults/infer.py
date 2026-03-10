"""Inference helpers for fault detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .model_zoo import build_fault_model


@dataclass
class FaultDetector:
    """Load a classifier checkpoint and return predictions."""

    model_name: str
    num_classes: int
    checkpoint_path: Path | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model = build_fault_model(self.model_name, num_classes=self.num_classes)
        if self.checkpoint_path and self.checkpoint_path.exists():
            state = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, images: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for image in images:
            logits = self.model(image.to(self.device).unsqueeze(0))
            outputs.append(torch.softmax(logits, dim=1).squeeze(0).cpu())
        return outputs
