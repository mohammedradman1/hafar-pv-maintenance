"""Segmentation metric utilities."""

from __future__ import annotations

import torch


class SegmentationMetrics:
    """Callable wrapper computing IoU and Dice."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        dice = (2 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6)
        iou = (intersection + 1e-6) / (union + 1e-6)

        return {
            "dice": float(dice.mean().item()),
            "iou": float(iou.mean().item()),
        }
