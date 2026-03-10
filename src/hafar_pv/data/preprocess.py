"""Preprocessing utilities for solar panel imagery."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixels to [0, 1] float32."""

    if image.dtype != np.float32:
        image = image.astype(np.float32)
    max_value = image.max() or 1.0
    return image / max_value


def build_panel_crops(
    frame: np.ndarray, masks: Iterable[np.ndarray], resize_to: Tuple[int, int] | None = None
) -> list[np.ndarray]:
    """Apply binary masks to isolate panels and return cropped tensors."""

    crops: list[np.ndarray] = []
    for mask in masks:
        if mask.ndim == 2:
            mask = mask[:, :, None]
        masked = frame * mask
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        crop = masked[y : y + h, x : x + w]
        if resize_to is not None:
            crop = cv2.resize(crop, resize_to, interpolation=cv2.INTER_AREA)
        crops.append(crop)
    return crops


def persist_npz(tensor: np.ndarray, destination: Path) -> None:
    """Persist numpy arrays to NPZ format."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, tensor=tensor)
