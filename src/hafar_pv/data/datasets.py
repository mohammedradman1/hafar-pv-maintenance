"""Dataset wrappers for PyTorch dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PanelDataset(Dataset[dict[str, torch.Tensor]]):
    """Simple dataset that loads NPZ tensors for panels or full scenes."""

    def __init__(
        self,
        items: list[Path],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_key: Optional[str] = None,
    ) -> None:
        self.items = items
        self.transform = transform
        self.target_transform = target_transform
        self.target_key = target_key

    def __len__(self) -> int:  # noqa: D401
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.items[index]
        sample = np.load(path)
        image_np = sample["image"]
        target_np = self._resolve_target(sample)

        image = torch.from_numpy(self._to_chw(image_np)).float()
        target = torch.from_numpy(self._prepare_target(target_np)).float()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return {"image": image, "target": target}

    def _resolve_target(self, sample: np.lib.npyio.NpzFile) -> np.ndarray:
        if self.target_key is not None:
            return sample[self.target_key]
        if "mask" in sample:
            return sample["mask"]
        if "label" in sample:
            return sample["label"]
        return np.array(0, dtype=np.float32)

    @staticmethod
    def _to_chw(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image[None, ...]
        if image.ndim == 3 and image.shape[0] <= 4:
            # Already channel-first (assumed) if first dim is small
            return image
        if image.ndim == 3:
            return np.transpose(image, (2, 0, 1))
        raise ValueError(f"Unsupported image dimensionality: {image.shape}")

    @staticmethod
    def _prepare_target(target: np.ndarray) -> np.ndarray:
        if target.ndim == 0:
            return np.array([target], dtype=np.float32)
        if target.ndim == 1:
            return target.astype(np.float32)
        if target.ndim == 2:
            return target[None, ...]
        if target.ndim == 3 and target.shape[0] <= 4:
            return target.astype(np.float32)
        if target.ndim == 3:
            return np.transpose(target, (2, 0, 1)).astype(np.float32)
        raise ValueError(f"Unsupported target dimensionality: {target.shape}")
