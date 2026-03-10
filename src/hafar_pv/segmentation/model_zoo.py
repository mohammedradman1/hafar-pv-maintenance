"""Factory functions for segmentation architectures."""

from __future__ import annotations

from functools import lru_cache

import segmentation_models_pytorch as smp
import torch.nn as nn


@lru_cache(maxsize=8)
def build_segmentation_model(name: str = "unet-resnet34") -> nn.Module:
    """Return a segmentation model by name."""

    if name == "unet-resnet34":
        return smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
    if name == "deeplab-resnet50":
        return smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=3, classes=1)
    raise ValueError(f"Unknown segmentation model: {name}")
