"""Fault detection model factory."""

from __future__ import annotations

from functools import lru_cache

import torch.nn as nn
import torchvision.models as models


@lru_cache(maxsize=8)
def build_fault_model(name: str, num_classes: int) -> nn.Module:
    """Return a classifier backbone with adjusted output layer."""

    if name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unknown fault detection model: {name}")
