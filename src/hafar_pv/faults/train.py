"""Fault detection training helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader, Dataset

from .model_zoo import build_fault_model

logger = logging.getLogger(__name__)


@dataclass
class FaultDetectionConfig:
    """Configuration for panel fault detection."""

    model_name: str = "efficientnet_v2_s"
    learning_rate: float = 1e-4
    batch_size: int = 16
    class_weights: Sequence[float] | None = None
    max_epochs: int = 20


class FaultDetectionModule(pl.LightningModule):
    """Lightning module for classification of panel defects."""

    def __init__(self, config: FaultDetectionConfig, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = build_fault_model(config.model_name, num_classes=num_classes)
        weights = None
        if config.class_weights is not None:
            weights = torch.tensor(config.class_weights, dtype=torch.float32)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(images)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        images = batch["image"]
        labels = batch["target"].long()
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        images = batch["image"]
        labels = batch["target"].long()
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [scheduler]


class FaultDetectionTrainer:
    """Wrapper for fault detection experiments."""

    def __init__(self, config: FaultDetectionConfig | None = None) -> None:
        self.config = config or FaultDetectionConfig()

    def fit(self, train_ds: Dataset, val_ds: Dataset | None = None, num_classes: int = 2) -> pl.Trainer:
        logger.info("Training fault detector with %s", self.config.model_name)
        module = FaultDetectionModule(self.config, num_classes=num_classes)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(val_ds, batch_size=self.config.batch_size)
        trainer = pl.Trainer(max_epochs=self.config.max_epochs)
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return trainer
