"""Segmentation training loop skeleton."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import SegmentationMetrics
from .model_zoo import build_segmentation_model

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    """Configuration options for segmentation training."""

    model_name: str = "unet-resnet34"
    learning_rate: float = 1e-3
    batch_size: int = 4
    max_epochs: int = 25


class SegmentationModule(pl.LightningModule):
    """PyTorch Lightning module for panel segmentation."""

    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = build_segmentation_model(config.model_name)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.metrics = SegmentationMetrics()

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(images)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        images = batch["image"]
        masks = batch["target"]
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        images = batch["image"]
        masks = batch["target"]
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        metrics = self.metrics(outputs, masks)
        self.log_dict({"val_loss": loss, **metrics})
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class SegmentationTrainer:
    """Convenience wrapper to run segmentation experiments."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.config = config or SegmentationConfig()

    def fit(self, train_ds: Dataset, val_ds: Dataset | None = None) -> pl.Trainer:
        logger.info("Starting segmentation training with model %s", self.config.model_name)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(val_ds, batch_size=self.config.batch_size)
        module = SegmentationModule(self.config)
        trainer = pl.Trainer(max_epochs=self.config.max_epochs)
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return trainer


def train_from_config(train_ds: Dataset, val_ds: Dataset | None = None) -> None:
    """Hook for Hydra-powered experiments."""

    trainer = SegmentationTrainer()
    trainer.fit(train_ds, val_ds)
