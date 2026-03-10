"""Data acquisition and preprocessing utilities."""

from .datasets import PanelDataset
from .download import DatasetRegistry, download_datasets
from .photovoltaic_thermography import (
    PhotovoltaicThermographyPreprocessor,
    load_panels_manifest,
    load_scenes_manifest,
)
from .preprocess import build_panel_crops, normalize_image

__all__ = [
    "DatasetRegistry",
    "PanelDataset",
    "PhotovoltaicThermographyPreprocessor",
    "build_panel_crops",
    "download_datasets",
    "load_panels_manifest",
    "load_scenes_manifest",
    "normalize_image",
]
