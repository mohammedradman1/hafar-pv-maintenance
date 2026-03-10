"""Preprocessing utilities for the Kaggle Photovoltaic System Thermography dataset."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .preprocess import normalize_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes


@dataclass(frozen=True)
class ModuleAnnotation:
    """Single photovoltaic module annotation."""

    polygon: np.ndarray  # (N, 2)
    defective: bool


@dataclass(frozen=True)
class SceneAnnotation:
    """Container for a scene image and its annotated modules."""

    image_path: Path
    annotation_path: Path
    modules: list[ModuleAnnotation]


@dataclass(frozen=True)
class PanelRecord:
    """Metadata describing a processed panel crop."""

    panel_id: str
    scene_id: str
    npz_path: Path
    defective: bool
    area_px: int
    mean_temperature: float
    max_temperature: float


@dataclass(frozen=True)
class SceneRecord:
    """Metadata describing a processed scene image."""

    scene_id: str
    npz_path: Path
    num_modules: int
    num_defective: int
    temperature_mean: float
    temperature_max: float


# ---------------------------------------------------------------------------
# Helper functions


def _try_import_radiometric_extractors():
    """Return callable extractors for radiometric JPG files.

    The dataset stores thermal data in FLIR R-JPG containers. We try importing
    `flirimageextractor` first (widely used) and fall back to `flyr` if available.
    The returned list contains callables that accept a path-like object and
    return an `np.ndarray` of the thermal matrix.
    """

    extractors: list = []

    try:
        from flirimageextractor import FlirImageExtractor  # type: ignore

        def _flir(path: Path) -> np.ndarray:
            extractor = FlirImageExtractor()
            extractor.process_image(str(path))
            return extractor.get_thermal_np().astype(np.float32)

        extractors.append(_flir)
    except (ImportError, ModuleNotFoundError):
        logger.debug("flirimageextractor not available; using fallback readers")

    try:
        from flyr import RawImage  # type: ignore

        def _flyr(path: Path) -> np.ndarray:
            raw = RawImage.open(str(path))
            return raw.raw_image.astype(np.float32)

        extractors.append(_flyr)
    except (ImportError, ModuleNotFoundError):
        logger.debug("flyr not available; using fallback readers")

    return extractors


_RADIOMETRIC_EXTRACTORS = _try_import_radiometric_extractors()


def load_radiometric_frame(path: Path) -> np.ndarray:
    """Load a radiometric JPEG into a float32 numpy array.

    If specialized extractors are unavailable, fall back to 8-bit grayscale using
    OpenCV. The downstream pipeline normalizes intensities, so this still allows
    experimentation albeit without absolute temperature fidelity.
    """

    for extractor in _RADIOMETRIC_EXTRACTORS:
        try:
            frame = extractor(path)
            logger.debug("Loaded radiometric frame via %s", extractor.__qualname__)
            return frame
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Radiometric extractor %s failed for %s: %s", extractor, path, exc)

    logger.warning("Falling back to 8-bit grayscale load for %s", path)
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise FileNotFoundError(f"Unable to read image from {path}")
    return frame.astype(np.float32)


def parse_scene(annotation_path: Path, image_candidates: Sequence[Path]) -> SceneAnnotation:
    """Parse annotation JSON and pair it with the corresponding image."""

    with annotation_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    # Heuristics to identify the matching image
    stem = annotation_path.stem
    image_path = None
    for candidate in image_candidates:
        if candidate.stem == stem:
            image_path = candidate
            break

    if image_path is None:
        raise FileNotFoundError(f"No image found for annotation {annotation_path}")

    modules: list[ModuleAnnotation] = []
    for item in payload.get("instances", []):
        corners = item.get("corners") or []
        if len(corners) < 3:
            logger.debug("Skipping malformed annotation in %s: %s", annotation_path, item)
            continue
        polygon = np.array([(pt["x"], pt["y"]) for pt in corners], dtype=np.float32)
        defective = bool(item.get("defected_modules", False))
        modules.append(ModuleAnnotation(polygon=polygon, defective=defective))

    if not modules:
        logger.warning("Annotation %s contains no valid modules", annotation_path)

    return SceneAnnotation(image_path=image_path, annotation_path=annotation_path, modules=modules)


def _rasterize_polygon(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _compute_temperature_stats(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    return (float(values.mean()), float(values.max()))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Preprocessor implementation


class PhotovoltaicThermographyPreprocessor:
    """Convert raw Kaggle dataset into NPZ tensors consumable by the project."""

    def __init__(
        self,
        raw_root: Path,
        output_root: Path,
        resize_panels_to: tuple[int, int] | None = None,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        random_state: int = 42,
    ) -> None:
        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
            raise ValueError("Validation and test fractions must be non-negative and sum to < 1")
        self.raw_root = raw_root
        self.output_root = output_root
        self.resize_panels_to = resize_panels_to
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.random_state = random_state

    # Public API -----------------------------------------------------------------

    def run(self) -> None:
        logger.info("Starting preprocessing of Photovoltaic Thermography dataset")

        scenes = self._discover_scenes()
        if not scenes:
            logger.warning("No scenes discovered under %s", self.raw_root)
            return

        frames_dir = self.output_root / "frames"
        panels_dir = self.output_root / "panels"
        meta_dir = self.output_root / "metadata"
        ensure_dir(frames_dir)
        ensure_dir(panels_dir)
        ensure_dir(meta_dir)

        panel_records: list[PanelRecord] = []
        scene_records: list[SceneRecord] = []

        for scene in scenes:
            frame = load_radiometric_frame(scene.image_path)
            normalized_frame = normalize_image(frame)
            height, width = frame.shape[:2]

            aggregate_mask = np.zeros((height, width), dtype=np.uint8)
            num_defective = 0
            per_module_masks: list[np.ndarray] = []

            for idx, module in enumerate(scene.modules):
                mask = _rasterize_polygon(module.polygon, (height, width))
                if not mask.any():
                    logger.debug(
                        "Skipping empty mask for module %d in %s", idx, scene.annotation_path
                    )
                    continue
                aggregate_mask = np.maximum(aggregate_mask, mask)
                per_module_masks.append(mask)
                if module.defective:
                    num_defective += 1

                y, x = np.where(mask > 0)
                panel_area = int(mask.sum())
                masked_values = frame[y, x] if panel_area else np.array([], dtype=frame.dtype)
                mean_temp, max_temp = _compute_temperature_stats(masked_values)

                x0, y0, w, h = cv2.boundingRect(mask)
                panel_crop = frame[y0 : y0 + h, x0 : x0 + w]
                mask_crop = mask[y0 : y0 + h, x0 : x0 + w]
                if self.resize_panels_to is not None:
                    panel_crop = cv2.resize(
                        panel_crop, self.resize_panels_to, interpolation=cv2.INTER_AREA
                    )
                    mask_crop = cv2.resize(
                        mask_crop, self.resize_panels_to, interpolation=cv2.INTER_NEAREST
                    )

                panel_crop = normalize_image(panel_crop)
                mask_crop = mask_crop.astype(np.float32)

                scene_id = scene.image_path.stem
                panel_id = f"{scene_id}_{idx:03d}"
                panel_path = panels_dir / f"{panel_id}.npz"
                np.savez_compressed(
                    panel_path,
                    image=panel_crop.astype(np.float32),
                    mask=mask_crop,
                    label=np.array(int(module.defective), dtype=np.int64),
                )

                panel_records.append(
                    PanelRecord(
                        panel_id=panel_id,
                        scene_id=scene_id,
                        npz_path=panel_path.relative_to(self.output_root),
                        defective=module.defective,
                        area_px=panel_area,
                        mean_temperature=mean_temp,
                        max_temperature=max_temp,
                    )
                )

            # Persist scene-level tensor only if we have masks
            scene_id = scene.image_path.stem
            scene_npz = frames_dir / f"{scene_id}.npz"
            np.savez_compressed(
                scene_npz,
                image=normalized_frame.astype(np.float32),
                mask=aggregate_mask.astype(np.float32),
            )

            mean_scene_temp, max_scene_temp = _compute_temperature_stats(frame[aggregate_mask > 0])
            scene_records.append(
                SceneRecord(
                    scene_id=scene_id,
                    npz_path=scene_npz.relative_to(self.output_root),
                    num_modules=len(per_module_masks),
                    num_defective=num_defective,
                    temperature_mean=mean_scene_temp,
                    temperature_max=max_scene_temp,
                )
            )

        if panel_records:
            panels_df = self._build_panel_dataframe(panel_records)
            panels_df.to_csv(meta_dir / "panels_manifest.csv", index=False)
            logger.info("Wrote panel manifest with %d entries", len(panels_df))
        else:
            logger.warning("Panel manifest is empty; check annotations")

        if scene_records:
            scenes_df = pd.DataFrame([scene.__dict__ for scene in scene_records])
            scenes_df.to_csv(meta_dir / "scenes_manifest.csv", index=False)
            logger.info("Wrote scenes manifest with %d entries", len(scenes_df))

    # Internal helpers -----------------------------------------------------------

    def _discover_scenes(self) -> list[SceneAnnotation]:
        annotations = sorted(self.raw_root.rglob("*.json"))
        images = list(self._iter_images(self.raw_root))
        logger.info("Discovered %d annotation files and %d images", len(annotations), len(images))
        scenes = []
        for annotation_path in annotations:
            try:
                scene = parse_scene(annotation_path, images)
            except FileNotFoundError as exc:
                logger.warning("Skipping annotation %s: %s", annotation_path, exc)
                continue
            scenes.append(scene)
        return scenes

    @staticmethod
    def _iter_images(root: Path) -> Iterator[Path]:
        for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
            yield from root.rglob(ext)

    def _build_panel_dataframe(self, records: Iterable[PanelRecord]) -> pd.DataFrame:
        df = pd.DataFrame([record.__dict__ for record in records])
        if df.empty:
            return df

        # Stratified split by defect label
        labels = df["defective"].astype(int)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_fraction, random_state=self.random_state
        )
        train_val_idx, test_idx = next(splitter.split(df, labels))

        df.loc[df.index[test_idx], "split"] = "test"

        remaining = df.index[train_val_idx]
        val_fraction_adjusted = self.val_fraction / (1 - self.test_fraction)
        val_fraction_adjusted = min(max(val_fraction_adjusted, 0.0), 1.0)

        if math.isclose(val_fraction_adjusted, 0.0):
            df.loc[remaining, "split"] = "train"
            return df

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction_adjusted, random_state=self.random_state
        )
        train_idx, val_idx = next(splitter.split(df.loc[remaining], labels.loc[remaining]))
        train_indices = remaining[train_idx]
        val_indices = remaining[val_idx]

        df.loc[train_indices, "split"] = "train"
        df.loc[val_indices, "split"] = "val"
        return df


# ---------------------------------------------------------------------------
# Convenience loaders


def load_panels_manifest(processed_root: Path) -> pd.DataFrame:
    """Load panel manifest CSV as a DataFrame."""

    manifest_path = processed_root / "metadata" / "panels_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Panel manifest not found at {manifest_path}. Run the preprocessor first."
        )
    df = pd.read_csv(manifest_path)
    df["npz_path"] = df["npz_path"].apply(lambda p: processed_root / Path(p))
    return df


def load_scenes_manifest(processed_root: Path) -> pd.DataFrame:
    """Load scene manifest CSV as a DataFrame."""

    manifest_path = processed_root / "metadata" / "scenes_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Scene manifest not found at {manifest_path}. Run the preprocessor first."
        )
    df = pd.read_csv(manifest_path)
    df["npz_path"] = df["npz_path"].apply(lambda p: processed_root / Path(p))
    return df
