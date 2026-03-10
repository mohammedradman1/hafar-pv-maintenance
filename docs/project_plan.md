# Project Plan

## Roadmap

1. **Phase 0 – Foundation**
   - Finalize repository scaffolding, dependency management, pre-commit hooks, and contribution guide.
   - Stand up placeholder Streamlit view and continuous integration smoke checks.
2. **Phase 1 – Data Sourcing**
   - Rank open datasets for segmentation and fault detection tasks.
   - Script downloads, license validations, and storage under `data/` (raw/interim/processed split).
   - Create lightweight EDA notebooks validating label formats, resolutions, and class balance.
3. **Phase 2 – Segmentation**
   - Curate aerial/thermal imagery and corresponding panel masks.
   - Build PyTorch `Dataset` objects with augmentations (albumentations) and tiling.
   - Train a baseline model (e.g., UNet) and track metrics such as mIoU, pixel recall, and panel-level F1.
4. **Phase 3 – Fault Detection**
   - Pair segmented panels with defect annotations across modalities (EL, IR, RGB).
   - Train classifiers/detectors (CNN, ViT, anomaly detection) with calibration.
   - Evaluate per-defect precision/recall and composite health scores.
5. **Phase 4 – Inference Pipeline**
   - Chain ingestion → segmentation → cropping → fault detection with retry/logging.
   - Export TorchScript/ONNX variants and create batch/real-time runners.
6. **Phase 5 – App & Ops**
   - Expand Streamlit experience (upload, gallery, metrics, reporting).
   - Add API wrapper, monitoring hooks, and deployment automation (Streamlit Cloud/Azure).

## Target Datasets

- **ELPV (Kaggle)** – 2.7k electroluminescence tiles labelled for micro-cracks, finger interruptions, corrosion; ideal for supervised fault detection.
- **Infrared Solar Panel Faults (Kaggle)** – ~6k thermographic frames grouped by hotspot, bypassed, and healthy panels; aligns with thermal analysis.
- **Solar Panel Surface Defects (Kaggle)** – RGB close-up imagery for dust, scratches, discoloration, bird droppings; covers surface-level classification.
- **Roboflow Universe – Solar Panels Semantic Segmentation** – Drone/RGB imagery with per-panel masks exportable to COCO/Pascal; bootstraps the segmentation stage.
- **PV Module Dirt/Soiling Dataset (Mendeley)** – Clean vs soiled samples with soiling ratio metadata; supports robustness checks and regression metrics.
- **Custom Annotation Pipeline** – If coverage gaps exist, leverage CVAT/Label Studio on aerial datasets (SolarNet, Google Earth Engine) to extend masks.

## Repository Structure

```
hafar-pv-maintenance/
  pyproject.toml
  README.md
  data/
    raw/          # immutable source dumps (gitignored)
    interim/      # cleaned/filtered versions
    processed/    # model-ready tensors/features
    external/     # third-party metadata
  models/         # trained weights + metadata (gitignored)
  notebooks/     # exploratory notebooks (tracked selectively)
  experiments/   # configs and results for segmentation & fault studies
  src/
    hafar_pv/
      config/       # Pydantic settings, path registry
      data/         # downloaders, preprocessors, dataset classes
      segmentation/ # training + inference for panel masks
      faults/       # defect classifiers/detectors
      pipeline/     # orchestration glue
      app/          # Streamlit UI and components
      utils/        # shared utilities (logging, viz)
  scripts/        # CLI helpers for data prep and inference
  tests/          # pytest suites for data/models/app
  docs/           # extended documentation (this plan, research notes)
  .github/workflows/
```
