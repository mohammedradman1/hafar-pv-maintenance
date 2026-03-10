# Photovoltaic System Thermography Integration Plan

## Dataset Snapshot
- **Source:** Kaggle – [Photovoltaic system thermography](https://www.kaggle.com/datasets/marcosgabriel/photovoltaic-system-thermography)
- **License:** CC0 1.0 (public domain)
- **Size:** ~90 MB (ZIP) containing ~120 radiometric FLIR R-JPG images + JSON annotations
- **Modality:** Thermal infrared imagery, drone-captured over a 6 MWp PV plant
- **Labels:** Instance annotations per module with quadrilateral corners and `defected_modules` boolean flag
- **Use Cases:** Thermal panel segmentation (module localization) and binary fault tagging (hotspot vs. healthy)

## Viability Assessment
- **Strengths:**
  - Legally frictionless (CC0) and ready for redistribution within the project.
  - Provides both localization and defect labels, matching Phase 1 (data sourcing), Phase 2 (segmentation), and Phase 3 (fault detection) objectives.
  - Radiometric data preserves absolute temperature, enabling physics-aware features (temperature deltas, mean module heat).
- **Limitations:**
  - Small sample count; requires transfer learning, heavy augmentation, and possibly additional datasets for robustness.
  - Single-site capture may cause domain shift when applied to different plants or cameras.
  - Annotation format (quadrilateral corners) needs conversion to masks; no pre-generated segmentation masks.
  - Mixed provenance (research paper + community GitHub) may introduce varying resolutions and noise levels.
- **Conclusion:** Viable as a baseline thermal dataset and for rapid prototyping. Recommended to combine with additional thermal datasets later to improve generalization.

## Ingestion Plan
1. **Credential Setup:**
   - Obtain Kaggle API token (`kaggle.json`) and place it in the environment where downloads will run (e.g., Kaggle notebook, local machine, or Colab).
   - Configure environment variable `KAGGLE_CONFIG_DIR` when running outside Kaggle.
2. **Download Script:**
   - Implement `scripts/download_photovoltaic_thermography.py` (future work) using the Kaggle API (`kaggle datasets download marcosgabriel/photovoltaic-system-thermography`).
   - Store the raw ZIP under `data/external/photovoltaic-system-thermography/raw.zip` and extract contents into `data/external/photovoltaic-system-thermography/raw/`.
3. **Integrity Checks:**
   - Record Kaggle dataset version, SHA256 checksum, and download timestamp in a manifest file (e.g., `data/external/photovoltaic-system-thermography/manifest.json`).
4. **Version Control:**
   - Keep raw data out of Git (already handled by `.gitignore`). Document dataset usage in `docs/data_sources.md` with license and retrieval details.

## Preprocessing Pipeline Outline
1. **Radiometric Extraction:**
   - Use the Flyr library to read R-JPG metadata and extract the radiometric temperature matrix.
   - Convert to `float32` arrays and normalize using `normalize_image` in `src/hafar_pv/data/preprocess.py` (retain temperature statistics in metadata).
2. **Annotation Parsing:**
   - Load JSON annotations and parse `instances`→`corners` and `defected_modules`.
   - Generate binary masks per instance via polygon rasterization (e.g., `shapely` + `skimage.draw.polygon` or OpenCV fill).
   - Aggregate module masks into scene-level mask arrays for semantic segmentation targets.
3. **Panel Crops:**
   - Apply `build_panel_crops` or a grid-based cropper to extract individual module images.
   - Store each crop as compressed NPZ (keys: `image`, `mask`, `label`) under `data/processed/photovoltaic-system-thermography/panels/`.
   - Attach metadata (source image id, polygon id, defect flag, temperature stats) in a CSV or Parquet manifest.
4. **Splitting & Augmentation Prep:**
   - Create stratified train/val/test splits ensuring defect representation in each split.
   - Save split manifests referencing panel NPZ paths.
5. **Quality Assurance:**
   - Visual spot checks: overlay masks on thermal frames, inspect random crops, verify label balance.
   - Compute dataset stats (panel count, defect ratio, temperature range) and log them for reports.

## Training Workflow Strategy
1. **Execution Environment:**
   - Prefer Kaggle Notebooks for proximity to dataset, or Google Colab with Drive-mounted processed data for longer runs.
2. **Segmentation Baseline:**
   - Model: `unet-resnet34` from `segmentation_models_pytorch`.
   - Inputs: radiometric frames + generated masks.
   - Metrics: mIoU, Dice, pixel recall.
3. **Fault Detection Baseline:**
   - Model: `efficientnet_v2_s` fine-tune on panel crops.
   - Metrics: accuracy, F1, ROC-AUC, calibration plots.
4. **Outputs:**
   - Store checkpoints (`.ckpt`), metrics JSON, confusion matrices.
   - Upload to Kaggle output or Drive, then copy into `/models` (ignoring large files if necessary, but keep references/metadata).
5. **Reproducibility:**
   - Log hyperparameters and random seeds.
   - Version processed datasets (include manifest hash) to ensure training notebooks remain reproducible.

## Prerequisites & Open Questions
- **Dependencies:** Confirm allowance for Flyr, Shapely, scikit-image, Albumentations, and wandb (optional) in project requirements.
- **Hardware:** Determine preferred GPU tier (Kaggle P100/T4 vs. Colab A100 for Pro users) to tune batch sizes.
- **Data Storage:** Decide on long-term processed data location (Google Drive vs. Kaggle dataset vs. on-prem storage).
- **Scaling Plan:** Identify complementary datasets to mitigate small-sample limitations (e.g., Infrared Solar Panel Faults).
- **Annotation Consistency:** Validate that all JSON files follow the modified schema; note any missing/duplicated entries.

## Next Actions
1. Implement automated downloader + manifest generation.
2. Prototype radiometric extraction and mask rasterization script.
3. Finalize train/val/test split manifests and validate data integrity.
4. Run exploratory data analysis notebook to document findings and confirm defect balance.
5. Execute baseline training using the provided notebook template (see `notebooks/photovoltaic_thermography_training_template.ipynb`).
