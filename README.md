# Hafar PV Maintenance

Solar panel monitoring toolkit covering segmentation of photovoltaic arrays and downstream fault detection. The stack is PyTorch-first with a Streamlit interface planned for visualization and operations support.

## Project Vision

- **Stage 1 – Segmentation:** isolate individual panels or strings from aerial, thermal, or ground imagery.
- **Stage 2 – Fault Detection:** classify cropped panels for defects such as cracks, hotspots, corrosion, or soiling.
- **Delivery:** expose the full pipeline through a Streamlit dashboard and optional API for batch inference.

## Getting Started

1. **Install dependencies** *(after choosing a virtual environment)*:

   ```bash
   pip install --upgrade pip
   pip install .[dev,tests]
   ```

2. **Run quality gates**:

   ```bash
   ruff check src tests
   black --check src tests
   pytest
   ```

3. **Streamlit placeholder**:

   ```bash
   streamlit run src/hafar_pv/app/streamlit_app.py
   ```

## Repository Layout

```
data/                # raw, interim, processed, external datasets (gitignored)
docs/                # project plan, research notes
experiments/         # tracked experiments for segmentation and fault models
notebooks/           # exploratory analysis (git-kept via placeholder)
scripts/             # CLI entry points for data prep and inference
src/hafar_pv/        # core python package
tests/               # pytest suites
```

Refer to `docs/project_plan.md` for detailed roadmap and dataset references.
