# src/ — Core Modules

Reusable Python modules for the anemia cascade CDS pipeline.
Imported by the notebooks in `notebooks/`.

## Configuration

`cds_config.py` assumes a Google Drive project layout:

    MyDrive/0000_Kaan_CDS/        <- project root (cds_root)
    ├── src/                       <- this folder
    └── frozen_models/             <- trained models (not in repo)

Adjust the `cds_root` / `ml_root` paths in `cds_config.py` to match
your own setup before running the notebooks.

## Modules

| File | Purpose |
|------|---------|
| `cds_config.py` | Project paths, experiment registry, notebook setup |
| `plot_style.py` | Academic figure styling (palette, save helpers) |
| `m1_calibration.py` | Probability calibration |
| `m2_thresholds.py` | Decision threshold optimization |
| `m3_errors.py` | Error / confusion-matrix analysis |
| `m4_e2e.py` | End-to-end cascade pipeline |
| `m5_shap.py` | SHAP explainability |
| `m6_conformal.py` | Conformal prediction (APS) |
| `m7_cascade.py` | Cascade simulation + reflex recommendations |
| `m8_cds_report.py` | CDS patient report generation |
