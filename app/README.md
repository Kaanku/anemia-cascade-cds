[app_README.md](https://github.com/user-attachments/files/29481450/app_README.md)
# app/ — Anemia Cascade CDS (Streamlit)

Interactive demo of the two-stage anemia cascade clinical decision support
system: Stage 1 separates associated (AAC) from other (OAC) anemia causes,
and Stage 2 classifies the AAC subtype (IDA / HA / HGB HTZ / Normal). The app
shows confidence zones, conformal prediction sets, and reflex-test
recommendations.

> Research demonstration only — not a medical device.

## Run locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app loads the lightweight deployment models from `deploy_models/` and its
assets from `app/assets/`. No real patient data is required to run it.

## Run on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. At share.streamlit.io, create a new app pointing to
   `app/streamlit_app.py` on the `main` branch.

The deployment models total ~73 MB, well within Community Cloud limits.

## Input modes

- **Demo patient** — pick a cohort (train / test / temporal) and data type
  (anonymized real or synthetic), then a patient. Raw values are shown and the
  reference label is displayed for comparison (it is never fed to the model).
- **Manual entry** — type raw measured values directly.

## Scenarios

- **CBC only** — 14 CBC parameters.
- **CBC + biochemistry** — adds ferritin, serum iron, LD, and UIBC.

## What ships here

| Path | Contents |
|------|----------|
| `streamlit_app.py` | The Streamlit UI |
| `cascade_engine.py` | Model loading, feature derivation, cascade, zones, conformal, reflex |
| `assets/threshold_config.json` | Stage-1 thresholds + Stage-2 zone boundaries |
| `assets/conformal_qhat.json` | Pre-computed APS conformal thresholds |
| `assets/reflex_rules.json` | Reflex-test recommendation rules |
| `assets/calibrators/` | Calibration registry + isotonic calibrator |
| `assets/demo_data/` | Anonymized real + synthetic demo cohorts (no identifiable data) |

Derived ratio features (e.g. `hgb_g_d_l_div_mcv_f_l`) are computed from raw
inputs inside `cascade_engine.py`; only raw measured values are entered.
