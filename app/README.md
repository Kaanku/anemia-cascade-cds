# Two-tier anaemia cascade · research demo

Interactive demonstration of the two-tier clinical decision support cascade of our study on the differential
diagnosis of acquired and hereditary erythrocyte disorders (manuscript under review).

> **Research use only. Not a medical device.** The models in this app are trained on synthetic data; their
> outputs are not those of the study model and must not be used for patient care.

## What it does

* **Tier 1** uses the full blood count and reticulocyte panel of a Sysmex XN analyser.
  Stage 1 separates anaemias with a classifiable cause (iron deficiency anaemia, haemolytic anaemia,
  heterozygous haemoglobinopathy, normal red cell profile) from anaemias of another cause; Stage 2 assigns one
  of the four classes. A case is finalised at Tier 1 when Stage 1 says "classifiable" and the Stage 2 top-class
  probability reaches the locked HIGH cut-off.
* **Tier 2** adds the biochemistry panel (ferritin, iron, UIBC, LDH). Missing analytes are imputed with
  label-free KNN, as in the study.
* The app shows the probabilities (after the locked display calibration), the confidence zone
  (HIGH / MEDIUM / LOW), the 90 % conformal prediction set (APS, conservative version) and the reflex-test
  recommendation of the study's 14 rules.

## Models and locked decisions

* **Model:** TabPFN-3.5-fast (Prior Labs) with 2 ensemble members, seed 42, CPU; one model per stage and tier,
  with the feature lists selected in the study (`data/features.json`). The study itself used TabPFN-3.5 with its
  default ensemble. The lighter model keeps the demo usable on a 2-core server: about 20 s for Tier 1 and
  20 s more for Tier 2.
* **Lock** (`data/lock.json`): Stage 1 thresholds, HIGH cut-offs, LOW < 0.35, conformal quantiles and display
  calibration, set with the study's rules on real development patients. Each patient was predicted by a model
  of the same kind trained on synthetic data generated without that patient (5 folds: the synthetic copy of four
  folds predicts the fifth). Calibration is parametric only (Platt or temperature), so the lock holds no
  patient-level values.
* **One rule differs from the study.** The HIGH cut-off is the 85 % accuracy point; the study uses 90 %.
  Models trained on synthetic data are less accurate on real patients than the study model, and none of them
  reached 90 % accuracy at any cut-off (this held for every setting tried, including full TabPFN-3.5). With the
  study's rule the demo would never finalise a case at Tier 1.

## Why synthetic training data

TabPFN keeps its training rows as in-context examples, so an app built on the study data would publish the
patients' blood counts. The app is therefore trained on a synthetic cohort (`data/synthetic_cohort.parquet`,
863 records, 657 with biochemistry) generated class by class with a Gaussian copula: empirical marginals,
normal-score correlations, analyser identities recomputed (HCT, HGB, MCH, RET%, LFR, MFR, RBC-He, WBC), values
rounded to the analyser's resolution.

Checks before release (holdout design, generator fitted on a random half, five repeats):

* **Privacy:** synthetic records were closer to the training half than to the held-out half in 54.8 % of
  cases (50 % = no memorisation). Their distance to the closest real record was larger than that of real
  held-out records, and no synthetic record matches a real one.
* **Fidelity:** median Kolmogorov–Smirnov distance 0.06 per class and variable; real-vs-synthetic
  discriminator AUC 0.69.
* **Utility:** models trained on synthetic data and tested on real patients reached the AUCs of models trained
  on real data (e.g. Stage 2 CBC 0.830 vs 0.833).

## Performance on real patients

| Measure | Development (cross-generation) | Temporal (this app's models) |
|---|---|---|
| Tier 1 · Stage 2 macro AUC | 0.848 | 0.915 |
| Tier 1 · Stage 2 accuracy | 0.656 | 0.722 |
| Tier 2 · Stage 2 macro AUC | 0.868 | 0.941 |
| Tier 2 · Stage 2 accuracy | 0.684 | 0.742 |
| Cascade · accuracy | 0.568 | 0.680 |
| Cascade · finalised at Tier 1 | 10.7 % (accuracy 81.4 %) | 13.4 % (accuracy 92.3 %) |

The temporal cohort (97 patients) is independent and has no other-cause patients. The development figures
come from the cross-generation predictions on which the cut-offs were chosen, so they are slightly optimistic.
The cascade rows count patients with the biochemistry panel measured. The full table is in
`data/tstr_summary.csv` and in the app's "About this demo" panel.

## Files

| File | Content |
|---|---|
| `streamlit_app.py` | Streamlit interface |
| `engine.py` | features, imputation, models, locked decisions, cascade |
| `app_selftest.py` | end-to-end check with the real backend (timings, memory) |
| `data/synthetic_cohort.parquet` | synthetic training records |
| `data/features.json` | feature list of each model |
| `data/lock.json` | thresholds, HIGH cut-offs, conformal quantiles, calibration |
| `data/reflex_rules.json` | the 14 reflex rules |
| `data/examples.json` | one synthetic example per class (separate draw, near the class median) |
| `data/tstr_summary.csv` | the app's models on the real patients |

## Run locally

```bash
pip install -r requirements.txt          # from the repository root
export TABPFN_TOKEN=<your Prior Labs key>   # TabPFN-3.5 licence accepted at https://ux.priorlabs.ai
streamlit run app/streamlit_app.py
```

On first start the app downloads the TabPFN-3.5-fast weights and fits the four models in a few seconds.

## Deploy on Streamlit Community Cloud

Create an app from this repository with `app/streamlit_app.py` as the main file. Add the secret
`TABPFN_TOKEN = "<your Prior Labs key>"` under the app's settings. `requirements.txt` in the repository root
installs the CPU build of PyTorch. The four stages share one TabPFN model in memory (`TABPFN_MODEL_CACHE_SIZE=1`),
which keeps the app at about 1.9 GB, within the platform's 2.7 GB limit.

## Licences

* TabPFN-3.5 weights: TabPFN-3.5 licence (non-commercial). This demo is for research and education only.
* Code and synthetic data: GPL-3.0 (repository licence).
