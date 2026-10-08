# anemia-cascade-cds

Two-tier cascade clinical decision support for the differential diagnosis of anaemia from the full blood count
and reticulocyte panel (Sysmex XN), with a first-line biochemistry panel as the second tier. Associated
manuscript under review.

> **Research use only. Not a medical device.**

**Demo:** <https://anemia-cascade-cds-2z85cbsumfyft2ckde6y75.streamlit.app> (models trained on synthetic data)

## Contents

| Folder | Content |
|---|---|
| `app/` | Interactive research demo (Streamlit). TabPFN-3.5-fast models trained on a **synthetic** cohort; decisions locked with the study's rules on real development patients; SHAP explanation of each result. See [`app/README.md`](app/README.md). |
| `analysis/` | The analysis code of the manuscript: cohort, features, nested cross-validation (TabPFN-3.5; AutoGluon as comparator), locked decisions, evaluation, Shapley values, sensitivity analyses, the demo's synthetic cohort, and the manuscript's tables and figures. See [`analysis/README.md`](analysis/README.md). |
| `legacy/` | Notebooks and modules of an earlier version of the analysis (M1–M11), kept for reference. |

## The cascade

* **Tier 1 — full blood count.** Stage 1 separates anaemias with a classifiable cause (iron deficiency
  anaemia, haemolytic anaemia, heterozygous haemoglobinopathy, normal red cell profile) from anaemias of
  another cause. Stage 2 assigns one of the four classes, with a confidence zone (HIGH / MEDIUM / LOW) and a
  conformal prediction set. A case is finalised at Tier 1 when Stage 1 says "classifiable" and the Stage 2
  result is in the HIGH zone.
* **Tier 2 — plus biochemistry** (ferritin, iron, UIBC, LDH) for the cases Tier 1 does not finalise.
* **Reflex recommendations:** 14 rules map tier × class × zone to the next laboratory step.

## Data availability

Patient data are not shared (ethics approval). The analysis code is in `analysis/`; it runs on the original
data only. The demo is trained on synthetic data only; the synthetic cohort, its generation and its privacy,
fidelity and utility checks are described in `app/README.md`.

## Running the demo

```bash
pip install -r requirements.txt
export TABPFN_TOKEN=<your Prior Labs key>   # TabPFN-3.5 licence accepted at https://ux.priorlabs.ai
streamlit run app/streamlit_app.py
```

## Licences

* Code and synthetic data: GPL-3.0 (see `LICENSE`).
* TabPFN-3.5 weights: TabPFN-3.5 licence (non-commercial); the demo is for research and education only.
