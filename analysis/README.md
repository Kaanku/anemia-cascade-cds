# Analysis code

The scripts that produced the results of the manuscript: cohort, biochemistry linkage, folds, features,
models, locked decisions, evaluation, Shapley values, sensitivity and post hoc analyses, the preparation of
the demo app, and the manuscript's tables and figures.

Patient data are not shared (ethics approval), so the scripts document the analysis and can be rerun only
with the original data. Input paths are those of the study's own data folder (`--root`); every output goes to
one study folder (`--out`).

## Pipeline

| Step | Script | What it does | Ran on |
|---|---|---|---|
| 1 | `s01_build_cohort.py` | analysis cohort from the laboratory export and the chart review, exclusions, STARD flow counts, secondary (out-of-scope) set | local |
| 2 | `s02_link_biochemistry.py` | ferritin, iron, UIBC and LDH linked to each sample (closest result within ±10 days) | local |
| 3 | `s03_split.py` | patient-level nested cross-validation folds and learning-curve subsets | local |
| 4 | `s04_features.py` | ratio features, Boruta selection and label-free KNN imputation, fitted on the training patients of each fold-set | local |
| 5 | `s05_train.py` (`run_jobs.py`, `run_lc.py`) | AutoGluon 1.5.0 models of the nested cross-validation (pre-specified engine, reported as comparator) | Colab, CPU |
| 5b | `s05b_tabpfn.py` | TabPFN v2 on the same fold-sets, rows and features (comparator of the first evaluation) | Colab, GPU |
| 5c | `s05c_tabpfn.py` | TabPFN-3.5, the main model, on the same fold-sets, rows and features | Colab, GPU |
| 6 | `s06_temporal.py` | temporal cohort and its matrices, with the transforms of the final fold-set | local |
| 7 | `s07_lock.py` | locked decisions per fold-set from inner out-of-fold predictions only: Stage 1 threshold, HIGH cut-off, calibration, conformal quantiles | local |
| 8 | `s08_indices.py` | the seven published red cell indices for IDA vs HGB HTZ | local |
| 9 | `s09_evaluate.py` | outer folds and temporal cohort scored with the locked decisions; bootstrap CIs, DeLong and McNemar tests, Holm correction | local |
| 11 | `s11_shap.py` (`shap_settings.json`) | Shapley values with shapiq (marginal imputation, 20 background patients) | Colab, GPU |
| 12 | `s12_shap_summary.py` | aggregate Shapley summaries, stability across folds, temporal agreement | local |
| 13 | `s13_sensitivity_noage.py` | the main model compared with the same model without age | local |
| 14 | `s14_rwo.py` | the analyzer's rule-based RBC workflow algorithm, recomputed | local |
| 15 | `s15_reflex.py` | reflex-test recommendations of the cascade | local |
| 16–18 | `s16_synthetic.py`, `s16b_examples.py`, `s16c_crossgen.py`, `s17_app_prep.py`, `s17b_crossgen.py`, `s18_app_lock.py`, `app_cpu_options.py` | synthetic cohort of the demo app and its checks, cross-generation predictions, the app's lock file, CPU settings | local and Colab |
| 19 | `s19_extra.py` | the cascade at the other locked operating points, class-conditional conformal coverage, subgroups by sex and age, Stage 2 recall per class with and without biochemistry | local |
| 20 | `s20_secondary.py` | the frozen final models on the secondary set (post hoc, exploratory) | local and Colab |
| — | `manuscript/make_tables.py`, `make_supp_tables.py`, `make_figures.py` | the manuscript's tables and figures, from the outputs above (study folder in `CDS_OUT`) | local |

Each script's docstring lists its inputs, outputs and rules, and its command line. Rule identifiers in the
code (for example D5, B4, V5, R21) refer to the investigators' decision log; the decisions that shaped the
analysis are listed with their dates in Supplementary Methods S1.

Two local inputs are not part of the code because they identify records: the documented correction of one
temporal sample (`s06_temporal.py`, rule V5: `<out>/private/temporal_corrections.json` or `--corrections`) and
the site's barcode file (`s02_link_biochemistry.py --barcode-file`).

## Environment

* Local steps: Python 3.11 with pandas 2.3, NumPy 2.4, scikit-learn 1.8.0, SciPy 1.17, boruta 0.4.3, joblib,
  openpyxl, xlrd, pyarrow and matplotlib 3.10.
* Colab steps: Python 3.13 with tabpfn 9.0.0 and PyTorch 2.11 on an NVIDIA T4 or A100 GPU, AutoGluon 1.5.0
  (`autogluon.tabular[all]==1.5.0`) on CPU, shapiq 1.7.0. TabPFN-3.5 needs a Prior Labs licence key
  (`TABPFN_TOKEN`); its weights are under the non-commercial TabPFN-3.5 licence.
