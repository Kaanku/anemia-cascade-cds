"""make_supp_tables.py — Supplementary Tables S1–S19 of the v4.3 manuscript.

Numbers come only from the study outputs. S1 (STARD-AI), S2 (analytical performance, from the laboratory's quality
control records, as reported for the earlier version of the system) and S5 (data used at each step) are text tables. S15 (Shapley
values) and S19 (secondary set) are added after the remaining Colab run; their slots are written as placeholders.
Tables are numbered in the order of their first citation in the manuscript.
Writes ms/supp_tables.json (for the Word builder) and ms/supp_tables.md (preview).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from decimal import Decimal, ROUND_HALF_UP


def fr(v, d):
    """Half-up rounding of the decimal value (binary noise removed first), as text."""
    q = Decimal(1).scaleb(-int(d))
    return str(Decimal(repr(round(float(v), 9))).quantize(q, rounding=ROUND_HALF_UP))

O = Path(os.environ.get("CDS_OUT", Path(__file__).resolve().parent.parent))   # the study folder
R = O / "reports"
MS = O / "ms"
m = pd.read_parquet(R / "figures_data/metrics.parquet")
S2C = ["IDA", "HA", "HGB_HTZ", "NORMAL"]
LAB = {"IDA": "IDA", "HA": "HA", "HGB_HTZ": "HGB HTZ", "NORMAL": "Normal", "OAC": "OAC", "all": "All"}
STAGE = {"S1": "Stage 1", "S2": "Stage 2", "MX": "IDA vs HGB HTZ"}
tables = []
MINUS = "−"


def T(engine="tabpfn", analysis="A1", fs="FULL", unit="first sample"):
    return m[(m.engine == engine) & (m.analysis == analysis) & (m.feature_set == fs) & (m.unit == unit)]


def g(df, blk, sset, metric, scen=None, pop=None):
    x = df[(df.block == blk) & (df.set == sset) & (df.metric == metric)]
    if scen:
        x = x[x.scenario == scen]
    if pop:
        x = x[x.population == pop]
    if len(x) != 1:
        return None
    r = x.iloc[0]
    return float(r.value), float(r.ci_low), float(r.ci_high), int(r.n)


def num(v, d=3):
    """Half-up rounding; a non-zero value that rounds to zero keeps one more decimal so its sign stays visible."""
    s = fr(v, d)
    if float(s) == 0 and v != 0 and abs(v) >= 5 * 10 ** -(d + 2):
        s = fr(v, d + 1)
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", MINUS)


def ci(t, pct=False, d=3):
    if t is None or not np.isfinite(t[0]):
        return "—"
    v, lo, hi = t[:3]
    k = 100 if pct else 1
    dd = 1 if pct else d
    if not np.isfinite(lo):
        return num(k * v, dd)
    return f"{num(k * v, dd)} ({num(k * lo, dd)}–{num(k * hi, dd)})"


def dif(v, lo, hi, d=3, k=1):
    dd = 1 if k == 100 else d
    return f"{num(k * v, dd)} ({num(k * lo, dd)} to {num(k * hi, dd)})"


def p_fmt(p):
    """Two decimals, three below 0.01 or close to 0.05; '<0.001' below 0.001."""
    if p is None or not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    if p < 0.01 or 0.045 <= p < 0.055:
        return f"{fr(p, 3)}"
    return f"{fr(p, 2)}"


def add(tid, title, header, rows, notes, panels=None, landscape=False):
    t = {"id": tid, "title": title, "notes": notes, "landscape": landscape}
    if panels:
        t["panels"] = panels
    else:
        t["panels"] = [{"label": None, "header": header, "rows": rows}]
    tables.append(t)


# ═════════════════════════════════════════ S1 STARD-AI checklist
ST = [
    ("Title or abstract", None, None),
    ("1†", "Identification as a study of AI-centred diagnostic accuracy, with at least one measure of accuracy",
     "Title; Abstract (AUC, accuracy)"),
    ("Abstract", None, None),
    ("2", "Structured summary of study design, methods, results and conclusions", "Abstract"),
    ("Introduction", None, None),
    ("3†", "Scientific and clinical background, including the intended use of the index test and its place in the "
           "clinical pathway", "Introduction, paragraphs 1–3; Section 2.8; Discussion, paragraph 8"),
    ("4", "Study objectives and hypotheses", "Introduction, last paragraph"),
    ("Methods — study design", None, None),
    ("5", "Whether data collection was planned before (prospective) or after (retrospective) the index test and "
          "reference standard were performed", "Section 2.1 (retrospective)"),
    ("Methods — ethics", None, None),
    ("6*", "Approval by an ethics committee", "Section 2.1 (decision No. 2025/441)"),
    ("Methods — participants", None, None),
    ("7†", "Eligibility criteria at participant and data level, in the order applied",
     "Sections 2.2, 2.12; Fig. 1"),
    ("8", "On what basis potentially eligible participants were identified",
     "Section 2.2 (reticulocyte-channel analysis in the laboratory information system)"),
    ("9", "Where and when potentially eligible participants were identified",
     "Sections 2.1, 2.2 (21 July 2025–2 January 2026), 2.12 (1 February–7 April 2026)"),
    ("10", "Whether participants formed a consecutive, random or convenience series",
     "Sections 2.2, 2.12 (consecutive)"),
    ("Methods — dataset", None, None),
    ("11*", "Source of the data; routinely collected, purpose-collected or open repository",
     "Sections 2.2, 2.4 (routinely collected; laboratory information system, analyzer exports, health record)"),
    ("12*", "Who annotated the dataset (experience, background) and how",
     "Section 2.3 (investigators and an internal medicine specialist, chart review)"),
    ("13*", "Devices used to capture the data; software (versions) used to build the index test",
     "Sections 2.4, 2.6, 2.13"),
    ("14*", "Data acquisition protocols and pre-processing, in enough detail for replication",
     "Sections 2.4, 2.5; Supplementary Tables S3–S5; Supplementary Methods S1"),
    ("Methods — test methods", None, None),
    ("15a", "Index test, in enough detail for replication", "Sections 2.5–2.9; Fig. 2; analysis code [36]"),
    ("15b*", "How the index test was developed: training, validation, testing and their sample sizes",
     "Section 2.7; Fig. 2; Supplementary Table S5"),
    ("15c", "Definition of and rationale for test positivity cut-offs or result categories; pre-specified or exploratory",
     "Section 2.8; Supplementary Table S6; Supplementary Methods S1"),
    ("15d*", "Intended end user and level of expertise required",
     "Discussion, paragraph 8 (laboratory professionals; ordering of the second tier and manual review)"),
    ("16a", "Reference standard, in enough detail for replication", "Section 2.3"),
    ("16b", "Rationale for choosing the reference standard", "Section 2.3 (final diagnosis after chart review); "
                                                             "Discussion, limitations (incorporation bias)"),
    ("16c", "Definition of and rationale for reference standard categories",
     "Section 2.3 (five classes; composition of HGB HTZ and OAC; secondary set)"),
    ("17a", "Whether clinical information and reference standard results were available to the performers or readers "
            "of the index test", "Sections 2.5, 2.7 (automated test; the class label was used only for training)"),
    ("17b", "Whether clinical information and index test results were available to the assessors of the reference "
            "standard", "Section 2.3 (review before model development; CBC and biochemistry available to the "
                        "reviewers); Discussion, limitations"),
    ("Methods — analysis", None, None),
    ("18", "Methods for estimating or comparing measures of diagnostic accuracy", "Section 2.13"),
    ("19", "How indeterminate index test or reference standard results were handled",
     "Section 2.8 (MEDIUM and LOW zones escalated to Tier 2); Section 2.2 (unclear diagnoses in the secondary set); "
     "Section 3.11; Supplementary Table S19"),
    ("20", "How missing data on the index test and reference standard were handled",
     "Section 2.5; Supplementary Tables S3, S4"),
    ("21", "Analyses of variability in diagnostic accuracy, pre-specified or exploratory",
     "Section 2.13; Supplementary Tables S16–S18"),
    ("22", "Intended sample size and how it was determined", "Section 2.13 (no formal calculation; learning curves, "
                                                             "Supplementary Fig. S4)"),
    ("23*", "Error analysis and assessment of algorithmic bias and fairness",
     "Section 3.2 and Supplementary Table S8 (confusion matrices); Section 3.11 and Supplementary Table S18 (sex and "
     "age groups); Discussion, paragraph 7"),
    ("Results — participants and dataset", None, None),
    ("24", "Flow of participants, with a diagram", "Fig. 1"),
    ("25†", "Baseline demographic, clinical and technical characteristics of the datasets",
     "Table 1; Supplementary Table S3"),
    ("26a", "Distribution of severity of disease in those with the target condition",
     "Table 1 (hemoglobin and biochemistry by class)"),
    ("26b", "Distribution of alternative diagnoses in those without the target condition",
     "Section 2.3 (composition of OAC)"),
    ("27", "Time interval and any clinical interventions between index test and reference standard",
     "Section 2.2 (first sample, before treatment); Section 2.4 (biochemistry within ±10 days)"),
    ("28*", "Whether the datasets represent the distribution of the target condition in the intended-use population",
     "Discussion, limitations (exclusions, single center)"),
    ("29*", "For a temporal or external evaluation, how the dataset differs from the development data",
     "Sections 2.12, 3.10; Table 1; Discussion, paragraph 5"),
    ("Results — test results", None, None),
    ("30", "Cross-tabulation of index test results by the reference standard",
     "Supplementary Tables S8 and S14; Table 3"),
    ("31", "Estimates of diagnostic accuracy and their precision (95% CI)", "Tables 2, 3; Results"),
    ("32", "Adverse events from the index test or reference standard", "Not applicable (retrospective)"),
    ("Discussion", None, None),
    ("33", "Limitations, including sources of bias, statistical uncertainty and generalizability",
     "Discussion, last paragraph"),
    ("34", "Implications for practice, including the intended use and clinical role of the index test",
     "Discussion, paragraphs 4 and 8; Conclusions"),
    ("35*", "Ethical considerations, including fairness",
     "Sections 2.1, 2.14 (synthetic training data for the public demonstration); Supplementary Table S18"),
    ("Other information", None, None),
    ("36", "Registration number and name of registry", "Not registered (retrospective diagnostic accuracy study)"),
    ("37", "Where the full study protocol can be accessed",
     "Supplementary Methods S1 (decision log); analysis code [36]"),
    ("38", "Sources of funding and other support; role of funders", "Funding"),
    ("39*", "Commercial interests", "Declaration of competing interest; Discussion (TabPFN-3.5 licence)"),
    ("40a*", "Availability of datasets and code; restrictions on reuse", "Data availability"),
    ("40b*", "Whether outputs are stored and auditable",
     "Section 2.7 (lock file with the hash of every out-of-fold file, held-out predictions opened only after locking); "
     "Supplementary Methods S1; Data availability"),
]
rows = []
for a, b, c in ST:
    rows.append([a, "", ""] if b is None else [a, b, c])
add("S1", "Table S1. STARD-AI checklist.", ["Item", "STARD-AI item", "Reported in"], rows, [
    "* New item relative to STARD 2015; † modified item. Item wording abridged from Sounderajah et al. [S2]; rows without "
    "an item number are section headings. AUC, area under the receiver operating characteristic curve; CBC, complete "
    "blood count; HGB HTZ, heterozygous hemoglobinopathy; OAC, other anemia causes."])
tables[-1]["section_rows"] = True

# ═════════════════════════════════════════ S2 analytical performance
AP = [
    ("Red blood cells", "×10¹²/L", "XN-1000", "Hydrodynamic impedance", "2.68 (1.64)", "4.25 (1.85)", "5.00 (1.75)", "3.9"),
    ("Hemoglobin", "g/dL", "XN-1000", "Spectrophotometric", "5.94 (1.77)", "11.45 (1.53)", "14.98 (1.69)", "−3.1"),
    ("Hematocrit", "%", "XN-1000", "Cumulative impedance", "17.40 (2.66)", "34.84 (3.85)", "43.82 (2.69)", "−4.7"),
    ("MCV", "fL", "XN-1000", "Calculated", "68.87 (1.54)", "80.49 (1.19)", "87.82 (1.30)", "1.7"),
    ("MCH", "pg", "XN-1000", "Calculated", "22.89 (1.76)", "27.19 (2.29)", "33.53 (2.01)", "−3.6"),
    ("MCHC", "g/dL", "XN-1000", "Calculated", "32.61 (2.65)", "33.86 (4.03)", "34.20 (4.10)", "−2.9"),
    ("RDW-SD", "fL", "XN-1000", "Impedance histogram", "49.00 (1.44)", "48.02 (3.20)", "50.70 (2.44)", "3.4"),
    ("RDW-CV", "%", "XN-1000", "Impedance histogram", "19.59 (1.79)", "16.13 (1.03)", "15.32 (1.51)", "5.1"),
    ("NRBC#", "×10³/µL", "XN-1000", "Fluorescence flow cytometry", "0.14 (5.06)", "0.42 (5.89)", "1.10 (5.23)", "—ᵃ"),
    ("Reticulocytes (RET#)", "×10⁹/L", "XN-1000", "Fluorescence flow cytometry", "135.32 (4.22)", "97.99 (4.54)", "57.47 (4.48)", "−14.9"),
    ("Reticulocytes (RET%)", "%", "XN-1000", "Fluorescence flow cytometry", "5.02 (4.20)", "2.12 (3.76)", "1.07 (4.10)", "—ᵃ"),
    ("LFR", "%", "XN-1000", "Fluorescence flow cytometry", "74.92 (3.99)", "71.93 (2.96)", "76.57 (3.62)", "—ᵃ"),
    ("MFR", "%", "XN-1000", "Fluorescence flow cytometry", "21.51 (1.19)", "13.13 (1.36)", "11.75 (1.36)", "—ᵃ"),
    ("HFR", "%", "XN-1000", "Fluorescence flow cytometry", "1.14 (2.88)", "1.79 (2.24)", "1.71 (3.06)", "—ᵃ"),
    ("IRF", "%", "XN-1000", "Fluorescence flow cytometry", "25.08 (1.19)", "14.92 (1.35)", "13.46 (1.31)", "—ᵃ"),
    ("RET-He", "pg", "XN-1000", "Scattergram", "24.91 (1.63)", "25.16 (1.50)", "26.49 (1.55)", "—ᵃ"),
    ("Iron", "µg/dL", "Cobas c702", "Colorimetric", "109.08 (3.94)", "242.52 (4.06)", "—ᵇ", "4.5"),
    ("UIBC", "µg/dL", "Cobas c702", "Colorimetric", "241.87 (6.33)", "293.66 (5.68)", "—ᵇ", "−5.2"),
    ("Ferritin", "µg/L", "Cobas e801", "ECLIA", "126.28 (4.85)", "698.60 (6.25)", "—ᵇ", "−5.0"),
    ("LDH", "U/L", "Cobas c702", "UV kinetic", "164.77 (3.64)", "288.04 (3.61)", "—ᵇ", "4.5"),
]
add("S2", "Table S2. Analytical performance of the measured parameters.",
    ["Parameter", "Unit", "Analyzer", "Method", "IQC level 1, mean (CV%)", "IQC level 2, mean (CV%)",
     "IQC level 3, mean (CV%)", "EQA bias, %"], [list(r) for r in AP], [
        "Hematology on a Sysmex XN-1000 (Sysmex Corporation, Kobe, Japan); biochemistry on the c702 and e801 modules of a "
        "Cobas 8000 system (Roche Diagnostics, Mannheim, Germany). All measurements were made after routine daily quality "
        "control. Values are internal quality control (IQC) means with imprecision (CV%) at up to three levels, and the "
        "bias in the external quality assessment (EQA) scheme in which the laboratory participates.",
        "ᵃ Not assessed in the EQA scheme. ᵇ Two IQC levels in routine use.",
        "CV, coefficient of variation; ECLIA, electrochemiluminescence immunoassay; HFR, LFR and MFR, high-, low- and "
        "medium-fluorescence reticulocytes; IRF, immature reticulocyte fraction; LDH, lactate dehydrogenase; MCH, mean "
        "corpuscular hemoglobin; MCHC, mean corpuscular hemoglobin concentration; MCV, mean corpuscular volume; NRBC, "
        "nucleated red blood cells; RDW-CV and RDW-SD, red cell distribution width (coefficient of variation, standard "
        "deviation); RET-He, reticulocyte hemoglobin equivalent; UIBC, unsaturated iron-binding capacity; UV, ultraviolet."])

# ═════════════════════════════════════════ S3 inputs, missing values, imputation
NAMES = {
    "age": ("Age", "years", "base"), "hgb_g_d_l": ("Hemoglobin", "g/dL", "base"),
    "rbc_10_6_u_l": ("Red blood cells", "×10⁶/µL", "base"), "ret_number_10_6_l": ("Reticulocytes (RET#)", "×10⁶/µL", "base"),
    "mcv_f_l": ("MCV", "fL", "base"), "mchc_g_dl": ("MCHC", "g/dL", "base"), "rdw_sd_fl": ("RDW-SD", "fL", "base"),
    "ret_he_pg": ("RET-He", "pg", "base"), "irf_pct": ("Immature reticulocyte fraction (IRF)", "%", "base"),
    "micro_macro_ratio": ("MicroR/MacroR ratio", "—", "base"), "nrbc_pct": ("Nucleated RBC (NRBC%)", "/100 WBC", "base"),
    "delta_he_pg": ("Delta-He", "pg", "base"), "frc_perc": ("Fragmented red cells (FRC#/RBC)", "fraction", "base"),
    "hct_pct": ("Hematocrit", "%", "ext"), "mch_pg": ("MCH", "pg", "ext"), "rdw_cv_pct": ("RDW-CV", "%", "ext"),
    "nrbc_number_10_3_u_l": ("Nucleated RBC (NRBC#)", "×10³/µL", "ext"), "wbc_10_3_u_l": ("White blood cells", "×10³/µL", "ext"),
    "neut_number_10_3_u_l": ("Neutrophils", "×10³/µL", "ext"), "lymph_number_10_3_u_l": ("Lymphocytes", "×10³/µL", "ext"),
    "mono_number_10_3_u_l": ("Monocytes", "×10³/µL", "ext"), "eo_number_10_3_u_l": ("Eosinophils", "×10³/µL", "ext"),
    "baso_number_10_3_u_l": ("Basophils", "×10³/µL", "ext"), "ig_number_10_3_u_l": ("Immature granulocytes", "×10³/µL", "ext"),
    "plt_10_3_u_l": ("Platelets", "×10³/µL", "ext"), "mpv_fl": ("MPV", "fL", "ext"), "pdw_fl": ("PDW", "fL", "ext"),
    "p_lcr_pct": ("P-LCR", "%", "ext"), "ret_pct": ("Reticulocytes (RET%)", "%", "ext"),
    "lfr_pct": ("Low-fluorescence reticulocytes (LFR)", "%", "ext"), "mfr_pct": ("Medium-fluorescence reticulocytes (MFR)", "%", "ext"),
    "hfr_pct": ("High-fluorescence reticulocytes (HFR)", "%", "ext"), "rbc_he_pg": ("RBC-He", "pg", "ext"),
    "micro_r_pct": ("Microcytic red cells (MicroR)", "%", "ext"), "macro_r_pct": ("Macrocytic red cells (MacroR)", "%", "ext"),
    "ret_y_ch": ("RET-Y (research parameter)", "channel", "ext"), "ret_rbc_y_ch": ("RET-RBC-Y (research parameter)", "channel", "ext"),
    "irf_y_ch": ("IRF-Y (research parameter)", "channel", "ext"),
    "ferritin": ("Ferritin", "µg/L", "bio"), "iron": ("Iron", "µg/dL", "bio"), "uibc": ("UIBC", "µg/dL", "bio"),
    "ldh": ("LDH", "U/L", "bio")}
md = pd.read_csv(R / "manuscript/missing_data.csv")
ap1 = pd.read_parquet(O / "data/analysis/analysis_primary.parquet")
BIO = ["iron", "uibc", "ferritin", "ldh"]
meas = ap1[ap1[BIO].notna().any(axis=1)]
assert len(meas) == 657
imp = meas[BIO].isna().sum()
n_imp_any = int(meas[BIO].isna().any(axis=1).sum())
n_complete = int(meas[BIO].notna().all(axis=1).sum())
derived = {c: int(ap1[f"derived_{c}"].astype("boolean").fillna(False).sum()) for c in ("iron", "uibc")}
censored = {c: int(ap1[f"censored_{c}"].astype("boolean").fillna(False).sum()) for c in BIO}
grp = {"base": "Base inputs (age and 12 CBC parameters)", "ext": "Further parameters of the same analyzer run",
       "bio": "Biochemistry panel (Tier 2)"}
panels = []
for key in ("base", "ext", "bio"):
    rows = []
    for r in md.itertuples():
        nm, unit, k = NAMES[r.variable]
        if k != key:
            continue
        imputed = (f"{int(imp[r.variable])} ({fr(100 * imp[r.variable] / len(meas), 1)})" if key == "bio" else "None")
        rows.append([nm, unit, f"{r.missing_dev_n} ({fr(r.missing_dev_pct, 1)})",
                     f"{r.missing_temporal_n} ({fr(r.missing_temporal_pct, 1)})", imputed])
    panels.append({"label": grp[key], "header": ["Parameter", "Unit of the model input", "Missing, development (n = 863)",
                                                  "Missing, temporal (n = 97)", "Imputed (n = 657)ᵃ"], "rows": rows})
add("S3", "Table S3. Model inputs, missing values and imputation, n (%).", None, None, [
    "Model inputs are in the units reported by the analyzer and the laboratory information system (Table 1 gives SI "
    "units). Base inputs were complete by design: records with a missing base input were excluded. Missing values of the "
    "further parameters are results that the analyzer did not report (flagged); they were left missing, because TabPFN "
    "handles missing values, and were filled with the training median only for the feature-selection step. Biochemistry "
    "counts as missing when no result lay within ±10 days of the CBC sample; a complete panel was an inclusion criterion "
    "of the temporal cohort. Temporal values refer to the 97 patients with an analyzer record. Sex was not used.",
    f"ᵃ Models with biochemistry were developed and evaluated in the 657 patients with at least one analyte measured; "
    f"{n_complete} had a complete panel and {n_imp_any} ({fr(100 * n_imp_any / len(meas), 1)}%) had at least one analyte "
    f"imputed by k-nearest neighbors (k = 5) on standardized age, base CBC parameters and analytes (ferritin and LDH "
    f"log-transformed), fitted on "
    f"the training patients of each fold-set without the class label. No CBC parameter was imputed for the models. Among "
    f"the first samples, {derived['iron']} iron and {derived['uibc']} UIBC results were calculated from total iron-binding "
    f"capacity (TIBC = iron + UIBC in this laboratory), and {censored['ferritin']} ferritin and {censored['uibc']} UIBC "
    f"results reported beyond the measuring range were set to the reported limit.",
    "Candidate ratio features (a/b for each pair, in a fixed order): hemoglobin, RBC, RET#, MCV, MCHC, RDW-SD, RET-He, IRF "
    "and MicroR/MacroR (36 ratios) and, with biochemistry, these nine and the four analytes (78 ratios). NRBC%, Delta-He, "
    "FRC#/RBC and age were not used in ratios because they can be zero or negative.",
    "LDH, lactate dehydrogenase; MCH, mean corpuscular hemoglobin; MCHC, mean corpuscular hemoglobin concentration; MCV, "
    "mean corpuscular volume; MPV, mean platelet volume; PDW, platelet distribution width; P-LCR, platelet large cell "
    "ratio; RBC, red blood cells; RBC-He, red blood cell hemoglobin equivalent; RDW-CV and RDW-SD, red cell distribution "
    "width (coefficient of variation, standard deviation); RET-He, reticulocyte hemoglobin equivalent; UIBC, unsaturated "
    "iron-binding capacity; WBC, white blood cells."], panels=panels)

# ═════════════════════════════════════════ S4 biochemistry by class, linkage
bl = pd.ExcelFile(R / "biochem_linkage_report.xlsx")
mp = bl.parse("missing_%_primary")
rows = [[LAB.get(r.cls, r.cls), f"{fr(r.iron, 1)}", f"{fr(r.uibc, 1)}", f"{fr(r.ferritin, 1)}", f"{fr(r.ldh, 1)}",
         f"{fr(r._6, 1)}"] for r in mp.itertuples()]
w3 = bl.parse("within_3_days")
w3 = dict(zip(w3.iloc[:, 0], w3.iloc[:, 1]))
AN = {"iron": "iron", "uibc": "UIBC", "ferritin": "ferritin", "ldh": "LDH"}
lm = dict(zip(*[bl.parse("link_method")[c] for c in ("method", "n")]))
n_link = int(bl.parse("summary").set_index("item").loc["records linked (all samples + secondary)", "n"])
assert n_link == sum(lm.values()) == 1143
add("S4", "Table S4. Missing biochemistry by diagnostic class in the development cohort (first samples), %.",
    ["Class", "Iron", "UIBC", "Ferritin", "LDH", "All four missing"], rows, [
        "An analyte counts as missing when no result was available within ±10 days of the CBC sample; per analyte, the "
        "result closest in time was used (ties: the earlier result). Linked results within ±3 days of the CBC sample, "
        "over all 1,143 linked records: " + ", ".join(f"{AN[k]} {fr(v, 1)}%" for k, v in w3.items()) + ".",
        f"Linkage of the {n_link:,} records (all valid samples of the primary cohort and the first samples of the "
        f"secondary set): the hospital file number came from the sample barcode file for {lm['barcode file']} records and, "
        f"for {lm['name + sex + date of birth']}, from a unique match on name and sex in the biochemistry export with a date "
        f"of birth consistent with the recorded age (±1 year); {lm['not found in biochemistry export']} records had no "
        f"entry in the biochemistry export. Rejected specimens and non-numeric results were discarded.",
        "Because missingness depended on the class, models with biochemistry were developed and evaluated only in patients "
        "with at least one analyte measured (657 patients); gaps within a measured panel were imputed without the class "
        "label (Supplementary Table S3).",
        "HA, hemolytic anemia; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency anemia; LDH, lactate "
        "dehydrogenase; OAC, other anemia causes; UIBC, unsaturated iron-binding capacity."])

# ═════════════════════════════════════════ S5 data used at each step
S5 = [
    ("Assignment to folds", "All 863 patients: five outer folds stratified by class and by whether the patient had "
     "repeat samples; within the training patients of each outer fold, five inner folds with the same stratification "
     "(seed 42).", "—"),
    ("Imputation of biochemistry (k-nearest neighbors)", "Training patients of the fold-set with at least one analyte "
     "measured; fitted once per fold-set, without the class label.", "Held-out patients (outer fold or temporal cohort): "
     "transformed only."),
    ("Feature selection (Boruta)", "Training patients of the fold-set, separately for each model; fitted once per "
     "fold-set (not refitted within the inner folds).", "Held-out patients."),
    ("Inner out-of-fold predictions", "Five TabPFN-3.5 fits, each on four inner folds, predicting the fifth.",
     "Outer fold, temporal cohort."),
    ("Operating decisions: Stage 1 and IDA vs HGB HTZ thresholds, confidence-zone cut-offs, conformal quantiles, display "
     "calibration", "Inner out-of-fold predictions of the training patients' first samples (uncalibrated scores for every "
     "decision except calibration). Recorded in the lock file with the hash of each out-of-fold file.",
     "Outer fold, temporal cohort."),
    ("Outer-fold model", "One TabPFN-3.5 fit on all training patients of the fold-set (A1: first samples; A2: all valid "
     "samples).", "Outer fold."),
    ("Outer-fold evaluation (nested cross-validation)", "First sample of each outer-fold patient, predicted once with "
     "the frozen decisions; the five outer folds are pooled (863 patients; 657 with biochemistry).", "—"),
    ("Final models", "All 863 patients (657 with biochemistry); decisions locked on their out-of-fold predictions over "
     "the five outer folds.", "Temporal cohort."),
    ("Temporal evaluation", "97 patients with an analyzer record (full feature set) or 105 (13 base inputs), predicted "
     "once by the final models.", "—"),
    ("Learning curves", "Nested random subsets (25%, 50%, 75%, 100%) of the training patients of each outer fold, with "
     "feature selection repeated on each subset; evaluated on the outer fold.", "—"),
]
add("S5", "Table S5. Data used at each step of model development and evaluation.",
    ["Step", "Data used", "Not used"], [list(r) for r in S5], [
        "A fold-set is one outer fold's training patients (outer folds 1–5) or all patients (final models). The class "
        "label was used only for model fitting, feature selection and the operating decisions, always on training "
        "patients. Outer-fold and temporal predictions were opened only after the lock file was complete (Supplementary "
        "Methods S1). A1, one (first) sample per patient for training; A2, all valid "
        "samples; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency anemia."])

# ═════════════════════════════════════════ S6 locked choices (cited in Section 2.8)
L = json.loads((O / "lock/lock.json").read_text())["engines"]["tabpfn"]
rows = []
for cfg, lab in (("A1_FULL_CBC_S1", "Stage 1, CBC"), ("A1_FULL_CBC_BIO_S1", "Stage 1, CBC + biochemistry"),
                 ("A1_FULL_CBC_MX", "IDA vs HGB HTZ"), ("A1_FULL_CBC_S2", "Stage 2, CBC"),
                 ("A1_FULL_CBC_BIO_S2", "Stage 2, CBC + biochemistry")):
    for fs_ in ["outer0", "outer1", "outer2", "outer3", "outer4", "final"]:
        e = L[cfg][fs_]
        thr = f"{fr(e['threshold'], 2)} / {fr(e['threshold_youden'], 2)}" if "threshold" in e else "—"
        ops = e.get("operating_points")
        opstr = ("; ".join(f"{int(round(float(k) * 100))}%: {'—' if v is None else fr(v['cutoff'], 2)}"
                           for k, v in ops.items()) if ops else "—")
        q = f"{fr(e['conformal_qhat']['0.1'], 3)}" if "conformal_qhat" in e else "—"
        rows.append([lab, "Final" if fs_ == "final" else f"Outer {int(fs_[-1]) + 1}", str(e["n_oof_rows"]), thr, opstr, q,
                     {"uncalibrated": "None", "platt": "Platt", "temperature": "Temperature", "isotonic": "Isotonic"}[e["calibration"]]])
add("S6", "Table S6. Operating decisions locked on the inner out-of-fold predictions of each fold-set (TabPFN-3.5).",
    ["Model", "Fold-set", "OOF patients", "Threshold, macro F1 / Youden", "Cut-off for the target accuracy (top-class probability)",
     "Conformal quantile, α = 0.10", "Display calibration"], rows, [
        "Each outer fold-set's decisions were locked on the out-of-fold predictions of its training patients (first "
        "samples) and applied unchanged to its held-out fold; the final models' decisions were applied to the temporal "
        "cohort. Thresholds: maximum macro F1 (used), with the Youden threshold as a secondary operating point; HGB HTZ is "
        "the positive class of the IDA vs HGB HTZ model. Cut-offs: the lowest top-class probability (max(p, 1 − p) for the "
        "binary model) at which the out-of-fold accuracy of the retained patients reached the target with at least 20 "
        "patients; '—', no cut-off reached the target. The 90% cut-off defines the HIGH zone, and LOW is a top-class "
        "probability below 0.35 in every fold-set. Display calibration: Platt scaling (binary models), temperature scaling "
        "(Stage 2) or isotonic regression, kept only if it lowered the cross-validated expected calibration error of the "
        "out-of-fold predictions by more than 0.005; it affects only the displayed probabilities. OOF, out-of-fold."],
    landscape=True)

# ═════════════════════════════════════════ S7 reflex rules and distribution
rr = pd.read_csv(R / "reflex/reflex_rules.csv")
rd = pd.read_csv(R / "reflex/reflex_distribution.csv")


def dist(rule, sset):
    x = rd[(rd.set == sset) & (rd.rule.str.split(" ").str[0] == rule)]
    if not len(x):
        return "0"
    r = x.iloc[0]
    s = f"{int(r.n)} ({fr(100 * r.share, 1)})"
    if np.isfinite(r.class_match):
        s += f"; {fr(100 * r.class_match, 0)}%"
    return s


def us(t):
    for a, b in (("haemolysis", "hemolysis"), ("Haemolysis", "Hemolysis"), ("Haemoglobin", "Hemoglobin"),
                 ("haematology", "hematology"), ("thalassaemia", "thalassemia"), ("counselling", "counseling"),
                 ("anaemia", "anemia"), ("haemolytic", "hemolytic"), ("LD ", "LDH "), ("LD)", "LDH)"), ("LD,", "LDH,"),
                 ("->", "→"), ("alpha-globin", "α-globin")):
        t = t.replace(a, b)
    return t


rows = []
for r in rr.itertuples():
    rows.append([r.rule, str(r.tier), LAB.get(r.predicted_class, r.predicted_class) if r.predicted_class != "any" else "Any",
                 "—" if str(r.zone).strip() in ("-", "", "nan") else r.zone, us(r.recommended_test), r.urgency.capitalize(),
                 dist(r.rule, "nested CV"), dist(r.rule, "temporal")])
add("S7", "Table S7. Reflex rules and their distribution in patients with measured biochemistry.",
    ["Rule", "Tier", "Predicted class", "Zone", "Recommended next step", "Urgency", "Nested CV (n = 657)ᵃ",
     "Temporal (n = 97)ᵃ"], rows, [
        "ᵃ n (%) of patients; after the semicolon, the share whose reference class matched the predicted class. Rules T1-5 "
        "to T1-7 are escalation steps, after which each patient receives one Tier 2 rule. The rules were applied to the "
        "cascade output without fitting. The temporal cohort contained no OAC patients, so every Tier 2 OAC result there "
        "was an error.",
        "CBC, complete blood count; CRP, C-reactive protein; DAT, direct antiglobulin test; eGFR, estimated glomerular "
        "filtration rate; HA, hemolytic anemia; HbA2, hemoglobin A2; HGB HTZ, heterozygous hemoglobinopathy; HPLC, "
        "high-performance liquid chromatography; IDA, iron deficiency anemia; LDH, lactate dehydrogenase; OAC, other anemia "
        "causes; TIBC, total iron-binding capacity; UIBC, unsaturated iron-binding capacity."], landscape=True)

# ═════════════════════════════════════════ S8 Stage 2 per class + confusion
t = T()
panels = []
for sset in ("nested CV", "temporal"):
    rows = []
    for scen, sl in (("CBC", "CBC"), ("CBC_BIO", "CBC + biochemistry")):
        for c in S2C:
            rows.append([sl, LAB[c], ci(g(t, "stage2", sset, f"recall_{c}", scen), pct=True),
                         ci(g(t, "stage2", sset, f"precision_{c}", scen), pct=True),
                         ci(g(t, "stage2", sset, f"f1_{c}", scen)), ci(g(t, "stage2", sset, f"auc_{c}", scen))])
    panels.append({"label": f"{'A. Nested cross-validation' if sset == 'nested CV' else 'B. Temporal cohort'}: per-class performance",
                   "header": ["Model", "Class", "Recall, % (95% CI)", "Precision, % (95% CI)", "F1 (95% CI)",
                              "AUC, one vs rest (95% CI)"], "rows": rows})
cm = pd.read_parquet(R / "figures_data/confusion_stage2.parquet")
cm = cm[(cm.engine == "tabpfn") & (cm.analysis == "A1") & (cm.feature_set == "FULL") & (cm.unit == "first sample")]
for sset in ("nested CV", "temporal"):
    rows = []
    for scen, sl in (("CBC", "CBC"), ("CBC_BIO", "CBC + biochemistry")):
        x = cm[(cm.set == sset) & (cm.scenario == scen)].set_index("true")
        for c in S2C:
            rows.append([sl, LAB[c]] + [str(int(x.loc[c, k])) for k in S2C] + [str(int(x.loc[c, S2C].sum()))])
    panels.append({"label": f"{'C. Nested cross-validation' if sset == 'nested CV' else 'D. Temporal cohort'}: confusion "
                            "matrix (rows, reference class; columns, predicted class)",
                   "header": ["Model", "Reference class"] + [LAB[c] for c in S2C] + ["Total"], "rows": rows})
add("S8", "Table S8. Stage 2 per-class performance and confusion matrices.", None, None, [
    "Patients of the four target classes (first samples); prediction = most probable class. CBC model: 637 patients in "
    "nested cross-validation; CBC + biochemistry model: 449 patients with measured biochemistry; temporal cohort: 97 "
    "patients. 95% CIs: patient bootstrap (2,000 resamples).",
    "AUC, area under the receiver operating characteristic curve; HA, hemolytic anemia; HGB HTZ, heterozygous "
    "hemoglobinopathy; IDA, iron deficiency anemia."], panels=panels)

# ═════════════════════════════════════════ S9 paired by class
pc = pd.read_csv(R / "extra/stage2_paired_by_class.csv")
rows = [[("Nested CV" if r.set == "nested CV" else "Temporal"), LAB.get(r._2, r._2), str(r.n), f"{fr(100 * r.recall_cbc, 1)}",
         f"{fr(100 * r.recall_cbc_bio, 1)}", f"{r.only_cbc_correct} / {r.only_bio_correct}", p_fmt(r.p_mcnemar)]
        for r in pc.itertuples()]
add("S9", "Table S9. Stage 2 recall with the CBC and with CBC + biochemistry in the same patients.",
    ["Set", "Class", "n", "Recall, CBC, %", "Recall, CBC + biochemistry, %", "Correct only with CBC / only with biochemistry",
     "P (McNemar)"], rows, [
        "Patients of the four target classes with measured biochemistry; both models predicted the same patients. 'All' "
        "is Stage 2 accuracy. P values are exact McNemar tests, not adjusted for multiplicity (descriptive).",
        "HA, hemolytic anemia; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency anemia."])

# ═════════════════════════════════════════ S10 AutoGluon vs TabPFN
cmp = pd.read_parquet(R / "figures_data/comparisons.parquet")
cmp = cmp[cmp.comparison == "AutoGluon vs TabPFN"]
rows = []
for r in cmp.itertuples():
    a, fs_, st = r.config.split("_")[0], r.config.split("_")[1], r.config[-2:]
    sc_ = "CBC + biochemistry" if "_CBC_BIO_" in r.config else "CBC"
    if st == "S2":
        auc, pd_ = f"{fr(r.macro_auc_a, 3)} / {fr(r.macro_auc_b, 3)}", "—"
    else:
        auc, pd_ = f"{fr(r.auc_a, 3)} / {fr(r.auc_b, 3)}", p_fmt(r.p_delong)
    rows.append([a, "Full" if fs_ == "FULL" else "13 base inputs", sc_, STAGE[st], str(r.n), auc, pd_,
                 f"{fr(100 * r.acc_a, 1)} / {fr(100 * r.acc_b, 1)}", p_fmt(r.p_mcnemar)])
cs = pd.read_parquet(R / "figures_data/confidence_shift.parquet")
cs = cs[(cs.analysis == "A1") & (cs.feature_set == "FULL")]
rows2 = []
for eng in ("autogluon", "tabpfn"):
    for (st, sc_), x in cs[cs.engine == eng].groupby(["stage", "scenario"], sort=False):
        x = x.set_index("part")
        sh = (f"{fr(100 * x.loc['inner OOF', 'share_high'], 1)} / {fr(100 * x.loc['outer fold', 'share_high'], 1)}"
              if np.isfinite(x.loc["inner OOF", "share_high"]) else "—")
        rows2.append(["AutoGluon" if eng == "autogluon" else "TabPFN-3.5", STAGE[st],
                      "CBC + biochemistry" if sc_ == "CBC_BIO" else "CBC",
                      f"{fr(x.loc['inner OOF', 'p90'], 3)} / {fr(x.loc['outer fold', 'p90'], 3)}",
                      f"{fr(100 * x.loc['inner OOF', 'share_ge_0.95'], 1)} / {fr(100 * x.loc['outer fold', 'share_ge_0.95'], 1)}", sh])
ta, tg = T("tabpfn"), T("autogluon")
rows3 = []
for lab, blk, metric, scen, pop in [
        ("Finalized at Tier 1, %", "cascade", "tier1_share", None, "measured biochemistry"),
        ("Accuracy of Tier 1 results, %", "cascade", "tier1_accuracy", None, "measured biochemistry"),
        ("Cascade accuracy, %", "cascade", "cascade_accuracy", None, "measured biochemistry"),
        ("Stage 2 patients in the HIGH zone, % (CBC)", "selective_S2", "coverage", "CBC", "target 0.90"),
        ("Accuracy in the HIGH zone, % (CBC)", "selective_S2", "accuracy_retained", "CBC", "target 0.90"),
        ("Conformal coverage, α = 0.10, % (CBC)", "conformal", "coverage", "CBC", "alpha 0.1"),
        ("Single-class sets, α = 0.10, % (CBC)", "conformal", "singleton_share", "CBC", "alpha 0.1")]:
    rows3.append([lab, ci(g(tg, blk, "nested CV", metric, scen, pop), pct=True), ci(g(ta, blk, "nested CV", metric, scen, pop), pct=True)])
add("S10", "Table S10. AutoGluon (pre-specified engine) versus TabPFN-3.5 (main model) in nested cross-validation.", None, None, [
    "Panel A: paired comparisons on the same patients' first samples (DeLong test for AUC; exact McNemar test for accuracy "
    "at each model's locked threshold or most probable class). A1, one sample per patient for training; A2, all valid "
    "samples. Panel B: confidence of the out-of-fold predictions on which the cut-offs were locked versus the outer-fold "
    "predictions of new patients (top-class probability for Stage 2; max(p, 1 − p) for the binary models). Each "
    "out-of-fold prediction of AutoGluon comes from a single bagged fold model, whereas its predictions for new patients "
    "average all fold models; its out-of-fold probabilities were more extreme, so its locked cut-offs retained few new "
    "patients (panel C). TabPFN-3.5 was fitted once per training set.",
    "AUC, area under the receiver operating characteristic curve; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron "
    "deficiency anemia; OOF, out-of-fold."], panels=[
    {"label": "A. Discrimination and accuracy (AutoGluon / TabPFN-3.5)",
     "header": ["Training unit", "Feature set", "Scenario", "Model", "n", "AUC or macro AUC", "P (DeLong)", "Accuracy, %",
                "P (McNemar)"], "rows": rows},
    {"label": "B. Out-of-fold versus outer-fold confidence (inner OOF / outer fold), A1, full feature set",
     "header": ["Engine", "Model", "Scenario", "90th percentile of confidence", "Share ≥0.95, %", "Share at or above the HIGH cut-off, %"],
     "rows": rows2},
    {"label": "C. Locked operating decisions applied to new patients (A1, full feature set; 95% CI)",
     "header": ["", "AutoGluon", "TabPFN-3.5"], "rows": rows3}], landscape=True)

# ═════════════════════════════════════════ S11 cascade by target
cb = pd.read_csv(R / "extra/cascade_by_target.csv")
rows = []
for r in cb.itertuples():
    rows.append([("Nested CV (n = 657)" if r.set == "nested CV" else "Temporal (n = 97)"), f"{fr(100 * float(r.target), 0)}%",
                 str(r.folds_with_cutoff) if r.set == "nested CV" else "—",
                 f"{r.n_tier1} ({fr(100 * r.tier1_share, 1)})",
                 ci((r.tier1_accuracy, r.tier1_accuracy_low, r.tier1_accuracy_high), pct=True),
                 ci((r.cascade_accuracy, r.cascade_accuracy_low, r.cascade_accuracy_high), pct=True),
                 ci((r.cbc_bio_accuracy, r.cbc_bio_accuracy_low, r.cbc_bio_accuracy_high), pct=True),
                 dif(r.accuracy_difference, r.accuracy_difference_low, r.accuracy_difference_high, k=100),
                 p_fmt(r.p_mcnemar)])
add("S11", "Table S11. The cascade with the Tier 1 cut-off taken from each locked operating point.",
    ["Set", "Target accuracy", "Folds with a cut-offᵃ", "Finalized at Tier 1, n (%)", "Tier 1 accuracy, % (95% CI)",
     "Cascade accuracy, % (95% CI)", "Biochemistry for all, % (95% CI)", "Difference, percentage points (95% CI)",
     "P (McNemar)"], rows, [
        "Patients with measured biochemistry; first samples. The 90% point is the pre-specified rule (Table 3); the other "
        "points were locked in the same way and examined post hoc. 95% CIs: patient bootstrap (2,000 resamples).",
        "ᵃ Outer folds in which a cut-off reached the target on the inner out-of-fold predictions (Supplementary Table "
        "S6); in the other folds no patient was finalized at Tier 1."], landscape=True)

# ═════════════════════════════════════════ S12 class-conditional coverage
cc = pd.read_csv(R / "extra/conformal_by_class.csv")
rows = []
for (sset, scen, a), x in cc.groupby(["set", "scenario", "alpha"], sort=False):
    x = x.set_index("class")
    row = [("Nested CV" if sset == "nested CV" else "Temporal"), ("CBC + biochemistry" if scen == "CBC_BIO" else "CBC"), f"{fr(a, 2)}"]
    for c in ["all"] + S2C:
        row.append(f"{fr(100 * x.loc[c, 'coverage'], 1)} / {fr(x.loc[c, 'mean_set_size'], 2)}")
    row.append(f"{fr(100 * x.loc['sets with IDA and HGB_HTZ', 'singleton_share'], 1)}")
    rows.append(row)
    if "all, patients with biochemistry" in x.index and x.loc["all, patients with biochemistry", "n"] < x.loc["all", "n"]:
        xb = x.loc["all, patients with biochemistry"]
        rows.append([row[0], f"CBC, patients with biochemistry (n = {int(xb.n)})", row[2],
                     f"{fr(100 * xb.coverage, 1)} / {fr(xb.mean_set_size, 2)}", "", "", "", "",
                     f"{fr(100 * x.loc['sets with IDA and HGB_HTZ, patients with biochemistry', 'singleton_share'], 1)}"])
add("S12", "Table S12. Conformal prediction by class: coverage, % / mean set size.",
    ["Set", "Model", "α", "All", "IDA", "HA", "HGB HTZ", "Normal", "Sets with IDA and HGB HTZ, %"], rows, [
        "Adaptive prediction sets with the locked quantiles (randomized; Stage 2, patients of the four target classes: "
        "637 and 449 in nested cross-validation, 97 in the temporal cohort). Rows 'CBC, patients with biochemistry' "
        "summarize the CBC model's sets in the patients evaluated with biochemistry, for a comparison on the same "
        "patients. Coverage is guaranteed only marginally (column 'All'); class-conditional coverage was computed "
        "descriptively, post hoc.",
        "HA, hemolytic anemia; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency anemia."], landscape=True)

# ═════════════════════════════════════════ S13 IDA vs HGB HTZ vs indices
mx = pd.read_parquet(R / "figures_data/mx_vs_indices.parquet")
mx = mx[(mx.engine == "tabpfn") & (mx.analysis == "A1") & (mx.feature_set == "FULL") & (mx.unit == "first sample")]
ORDER = ["Mentzer", "Green & King", "England & Fraser", "RDW index", "Shine & Lal", "Srivastava", "Ehsani"]
CUT = {"Mentzer": "MCV/RBC <13", "Green & King": "MCV² × RDW-CV/(100 × Hb) <65", "England & Fraser": "MCV − RBC − 5 × Hb − 3.4 <0",
       "RDW index": "MCV × RDW-CV/RBC <220", "Shine & Lal": "MCV² × MCH/100 <1530", "Srivastava": "MCH/RBC <3.8",
       "Ehsani": "MCV − 10 × RBC <15"}
panels = []
for sset in ("nested CV", "temporal"):
    rows = []
    for sub in ("all", "MCV<80"):
        x = mx[(mx.set == sset) & (mx.subgroup == sub)].set_index("index")
        assert x["n"].nunique() == 1
        mod_acc = g(T(), "mx", sset, "accuracy", "CBC", f"{sub} (model)")
        rows.append([("All" if sub == "all" else "MCV <80 fL"), "Model (locked threshold)", str(int(x["n"].iloc[0])),
                     f"{fr(x['auc_model'].iloc[0], 3)}", "—", f"{fr(100 * mod_acc[0], 1)}", "—"])
        for k in ORDER:
            r = x.loc[k]
            rows.append(["", f"{k} ({CUT[k]})", str(int(r.n)), f"{fr(r.auc_index, 3)}", p_fmt(r.p_delong_holm),
                         f"{fr(100 * r.acc_index, 1)}", p_fmt(r.p_mcnemar_holm)])
    panels.append({"label": "A. Nested cross-validation" if sset == "nested CV" else "B. Temporal cohort",
                   "header": ["Patients", "Model or index (HGB HTZ when)", "n", "AUC", "P vs model (DeLong, Holm)",
                              "Accuracy, %", "P vs model (McNemar, Holm)"], "rows": rows})
add("S13", "Table S13. Iron deficiency anemia versus heterozygous hemoglobinopathy: the model and seven discriminant indices.",
    None, None, [
        "Patients with IDA or HGB HTZ (first samples). Indices at their published cut-offs (nothing fitted) [S3]; MCH and "
        "RDW-CV from the analyzer record. Holm correction across the seven comparisons within each panel and subgroup.",
        "AUC, area under the receiver operating characteristic curve; Hb, hemoglobin (g/dL); HGB HTZ, heterozygous "
        "hemoglobinopathy; IDA, iron deficiency anemia; MCH, mean corpuscular hemoglobin (pg); MCV, mean corpuscular volume "
        "(fL); RBC, red blood cells (×10⁶/µL); RDW-CV, red cell distribution width, coefficient of variation (%)."],
    panels=panels, landscape=True)

# ═════════════════════════════════════════ S14 RWO
own = pd.read_csv(R / "rwo/rwo_own_scheme.csv").set_index("metric")
conf = pd.read_csv(R / "rwo/rwo_confusion.csv").set_index("true class (v4)")
rows = []
for c, lab in (("IDA", "IDA"), ("HGB HTZ", "HGB HTZ"), ("SCD", "Sickle cell"), ("HS", "Hereditary spherocytosis"), ("Other", "Other")):
    def f(k):
        v = own.loc[f"{k}_{c}"]
        return "—" if not np.isfinite(v.value) else f"{fr(v.value, 2)} ({fr(v.ci_low, 2)}–{fr(v.ci_high, 2)})"
    rows.append([lab, str(int(own.loc[f"precision_{c}", "n_true"])), str(int(own.loc[f"precision_{c}", "n_flagged"])),
                 f("sensitivity"), f("precision"), f("f1")])
acc = own.loc["accuracy"]
assert int(acc.n) == 701
rows_c = [[LAB.get(i, i)] + [str(int(conf.loc[i, k])) for k in conf.columns] + [str(int(conf.loc[i].sum()))]
          for i in conf.index]
add("S14", "Table S14. The manufacturer's RBC defect workflow optimization (RWO) algorithm recomputed in the development cohort.",
    None, None, [
        "Panel A: the algorithm's own categories in patients other than the Normal class (the algorithm has no normal "
        "output). Reference categories: IDA; HGB HTZ (thalassemia trait and other heterozygous variants); sickle cell "
        "(sickle cell trait carriers); hereditary spherocytosis (none in the primary cohort); other (HA and OAC). Panel B: "
        "RWO output for every reference class of the 863 patients. Published decision tree (Supplementary Methods S2); the "
        "Southeast Asian ovalocytosis branch could not be triggered (Hypo-He not exported), and an MCHC above 36.5 g/dL "
        "was taken as confirmed (optical MCHC not exported). 95% CIs: patient bootstrap (2,000 resamples).",
        "HA, hemolytic anemia; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency anemia; OAC, other anemia causes."],
    panels=[{"label": f"A. Own categories (n = 701; accuracy {fr(100 * acc.value, 1)}%, 95% CI {fr(100 * acc.ci_low, 1)}–{fr(100 * acc.ci_high, 1)})",
             "header": ["RWO category", "Reference n", "Flagged n", "Sensitivity (95% CI)", "Precision (95% CI)", "F1 (95% CI)"],
             "rows": rows},
            {"label": "B. RWO output by reference class (n = 863)", "header": ["Reference class"] + [
                {"SCD": "Sickle cell", "HS": "Spherocytosis"}.get(k, k) for k in conf.columns] + ["Total"], "rows": rows_c}])

# ═════════════════════════════════════════ S15 Shapley values (s12 outputs; placeholder until all jobs are done)
import os
SHAP = Path(os.environ.get("SHAP_DIR", R / "shap"))
SHAP_CFGS = [("A1_FULL_CBC_S1", "Stage 1, CBC", ["1"]), ("A1_FULL_CBC_MX", "IDA vs HGB HTZ, CBC", ["HGB_HTZ"]),
             ("A1_FULL_CBC_S2", "Stage 2, CBC", S2C), ("A1_FULL_CBC_BIO_S1", "Stage 1, CBC + biochemistry", ["1"]),
             ("A1_FULL_CBC_BIO_S2", "Stage 2, CBC + biochemistry", S2C)]
FEAT = {"age": "Age", "hgb_g_d_l": "HGB", "rbc_10_6_u_l": "RBC", "ret_number_10_6_l": "RET#", "mcv_f_l": "MCV",
        "mchc_g_dl": "MCHC", "rdw_sd_fl": "RDW-SD", "ret_he_pg": "RET-He", "irf_pct": "IRF",
        "micro_macro_ratio": "MicroR/MacroR", "nrbc_pct": "NRBC%", "delta_he_pg": "Delta-He", "frc_perc": "FRC",
        "hct_pct": "HCT", "mch_pg": "MCH", "rdw_cv_pct": "RDW-CV", "nrbc_number_10_3_u_l": "NRBC#",
        "wbc_10_3_u_l": "WBC", "neut_number_10_3_u_l": "NEUT#", "lymph_number_10_3_u_l": "LYMPH#",
        "mono_number_10_3_u_l": "MONO#", "eo_number_10_3_u_l": "EO#", "baso_number_10_3_u_l": "BASO#",
        "ig_number_10_3_u_l": "IG#", "plt_10_3_u_l": "PLT", "mpv_fl": "MPV", "pdw_fl": "PDW", "p_lcr_pct": "P-LCR",
        "ret_pct": "RET%", "lfr_pct": "LFR", "mfr_pct": "MFR", "hfr_pct": "HFR", "rbc_he_pg": "RBC-He",
        "micro_r_pct": "MicroR", "macro_r_pct": "MacroR", "ret_y_ch": "RET-Y", "ret_rbc_y_ch": "RET-RBC-Y",
        "irf_y_ch": "IRF-Y", "ferritin": "Ferritin", "iron": "Iron", "uibc": "UIBC", "ldh": "LDH"}


def feat_name(f):
    if "_div_" not in f:
        return FEAT.get(f, f)
    parts = [FEAT[x] for x in f.split("_div_")]
    return "/".join(f"({x})" if "/" in x else x for x in parts)


def shap_ready():
    return all((SHAP / f"{c}_global.csv").exists() for c, _, _ in SHAP_CFGS)


if shap_ready():
    TOP = 5
    rows_a = []
    for cfg, lab, classes in SHAP_CFGS:
        G = pd.read_csv(SHAP / f"{cfg}_global.csv", dtype={"explained_class": str})
        for c in classes:
            x = G[G.explained_class == c].sort_values("rank").head(TOP)
            tgt = {"1": "AAC", "HGB_HTZ": "HGB HTZ"}.get(c, LAB.get(c, c))
            for i, r in enumerate(x.itertuples()):
                rows_a.append([lab if i == 0 else "", tgt if i == 0 else "", str(int(r.rank)), feat_name(r.feature),
                               f"{fr(r.mean_abs_phi, 4)} ({fr(r.ci_low, 4)}–{fr(r.ci_high, 4)})",
                               num(r.spearman_value_phi, 2), f"{int(r.folds_selected)}/5"])
    rows_b = []
    for cfg, lab, classes in SHAP_CFGS[3:]:
        GR = pd.read_csv(SHAP / f"{cfg}_groups.csv", dtype={"explained_class": str})
        for c in classes:
            x = GR[GR.explained_class == c]
            bio = x[x.group.isin(["Biochemistry", "Ratios with biochemistry"])].mean_abs_phi.sum()
            rows_b.append([lab, {"1": "AAC"}.get(c, LAB.get(c, c)), f"{fr(100 * bio / x.mean_abs_phi.sum(), 1)}"])
    rows_c = []
    for cfg, lab, classes in SHAP_CFGS:
        ST = pd.read_csv(SHAP / f"{cfg}_stability.csv", dtype={"explained_class": str})
        TG = pd.read_csv(SHAP / f"{cfg}_temporal_global.csv", dtype={"explained_class": str}) \
            if (SHAP / f"{cfg}_temporal_global.csv").exists() else None
        for c in classes:
            st_ = ST[ST.explained_class == c].spearman
            rho_t = TG[TG.explained_class == c].rho_cv_temporal.iloc[0] if TG is not None and (TG.explained_class == c).any() else np.nan
            rows_c.append([lab, {"1": "AAC", "HGB_HTZ": "HGB HTZ"}.get(c, LAB.get(c, c)),
                           f"{num(st_.median(), 2)} ({num(st_.min(), 2)}–{num(st_.max(), 2)})", num(rho_t, 2)])
    add("S15", "Table S15. Shapley values of the outer-fold models (nested cross-validation).", None, None, [
        "Shapley values on the probability scale (shapiq, marginal imputation from 20 background training patients, "
        "OddSHAP estimator); each outer-fold model explained its own outer-fold patients, and the five folds were pooled. "
        "Mean |φ|, mean absolute Shapley value with a patient bootstrap 95% CI. ρ, Spearman correlation between the "
        "feature value and its Shapley value (positive: higher values push towards the explained class). Folds: outer "
        "folds whose feature selection retained the feature. Panel B: share of the total mean |φ| carried by the four "
        "analytes and the ratios that contain them. Panel C: agreement of the importance rankings between pairs of outer "
        "folds and between nested cross-validation and the temporal cohort (final models).",
        "AAC, associated anemia causes; HA, hemolytic anemia; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency "
        "anemia. Feature abbreviations as in Supplementary Table S3; A/B denotes a ratio."], panels=[
        {"label": f"A. The {TOP} features with the largest mean |φ| per model and explained class",
         "header": ["Model", "Explained class", "Rank", "Feature", "Mean |φ| (95% CI)", "ρ", "Folds"], "rows": rows_a},
        {"label": "B. Share of the attributions carried by biochemistry", "header": ["Model", "Explained class", "Share, %"],
         "rows": rows_b},
        {"label": "C. Stability of the importance rankings", "header": ["Model", "Explained class",
                                                                        "Spearman between folds, median (range)",
                                                                        "Spearman, nested CV vs temporal"], "rows": rows_c}],
        landscape=True)
else:
    add("S15", "Table S15. Shapley values of the outer-fold models: the most important features per model and class.",
        ["Model", "Rank", "Feature", "Mean |φ|", "Direction"], [["[PENDING: filled from s12 after the remaining Shapley jobs]", "", "", "", ""]],
        ["[PENDING]"])
    tables[-1]["pending"] = True

# ═════════════════════════════════════════ S16 A2 and T13
cmpa = pd.read_parquet(R / "figures_data/comparisons.parquet")
cmpa = cmpa[cmpa.engine == "tabpfn"]
rows = []
for name, label in (("A1 vs A2", "All samples (A2)"), ("FULL vs T13", "13 base inputs (T13)")):
    x = cmpa[cmpa.comparison == name]
    x = x[x.feature_set == "FULL"] if name == "A1 vs A2" else x[x.analysis == "A1"]
    for r in x.itertuples():
        sc_ = r.scenario if name == "A1 vs A2" else "CBC"
        if r.stage == "S2":
            auc, pdl = f"{fr(r.macro_auc_a, 3)} / {fr(r.macro_auc_b, 3)}", "—"
        else:
            auc, pdl = f"{fr(r.auc_a, 3)} / {fr(r.auc_b, 3)}", p_fmt(r.p_delong)
        rows.append([label, "CBC + biochemistry" if sc_ == "CBC_BIO" else "CBC", STAGE[r.stage], str(r.n), auc, pdl,
                     f"{fr(100 * r.acc_a, 1)} / {fr(100 * r.acc_b, 1)}", p_fmt(r.p_mcnemar)])
t2 = T(analysis="A2")
rows2 = []
for lab, blk, metric, scen, pop in [("Finalized at Tier 1, %", "cascade", "tier1_share", None, "measured biochemistry"),
                                    ("Accuracy of Tier 1 results, %", "cascade", "tier1_accuracy", None, "measured biochemistry"),
                                    ("Cascade accuracy, %", "cascade", "cascade_accuracy", None, "measured biochemistry"),
                                    ("Biochemistry for all, %", "cascade", "cbc_bio_accuracy", None, "measured biochemistry"),
                                    ("Conformal coverage, α = 0.10, CBC, %", "conformal", "coverage", "CBC", "alpha 0.1")]:
    rows2.append([lab, ci(g(T(), blk, "nested CV", metric, scen, pop), pct=True), ci(g(t2, blk, "nested CV", metric, scen, pop), pct=True),
                  ci(g(T(), blk, "temporal", metric, scen, pop), pct=True), ci(g(t2, blk, "temporal", metric, scen, pop), pct=True)])
ta2 = T(analysis="A2", unit="all samples")
rows3 = []
for lab, blk, metric, scen, pop in [("Stage 1 AUC, CBC", "stage1", "auc", "CBC", None),
                                    ("Stage 2 macro AUC, CBC", "stage2", "macro_auc", "CBC", None),
                                    ("Stage 2 macro AUC, CBC + biochemistry", "stage2", "macro_auc", "CBC_BIO", None),
                                    ("IDA vs HGB HTZ AUC", "mx", "auc", "CBC", "all (model)")]:
    v = g(ta2, blk, "nested CV", metric, scen, pop)
    rows3.append([lab, ci(v), f"{v[3]:,}"])
add("S16", "Table S16. Pre-specified alternative analyses: training on all samples (A2; weighted equally with A1 in the protocol) and restriction to the 13 base inputs (T13).",
    None, None, [
        "Panel A: paired comparisons with the main model (A1, full feature set) on the same patients' first samples "
        "(main / alternative); DeLong test for AUC, exact McNemar test for accuracy. Panel B: locked decisions of the A2 "
        "models (95% CI). Panel C: A2 models evaluated on all 1,084 samples; 95% CIs by patient-cluster bootstrap; tests "
        "were not computed in this unit, because repeat samples of a patient are not independent.",
        "AUC, area under the receiver operating characteristic curve; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron "
        "deficiency anemia."], panels=[
        {"label": "A. Paired comparisons with the main model (main / alternative)",
         "header": ["Analysis", "Scenario", "Model", "n", "AUC or macro AUC", "P (DeLong)", "Accuracy, %", "P (McNemar)"], "rows": rows},
        {"label": "B. Cascade and conformal prediction with the A2 models",
         "header": ["", "A1, nested CV", "A2, nested CV", "A1, temporal", "A2, temporal"], "rows": rows2},
        {"label": "C. A2 models on all samples (nested cross-validation)", "header": ["", "Estimate (95% CI)", "Samples"], "rows": rows3}],
    landscape=True)

# ═════════════════════════════════════════ S17 no age
na = pd.read_csv(R / "sensitivity/noage_comparison.csv")
assert na.loc[na.set == "nested CV", "p_delong"].notna().sum() == 10
na = na[(na.analysis == "A1")]
rows = []
for r in na.itertuples():
    fs_ = "13 base inputs" if r.feature_set == "T13" else "Full"
    sc_ = "CBC + biochemistry" if r.scenario == "CBC_BIO" else "CBC"
    if np.isfinite(r.auc_with):
        auc = f"{fr(r.auc_with, 3)} / {fr(r.auc_without, 3)}"
        dd = dif(r.auc_diff, r.auc_diff_low, r.auc_diff_high)
    else:
        auc, dd = "—", "—"
    if r.stage == "S2":
        extra = (f"HIGH {fr(100 * r.high_share_with, 1)}% ({fr(100 * r.high_accuracy_with, 0)}%) / "
                 f"{fr(100 * r.high_share_without, 1)}% ({fr(100 * r.high_accuracy_without, 0)}%)")
    elif np.isfinite(r.specificity_with):
        extra = (f"Sens {fr(100 * r.sensitivity_with, 1)} / {fr(100 * r.sensitivity_without, 1)}; "
                 f"Spec {fr(100 * r.specificity_with, 1)} / {fr(100 * r.specificity_without, 1)}")
    else:
        extra = f"Sens {fr(100 * r.sensitivity_with, 1)} / {fr(100 * r.sensitivity_without, 1)}"
    rows.append([("Nested CV" if r.set == "nested CV" else "Temporal"), fs_, sc_, STAGE[r.stage], str(r.n), auc, dd,
                 p_fmt(r.p_delong_holm), f"{fr(100 * r.accuracy_with, 1)} / {fr(100 * r.accuracy_without, 1)}",
                 p_fmt(r.p_mcnemar), extra])
add("S17", "Table S17. Sensitivity analysis without age (TabPFN-3.5, one sample per patient for training): with / without age.",
    ["Set", "Feature set", "Scenario", "Model", "n", "AUC or macro AUC", "Difference (95% CI)", "P (DeLong, Holm)",
     "Accuracy, %", "P (McNemar)", "Operating point"], rows, [
        "Post hoc. The models were refitted with age removed from each model's feature list and nothing else changed "
        "(same folds, patients and other features; no ratio contains age) and locked with the same rules. AUC differences "
        "with patient bootstrap 95% CIs (2,000 resamples). DeLong tests with Holm correction across the ten nested-CV tests "
        "of the A1 and A2 analyses (A2 rows not shown); the temporal cohort has no OAC patients, so its Stage 1 AUC is not "
        "defined. Operating point: Stage 1 and IDA vs HGB HTZ at each model's locked threshold; Stage 2 share of patients "
        "in the HIGH zone (accuracy).",
        "HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency anemia; OAC, other anemia causes; Sens, sensitivity; "
        "Spec, specificity."], landscape=True)

# ═════════════════════════════════════════ S18 subgroups
sg = pd.read_csv(R / "extra/subgroups.csv")
rows = []
for r in sg.itertuples():
    lab = {"female": "Female", "male": "Male", "<18": "<18 years", "18-39": "18–39 years", "40-64": "40–64 years",
           ">=65": "≥65 years"}[r.group]
    rows.append([("Sex" if r.variable == "sex" else "Age"), lab, f"{r.n_patients} ({r.n_oac})",
                 ci((r.s1_auc, r.s1_auc_low, r.s1_auc_high)),
                 ci((r.s1_sensitivity, r.s1_sensitivity_low, r.s1_sensitivity_high), pct=True),
                 ci((r.s1_specificity, r.s1_specificity_low, r.s1_specificity_high), pct=True),
                 ci((r.s2_macro_auc, r.s2_macro_auc_low, r.s2_macro_auc_high)),
                 ci((r.s2_accuracy, r.s2_accuracy_low, r.s2_accuracy_high), pct=True),
                 ci((r.e2e_accuracy, r.e2e_accuracy_low, r.e2e_accuracy_high), pct=True)])
add("S18", "Table S18. Performance with the CBC by sex and age group (nested cross-validation).",
    ["Variable", "Group", "Patients (OAC)", "Stage 1 AUC", "Stage 1 sensitivity, %", "Stage 1 specificity, %",
     "Stage 2 macro AUC", "Stage 2 accuracy, %", "Five-class accuracy, %"], rows, [
        "Post hoc, descriptive. Same outer-fold predictions and locked thresholds as in the main analysis; sex was not a "
        "model input. Sensitivity refers to the four target classes (AAC) and specificity to OAC. 95% CIs: DeLong (Stage 1 "
        "AUC) and patient bootstrap (2,000 resamples).",
        "AAC, associated anemia causes; AUC, area under the receiver operating characteristic curve; OAC, other anemia "
        "causes."], landscape=True)

# ═════════════════════════════════════════ S19 secondary set (s20 outputs; placeholder until the Colab run)
SEC = Path(os.environ.get("SEC_DIR", R / "extra"))
if (SEC / "secondary_set.csv").exists():
    ss = pd.read_csv(SEC / "secondary_set.csv").set_index("category")
    chk = pd.read_csv(SEC / "secondary_refit_check.csv")
    CAT = [("Megaloblastic anemia", "Megaloblastic anemia"), ("Diagnosis unclear", "Diagnosis unclear"),
           ("Homozygous or major haemoglobinopathy", "Homozygous or major hemoglobinopathy"),
           ("Hereditary spherocytosis", "Hereditary spherocytosis"),
           ("Combined IDA + ACD", "Combined IDA and chronic disease anemia"), ("all", "All")]
    def cls_list(t):
        if not isinstance(t, str) or not t.strip():
            return "—"
        return "; ".join(f"{LAB.get(k, k)} {v}" for k, v in (x.rsplit(" ", 1) for x in t.split("; ")))
    rows = []
    for key, lab in CAT:
        if key not in ss.index:
            continue
        r = ss.loc[key]
        rows.append([lab, str(int(r.n)), str(int(r.stage1_oac)), f"{int(r.s2_high)} / {int(r.s2_medium)} / {int(r.s2_low)}",
                     f"{int(r.tier1_finalised)}" + (f" ({cls_list(r.tier1_classes)})" if int(r.tier1_finalised) else ""),
                     str(int(r.with_biochemistry)), f"{int(r.tier2_oac)} / {int(r.tier2_high)} / {int(r.tier2_medium_low)}",
                     fr(r.median_set_size_cbc, 0)])
    add("S19", "Table S19. The frozen final models applied to the secondary set (59 patients outside the four target classes).",
        ["Category", "n", "Stage 1 OAC", "Stage 1 AAC: Stage 2 HIGH / MEDIUM / LOW", "Finalized at Tier 1 (class)",
         "Biochemistry measured", "Tier 2: OAC / HIGH / MEDIUM or LOW", "Median CBC set size, α = 0.10"], rows, [
            "Post hoc, exploratory. First sample of each secondary-set patient, scored by the final models (A1, full feature "
            "set) with the locked final decisions (Stage 1 threshold, HIGH cut-off, LOW <0.35, conformal quantile); the "
            "Tier 2 columns refer to the patients with biochemistry measured. These patients belong to none of the four "
            "target classes: a Stage 1 OAC result or a MEDIUM or LOW Stage 2 zone sends them to Tier 2, whereas a Tier 1 "
            "finalization assigns one of the target classes (for homozygous hemoglobinopathy and combined deficiency, a "
            "related class). The final models were "
            f"refitted for this analysis and reproduced the stored temporal predictions (maximum absolute difference in "
            f"probability {num(chk.max_abs_dp.max(), 4)}; agreement of the predicted class {fr(100 * chk.argmax_agreement.min(), 1)}% or more).",
            "AAC, associated anemia causes; HA, hemolytic anemia; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency "
            "anemia; OAC, other anemia causes."], landscape=True)
else:
    add("S19", "Table S19. The frozen final models applied to the secondary set (59 patients outside the four target classes).",
        ["Category", "n", "Stage 1 OAC", "Stage 1 AAC", "Stage 2 HIGH zone", "Finalized at Tier 1"],
        [["[PENDING: filled from s20 after the Colab run]", "", "", "", "", ""]], ["[PENDING]"])
    tables[-1]["pending"] = True

ids = [int(t["id"][1:]) for t in tables]
assert ids == list(range(1, 20)), ids
(MS / "supp_tables.json").write_text(json.dumps(tables, indent=1, ensure_ascii=False))


def to_md(t):
    out = [f"**{t['title']}**", ""]
    for p in t["panels"]:
        if p["label"]:
            out += [f"*{p['label']}*", ""]
        out += ["| " + " | ".join(p["header"]) + " |", "|" + "---|" * len(p["header"])]
        out += ["| " + " | ".join(r) + " |" for r in p["rows"]] + [""]
    return "\n".join(out + t["notes"] + [""])


(MS / "supp_tables.md").write_text("\n".join(to_md(t) for t in tables))
print("tables:", [t["id"] for t in tables])
print(f"imputed any {n_imp_any}/{len(meas)}; complete {n_complete}; derived {derived}; censored {censored}")
