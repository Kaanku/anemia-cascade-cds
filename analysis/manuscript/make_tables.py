"""make_tables.py — main-text Tables 1–3 of the v4.3 manuscript, built only from the study outputs.

Writes ms/tables_main.json (structure for the Word builder) and ms/tables_main.md (preview).
Nothing is typed by hand: every value is read from data/, reports/figures_data/, reports/extra/, reports/rwo/.
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
T = m[(m.engine == "tabpfn") & (m.analysis == "A1") & (m.feature_set == "FULL") & (m.unit == "first sample")]
CLS = ["IDA", "HA", "HGB_HTZ", "NORMAL", "OAC"]
LAB = {"IDA": "IDA", "HA": "HA", "HGB_HTZ": "HGB HTZ", "NORMAL": "Normal", "OAC": "OAC"}
FE = 0.1791                                                     # iron/UIBC µg/dL -> µmol/L


def get(blk, sset, metric, scen=None, pop=None):
    x = T[(T.block == blk) & (T.set == sset) & (T.metric == metric)]
    if scen:
        x = x[x.scenario == scen]
    if pop:
        x = x[x.population == pop]
    assert len(x) == 1, (blk, sset, metric, scen, pop, len(x))
    r = x.iloc[0]
    return float(r.value), float(r.ci_low), float(r.ci_high), int(r.n)


def ci(v, lo, hi, pct=False, d=3):
    if pct:
        return f"{fr(100 * v, 1)} ({fr(100 * lo, 1)}–{fr(100 * hi, 1)})"
    return f"{fr(v, d)} ({fr(lo, d)}–{fr(hi, d)})"


def num(v, d):
    if not np.isfinite(v):
        return "—"
    if d == "auto":
        return f"{fr(v, 0)}" if abs(v - round(v)) < 1e-9 else f"{fr(v, 1)}"
    return fr(v, d)


def med(x, d):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "—"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{num(q2, d)} ({num(q1, d)}–{num(q3, d)})"


# ───────────────────────────────────────────── Table 1
dev = pd.read_parquet(O / "data/analysis/analysis_primary.parquet")
tem = pd.read_parquet(O / "data/temporal/temporal_cohort.parquet")
VARS = [  # label, column, transform, decimals
    ("Age, years", "age", 1, "auto"),
    ("Hemoglobin, g/L", "hgb_g_d_l", 10, 0),
    ("Red blood cells, ×10¹²/L", "rbc_10_6_u_l", 1, 2),
    ("MCV, fL", "mcv_f_l", 1, 1),
    ("MCHC, g/L", "mchc_g_dl", 10, 0),
    ("RDW-SD, fL", "rdw_sd_fl", 1, 1),
    ("Reticulocytes, ×10⁹/L", "ret_number_10_6_l", 1000, 0),
    ("RET-He, pg", "ret_he_pg", 1, 1),
    ("Microcytic red cells (MicroR), %", "micro_r_pct", 1, 1),
    ("Ferritin, µg/L", "ferritin", 1, 0),
    ("Iron, µmol/L", "iron", FE, 1),
    ("UIBC, µmol/L", "uibc", FE, 1),
    ("LDH, U/L", "ldh", 1, 0),
]
BIO = ["ferritin", "iron", "uibc", "ldh"]


def cohort_block(df, classes, female):
    rows = []
    n_all = len(df)
    groups = [(c, df[df["cls"] == c]) for c in classes] + [("All", df)]
    rows.append(["Patients, n (%)"] + [f"{len(g)} ({fr(100 * len(g) / n_all, 1)})" if c != "All" else f"{len(g)}"
                                       for c, g in groups])
    if female:
        rows.append(["Female, n (%)"] + [f"{int((g['sex'] == 'K').sum())} ({fr(100 * (g['sex'] == 'K').mean(), 1)})"
                                         for c, g in groups])
    else:
        rows.append(["Female, n (%)"] + ["n.r."] * len(groups))
    for lab, col, f, d in VARS[:1]:
        rows.append([f"{lab}, median (IQR)"] + [med(g[col] * f, d) for c, g in groups])
    rows.append(["Age <18 years, n (%)"] + [f"{int((g['age'] < 18).sum())} ({fr(100 * (g['age'] < 18).mean(), 1)})"
                                            for c, g in groups])
    for lab, col, f, d in VARS[1:9]:
        rows.append([lab] + [med(g[col] * f, d) for c, g in groups])
    rows.append(["Biochemistry measured, n (%)ᵃ"] + [
        f"{int((g[BIO].notna().sum(1) > 0).sum())} ({fr(100 * (g[BIO].notna().sum(1) > 0).mean(), 1)})" for c, g in groups])
    for lab, col, f, d in VARS[9:]:
        rows.append([lab] + [med(g[col] * f, d) for c, g in groups])
    return rows


cols = [LAB[c] for c in CLS] + ["All"]
t1_dev = cohort_block(dev, CLS, female=True)
t1_tem = cohort_block(tem, ["IDA", "HA", "HGB_HTZ", "NORMAL"], female=False)
t1_tem = [r[:5] + ["—"] + r[5:] for r in t1_tem]                  # no OAC column in the temporal cohort
table1 = {
    "title": "Table 1. Characteristics of the development and temporal cohorts by diagnostic class.",
    "header": ["Characteristic"] + cols,
    "panels": [{"label": f"Development cohort (nested cross-validation), n = {len(dev)}", "rows": t1_dev},
               {"label": f"Temporal cohort (same center, Feb–Apr 2026), n = {len(tem)}", "rows": t1_tem}],
    "notes": [
        "Values are median (interquartile range) unless stated otherwise; one (first) sample per patient. Biochemistry "
        "values are summarized over patients with the analyte measured.",
        "ᵃ At least one of ferritin, iron, UIBC and LDH measured within ±10 days of the CBC sample. In the temporal cohort a "
        "complete panel was an inclusion criterion.",
        "Conventional units: hemoglobin and MCHC g/dL = g/L ÷ 10; iron and UIBC µg/dL = µmol/L ÷ 0.1791; ferritin ng/mL = µg/L.",
        "IDA, iron deficiency anemia; HA, hemolytic anemia; HGB HTZ, heterozygous hemoglobinopathy; OAC, other anemia "
        "causes; IQR, interquartile range; MCV, mean corpuscular volume; MCHC, mean corpuscular hemoglobin concentration; "
        "RDW-SD, red cell distribution width (standard deviation); RET-He, reticulocyte hemoglobin equivalent; UIBC, "
        "unsaturated iron-binding capacity; LDH, lactate dehydrogenase; n.r., not recorded."],
}

# ───────────────────────────────────────────── Table 2
def cal(stage, sset, scen):
    x = T[(T.block == f"calibration_{stage}") & (T.set == sset) & (T.scenario == scen)].set_index("metric")["value"]
    return x


def row_binary(label, blk, sset, scen, pop=None, temporal_s1=False):
    if temporal_s1:
        s = get(blk, sset, "sensitivity", scen, pop)
        return [label, str(s[3]), "—ᵇ", ci(*s[:3], pct=True), f"{fr(100 * s[0], 1)} / —ᵇ", "—ᵇ", "—ᵇ"]
    a = get(blk, sset, "auc", scen, pop)
    acc = get(blk, sset, "accuracy", scen, pop)
    se = get(blk, sset, "sensitivity", scen, pop)[0]
    sp = get(blk, sset, "specificity", scen, pop)[0]
    st = "S1" if blk == "stage1" else "MX"
    c = cal(st, sset, scen)
    return [label, str(a[3]), ci(*a[:3]), ci(*acc[:3], pct=True), f"{fr(100 * se, 1)} / {fr(100 * sp, 1)}",
            f"{fr(c['uncalibrated_ECE'], 3)} / {fr(c['calibrated_ECE'], 3)}", f"{fr(c['calibrated_Brier'], 3)}"]


def row_s2(label, sset, scen):
    a = get("stage2", sset, "macro_auc", scen)
    acc = get("stage2", sset, "accuracy", scen)
    f1 = get("stage2", sset, "macro_f1", scen)[0]
    c = cal("S2", sset, scen)
    return [label, str(a[3]), ci(*a[:3]), ci(*acc[:3], pct=True), f"{fr(f1, 3)}",
            f"{fr(c['uncalibrated_ECE'], 3)} / {fr(c['calibrated_ECE'], 3)}", f"{fr(c['calibrated_Brier'], 3)}"]


def row_e2e(label, sset, scen):
    acc = get("end_to_end", sset, "accuracy", scen)
    f1 = get("end_to_end", sset, "macro_f1", scen)[0]
    return [label, str(acc[3]), "—", ci(*acc[:3], pct=True), f"{fr(f1, 3)}", "—", "—"]


table2 = {
    "title": "Table 2. Discrimination and calibration of the TabPFN-3.5 models in nested cross-validation and in the "
             "temporal cohort.",
    "header": ["Model", "n", "AUC (95% CI)", "Accuracy, % (95% CI)", "Sens / Spec, % or macro F1ᵃ",
               "ECE, raw / calibratedᶜ", "Brierᶜ"],
    "panels": [
        {"label": "Nested cross-validation (development cohort)", "rows": [
            row_binary("Stage 1 (AAC vs OAC), CBC", "stage1", "nested CV", "CBC"),
            row_binary("Stage 1, CBC + biochemistry", "stage1", "nested CV", "CBC_BIO"),
            row_s2("Stage 2 (four classes), CBC", "nested CV", "CBC"),
            row_s2("Stage 2, CBC + biochemistry", "nested CV", "CBC_BIO"),
            row_binary("IDA vs HGB HTZ, CBC", "mx", "nested CV", "CBC", pop="all (model)"),
            row_e2e("Five classes, Stage 1 → Stage 2, CBC", "nested CV", "CBC")]},
        {"label": "Temporal cohort (final models)", "rows": [
            row_binary("Stage 1 (AAC vs OAC), CBC", "stage1", "temporal", "CBC", temporal_s1=True),
            row_binary("Stage 1, CBC + biochemistry", "stage1", "temporal", "CBC_BIO", temporal_s1=True),
            row_s2("Stage 2 (four classes), CBC", "temporal", "CBC"),
            row_s2("Stage 2, CBC + biochemistry", "temporal", "CBC_BIO"),
            row_binary("IDA vs HGB HTZ, CBC", "mx", "temporal", "CBC", pop="all (model)"),
            row_e2e("Five classes, Stage 1 → Stage 2, CBC", "temporal", "CBC")]}],
    "notes": [
        "Patients' first samples. Stage 1 and the IDA vs HGB HTZ model at the locked threshold (maximum out-of-fold macro "
        "F1); Stage 2 by the most probable class in patients of the four target classes. Models with biochemistry: "
        "patients with at least one analyte measured. 95% CIs: DeLong for the AUCs of the binary models; patient "
        "bootstrap (2,000 resamples) for the Stage 2 macro-AUC and all other measures.",
        "ᵃ Sensitivity / specificity for the binary models (positive class AAC for Stage 1 and HGB HTZ for IDA vs HGB HTZ); "
        "macro F1 for Stage 2 and the five-class result (in the temporal cohort, the mean over the four classes present). "
        "Stage 2 AUC is the mean of the one-vs-rest AUCs.",
        "ᵇ The temporal cohort contains no OAC patients, so Stage 1 AUC, specificity and calibration are not defined; "
        "accuracy equals sensitivity.",
        "ᶜ Expected calibration error (ten bins; for Stage 2, of the top-class probability) before and after the locked "
        "calibrator; Brier score of the calibrated probabilities (for Stage 2, the squared error summed over the four "
        "classes and divided by four).",
        "AAC, the four target classes; OAC, other anemia causes; AUC, area under the receiver operating characteristic "
        "curve; CBC, complete blood count; ECE, expected calibration error; HGB HTZ, heterozygous hemoglobinopathy; IDA, "
        "iron deficiency anemia; Sens, sensitivity; Spec, specificity."],
}

# ───────────────────────────────────────────── Table 3
casc = pd.read_csv(R / "extra/cascade_by_target.csv").set_index(["set", "target"])
rfx = pd.read_csv(R / "reflex/reflex_distribution.csv")


def casc_col(sset):
    r = casc.loc[(sset, 0.9)] if (sset, 0.9) in casc.index else casc.loc[(sset, "0.90")]
    n = int(r["n"])
    t1 = int(r["n_tier1"])
    cbc_only = get("cascade", sset, "cbc_only_accuracy", None, "measured biochemistry")
    return {
        "Finalized at Tier 1, n (%) [95% CI]": f"{t1} ({fr(100 * t1 / n, 1)}) [{fr(100 * r['tier1_share_low'], 1)}–{fr(100 * r['tier1_share_high'], 1)}]",
        "Accuracy of Tier 1 results, % (95% CI)": ci(r["tier1_accuracy"], r["tier1_accuracy_low"], r["tier1_accuracy_high"], pct=True),
        "Escalated to Tier 2, n (%)": f"{n - t1} ({fr(100 * (n - t1) / n, 1)})",
        "Cascade accuracy, % (95% CI)": ci(r["cascade_accuracy"], r["cascade_accuracy_low"], r["cascade_accuracy_high"], pct=True),
        "Biochemistry for all patients, % (95% CI)": ci(r["cbc_bio_accuracy"], r["cbc_bio_accuracy_low"], r["cbc_bio_accuracy_high"], pct=True),
        "CBC only, % (95% CI)": ci(*cbc_only[:3], pct=True),
        "P, cascade vs biochemistry for allᵇ": f"{fr(r['p_mcnemar'], 2)}",
        "_n": n}


cn, ct = casc_col("nested CV"), casc_col("temporal")
rowsA = [[k, cn[k], ct[k]] for k in cn if not k.startswith("_")]

conf = pd.read_csv(R / "extra/conformal_by_class.csv")


def conf_cells(sset, scen):
    x = conf[(conf.set == sset) & (conf.scenario == scen) & (np.isclose(conf.alpha, 0.1)) & (conf["class"] == "all")].iloc[0]
    lo = conf[(conf.set == sset) & (conf.scenario == scen) & (np.isclose(conf.alpha, 0.1))
              & (conf["class"].isin(["HA", "HGB_HTZ", "IDA", "NORMAL"]))]["coverage"]
    return [f"{fr(100 * x.coverage, 1)}", f"{fr(100 * lo.min(), 1)}–{fr(100 * lo.max(), 1)}", f"{fr(x.mean_set_size, 2)}",
            f"{fr(100 * x.singleton_share, 1)}"]


rowsB = []
for sset, lab in (("nested CV", "Nested CV"), ("temporal", "Temporal")):
    for scen, sl in (("CBC", "CBC"), ("CBC_BIO", "CBC + biochemistry")):
        n = int(conf[(conf.set == sset) & (conf.scenario == scen) & (conf["class"] == "all")].iloc[0]["n"])
        rowsB.append([f"{lab}, {sl} (n = {n})"] + conf_cells(sset, scen))

rwo = pd.read_csv(R / "rwo/rwo_vs_model.csv").set_index("metric")


def rw(metric):
    r = rwo.loc[metric]
    return (f"{fr(100 * r['rwo'], 1)} ({fr(100 * r['rwo_low'], 1)}–{fr(100 * r['rwo_high'], 1)})",
            f"{fr(100 * r['model'], 1)} ({fr(100 * r['model_low'], 1)}–{fr(100 * r['model_high'], 1)})")


diff = rwo.loc["accuracy difference (model - RWO)"]
mc = rwo.loc["McNemar: only model correct / only RWO correct / p"]
rowsC = [["Accuracy, % (95% CI)", *rw("accuracy")],
         ["Sensitivity for IDA, %", *rw("sensitivity_IDA")],
         ["Sensitivity for HGB HTZ, %", *rw("sensitivity_HGB HTZ")],
         ["Sensitivity for other, %", *rw("sensitivity_Other")],
         ["Precision for HGB HTZ, %", *rw("precision_HGB HTZ")],
         ["Accuracy difference (cascade − RWO), percentage points", "—",
          f"{fr(100 * diff['model'], 1)} ({fr(100 * diff['model_low'], 1)} to {fr(100 * diff['model_high'], 1)})".replace("-", "−")],
         ["Discordant patients (only RWO correct / only cascade correct); P", "—",
          f"{int(mc['rwo'])} / {int(mc['model'])}; P = {fr(mc['model_high'], 2)}"]]

table3 = {
    "title": "Table 3. Two-tier cascade, conformal prediction and comparison with the manufacturer's rule-based algorithm.",
    "panels": [
        {"label": "A. Two-tier cascade at the pre-specified 90% rule (patients with measured biochemistry)",
         "header": ["", f"Nested CV (n = {cn['_n']})", f"Temporal (n = {ct['_n']})"], "rows": rowsA},
        {"label": "B. Conformal prediction, Stage 2, adaptive prediction sets, α = 0.10",
         "header": ["", "Coverage, %", "Coverage by class, %", "Mean set size", "Single-class sets, %"], "rows": rowsB},
        {"label": "C. RWO algorithm vs CBC-only cascade, common three-class scheme (nested CV, n = 863)ᶜ",
         "header": ["", "RWO algorithm", "CBC-only cascade"], "rows": rowsC}],
    "notes": [
        "ᵃ Tier 1 finalizes a patient when Stage 1 indicates one of the four target classes and the Stage 2 top-class "
        "probability reaches the HIGH cut-off locked on inner out-of-fold predictions for 90% accuracy; all other patients "
        "receive the biochemistry panel and are re-classified at Tier 2.",
        "ᵇ Exact McNemar test on paired patient-level results.",
        "ᶜ IDA; HGB HTZ including sickle cell trait; other = hemolytic anemia, other anemia causes and Normal. Cascade: "
        "Stage 1 at its locked threshold, then the Stage 2 class.",
        "CBC, complete blood count; CV, cross-validation; HGB HTZ, heterozygous hemoglobinopathy; IDA, iron deficiency "
        "anemia; RWO, RBC defect workflow optimization."],
}
table3["panels"][0]["label"] = table3["panels"][0]["label"].replace("90% rule", "90% ruleᵃ")

tables = {"table1": table1, "table2": table2, "table3": table3}
(MS / "tables_main.json").write_text(json.dumps(tables, indent=1, ensure_ascii=False))


def md(t):
    out = [f"**{t['title']}**", ""]
    for p in t["panels"]:
        hdr = p.get("header", t.get("header"))
        out += [f"*{p['label']}*", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        out += ["| " + " | ".join(r) + " |" for r in p["rows"]]
        out.append("")
    out += t["notes"] + [""]
    return "\n".join(out)


(MS / "tables_main.md").write_text("\n".join(md(t) for t in tables.values()))
print((MS / "tables_main.md").read_text())
