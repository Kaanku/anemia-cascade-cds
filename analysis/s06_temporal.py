"""
s06_temporal.py — Anemia cascade CDS, nested cross-validation (v4)
===================================================================

Builds the temporal validation cohort (Feb – Apr 2026) with the same rules as
the development cohort and writes model-ready matrices for the FINAL models of
every configuration, using the transformations fitted in s04 on all
development patients (ratios, imputer) without refitting anything.

Inputs (read-only):
    <ROOT>/#Yayın/Temporal Validasyon/Temporal Veriler.xls
        record-level diagnosis (chart review), age, iron/UIBC/ferritin/LDH
        (entered from the hospital information system), CBC as entered
    <ROOT>/#Yayın/Temporal Validasyon/1 SUBAT - 3 NISAN KAAN KUZU.csv
    <ROOT>/#Yayın/Temporal Validasyon/3 NISAN - 7 NISAN KAAN KUZU.csv
        Sysmex XN analyzer exports (source of truth for CBC values)
    <OUT>/data/features/<A>_<FS>_<SC>/final/features_*.json, imputer.joblib   (s04)

Outputs:
    <OUT>/data/temporal/temporal_cohort.parquet / .xlsx
    <OUT>/data/features/<A>_<FS>_<SC>/final/temporal.parquet
    <OUT>/reports/temporal_build_report.xlsx

Rules (DECISIONS.md V1–V7):
    V1  Temporal patients who also appear in the development cohort were
        excluded by the investigators when the cohort was assembled; each
        patient contributes one sample (verified by the investigators; the
        exports carry no patient identifiers).
    V2  Same cohort rules as development; CBC values from the analyzer export
        (last run of the sample), never from manually entered cells.
    V3  Consecutive patients (Feb 1 – Apr 7, 2026) with a confirmed diagnosis in
        one of the four target classes and a complete biochemistry panel; no
        OAC patients, so Stage 1 can only be evaluated for sensitivity.
    V4  Samples without an analyzer record keep the entered CBC values, which
        the investigators verified against the source record; FRC % (entered as
        per cent) is converted to the FRC#/RBC fraction used in development.
        They have none of the added parameters (N5), so they are flagged
        (has_analyzer_record = False) and evaluated with the T13 models only.
    V5  Documented corrections (source verified by the investigators) are applied
        in code and reported. They identify samples by their laboratory sample
        number, so they are kept out of the code in a local file
        (<OUT>/private/temporal_corrections.json, or --corrections): a list of
        {"sample_no", "feature", "entered", "corrected", "note"}; the study
        applied one (an MCH value entered in the MCV column).
    V6  A sample with any required CBC input missing or non-numeric is excluded,
        as in development.
    V7  All HGB HTZ patients are heterozygous (verified by the investigators).

Usage:
    python s06_temporal.py --root "/path/to/Kaan" --out "/path/to/Kaan/CDS_v2" [--corrections file.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from s01_build_cohort import EXTRA_RAW
from s04_features import BIO, CBC_BASE, MATRICES, add_ratios, apply_imputer, base_features, ratio_pairs

TDIR = Path("#Yayın") / "Temporal Validasyon"
LABELLED = TDIR / "Temporal Veriler.xls"
EXPORTS = [TDIR / "1 SUBAT - 3 NISAN KAAN KUZU.csv", TDIR / "3 NISAN - 7 NISAN KAAN KUZU.csv"]
CLASS_CODE = {0: "IDA", 1: "HA", 2: "HGB_HTZ", 3: "NORMAL"}
BIO_COLS = {"Ferritin": "ferritin", "Demir": "iron", "LDH": "ldh", "UIBC": "uibc"}
ANALYZER = {                       # analyzer column -> development feature (s01 definitions)
    "RBC(10^6/uL)": "rbc_10_6_u_l", "HGB(g/dL)": "hgb_g_d_l", "MCV(fL)": "mcv_f_l",
    "MCHC(g/dL)": "mchc_g_dl", "RDW-SD(fL)": "rdw_sd_fl", "NRBC%(%)": "nrbc_pct",
    "RET#(10^9/L)": "ret_number_10_9_l", "IRF(%)": "irf_pct", "RET-He(pg)": "ret_he_pg",
    "MicroR(%)": "micro_r_pct", "MacroR(%)": "macro_r_pct", "Delta-He(pg)": "delta_he_pg",
    "[FRC#(10^6/uL)]": "frc_number_10_6_u_l",
}
FILE_CBC = {                       # entered column -> development feature (V4)
    "RBC(10^6/uL)": "rbc_10_6_u_l", "HGB(g/dL)": "hgb_g_d_l", "MCV(fL)": "mcv_f_l",
    "MCHC(g/dL)": "mchc_g_dl", "RDW-SD(fL)": "rdw_sd_fl", "NRBC%(%)": "nrbc_pct",
    "IRF(%)": "irf_pct", "RET-He(pg)": "ret_he_pg", "Delta-He(pg)": "delta_he_pg",
}
# V5: corrections verified by the investigators against the source record (local file, see V5)
def load_corrections(path: Path) -> list[dict]:
    assert path.exists(), f"{path} not found: the documented V5 corrections are needed to rebuild the cohort"
    return json.loads(path.read_text(encoding="utf-8"))


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def load_exports(root: Path) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(root / f, skiprows=1, sep=",", encoding="latin-1", low_memory=False)
                   for f in EXPORTS], ignore_index=True)
    d["sample_no"] = d["Sample No."].astype(str).str.strip()
    d["run_time"] = pd.to_datetime(d["Date"].astype(str) + " " + d["Time"].astype(str),
                                   errors="coerce", dayfirst=True)
    d["n_runs"] = d.groupby("sample_no")["sample_no"].transform("size")
    return d.sort_values(["sample_no", "run_time"]).groupby("sample_no").tail(1).set_index("sample_no")


def build(root: Path, out: Path, corrections: Path | None = None) -> None:
    (out / "data" / "temporal").mkdir(parents=True, exist_ok=True)
    lab = pd.read_excel(root / LABELLED)
    lab["sample_no"] = lab["Sample No."].astype(str).str.strip()
    assert lab["sample_no"].is_unique
    an = load_exports(root)
    n0 = len(lab)

    rows = []
    for i, r in lab.reset_index(drop=True).iterrows():
        rec = {"record_id": f"T{i + 1:03d}", "patient_id": f"T{i + 1:03d}", "sample_no": r["sample_no"],
               "cls": CLASS_CODE[int(r["Tanı"])], "age": float(r["Yaş"])}
        for src, dst in BIO_COLS.items():
            rec[dst] = float(to_num(pd.Series([r[src]]))[0])
        if r["sample_no"] in an.index:                                      # V2
            a = an.loc[r["sample_no"]]
            raw = {dst: float(to_num(pd.Series([a[src]]))[0]) for src, dst in ANALYZER.items()}
            rec.update({k: raw[k] for k in ["rbc_10_6_u_l", "hgb_g_d_l", "mcv_f_l", "mchc_g_dl",
                                              "rdw_sd_fl", "nrbc_pct", "irf_pct", "ret_he_pg", "delta_he_pg"]})
            rec["ret_number_10_6_l"] = raw["ret_number_10_9_l"] / 1000
            rec["frc_perc"] = raw["frc_number_10_6_u_l"] / raw["rbc_10_6_u_l"]
            rec["micro_macro_ratio"] = (raw["micro_r_pct"] / raw["macro_r_pct"]
                                        if raw["macro_r_pct"] else np.nan)
            for src, dst in EXTRA_RAW.items():                                  # N5
                rec[dst] = float(to_num(pd.Series([a[src]]))[0])
            rec["sample_date"] = a["run_time"].normalize()
            rec["cbc_source"] = "analyzer export"
            rec["has_analyzer_record"] = True
            rec["n_runs"] = int(a["n_runs"])
        else:                                                               # V4
            for src, dst in FILE_CBC.items():
                rec[dst] = float(to_num(pd.Series([r[src]]))[0])
            rec["ret_number_10_6_l"] = float(to_num(pd.Series([r["RET#(10^9/L)"]]))[0]) / 1000
            rec["frc_perc"] = float(to_num(pd.Series([r["FRC%"]]))[0]) / 100
            rec["micro_macro_ratio"] = float(to_num(pd.Series([r["Micro/Macro"]]))[0])
            for dst in EXTRA_RAW.values():
                rec[dst] = np.nan
            rec["sample_date"] = pd.NaT
            rec["cbc_source"] = "entered, verified by investigators"
            rec["has_analyzer_record"] = False
            rec["n_runs"] = 0
        rows.append(rec)
    t = pd.DataFrame(rows)

    # V5 corrections
    corr_log = []
    for c in load_corrections(corrections or out / "private" / "temporal_corrections.json"):
        m = t["sample_no"] == c["sample_no"]
        assert m.sum() == 1, f"correction target not found: {c['sample_no']}"
        assert np.isclose(t.loc[m, c["feature"]].iloc[0], c["entered"]), "value differs from the documented entry"
        t.loc[m, c["feature"]] = c["corrected"]
        corr_log.append({**c, "cbc_source": t.loc[m, "cbc_source"].iloc[0]})

    # V6 analytic validity (same rule as development)
    invalid = t[CBC_BASE].isna().any(axis=1)
    excl = t[invalid].assign(reason=lambda d: "Missing or non-numeric analyzer result ("
                             + d[CBC_BASE].isna().apply(lambda r: ", ".join(r.index[r]), axis=1) + ")")
    t = t[~invalid].reset_index(drop=True)
    assert t[BIO].notna().all().all(), "temporal biochemistry expected complete (V3)"
    t["n_bio_measured"] = t[BIO].notna().sum(axis=1)
    t["y_s1"] = 1
    t["is_index_sample"] = True
    t["split"] = "temporal"
    t.to_parquet(out / "data" / "temporal" / "temporal_cohort.parquet", index=False)
    t.to_excel(out / "data" / "temporal" / "temporal_cohort.xlsx", index=False)

    # Model-ready matrices for the final models (nothing refitted) --------
    meta = ["record_id", "patient_id", "is_index_sample", "cls", "y_s1", "n_bio_measured",
            "has_analyzer_record"]
    for (analysis, fs, scenario) in MATRICES:
        fdir = out / "data" / "features" / f"{analysis}_{fs}_{scenario}" / "final"
        x = t.copy()
        if scenario == "CBC_BIO":
            x = apply_imputer(joblib.load(fdir / "imputer.joblib"), x)       # no-op: panel complete
        else:
            x = x.drop(columns=BIO)
        x, ratios = add_ratios(x, ratio_pairs(scenario))
        feats = set().union(*(json.loads(f.read_text()) for f in fdir.glob("features_*.json")))
        missing = feats - set(x.columns)
        assert not missing, f"{fdir}: missing features {missing}"
        base = base_features(fs, scenario)
        keep = meta + base + ratios + [c for c in x.columns if c.startswith("imputed_")]
        x[keep].to_parquet(fdir / "temporal.parquet", index=False)

    # Report ---------------------------------------------------------------
    flow = pd.DataFrame([
        ("Labelled temporal samples (consecutive, four target classes, complete biochemistry)", n0),
        ("  CBC from analyzer export", int((pd.DataFrame(rows)["cbc_source"] == "analyzer export").sum())),
        ("  CBC entered, verified by investigators (no analyzer record)",
         int((pd.DataFrame(rows)["cbc_source"] != "analyzer export").sum())),
        ("Excluded: missing or non-numeric analyzer result", len(excl)),
        ("Temporal cohort", len(t)),
    ] + [(f"  {c}", int(n)) for c, n in t["cls"].value_counts().items()], columns=["item", "n"])
    with pd.ExcelWriter(out / "reports" / "temporal_build_report.xlsx") as xw:
        flow.to_excel(xw, sheet_name="flow", index=False)
        pd.DataFrame(corr_log).to_excel(xw, sheet_name="corrections", index=False)
        excl[["record_id", "sample_no", "cls", "reason"]].to_excel(xw, sheet_name="excluded", index=False)
        t[["record_id", "sample_no", "cls", "cbc_source", "n_runs"]].to_excel(xw, sheet_name="source", index=False)
    print(flow.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--corrections", type=Path, default=None,
                    help="V5 corrections (default: <OUT>/private/temporal_corrections.json)")
    a = ap.parse_args()
    build(a.root, a.out, a.corrections)
