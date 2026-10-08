"""
s02_link_biochemistry.py — Anemia cascade CDS, clean rebuild (v2)
=================================================================

Links serum biochemistry to every CBC sample of the cohort built by
s01_build_cohort.py. No value is ever imputed here: a result is either a
measured value within the matching window or missing.

Inputs (read-only):
    <ROOT>/Veri/Tümü.xlsx                                raw CBC export (for linkage keys only)
    <ROOT>/R_Kaan_Tez/data/barkod_no.xlsx (--barcode-file) CBC sample no -> hospital file no
    <ROOT>/R_Kaan_Tez/data/add parameters/orj/*.xls*     monthly LIS exports per analyte
    <OUT>/data/cohort/cohort_all_samples.parquet         from s01
    <OUT>/data/cohort/cohort_secondary.parquet           from s01

Outputs:
    <OUT>/data/analysis/analysis_primary.parquet/.xlsx       PRIMARY analysis: one sample per patient
    <OUT>/data/analysis/analysis_all_samples.parquet/.xlsx   SENSITIVITY analysis: all valid samples
                                                             of the same patients
    <OUT>/data/analysis/analysis_secondary.parquet/.xlsx     indeterminate / out-of-scope set
    <OUT>/reports/biochem_linkage_report.xlsx                linkage + missingness tables

Rules (agreed 2026-09-27):
    B1  Patient file number (Dosya No) from the barcode file; otherwise from
        the biochemistry export by exact name + sex match that is unique and
        whose date of birth agrees with the recorded age (±1 year).
    B2  Per analyte, the result closest in time to the CBC sample within
        ±10 days (specimen acceptance date). Ties -> the earlier result.
    B3  Rejected specimens and non-numeric results (haemolysed, insufficient,
        frozen ...) are discarded. '<x' / '>x' are set to x and flagged.
    B4  TIBC is calculated in this laboratory as iron + UIBC. Where a
        specimen has TIBC and UIBC but no iron (export gap 7–15 Aug 2025),
        iron = TIBC − UIBC; where UIBC is missing, UIBC = TIBC − iron.
        Derived values are flagged.
    B5  Model inputs: iron, UIBC, ferritin, LDH. TIBC, vitamin B12 and folate
        are linked for description only.

Usage:
    python s02_link_biochemistry.py --root "/path/to/Kaan" --out "/path/to/Kaan/CDS_v2" [--barcode-file <file>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_DAYS = 10
BIO_DIR = Path("R_Kaan_Tez") / "data" / "add parameters" / "orj"
BARCODE_FILE = Path("R_Kaan_Tez") / "data" / "barkod_no.xlsx"          # site file name: --barcode-file
RAW_FILE = Path("Veri") / "Tümü.xlsx"

ANALYTES = {"Demir": "iron", "UIBC": "uibc", "TIBC": "tibc", "Ferritin": "ferritin",
            "LDH": "ldh", "Vitamin B12": "vitb12", "Folat": "folate"}
MODEL_BIO = ["iron", "uibc", "ferritin", "ldh"]
KEEP = ["Örnek", "Ad Soyad", "Test", "Sonuç", "Kabul Tar.", "Çalisma Tar.",
        "Dosya No", "Dogum Tar.", "Cinsiyet", "Rededildi"]


def norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)


def load_biochem(root: Path) -> pd.DataFrame:
    parts = []
    for f in sorted((root / BIO_DIR).glob("*.xls*")):
        if f.name.startswith("Transferr"):           # transferrin is not an analyte here; files are HTML
            continue
        d = pd.read_excel(f, engine="calamine", header=1, dtype=str)
        d.columns = [str(c).strip() for c in d.columns]
        missing = [c for c in KEEP if c not in d.columns]
        assert not missing, f"{f.name}: missing columns {missing}"
        parts.append(d[KEEP].dropna(subset=["Test"]).assign(source_file=f.name))
    b = pd.concat(parts, ignore_index=True)
    n0 = len(b)

    b["analyte"] = b["Test"].str.strip().map(ANALYTES)
    b = b[b["analyte"].notna()]
    b = b[b["Rededildi"].astype(str).str.strip() == "False"]          # B3 rejected / misaligned rows
    res = b["Sonuç"].astype(str).str.strip()
    b["censored"] = res.str.match(r"^[<>]")
    b["value"] = pd.to_numeric(res.str.lstrip("<>").str.replace(",", ".", regex=False), errors="coerce")
    b = b[b["value"].notna()]                                            # B3 non-numeric
    date = pd.to_datetime(b["Kabul Tar."], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    date = date.fillna(pd.to_datetime(b["Çalisma Tar."], format="%d.%m.%Y %H:%M:%S", errors="coerce"))
    b["bio_date"] = date.dt.normalize()
    b = b[b["bio_date"].notna()]
    b["dosya_no"] = b["Dosya No"].astype(str).str.strip()
    b["specimen"] = b["Örnek"].astype(str).str.strip()
    b["name"] = norm_name(b["Ad Soyad"])
    b["sex"] = b["Cinsiyet"].astype(str).str.strip().str.upper().str[0]
    b["dob"] = pd.to_datetime(b["Dogum Tar."], format="%d.%m.%Y", errors="coerce")
    b = b.drop_duplicates(["specimen", "analyte", "value"])
    print(f"biochemistry results: {n0} read, {len(b)} valid")
    return b[["specimen", "dosya_no", "name", "sex", "dob", "bio_date", "analyte", "value", "censored"]]


def derive_iron_uibc(b: pd.DataFrame) -> pd.DataFrame:
    """B4: TIBC = iron + UIBC in this laboratory; recover the missing member per specimen."""
    b = b.assign(derived=False)
    w = b.pivot_table(index=["specimen", "dosya_no", "bio_date"], columns="analyte",
                      values="value", aggfunc="first")
    new = []
    if {"tibc", "uibc"} <= set(w.columns):
        m = w["tibc"].notna() & w["uibc"].notna() & (w.get("iron", pd.Series(np.nan, index=w.index)).isna())
        v = (w.loc[m, "tibc"] - w.loc[m, "uibc"]).rename("value").reset_index()
        new.append(v.assign(analyte="iron"))
    if {"tibc", "iron"} <= set(w.columns):
        m = w["tibc"].notna() & w["iron"].notna() & (w.get("uibc", pd.Series(np.nan, index=w.index)).isna())
        v = (w.loc[m, "tibc"] - w.loc[m, "iron"]).rename("value").reset_index()
        new.append(v.assign(analyte="uibc"))
    if new:
        add = pd.concat(new, ignore_index=True)
        add = add[add["value"] > 0].assign(censored=False, derived=True)
        print(f"derived values: {add['analyte'].value_counts().to_dict()}")
        b = pd.concat([b, add], ignore_index=True)
    return b


def assign_file_numbers(rec: pd.DataFrame, b: pd.DataFrame, bar: pd.DataFrame) -> pd.DataFrame:
    """B1: barcode file first; otherwise name + sex candidates from the biochemistry
    export whose date of birth agrees with the recorded age (±1 y). Linked only
    when exactly one file number remains."""
    rec = rec.merge(bar, on="sample_no", how="left")
    rec["link_method"] = np.where(rec["dosya_no"].notna(), "barcode file", None)

    people = (b.dropna(subset=["name", "dob"])
                .drop_duplicates(["name", "sex", "dosya_no", "dob"])[["name", "sex", "dosya_no", "dob"]])
    cand = rec[rec["dosya_no"].isna()].drop(columns=["dosya_no"]).merge(
        people, on=["name", "sex"], how="inner")
    age_calc = np.floor((cand["sample_date"] - cand["dob"]).dt.days / 365.25)
    cand = cand[(age_calc - cand["age"]).abs() <= 1]
    n_ids = cand.groupby("record_id")["dosya_no"].nunique()
    cand = cand[cand["record_id"].isin(n_ids[n_ids == 1].index)].drop_duplicates("record_id")
    cand = cand[["record_id", "dosya_no"]]
    rec = rec.set_index("record_id")
    rec.loc[cand["record_id"], "dosya_no"] = cand["dosya_no"].values
    rec.loc[cand["record_id"], "link_method"] = "name + sex + date of birth"
    rec["link_method"] = rec["link_method"].fillna("not found in biochemistry export")
    return rec.reset_index()


def nearest_results(rec: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """B2: per record and analyte, closest result within ±WINDOW_DAYS (ties -> earlier)."""
    m = rec[["record_id", "dosya_no", "sample_date"]].dropna(subset=["dosya_no"]).merge(
        b[["dosya_no", "bio_date", "analyte", "value", "censored", "derived"]], on="dosya_no")
    m["days"] = (m["bio_date"] - m["sample_date"]).dt.days
    m = m[m["days"].abs() <= WINDOW_DAYS]
    m = m.assign(absd=m["days"].abs()).sort_values(["record_id", "analyte", "absd", "days"])
    m = m.drop_duplicates(["record_id", "analyte"], keep="first")
    out = m.pivot(index="record_id", columns="analyte", values="value")
    days = m.pivot(index="record_id", columns="analyte", values="days").add_prefix("days_")
    der = m.pivot(index="record_id", columns="analyte", values="derived").add_prefix("derived_")
    cen = m.pivot(index="record_id", columns="analyte", values="censored").add_prefix("censored_")
    return pd.concat([out, days, der, cen], axis=1).reset_index()


def build(root: Path, out: Path, barcode_file: Path = BARCODE_FILE) -> None:
    (out / "data" / "analysis").mkdir(parents=True, exist_ok=True)
    coh = out / "data" / "cohort"
    all_s = pd.read_parquet(coh / "cohort_all_samples.parquet")
    sec = pd.read_parquet(coh / "cohort_secondary.parquet")

    raw = pd.read_excel(root / RAW_FILE, dtype=str)
    keys = pd.DataFrame({
        "record_id": np.arange(1, len(raw) + 1),
        "sample_no": raw["Sample No."].astype(str).str.strip(),
        "name": norm_name(raw["AD-SOYAD"]),
        "sex": raw["Cinsiyet"].astype(str).str.strip().str.upper().str[0],
    })
    ids = pd.concat([all_s[["record_id", "sample_date", "age"]],
                     sec[["record_id", "sample_date", "age"]]], ignore_index=True)
    assert ids["record_id"].is_unique
    rec = ids.merge(keys, on="record_id", how="left")

    bar = pd.read_excel(root / barcode_file, dtype=str)
    bar = (bar.rename(columns={"sample_id_orj": "sample_no", "DOSYA_NO": "dosya_no"})
              [["sample_no", "dosya_no"]].dropna().drop_duplicates("sample_no"))
    bar["sample_no"] = bar["sample_no"].str.strip()
    bar["dosya_no"] = bar["dosya_no"].str.strip()

    b = derive_iron_uibc(load_biochem(root))
    rec = assign_file_numbers(rec, b, bar)
    linked = nearest_results(rec, b)
    rec = rec.drop(columns=["name", "sex", "sample_no"]).merge(linked, on="record_id", how="left")

    # Identity cross-check: s01 patient_id (name-based) vs hospital file number
    pid = pd.concat([all_s[["record_id", "patient_id"]], sec[["record_id", "patient_id"]]])
    chk = rec.merge(pid, on="record_id")[["patient_id", "dosya_no"]].dropna()
    pid_multi = int((chk.groupby("patient_id")["dosya_no"].nunique() > 1).sum())
    dos_multi = int((chk.groupby("dosya_no")["patient_id"].nunique() > 1).sum())

    bio_cols = [c for c in rec.columns if c != "record_id" and c not in ("sample_date", "age")]
    rec_out = rec.drop(columns=["sample_date", "age", "dosya_no"])     # file number stays out of outputs
    for name, base in [("analysis_all_samples", all_s),
                       ("analysis_primary", all_s[all_s["is_index_sample"]]),
                       ("analysis_secondary", sec)]:
        df = base.merge(rec_out, on="record_id", how="left", validate="one_to_one")
        for c in MODEL_BIO:
            if c not in df.columns:
                df[c] = np.nan
        df.to_parquet(out / "data" / "analysis" / f"{name}.parquet", index=False)
        df.to_excel(out / "data" / "analysis" / f"{name}.xlsx", index=False)

    # Report ---------------------------------------------------------------
    a = pd.read_parquet(out / "data" / "analysis" / "analysis_all_samples.parquet")
    p = pd.read_parquet(out / "data" / "analysis" / "analysis_primary.parquet")
    def miss(df, by):
        t = df.groupby(by)[MODEL_BIO].apply(lambda g: g.isna().mean() * 100).round(1)
        t.loc["All"] = (df[MODEL_BIO].isna().mean() * 100).round(1)
        t["all_four_missing_%"] = (df.assign(_m=df[MODEL_BIO].isna().all(axis=1))
                                     .groupby(by)["_m"].mean() * 100).round(1)
        t.loc["All", "all_four_missing_%"] = round(df[MODEL_BIO].isna().all(axis=1).mean() * 100, 1)
        return t
    with pd.ExcelWriter(out / "reports" / "biochem_linkage_report.xlsx") as xw:
        pd.DataFrame({
            "item": ["records linked (all samples + secondary)", "patient_id with >1 file number",
                     "file number with >1 patient_id"],
            "n": [len(rec), pid_multi, dos_multi]}).to_excel(xw, sheet_name="summary", index=False)
        rec["link_method"].value_counts().rename_axis("method").reset_index(name="n").to_excel(
            xw, sheet_name="link_method", index=False)
        miss(a, "cls").to_excel(xw, sheet_name="missing_%_all_samples")
        miss(p, "cls").to_excel(xw, sheet_name="missing_%_primary")
        dd = pd.concat({c: a[f"days_{c}"].describe() for c in MODEL_BIO if f"days_{c}" in a}, axis=1)
        dd.to_excel(xw, sheet_name="days_from_cbc")
        der = pd.Series({c: int(a.get(f"derived_{c}", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
                         for c in ["iron", "uibc"]}, name="derived_n")
        der.to_excel(xw, sheet_name="derived_values")
        within3 = pd.Series({c: round(float((a[f"days_{c}"].abs() <= 3).sum() / a[c].notna().sum() * 100), 1)
                             for c in MODEL_BIO}, name="% of linked results within ±3 days")
        within3.to_excel(xw, sheet_name="within_3_days")
    print("link methods:", rec["link_method"].value_counts().to_dict())
    print("identity cross-check: patient_id>1 file no:", pid_multi, "| file no>1 patient_id:", dos_multi)
    print("missing % (all samples):\n", miss(a, "cls").to_string())
    print("missing % (primary):\n", miss(p, "cls").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--barcode-file", type=Path, default=BARCODE_FILE,
                    help="barcode file (sample no -> file no), relative to --root or absolute")
    a = ap.parse_args()
    build(a.root, a.out, a.barcode_file)
