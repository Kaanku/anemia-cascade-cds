"""
s01_build_cohort.py — Anemia cascade CDS, clean rebuild (v2)
=============================================================

Builds the analysis cohort from the raw LIS export, reproducibly and with a
full audit trail. Every record in the raw file ends up in exactly one of:

    PRIMARY    -> enters model development (IDA / HA / HGB_HTZ / NORMAL / OAC)
    SECONDARY  -> "indeterminate / out-of-scope" set, used only to describe how
                  the frozen cascade behaves on cases outside the target classes
    EXCLUDED   -> with a single, documented reason

Inputs (read-only):
    <ROOT>/Veri/Tümü.xlsx                      raw LIS export, 2,289 records
    <ROOT>/R_Kaan_Tez/data/SadeX.xlsx [Sade]   record-level diagnoses adjudicated
                                               by chart review with a clinician
    <ROOT>/Veri/Orijinal Veriler/*.csv         original Sysmex XN exports (WBC/PLT block, D12)

Outputs (<OUT>/data/cohort and <OUT>/reports):
    cohort_primary.parquet / .xlsx     one row per patient (first valid sample), no names
    cohort_secondary.parquet / .xlsx   one row per patient, no names
    cohort_all_samples.parquet / .xlsx every valid sample of the primary-cohort
                                       patients (analysis A2)
    record_audit.xlsx                  all 2,289 records: status + reason
    flow_counts.xlsx                   STARD flow numbers, step by step

Decisions encoded here (agreed 2026-09-27):
    D1  Root file is Tümü; SadeX supplies the chart-review label per record.
    D2' One sample per patient: the FIRST (earliest) analytically valid sample,
        taken before any treatment could alter the blood count (replaces the
        most-recent rule, 2026-09-29). Patient identity = normalised name + sex,
        split when ages differ >1 y.
    D3  Non-anemia diagnoses (polycythaemia, thrombocytopenia, thrombophilia,
        high B12) are excluded.
    D4  'TANI?' (diagnosis unclear) -> SECONDARY.
    D5  Homozygous / major haemoglobinopathy (HbS > 50 %, HbF >= 20 %, or a
        'major' / 'intermedia' label) -> SECONDARY, never HGB_HTZ.
    D6  Hereditary spherocytosis, megaloblastic anemia, combined IDA+ACD
        -> SECONDARY (outside the four-class scope).
    D7  'ANEMİ' (anemia of undetermined cause) stays in OAC.
    D8  AIHA with coexisting CLL stays in HA.
    D12 Source of the WBC/PLT block (2026-10-06): the 11 WBC/PLT block parameters (WBC,
        NEUT#, LYMPH#, MONO#, EO#, BASO#, IG#, PLT, MPV, PDW, P-LCR) are taken from the
        original analyzer exports (<ROOT>/Veri/Orijinal Veriler), matching each record to
        its run by sample number and five red cell/reticulocyte results; the red cell and
        reticulocyte block is read from the raw file.
    D9  Question-marked thalassaemia labels stay in HGB_HTZ as adjudicated
        (diagnosis documented by an external electrophoresis report).
    D10' Two full analyses on the same patients, reported side by side:
        A1 = one (first) sample per patient; A2 = all valid samples of those
        patients. Any downstream split or CV must be grouped by patient_id.
    N5  Model inputs beyond the thesis set (full blood count and extended
        reticulocyte parameters, EXTRA_RAW) are carried as analyzer values;
        a non-numeric flag ('----', analyzer could not report) stays missing
        and does not exclude the sample (validity is judged on CBC_RAW only).

Usage:
    python s01_build_cohort.py --root "/path/to/Kaan" --out "/path/to/Kaan/CDS_v2"
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RAW_FILE = Path("Veri") / "Tümü.xlsx"
ADJ_FILE = Path("R_Kaan_Tez") / "data" / "SadeX.xlsx"
ADJ_SHEET = "Sade"

# Chart-review label (SadeX) -> analysis class
PRIMARY_MAP = {
    "DEA": "IDA",
    "HEMOLİTİK ANEMİ": "HA",
    "TALASEMİ": "HGB_HTZ",
    "HGB HTZ": "HGB_HTZ",
    "ORAK HÜCRELİ ANEMİ": "HGB_HTZ",
    "NORMAL": "NORMAL",
    "ANEMİ": "OAC",            # D7
    "KHA": "OAC",
    "APLASTİK ANEMİ": "OAC",
    "MDS": "OAC",
    "CA": "OAC",
    "PANSİTOPENİ": "OAC",
}
NON_ANEMIA = {"POLİSİTEMİ", "TROMBOSİTOPENİ", "TROMBOFİLİ", "VİTAMİN B12 YÜKSEKLİĞİ"}  # D3
SECONDARY_MAP = {                                                                     # D4, D6
    "TANI?": "Diagnosis unclear",
    "HS": "Hereditary spherocytosis",
    "MEGALOBLASTİK ANEMİ": "Megaloblastic anemia",
    "DEA+ KHA": "Combined IDA + ACD",
}

# Raw CBC columns required as model inputs (analytic validity check)
CBC_RAW = {
    "RBC(10^6/uL)": "rbc_10_6_u_l",
    "HGB(g/dL)": "hgb_g_d_l",
    "MCV(fL)": "mcv_f_l",
    "MCHC(g/dL)": "mchc_g_dl",
    "RDW-SD(fL)": "rdw_sd_fl",
    "NRBC%(%)": "nrbc_pct",
    "RET#(10^9/L)": "ret_number_10_9_l",
    "IRF(%)": "irf_pct",
    "RET-He(pg)": "ret_he_pg",
    "MicroR(%)": "micro_r_pct",
    "MacroR(%)": "macro_r_pct",
    "Delta-He(pg)": "delta_he_pg",
    "[FRC#(10^6/uL)]": "frc_number_10_6_u_l",
}
# Further analyzer parameters of the same run (N5): full blood count and the
# extended reticulocyte panel. Bracketed names are research-use parameters.
EXTRA_RAW = {
    "HCT(%)": "hct_pct", "MCH(pg)": "mch_pg", "RDW-CV(%)": "rdw_cv_pct",
    "NRBC#(10^3/uL)": "nrbc_number_10_3_u_l",
    "WBC(10^3/uL)": "wbc_10_3_u_l", "NEUT#(10^3/uL)": "neut_number_10_3_u_l",
    "LYMPH#(10^3/uL)": "lymph_number_10_3_u_l", "MONO#(10^3/uL)": "mono_number_10_3_u_l",
    "EO#(10^3/uL)": "eo_number_10_3_u_l", "BASO#(10^3/uL)": "baso_number_10_3_u_l",
    "IG#(10^3/uL)": "ig_number_10_3_u_l",
    "PLT(10^3/uL)": "plt_10_3_u_l", "MPV(fL)": "mpv_fl", "PDW(fL)": "pdw_fl", "P-LCR(%)": "p_lcr_pct",
    "RET%(%)": "ret_pct", "LFR(%)": "lfr_pct", "MFR(%)": "mfr_pct", "HFR(%)": "hfr_pct",
    "RBC-He(pg)": "rbc_he_pg", "MicroR(%)": "micro_r_pct", "MacroR(%)": "macro_r_pct",
    "[RET-Y(ch)]": "ret_y_ch", "[RET-RBC-Y(ch)]": "ret_rbc_y_ch", "[IRF-Y(ch)]": "irf_y_ch",
}
HB_COLS = {"HbA": "hba", "HbA2": "hba2", "HbF": "hbf", "HbS": "hbs", "HbC": "hbc",
           "Hb Elektroforez": "hb_electrophoresis"}

# Keyword rules for records excluded at chart review (reason for the flow chart).
# Evaluated in order; first match wins.
EXCLUSION_RULES = [
    ("Removed at final chart review",
     r"^OTOİMMÜN HEMOLİTİK ANEMİ|^HS$|^MEGALOBLASTİK|^VİTAMİN B12 EKSİKLİĞİ"),
    ("Diagnosis not established",
     r"\?|TANI ALAMAMIŞ|YANLIŞ YÖNLENDİRME"),
    ("Haematological malignancy or marrow disorder",
     r"\bKLL\b|\bAML\b|\bALL\b|\bKML\b|\bMDS\b|\bMM\b|LENFOMA|MYELOFİBROZİS|WALDENSTRÖM|MGUS|"
     r"PLASMASİTOM|\bKMH\b|ESANSİYEL TROMBOSİTOZ|POLİSİTEMİA VERA|MASTOSİTOZ|MYELOMONOSİTİK|"
     r"LENFOPROLİFERATİF|HEMOFAGOSİTİK|\bPNH\b|KAN VE KAN YAPICI"),
    ("Transplantation, chemo-/radiotherapy or dialysis",
     r"NAKİL|NAKLİ|KEMİK İLİĞİ ALICISI|KT SONRASI|KTYE|RT’YE|RT'YE|HEMODİYALİZ|POSTTRAN"),
    ("Solid malignancy",
     r"\bCA\b|SARKOM|\bTM\b|GLİOBLASTOMA|\bRCC\b|TİMOMA|KİTLE"),
    ("Multiple coexisting conditions",
     r"\+"),
    ("Microangiopathic or secondary haemolysis",
     r"\bTTP\b|\bHÜS\b|HELLP|MEKANİK HEMOLİZ|ENFEKSİYONA BAĞLI HEMOLİZ|\bDİK\b"),
    ("Non-anemic haematological finding",
     r"TROMBOSİTOPENİ|\bİTP\b|\bITP\b|LÖKOPENİ|NÖTROPENİ|LÖKOSİTOZ|LENFOSİTOZ|TROMBOSİTOZ|"
     r"POLİSİTEMİ|HEMOGLOBİN YÜKSEKLİĞİ|TROMBOFİLİ|HEMOFİLİ|FAKTÖR|F13|KANAMA|MORARMA|EKİMOZ|"
     r"PANSİTOPENİ|BİSİTOPENİ|SİTOPENİ|EOZİNOFİL|MCV YÜKSEKLİĞİ|B12 YÜKSEKLİĞİ|SPLENOMEGALİ|\bHSM\b|"
     r"\bLAP\b|\bHİT\b|GLANZMAN|HİPOFİBRİNOJENEMİ|PSÖDOTROMBOSİTOPENİ|SOLA KAYMA|LÖKOMOİD|"
     r"NÖTROSİTOZ|TROMBOEMBOLİ|\bPTE\b|KRİYOGLOBULİNEMİ|SPLENİK|DALAK|HEMOKROMATOZİS|SUPRESYON|"
     r"KEMİK İLİĞİ DONÖRÜ|KEMİL İLİĞİ DONÖRÜ|CROSS UYUMSUZLUĞU|KAN ÜRÜNÜ|RH UY"),
    ("Non-haematological or other out-of-scope diagnosis",
     r".*"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def norm_label(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


EXPORT_DIR = Path("Veri") / "Orijinal Veriler"       # original Sysmex XN exports (D12)
WBC_BLOCK = ["WBC(10^3/uL)", "NEUT#(10^3/uL)", "LYMPH#(10^3/uL)", "MONO#(10^3/uL)", "EO#(10^3/uL)",
             "BASO#(10^3/uL)", "IG#(10^3/uL)", "PLT(10^3/uL)", "MPV(fL)", "PDW(fL)", "P-LCR(%)"]
RUN_KEY = ["HGB(g/dL)", "MCV(fL)", "RDW-SD(fL)", "RET-He(pg)", "IRF(%)"]
RUN_CHECK = ["RBC(10^6/uL)", "MCHC(g/dL)", "MicroR(%)", "RET#(10^9/L)", "HCT(%)", "RBC-He(pg)"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def wbc_block_from_exports(src: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D12: take the WBC/PLT block of every record from the original analyzer export.

    Each record is matched to its analyzer run by sample number and five red cell / reticulocyte results
    (exact to 0.01); six further results are checked. Runs exported twice must agree on
    the block; if two runs disagree, the one whose values appear in the raw row is kept. A
    record without a matching run keeps the block missing."""
    files = sorted(f for f in (root / EXPORT_DIR).glob("*.csv") if not f.stem.endswith(("_2", "_3")))
    want = ["Sample No."] + RUN_KEY + RUN_CHECK + WBC_BLOCK
    parts = []
    for f in files:
        d = pd.read_csv(f, skiprows=1, sep=",", encoding="latin-1", low_memory=False)
        d.columns = [str(c).strip() for c in d.columns]
        assert all(c in d.columns for c in ["Sample No."] + RUN_KEY + WBC_BLOCK), f"{f.name}: columns missing"
        parts.append(d.reindex(columns=want))                 # a check column absent in a file stays NaN
    ex = pd.concat(parts, ignore_index=True)
    ex["sample_no"] = ex["Sample No."].astype(str).str.strip()
    for c in RUN_KEY + RUN_CHECK:
        ex[c] = _num(ex[c]).round(2)
    for c in WBC_BLOCK:                     # numeric; analyzer flags ('----') become missing, as in s01 to_num
        ex[c] = _num(ex[c])
    ex = ex.drop(columns="Sample No.").drop_duplicates()     # weekly exports overlap: same run twice
    rec = pd.DataFrame({"row": np.arange(len(src)), "sample_no": src["Sample No."].astype(str).str.strip()})
    for c in RUN_KEY + RUN_CHECK:
        rec[c] = _num(src[c]).round(2)
    m = rec.merge(ex, on=["sample_no"] + RUN_KEY, how="left", suffixes=("", "_run"))
    m["matched"] = m[WBC_BLOCK[0]].notna() | m[WBC_BLOCK[1:]].notna().any(axis=1)
    check_ok = np.ones(len(m), bool)
    for c in RUN_CHECK:
        a, b = m[c], m[c + "_run"]
        check_ok &= ~(a.notna() & b.notna() & (a != b)).to_numpy()
    m = m[~m["matched"] | check_ok]
    blk = m[m["matched"]].groupby("row")[WBC_BLOCK].apply(
        lambda g: pd.Series({c: g[c].dropna().unique() for c in WBC_BLOCK}, dtype=object))
    agree = blk.apply(lambda r: all(len(v) <= 1 for v in r), axis=1)
    fixed = src.copy()
    for c in WBC_BLOCK:
        fixed[c] = np.nan
        fixed[c] = fixed[c].astype(object)
    ok = blk.index[agree.to_numpy()]
    for c in WBC_BLOCK:
        fixed.loc[ok, c] = [v[0] if len(v) else np.nan for v in blk.loc[ok, c]]
    # Two different runs with identical red cell results: keep the run whose WBC, PLT and NEUT#
    # all appear in the raw row.
    i0 = list(src.columns).index(WBC_BLOCK[0])
    tie = []
    for r in blk.index[~agree.to_numpy()]:
        cand = m[m["row"] == r]
        cells = set(_num(src.iloc[r, i0 - 6:].astype(str)).dropna().round(2))
        hit = cand[cand.apply(lambda q: all(round(float(_num(pd.Series([q[c]]))[0]), 2) in cells
                                            for c in ["WBC(10^3/uL)", "PLT(10^3/uL)", "NEUT#(10^3/uL)"]
                                            if pd.notna(_num(pd.Series([q[c]]))[0])), axis=1)]
        if len(hit.drop_duplicates(subset=WBC_BLOCK)) == 1:
            for c in WBC_BLOCK:
                fixed.loc[r, c] = hit.iloc[0][c]
            tie.append(r)
    ok = ok.append(pd.Index(tie))
    old_wbc, new_wbc = _num(src["WBC(10^3/uL)"]), _num(fixed["WBC(10^3/uL)"])
    report = pd.DataFrame({"run_found": rec["row"].isin(ok).to_numpy(),
                           "runs_disagree": rec["row"].isin(blk.index[~agree.to_numpy()]).to_numpy(),
                           "resolved_from_raw_row": rec["row"].isin(tie).to_numpy(),
                           "wbc_changed": ~((old_wbc == new_wbc) | (old_wbc.isna() & new_wbc.isna())).to_numpy()})
    return fixed, report


def to_num(s: pd.Series) -> pd.Series:
    """Parse analyzer values; non-numeric flags such as '----' become NaN."""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False),
                         errors="coerce")


def record_key(df: pd.DataFrame) -> pd.DataFrame:
    """Key used to link the adjudication file to the raw export (no names)."""
    k = pd.DataFrame(index=df.index)
    k["sample_no"] = df["Sample No."].astype(str).str.strip()
    k["date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True).dt.normalize()
    for c in ["HGB(g/dL)", "MCV(fL)", "RDW-SD(fL)", "RET-He(pg)"]:
        k[c] = to_num(df[c]).round(2)
    return k


def classify_exclusion(label: str) -> str:
    for reason, pattern in EXCLUSION_RULES:
        if re.search(pattern, label):
            return reason
    return "Non-haematological or other out-of-scope diagnosis"


def patient_ids(name: pd.Series, sex: pd.Series, age: pd.Series) -> pd.Series:
    """Pseudonymous patient id: name + sex, split into separate persons when
    ages within the same name+sex differ by more than one year."""
    base = (name.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
            + "|" + sex.astype(str).str.upper().str.strip())
    out = pd.Series(index=name.index, dtype=object)
    for key, idx in base.groupby(base).groups.items():
        ages = age.loc[idx].sort_values()
        cluster, anchor = 0, None
        for i, a in ages.items():
            if anchor is not None and (pd.isna(a) or pd.isna(anchor) or a - anchor > 1):
                cluster += 1
                anchor = a
            if anchor is None:
                anchor = a
            out.loc[i] = hashlib.sha256(f"{key}|{cluster}".encode()).hexdigest()[:12]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build(root: Path, out: Path) -> None:
    (out / "data" / "cohort").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    flow: list[tuple[str, str, int]] = []

    # 0. Raw export --------------------------------------------------------
    src = pd.read_excel(root / RAW_FILE).reset_index(drop=True)
    src, align = wbc_block_from_exports(src, root)                            # D12
    sex = src["Cinsiyet"].astype(str).str.strip().str.upper()
    age = to_num(src["YAŞ"])
    # Working table: only what the pipeline needs (names never leave this function)
    raw = pd.DataFrame({
        "record_id": np.arange(1, len(src) + 1),
        "sample_no": src["Sample No."].astype(str).str.strip(),
        "raw_label": norm_label(src["Tanısı"]),
        "sample_date": pd.to_datetime(src["Date"], errors="coerce", dayfirst=True).dt.normalize(),
        "age": age,
        "sex": sex,
        "patient_id": patient_ids(src["AD-SOYAD"], sex, age),
        "adj_label": pd.Series([None] * len(src), dtype=object),
        "status": pd.Series([None] * len(src), dtype=object),
        "stage": pd.Series([None] * len(src), dtype=object),
        "reason": pd.Series([None] * len(src), dtype=object),
        "cls": pd.Series([None] * len(src), dtype=object),
    })
    flow.append(("0", "Records in LIS export (Jul 2025 – Jan 2026)", len(raw)))

    # 1. Link chart-review label -------------------------------------------
    adj = pd.read_excel(root / ADJ_FILE, sheet_name=ADJ_SHEET)
    adj["adj_label"] = norm_label(adj["Tanısı"])
    kr, ka = record_key(src), record_key(adj)
    keycols = list(kr.columns)
    ka = ka.assign(adj_label=adj["adj_label"].values)
    assert not ka.duplicated(keycols).any(), "Adjudication file has duplicate record keys"
    assert not kr.duplicated(keycols).any(), "Raw export has duplicate record keys"
    linked = kr.merge(ka, on=keycols, how="left")
    assert len(linked) == len(raw)
    raw["adj_label"] = linked["adj_label"].values
    n_adj = raw["adj_label"].notna().sum()
    assert n_adj == len(adj), f"Only {n_adj}/{len(adj)} adjudicated records linked to raw export"

    # 2. Exclusion at chart review -----------------------------------------
    m = raw["adj_label"].isna()
    raw.loc[m, "status"] = "EXCLUDED"
    raw.loc[m, "stage"] = "1 Chart review"
    raw.loc[m, "reason"] = raw.loc[m, "raw_label"].map(classify_exclusion)
    flow.append(("1", "Excluded at chart review (total)", int(m.sum())))
    for r, n in raw.loc[m, "reason"].value_counts().items():
        flow.append(("1", f"  – {r}", int(n)))

    # 3. Class assignment from adjudicated label ---------------------------
    act = raw["status"].isna()
    lab = raw["adj_label"]
    nonan = act & lab.isin(NON_ANEMIA)
    raw.loc[nonan, ["status", "stage", "reason"]] = ["EXCLUDED", "2 Scope", "Non-anemic diagnosis"]
    sec = act & lab.isin(SECONDARY_MAP)
    raw.loc[sec, "status"] = "SECONDARY"
    raw.loc[sec, "reason"] = lab[sec].map(SECONDARY_MAP)
    pri = act & lab.isin(PRIMARY_MAP)
    raw.loc[pri, "status"] = "PRIMARY"
    raw.loc[pri, "cls"] = lab[pri].map(PRIMARY_MAP)
    unmapped = raw["status"].isna()
    assert not unmapped.any(), f"Unmapped adjudicated labels: {sorted(lab[unmapped].unique())}"
    flow.append(("2", "Excluded: non-anemic diagnosis after chart review", int(nonan.sum())))

    # 4. Homozygous / major haemoglobinopathy -> SECONDARY (D5) -------------
    hbs, hbf = to_num(src["HbS"]), to_num(src["HbF"])
    homo = ((raw["cls"] == "HGB_HTZ")
            & ((hbs > 50) | (hbf >= 20)
               | raw["raw_label"].str.contains(r"MAJOR|İNTERMED|INTERMED", regex=True)))
    raw.loc[homo, ["status", "reason", "cls"]] = ["SECONDARY", "Homozygous or major haemoglobinopathy", None]
    flow.append(("3", "Moved to secondary set: homozygous/major haemoglobinopathy", int(homo.sum())))

    # 5. Analytic validity (all required CBC inputs numeric) ---------------
    cbc = pd.DataFrame({new: to_num(src[old]) for old, new in CBC_RAW.items()})
    invalid = cbc.isna().any(axis=1) | raw["age"].isna() | ~raw["sex"].isin(["E", "K"])
    inval = raw["status"].isin(["PRIMARY", "SECONDARY"]) & invalid
    raw.loc[inval, "reason"] = ("Missing or non-numeric analyzer result ("
                                + cbc[inval].isna().apply(lambda r: ", ".join(r.index[r]), axis=1)
                                + ")")
    raw.loc[inval, ["status", "stage"]] = ["EXCLUDED", "4 Analytic"]
    flow.append(("4", "Excluded: missing or non-numeric analyzer result", int(inval.sum())))

    # 6. One sample per patient: the first valid sample (D2') ---------------
    elig = raw["status"].isin(["PRIMARY", "SECONDARY"])
    order = raw[elig].sort_values(["patient_id", "sample_date", "record_id"])
    keep_idx = order.groupby("patient_id").head(1).index
    dup = elig & ~raw.index.isin(keep_idx)
    raw.loc[dup, ["status", "stage", "reason"]] = ["EXCLUDED", "5 Repeat", "Later sample of same patient"]
    flow.append(("5", "Excluded from one-sample set: later sample of a patient with repeat testing", int(dup.sum())))

    # 7. Final sets --------------------------------------------------------
    raw.loc[raw["status"] == "PRIMARY", "stage"] = "Included"
    raw.loc[raw["status"] == "SECONDARY", "stage"] = "Secondary set"
    for c, n in raw.loc[raw["status"] == "PRIMARY", "cls"].value_counts().items():
        flow.append(("6", f"Primary cohort – {c}", int(n)))
    flow.append(("6", "Primary cohort (total patients)", int((raw["status"] == "PRIMARY").sum())))
    for r, n in raw.loc[raw["status"] == "SECONDARY", "reason"].value_counts().items():
        flow.append(("7", f"Secondary set – {r}", int(n)))
    flow.append(("7", "Secondary set (total patients)", int((raw["status"] == "SECONDARY").sum())))

    # Features (definitions identical to the thesis R pipeline) ------------
    feat = pd.DataFrame(index=raw.index)
    feat["age"] = raw["age"]
    feat["sex"] = raw["sex"]
    for new in ["hgb_g_d_l", "rbc_10_6_u_l", "mcv_f_l", "mchc_g_dl", "nrbc_pct", "rdw_sd_fl",
                "irf_pct", "ret_he_pg", "delta_he_pg"]:
        feat[new] = cbc[new]
    feat["ret_number_10_6_l"] = cbc["ret_number_10_9_l"] / 1000
    feat["frc_perc"] = cbc["frc_number_10_6_u_l"] / cbc["rbc_10_6_u_l"]
    feat["micro_macro_ratio"] = cbc["micro_r_pct"] / cbc["macro_r_pct"].replace(0, np.nan)
    for old, new in EXTRA_RAW.items():                                        # N5
        feat[new] = to_num(src[old])
    hb = pd.DataFrame({new: (to_num(src[old]) if new != "hb_electrophoresis"
                             else src[old].astype(str).str.strip())
                       for old, new in HB_COLS.items()})

    meta = raw[["record_id", "patient_id", "sample_date", "raw_label", "adj_label", "cls", "reason"]]
    for status, fname in [("PRIMARY", "cohort_primary"), ("SECONDARY", "cohort_secondary")]:
        sel = raw["status"] == status
        df = pd.concat([meta[sel], feat[sel], hb[sel]], axis=1)
        if status == "PRIMARY":
            df = df.drop(columns=["reason"])
            assert df["patient_id"].is_unique
            assert df["cls"].notna().all()
        df.to_parquet(out / "data" / "cohort" / f"{fname}.parquet", index=False)
        df.to_excel(out / "data" / "cohort" / f"{fname}.xlsx", index=False)

    # All-samples set (analysis A2): every analytically valid sample of the
    # SAME patients as the primary cohort, each with its own chart-review
    # class. The index (first) sample is flagged, and the patient's class for
    # stratified splitting is the index sample's class.
    prim = raw["status"] == "PRIMARY"
    patient_cls = raw.loc[prim].set_index("patient_id")["cls"]
    allsel = (raw["patient_id"].isin(patient_cls.index)
              & raw["cls"].notna()
              & (prim | (raw["stage"] == "5 Repeat")))
    df = pd.concat([meta[allsel].drop(columns=["reason"]), feat[allsel], hb[allsel]], axis=1)
    df.insert(2, "is_index_sample", prim[allsel].values)
    df.insert(3, "patient_cls", df["patient_id"].map(patient_cls).values)
    df.insert(4, "n_samples_patient", df.groupby("patient_id")["record_id"].transform("size").values)
    assert df["is_index_sample"].sum() == prim.sum()
    assert set(df["patient_id"]) == set(patient_cls.index)
    df.to_parquet(out / "data" / "cohort" / "cohort_all_samples.parquet", index=False)
    df.to_excel(out / "data" / "cohort" / "cohort_all_samples.xlsx", index=False)
    flow.append(("8", "All-samples set: samples from primary-cohort patients", len(df)))
    flow.append(("8", "All-samples set: patients with >1 sample", int((df.groupby("patient_id").size() > 1).sum())))
    for c, n in df["cls"].value_counts().items():
        flow.append(("8", f"All-samples set – {c} (samples)", int(n)))

    # Audit trail (no names) -----------------------------------------------
    audit = raw[["record_id", "patient_id", "sample_no", "sample_date", "raw_label", "adj_label",
                 "status", "stage", "reason", "cls"]]
    audit.to_excel(out / "reports" / "record_audit.xlsx", index=False)
    al = pd.concat([raw[["record_id", "sample_date", "status"]], align], axis=1)            # D12
    with pd.ExcelWriter(out / "reports" / "wbc_block_from_exports.xlsx") as xw:
        al.to_excel(xw, sheet_name="records", index=False)
        al.groupby(al["sample_date"].dt.to_period("M").astype(str))[["run_found", "runs_disagree", "resolved_from_raw_row", "wbc_changed"]] \
            .sum().to_excel(xw, sheet_name="by_month")
        al.groupby("status")[["run_found", "runs_disagree", "resolved_from_raw_row", "wbc_changed"]].agg(["sum", "size"]).to_excel(xw, sheet_name="by_status")
    pd.DataFrame(flow, columns=["step", "description", "n"]).to_excel(
        out / "reports" / "flow_counts.xlsx", index=False)

    # Consistency checks ---------------------------------------------------
    assert raw["status"].notna().all()
    assert len(audit) == flow[0][2]
    print(pd.DataFrame(flow, columns=["step", "description", "n"]).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Folder that contains Veri/ and R_Kaan_Tez/")
    ap.add_argument("--out", required=True, type=Path, help="Output folder (CDS_v2)")
    a = ap.parse_args()
    build(a.root, a.out)
