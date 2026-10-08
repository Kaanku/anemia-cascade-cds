"""
s08_indices.py — Anemia cascade CDS, clean rebuild (v2)
=======================================================

Computes the seven classic discrimination indices for IDA vs HGB HTZ (DECISIONS
X3) for every development sample and every temporal sample. No label is read
and nothing is fitted: the indices use fixed literature cut-offs and are only
compared with the MX model in s09.

MCH and RDW-CV are not model inputs, so they are taken here from the analyzer
record (X3):
    development  <ROOT>/Veri/Tümü.xlsx, row = record_id (as in s01); Hb, RBC and
                 MCV of the same row must equal the cohort values (linkage check)
    temporal     Sysmex exports, last run of the sample (as in s06). For samples
                 without an analyzer record (entered values, V4) MCH is derived as
                 Hb / RBC × 10 (the analyzer's own definition) and RDW-CV is
                 missing, so Green & King and the RDW index are not available for
                 them.
Hb, RBC and MCV always come from the cohort files (s01 / s06), so documented
corrections (V5) are carried over.

Outputs:
    <OUT>/data/indices/indices_dev.parquet        record_id, mch_pg, rdw_cv_pct, idx_*
    <OUT>/data/indices/indices_temporal.parquet
    <OUT>/reports/indices_report.xlsx             definitions, sources, missing values

Usage:
    python s08_indices.py --root "/path/to/Kaan" --out "/path/to/Kaan/CDS_v2"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILE = Path("Veri") / "Tümü.xlsx"

# name -> (label, formula, cut-off, function); index below the cut-off = HGB HTZ (X3)
INDICES = {
    "mentzer":        ("Mentzer", "MCV / RBC", 13.0,
                       lambda d: d["mcv_f_l"] / d["rbc_10_6_u_l"]),
    "green_king":     ("Green & King", "MCV² × RDW-CV / (100 × Hb)", 65.0,
                       lambda d: d["mcv_f_l"] ** 2 * d["rdw_cv_pct"] / (100 * d["hgb_g_d_l"])),
    "england_fraser": ("England & Fraser", "MCV − RBC − 5 × Hb − 3.4", 0.0,
                       lambda d: d["mcv_f_l"] - d["rbc_10_6_u_l"] - 5 * d["hgb_g_d_l"] - 3.4),
    "rdwi":           ("RDW index", "MCV × RDW-CV / RBC", 220.0,
                       lambda d: d["mcv_f_l"] * d["rdw_cv_pct"] / d["rbc_10_6_u_l"]),
    "shine_lal":      ("Shine & Lal", "MCV² × MCH / 100", 1530.0,
                       lambda d: d["mcv_f_l"] ** 2 * d["mch_pg"] / 100),
    "srivastava":     ("Srivastava", "MCH / RBC", 3.8,
                       lambda d: d["mch_pg"] / d["rbc_10_6_u_l"]),
    "ehsani":         ("Ehsani", "MCV − 10 × RBC", 15.0,
                       lambda d: d["mcv_f_l"] - 10 * d["rbc_10_6_u_l"]),
}
KEY = ["hgb_g_d_l", "rbc_10_6_u_l", "mcv_f_l"]


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def add_indices(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for k, (_, _, _, fn) in INDICES.items():
        d[f"idx_{k}"] = fn(d)
    return d


def development(root: Path, out: Path) -> tuple[pd.DataFrame, dict]:
    src = pd.read_excel(root / RAW_FILE).reset_index(drop=True)
    raw = pd.DataFrame({"record_id": np.arange(1, len(src) + 1),
                        "mch_raw": to_num(src["MCH(pg)"]), "rdw_cv_pct": to_num(src["RDW-CV(%)"]),
                        "hgb_raw": to_num(src["HGB(g/dL)"]), "rbc_raw": to_num(src["RBC(10^6/uL)"]),
                        "mcv_raw": to_num(src["MCV(fL)"])})
    coh = pd.read_parquet(out / "data" / "model" / "A2.parquet")[["record_id"] + KEY]
    d = coh.merge(raw, on="record_id", how="left", validate="one_to_one")
    for c, r in zip(KEY, ["hgb_raw", "rbc_raw", "mcv_raw"]):                  # linkage check
        assert np.allclose(d[c], d[r], rtol=0, atol=1e-9), f"record_id linkage mismatch in {c}"
    derived = d["hgb_g_d_l"] / d["rbc_10_6_u_l"] * 10
    d["mch_source"] = np.where(d["mch_raw"].notna(), "analyzer", "derived Hb/RBC×10")
    d["mch_pg"] = d["mch_raw"].fillna(derived)
    info = {"n": len(d), "mch_missing_in_record": int(d["mch_raw"].isna().sum()),
            "rdw_cv_missing": int(d["rdw_cv_pct"].isna().sum()),
            "max_abs_diff_mch_vs_hb_rbc": round(float((d["mch_raw"] - derived).abs().max()), 3)}
    return add_indices(d[["record_id", "mch_pg", "mch_source", "rdw_cv_pct"] + KEY]), info


def temporal(root: Path, out: Path) -> tuple[pd.DataFrame, dict]:
    from s06_temporal import ANALYZER, load_exports
    t = pd.read_parquet(out / "data" / "temporal" / "temporal_cohort.parquet")
    an = load_exports(root)
    rows = []
    for _, r in t.iterrows():
        rec = {"record_id": r["record_id"], **{c: r[c] for c in KEY}}
        if r["cbc_source"] == "analyzer export":
            a = an.loc[r["sample_no"]]
            for src_col, dst in [("HGB(g/dL)", "hgb_g_d_l"), ("RBC(10^6/uL)", "rbc_10_6_u_l"), ("MCV(fL)", "mcv_f_l")]:
                assert ANALYZER[src_col] == dst
                assert np.isclose(float(to_num(pd.Series([a[src_col]]))[0]), r[dst]), "temporal linkage mismatch"
            rec["mch_pg"] = float(to_num(pd.Series([a["MCH(pg)"]]))[0])
            rec["rdw_cv_pct"] = float(to_num(pd.Series([a["RDW-CV(%)"]]))[0])
            rec["mch_source"] = "analyzer"
        else:
            rec["mch_pg"] = r["hgb_g_d_l"] / r["rbc_10_6_u_l"] * 10
            rec["rdw_cv_pct"] = np.nan
            rec["mch_source"] = "derived Hb/RBC×10 (no analyzer record)"
        rows.append(rec)
    d = pd.DataFrame(rows)
    info = {"n": len(d), "mch_derived": int((d["mch_source"] != "analyzer").sum()),
            "rdw_cv_missing": int(d["rdw_cv_pct"].isna().sum())}
    return add_indices(d[["record_id", "mch_pg", "mch_source", "rdw_cv_pct"] + KEY]), info


def build(root: Path, out: Path) -> None:
    idir = out / "data" / "indices"
    idir.mkdir(parents=True, exist_ok=True)
    dev, dinfo = development(root, out)
    tem, tinfo = temporal(root, out)
    dev.to_parquet(idir / "indices_dev.parquet", index=False)
    tem.to_parquet(idir / "indices_temporal.parquet", index=False)

    defs = pd.DataFrame([{"index": k, "name": v[0], "formula": v[1], "cutoff": v[2],
                          "rule": f"{v[0]} < {v[2]:g} → HGB HTZ"} for k, v in INDICES.items()])
    miss = pd.DataFrame([{"set": s, "index": k, "n_missing": int(df[f"idx_{k}"].isna().sum()), "n": len(df)}
                         for s, df in [("development", dev), ("temporal", tem)] for k in INDICES])
    inputs = {str(RAW_FILE): sha256(root / RAW_FILE),
              "A2.parquet": sha256(out / "data" / "model" / "A2.parquet"),
              "temporal_cohort.parquet": sha256(out / "data" / "temporal" / "temporal_cohort.parquet")}
    with pd.ExcelWriter(out / "reports" / "indices_report.xlsx") as xw:
        defs.to_excel(xw, sheet_name="definitions", index=False)
        miss.to_excel(xw, sheet_name="missing", index=False)
        pd.DataFrame([{"set": "development", **dinfo}, {"set": "temporal", **tinfo}]).to_excel(
            xw, sheet_name="sources", index=False)
        pd.DataFrame([{"file": k, "sha256": v} for k, v in inputs.items()]).to_excel(
            xw, sheet_name="inputs", index=False)
    print(defs[["name", "rule"]].to_string(index=False))
    print(json.dumps({"development": dinfo, "temporal": tinfo}, indent=1))
    print(miss[miss["n_missing"] > 0].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    build(a.root, a.out)
