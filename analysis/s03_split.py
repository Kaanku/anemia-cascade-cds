"""
s03_split.py — Anemia cascade CDS, nested cross-validation (v4)
================================================================

Assigns every patient of the cohort to the folds of a nested cross-validation.
There is no separate test split (DECISIONS N1): all 863 patients are used, and
performance is estimated on the outer folds. The unit is the PATIENT, so both
analyses share exactly the same partitions:

    A1  one (first) sample per patient   (analysis_primary)
    A2  all valid samples per patient    (analysis_all_samples)

Rules:
    N1  Outer loop: 5 patient-level folds, stratified by class x (has repeat
        samples), seed 42. Each outer fold is predicted by models that never
        saw its patients.
    N2  Inner loop: within the training patients of each outer fold, 5
        patient-level folds (same stratification, seed 42). AutoGluon receives
        them as the `groups` column (patient-grouped bagging); their OOF
        predictions are the only data used to fix thresholds, zones,
        conformal quantiles and calibration for that outer fold.
    N3  Final models are trained on all patients; their bagging folds are the
        outer folds. The temporal cohort is the only data they are evaluated on.
    N4  Learning curve: within each outer fold, nested random subsets of the
        training patients (25 % ⊂ 50 % ⊂ 75 % ⊂ 100 %), stratified by the same
        strata, seed 42; the subset keeps its inner-fold ids.

Inputs:
    <OUT>/data/analysis/analysis_primary.parquet
    <OUT>/data/analysis/analysis_all_samples.parquet

Outputs:
    <OUT>/data/split/patient_folds.csv           patient_id -> outer_fold, inner_fold_k, lc_rank
    <OUT>/data/model/A1.parquet / A2.parquet     analysis table + fold columns
    <OUT>/reports/split_report.xlsx

Usage:
    python s03_split.py --out "/path/to/Kaan/CDS_v4"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_OUTER = N_INNER = 5
CLASSES = ["IDA", "HA", "HGB_HTZ", "NORMAL", "OAC"]
MODEL_BIO = ["iron", "uibc", "ferritin", "ldh"]
LC_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def folds(ids: pd.Series, strata: pd.Series, n: int) -> pd.Series:
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED)
    out = pd.Series(-1, index=ids.index, dtype=int)
    for k, (_, idx) in enumerate(skf.split(ids, strata)):
        out.iloc[idx] = k
    return out


def build(out: Path) -> None:
    ana = out / "data" / "analysis"
    for d in ("split", "model"):
        (out / "data" / d).mkdir(parents=True, exist_ok=True)
    prim = pd.read_parquet(ana / "analysis_primary.parquet")
    alls = pd.read_parquet(ana / "analysis_all_samples.parquet")
    assert prim["patient_id"].is_unique and set(alls["patient_id"]) == set(prim["patient_id"])
    assert prim["cls"].isin(CLASSES).all() and alls["cls"].isin(CLASSES).all()

    pt = prim[["patient_id", "cls", "n_samples_patient"]].sort_values("patient_id").reset_index(drop=True)
    pt["has_repeats"] = pt["n_samples_patient"] > 1
    pt["stratum"] = pt["cls"] + "|" + np.where(pt["has_repeats"], "repeat", "single")
    small = pt["stratum"].value_counts()
    assert (small >= N_OUTER * 2).all(), f"Strata too small for stratification:\n{small}"

    pt["outer_fold"] = folds(pt["patient_id"], pt["stratum"], N_OUTER)                  # N1
    for k in range(N_OUTER):                                                             # N2
        col = f"inner_fold_{k}"
        pt[col] = pd.array([pd.NA] * len(pt), dtype="Int64")
        tr = pt["outer_fold"] != k
        pt.loc[tr, col] = folds(pt.loc[tr, "patient_id"], pt.loc[tr, "stratum"], N_INNER).values
    # N4: one random rank per patient within its stratum; a subset of fraction f of the
    # training patients of outer fold k = the lowest ceil(f * n) ranks of each stratum.
    rng = np.random.default_rng(SEED)
    pt["lc_u"] = rng.uniform(size=len(pt))
    for k in range(N_OUTER):
        tr = pt[pt["outer_fold"] != k]
        r = tr.groupby("stratum")["lc_u"].rank(method="first") / tr.groupby("stratum")["lc_u"].transform("size")
        pt.loc[tr.index, f"lc_q_{k}"] = r.values          # in (0, 1]; subset f = lc_q <= f
    pt = pt.drop(columns="lc_u")

    fold_cols = ["outer_fold"] + [f"inner_fold_{k}" for k in range(N_OUTER)] + [f"lc_q_{k}" for k in range(N_OUTER)]
    pt[["patient_id", "cls", "has_repeats", "stratum"] + fold_cols].to_csv(
        out / "data" / "split" / "patient_folds.csv", index=False)

    for name, df in [("A1", prim), ("A2", alls)]:
        m = df.merge(pt[["patient_id"] + fold_cols], on="patient_id", how="left", validate="many_to_one")
        assert m["outer_fold"].notna().all()
        m.to_parquet(out / "data" / "model" / f"{name}.parquet", index=False)

    # Checks ------------------------------------------------------------------
    for name in ["A1", "A2"]:
        df = pd.read_parquet(out / "data" / "model" / f"{name}.parquet")
        assert (df.groupby("patient_id")["outer_fold"].nunique() == 1).all()
        for k in range(N_OUTER):
            tr = df[df["outer_fold"] != k]
            assert tr[f"inner_fold_{k}"].notna().all() and df.loc[df["outer_fold"] == k, f"inner_fold_{k}"].isna().all()
            assert (pd.crosstab(tr[f"inner_fold_{k}"], tr["cls"]) > 0).all().all(), f"{name}: inner fold lacks a class"
            for f in LC_FRACTIONS:
                sub = tr[tr[f"lc_q_{k}"] <= f]
                assert (pd.crosstab(sub[f"inner_fold_{k}"], sub["cls"]) > 0).all().all(), \
                    f"{name}: learning-curve subset {f} of outer fold {k} has an inner fold without a class"
        assert (pd.crosstab(df["outer_fold"], df["cls"]) > 0).all().all()
    lc_nest = all((pt[f"lc_q_{k}"] <= a).le(pt[f"lc_q_{k}"] <= b).all()
                  for k in range(N_OUTER) for a, b in zip(LC_FRACTIONS, LC_FRACTIONS[1:]))
    assert lc_nest

    # Report --------------------------------------------------------------
    a1 = pd.read_parquet(out / "data" / "model" / "A1.parquet")
    a2 = pd.read_parquet(out / "data" / "model" / "A2.parquet")
    lc = pd.DataFrame([{"outer_fold": k, "fraction": f,
                        "patients": int(((pt["outer_fold"] != k) & (pt[f"lc_q_{k}"] <= f)).sum())}
                       for k in range(N_OUTER) for f in LC_FRACTIONS])
    with pd.ExcelWriter(out / "reports" / "split_report.xlsx") as xw:
        pd.crosstab(a1["cls"], a1["outer_fold"], margins=True).to_excel(xw, sheet_name="A1_patients_by_fold")
        pd.crosstab(a2["cls"], a2["outer_fold"], margins=True).to_excel(xw, sheet_name="A2_samples_by_fold")
        pd.crosstab(pt["stratum"], pt["outer_fold"], margins=True).to_excel(xw, sheet_name="strata_by_fold")
        a1.groupby("outer_fold")[MODEL_BIO].apply(lambda x: (x.isna().mean() * 100).round(1)).to_excel(
            xw, sheet_name="A1_bio_missing_by_fold")
        lc.to_excel(xw, sheet_name="learning_curve_sizes", index=False)
    print("A1 patients by class and outer fold\n", pd.crosstab(a1["cls"], a1["outer_fold"], margins=True).to_string())
    print("\nA2 samples by class and outer fold\n", pd.crosstab(a2["cls"], a2["outer_fold"], margins=True).to_string())
    print("\nLearning-curve training sizes (patients), outer fold 0:",
          lc[lc.outer_fold == 0].set_index("fraction")["patients"].to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="CDS_v4 folder")
    build(ap.parse_args().out)
