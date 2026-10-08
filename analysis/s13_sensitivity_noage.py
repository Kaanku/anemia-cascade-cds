"""
s13_sensitivity_noage.py — sensitivity analysis: TabPFN-3.5 without age (CDS v4.2, R24)
=========================================================================================

Age was the strongest single feature in the SHAP analysis (R23), and the classes
differ in age. This script asks how much of the main model's performance rests on
age: the main TabPFN-3.5 model (lock v2) is compared, patient by patient, with the
same model refitted without age (s05c --drop age; same folds, rows and other
features, Boruta not re-run).

The no-age predictions are locked and evaluated exactly like the main ones, in a
separate working folder (SENS) whose predictions_tabpfn35/ is the no-age run and
whose lock/ is its own (s07, then s09 on SENS). The main folder, its lock and its
unlock log are only read here.

Per configuration (16) and set (nested CV pooled outer folds; temporal cohort,
final models), first sample per patient:
    S1, MX  AUC with age, AUC without age, difference with a paired DeLong p and a
            patient bootstrap 95 % CI (B = 2000, seed 42); sensitivity / specificity /
            accuracy at each model's own locked threshold; McNemar on correctness
    S2      macro AUC (and per class) with the bootstrap CI of the difference; accuracy
            (argmax) with McNemar; share and accuracy of the HIGH zone (own locked cut-off)
Temporal: FULL configurations on patients with an analyzer record (N8); Stage 1 AUC
is not defined there (no OAC), sensitivity is reported.
Holm adjustment over the nested-CV DeLong tests (S1 and MX: 6 + 4 = 10 tests).

Outputs: MAIN/reports/sensitivity/noage_comparison.{csv,parquet,xlsx}

Usage:
    python s13_sensitivity_noage.py --main /path/CDS_v4 --sens /path/CDS_v4_noage [--boot 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import s07_lock as L
import s09_evaluate as V

ENGINE = "tabpfn"
AAC = V.AAC


def load(out: Path, need_log: bool) -> V.Data:
    lock = json.loads((out / "lock" / "lock.json").read_text())
    assert not lock["missing"], f"{out}: lock not final"
    for rel, h in lock["files_read_sha256"].items():
        assert L.sha256(out / rel) == h, f"{out}: OOF file changed after locking: {rel}"
    if need_log:
        assert (out / "lock" / "unlock_log.json").exists(), f"{out}: run s09 on this folder first"
    return V.Data(out, lock, joblib.load(out / "lock" / "calibrators.joblib"))


def boot_diff(fn, a: dict, b: dict, n: int, B: int) -> tuple[float, float, float]:
    """Point estimate and percentile CI of fn(a) - fn(b), resampling patients (rows) jointly."""
    point = fn(**a) - fn(**b)
    reps = []
    for idx in V.Boot(np.arange(n), B):
        reps.append(fn(**{k: v[idx] for k, v in a.items()}) - fn(**{k: v[idx] for k, v in b.items()}))
    reps = np.asarray(reps, float)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5]) if np.isfinite(reps).any() else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def paired(da: pd.DataFrame, db: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ka, kb = da.set_index("record_id"), db.set_index("record_id")
    assert set(ka.index) == set(kb.index), "the two runs predict different patients"
    ids = ka.index.sort_values()
    a, b = ka.loc[ids], kb.loc[ids]
    assert (a["cls"] == b["cls"]).all()
    return a, b


def compare(da, db, st: str, B: int) -> dict:
    a, b = paired(da, db)
    row = {}
    if st in ("S1", "MX"):
        if st == "MX":
            m = a["cls"].isin(["IDA", "HGB_HTZ"]).to_numpy()
            a, b = a[m], b[m]
            y = (a["cls"] == L.MX_POSITIVE).to_numpy(int)
        else:
            y = a["y_s1"].to_numpy(int)
        row["n"] = len(y)
        auc_a, auc_b, diff, p = V.delong_test(y, a["s"], b["s"])
        row.update(auc_with=auc_a, auc_without=auc_b, p_delong=p)
        if np.isfinite(diff):
            d, lo, hi = boot_diff(lambda y, s: V.auc_binary(y, s), dict(y=y, s=a["s"].to_numpy()),
                                  dict(y=y, s=b["s"].to_numpy()), len(y), B)
            row.update(auc_diff=d, auc_diff_low=lo, auc_diff_high=hi)
        for tag, x in (("with", a), ("without", b)):
            bm = V.binary_metrics(y, x["s"].to_numpy(), x["thr"].to_numpy())
            for k in ("sensitivity", "specificity", "accuracy"):
                row[f"{k}_{tag}"] = bm[k]
        ca = (a["s"].to_numpy() >= a["thr"].to_numpy()).astype(int) == y
        cb = (b["s"].to_numpy() >= b["thr"].to_numpy()).astype(int) == y
    else:
        m = a["cls"].isin(AAC).to_numpy()
        a, b = a[m], b[m]
        y = V.y2(a)
        row["n"] = len(y)
        Pa, Pb = V.P2(a), V.P2(b)
        ma, mb = V.multiclass_metrics(y, Pa), V.multiclass_metrics(y, Pb)
        row.update(auc_with=ma["macro_auc"], auc_without=mb["macro_auc"])
        d, lo, hi = boot_diff(lambda y, P: V.multiclass_metrics(y, P)["macro_auc"], dict(y=y, P=Pa), dict(y=y, P=Pb),
                              len(y), B)
        row.update(auc_diff=d, auc_diff_low=lo, auc_diff_high=hi)
        for c in V.S2:
            row[f"auc_{c}_with"], row[f"auc_{c}_without"] = ma[f"auc_{c}"], mb[f"auc_{c}"]
        row["accuracy_with"], row["accuracy_without"] = ma["accuracy"], mb["accuracy"]
        row["balanced_accuracy_with"], row["balanced_accuracy_without"] = ma["balanced_accuracy"], mb["balanced_accuracy"]
        ca, cb = Pa.argmax(1) == y, Pb.argmax(1) == y
        for tag, x, P, hit in (("with", a, Pa, ca), ("without", b, Pb, cb)):
            hi_ = P.max(1) >= x["high"].to_numpy()
            row[f"high_share_{tag}"] = float(hi_.mean())
            row[f"high_accuracy_{tag}"] = float(hit[hi_].mean()) if hi_.any() else np.nan
    row["only_with_correct"], row["only_without_correct"], row["p_mcnemar"] = V.mcnemar(ca, cb)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, type=Path)
    ap.add_argument("--sens", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=V.B_DEFAULT)
    a = ap.parse_args()
    DA, DB = load(a.main, True), load(a.sens, True)
    first = lambda d: d[d["is_index_sample"].astype(bool)]
    rows = []
    for cfg in DA.lock["engines"][ENGINE]:
        assert cfg in DB.lock["engines"][ENGINE], cfg
        st, an, fs = cfg[-2:], cfg.split("_")[0], cfg.split("_")[1]
        sc = "CBC_BIO" if "_CBC_BIO_" in cfg else "CBC"
        for set_name in ("nested CV", "temporal"):
            if set_name == "nested CV":
                da, db = DA.pooled(ENGINE, cfg), DB.pooled(ENGINE, cfg)
            else:
                da, db = DA.temporal(ENGINE, cfg), DB.temporal(ENGINE, cfg)
                if da is None:
                    continue
                if fs == "FULL":
                    da, db = (x[x["has_analyzer_record"].astype(bool)] for x in (da, db))
            r = compare(first(da), first(db), st, a.boot)
            rows.append({"set": set_name, "config": cfg, "analysis": an, "feature_set": fs, "scenario": sc,
                         "stage": st, **r})
            print(f"{set_name:9s} {cfg:22s} n={r['n']:4d}  AUC {r['auc_with']:.3f} -> {r['auc_without']:.3f}"
                  f"  diff {r.get('auc_diff', np.nan):+.3f} ({r.get('auc_diff_low', np.nan):+.3f} to "
                  f"{r.get('auc_diff_high', np.nan):+.3f})  p_DeLong {r.get('p_delong', np.nan):.3g}", flush=True)
    T = pd.DataFrame(rows)
    cv = (T["set"] == "nested CV") & T["stage"].isin(["S1", "MX"])
    T["p_delong_holm"] = np.nan
    T.loc[cv, "p_delong_holm"] = V.holm(T.loc[cv, "p_delong"].to_numpy())
    T["sens_lock_sha256"] = L.sha256(a.sens / "lock" / "lock.json")
    T["main_lock_sha256"] = L.sha256(a.main / "lock" / "lock.json")
    od = a.main / "reports" / "sensitivity"
    od.mkdir(parents=True, exist_ok=True)
    T.to_csv(od / "noage_comparison.csv", index=False)
    T.to_parquet(od / "noage_comparison.parquet", index=False)
    T.to_excel(od / "noage_comparison.xlsx", index=False)
    print(f"written: {od}/noage_comparison.* ({len(T)} rows)")


if __name__ == "__main__":
    main()
