"""
s18_app_lock.py — lock file of the demo app (DECISIONS R31, R34)
================================================================

Applies the study's lock rules (s07.lock_one: Stage 1 threshold by macro F1, HIGH cut-off at the 90 %
accuracy point (--high-target: the demo uses 85 %, R34), LOW < 0.35, APS conformal quantiles, display
calibration by cross-fitted ECE) to the
predictions of the app's model and writes them as plain JSON for the app (calibrators as parameters, no
pickles). Two sources:
  --xgen  (R34, used by the app) cross-generation predictions on the real development patients (s16c/s17b):
          each patient predicted by a model trained on synthetic data generated without them. The two halves
          (or the K folds of s16c --folds K) are the calibration folds; calibration is parametric only (Platt /
          temperature), so the lock holds no patient-level map.
  --oof   (R31) out-of-fold predictions on the synthetic cohort (s17); over-confident on real patients.
With --tstr it also summarises how the app's model (fitted on the whole synthetic cohort, s17 TSTR) does on
the temporal cohort (97 patients with an analyzer record) under the locked rules: AUC per stage, Stage 2
accuracy and macro F1, and the whole cascade. The development-patient rows of the summary come from the
cross-generation predictions when --xgen is given (no patient informs the model that predicts them),
otherwise from the TSTR predictions on the development patients.

Usage: python s18_app_lock.py --xgen /path/CDS_v43/app_xgen_k5_fast_n2 --tstr /path/CDS_v43/app_fast_n2/tstr \
           --app /path/CDS_v43/app --version v3.5-fast --n-estimators 2 --high-target 0.85
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import s07_lock as L

CONFIGS = ["A1_FULL_CBC_S1", "A1_FULL_CBC_S2", "A1_FULL_CBC_MX", "A1_FULL_CBC_BIO_S1", "A1_FULL_CBC_BIO_S2"]
S2 = L.S2_CLASSES
POP = {"S1": None, "S2": S2, "MX": [L.MX_POSITIVE, L.MX_NEGATIVE]}


def read_xgen(path: Path, stage: str) -> pd.DataFrame:
    """Cross-generation predictions (s17b) in the s07 OOF layout: the stage population, the halves as folds."""
    d = pd.read_parquet(path)
    if POP[stage] is not None:
        d = d[d["cls"].isin(POP[stage])]
    folds = d["fold"].astype(int) if "fold" in d else d["half"].map({"A": 0, "B": 1}).astype(int)
    d = d.assign(is_index_sample=True, inner_fold=folds)
    assert d["patient_id"].is_unique and d["inner_fold"].nunique() >= 2
    return d.reset_index(drop=True)


def export_cal(cal: dict) -> dict:
    m, model = cal["method"], cal["model"]
    if m == "uncalibrated":
        return {"method": m}
    if m == "platt":
        return {"method": m, "coef": float(model.lr.coef_[0][0]), "intercept": float(model.lr.intercept_[0])}
    if m == "temperature":
        return {"method": m, "T": float(model.T)}
    if m == "isotonic" and cal["binary"]:
        return {"method": m, "x": model.iso.X_thresholds_.tolist(), "y": model.iso.y_thresholds_.tolist()}
    if m == "isotonic":
        return {"method": m, "per_class": [{"x": i.X_thresholds_.tolist(), "y": i.y_thresholds_.tolist()}
                                           for i in model.isos]}
    raise ValueError(m)


def stage1(d: pd.DataFrame, e: dict) -> np.ndarray:
    return d["prob_1"].to_numpy(float) >= e["threshold"]


def tstr_summary(tdir: Path, lock: dict, xdir: Path | None = None) -> pd.DataFrame:
    rows = []
    for which in ("dev", "temporal"):
        if which == "dev" and xdir is not None:
            f = {c: pd.read_parquet(xdir / f"{c}.parquet").set_index("record_id") for c in CONFIGS}
        else:
            f = {c: pd.read_parquet(tdir / f"{c}_{which}.parquet").set_index("record_id") for c in CONFIGS}
        if which == "temporal":
            f = {c: d[d["has_analyzer_record"].astype(bool)] for c, d in f.items()}
        for c, d in f.items():
            st = c[-2:]
            r = {"set": which, "source": "cross-generation" if which == "dev" and xdir is not None else "TSTR",
                 "config": c}
            if st == "S1":
                r["n"] = len(d)
                if d["y_s1"].nunique() == 2:
                    r["auc"] = roc_auc_score(d["y_s1"], d["prob_1"])
                r["sensitivity_aac"] = float(stage1(d, lock[c])[(d["y_s1"] == 1).to_numpy()].mean())
            elif st == "S2":
                d = d[d["cls"].isin(S2)]
                P = d[[f"prob_{k}" for k in S2]].to_numpy(float)
                y = d["cls"].map({k: i for i, k in enumerate(S2)}).to_numpy()
                r.update(n=len(d), auc=float(np.mean([roc_auc_score(y == k, P[:, k]) for k in range(4) if (y == k).any()])),
                         accuracy=float((P.argmax(1) == y).mean()), macro_f1=float(f1_score(y, P.argmax(1), average="macro")))
            else:
                d = d[d["cls"].isin(["IDA", "HGB_HTZ"])]
                r.update(n=len(d), auc=roc_auc_score(d["cls"] == "HGB_HTZ", d["prob_HGB_HTZ"]))
            rows.append(r)
        # the whole cascade with the app's rules, patients with biochemistry measured (as s09 block "cascade")
        c1, c2, b1, b2 = (f[c] for c in ("A1_FULL_CBC_S1", "A1_FULL_CBC_S2", "A1_FULL_CBC_BIO_S1", "A1_FULL_CBC_BIO_S2"))
        ids = c1.index.intersection(b1.index)
        lab = np.array(S2 + ["OAC"])
        Pc = c2.loc[ids, [f"prob_{x}" for x in S2]].to_numpy(float)
        Pb = b2.loc[ids, [f"prob_{x}" for x in S2]].to_numpy(float)
        s1c = c1.loc[ids, "prob_1"].to_numpy() >= lock["A1_FULL_CBC_S1"]["threshold"]
        s1b = b1.loc[ids, "prob_1"].to_numpy() >= lock["A1_FULL_CBC_BIO_S1"]["threshold"]
        hc = lock["A1_FULL_CBC_S2"]["high_cutoff"]                 # None: no cut-off reaches 90 % -> no Tier 1
        tier1 = s1c & (Pc.max(1) >= hc) if hc is not None else np.zeros(len(ids), bool)
        pred = np.where(tier1, lab[Pc.argmax(1)], np.where(s1b, lab[Pb.argmax(1)], "OAC"))
        pred_bio = np.where(s1b, lab[Pb.argmax(1)], "OAC")
        y5 = c1.loc[ids, "cls"].to_numpy()
        rows.append({"set": which, "source": rows[-1]["source"], "config": "cascade", "n": len(ids),
                     "accuracy": float((pred == y5).mean()),
                     "cbc_bio_accuracy": float((pred_bio == y5).mean()), "tier1_share": float(tier1.mean()),
                     "tier1_accuracy": float((pred[tier1] == y5[tier1]).mean()) if tier1.any() else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--xgen", type=Path, help="cross-generation predictions (s17b; R34, the app's lock)")
    src.add_argument("--oof", type=Path, help="synthetic OOF predictions (s17; R31)")
    ap.add_argument("--app", required=True, type=Path)
    ap.add_argument("--tstr", type=Path)
    ap.add_argument("--engine", default="tabpfn-3.5", help="engine that produced the predictions (proxy = interface test)")
    ap.add_argument("--version", default="v3.5", help="TabPFN model version of the app's model (v3.5 or v3.5-fast)")
    ap.add_argument("--n-estimators", type=int, default=None, help="fixed ensemble size of the app's model")
    ap.add_argument("--parametric-only", action="store_true", help="no isotonic calibration (always with --xgen)")
    ap.add_argument("--high-target", type=float, default=None,
                    help="accuracy target of the HIGH cut-off (study rule: 0.90; the demo uses 0.85, R34)")
    a = ap.parse_args()
    if a.high_target is not None:
        assert any(abs(a.high_target - t) < 1e-9 for t in L.TARGETS), f"--high-target must be one of {L.TARGETS}"
        L.HIGH_TARGET = a.high_target
    if a.xgen is not None or a.parametric_only:            # no patient-level map in a published lock
        L.METHODS = {b: {k: v for k, v in m.items() if k != "isotonic"} for b, m in L.METHODS.items()}
    lock, report = {"_meta": {"engine": a.engine, "model_version": a.version, "n_estimators": a.n_estimators,
                              "rules": "s07.lock_one (study lock rules)" + (
                                  "" if abs(L.HIGH_TARGET - 0.90) < 1e-9
                                  else f"; HIGH cut-off at the {L.HIGH_TARGET:.0%} accuracy point (study: 90 %)"),
                              "high_target": L.HIGH_TARGET,
                              "locked_on": ("cross-generation predictions on the real development patients "
                                            "(s16c/s17b; R34)") if a.xgen is not None
                              else "out-of-fold predictions on the synthetic cohort (s17; R31)",
                              "calibration_methods": {("binary" if b else "multiclass"): sorted(m)
                                                      for b, m in L.METHODS.items()},
                              "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}}, {}
    for c in CONFIGS:
        if a.xgen is not None:
            f = a.xgen / f"{c}.parquet"
            entry, cal, _ = L.lock_one(read_xgen(f, c[-2:]), c[-2:])
        else:
            f = a.oof / f"{c}_oof.parquet"
            entry, cal, _ = L.lock_one(L.read_oof(f), c[-2:])
        entry["calibration"] = export_cal(cal)
        entry["predictions_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        lock[c] = entry
        report[c] = {k: entry.get(k) for k in ("threshold", "high_cutoff", "conformal_qhat", "oof_accuracy_all",
                                               "oof_at_threshold", "cv_ECE")}
        report[c]["calibration"] = entry["calibration"]["method"]
    (a.app / "data").mkdir(parents=True, exist_ok=True)
    (a.app / "data" / "lock.json").write_text(json.dumps(lock, indent=1))
    print(json.dumps(report, indent=1))
    if a.tstr:
        t = tstr_summary(a.tstr, {k: v for k, v in lock.items() if k != "_meta"}, a.xgen)
        t.to_csv(a.app / "data" / "tstr_summary.csv", index=False)
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()
