"""
s05c_tabpfn.py — TabPFN-3.5 as the single model of the anemia cascade CDS (v4.1)
=================================================================================

After the first evaluation TabPFN was chosen as the single model
(DECISIONS R12): AutoGluon's inner-OOF probabilities did not carry over to its
outer-fold predictions (R11), whereas TabPFN's did. This script refits every
job with the newest open TabPFN release, TabPFN-3.5 (Prior Labs, September
2026; weights under the non-commercial TabPFN-3.5 licence), on exactly the
fold-sets, rows and features of s05/s05b. TabPFN is not tuned: default
settings of TabPFNClassifier.create_default_for_version(V3_5), seed 42.

Jobs (finished jobs are skipped, so the script can be re-started):
    main   A1 and A2 × FULL (CBC: S1, S2, MX; CBC_BIO: S1, S2) × 6 fold-sets
    t13    A1 and A2 × T13 CBC (S1, S2, MX) × 6 fold-sets
    lc     learning curve: A1 and A2 × FULL CBC (S1, S2, MX) × 5 outer folds ×
           25/50/75/100 % of the training patients (the subsets and features of s04)

For every job:
    inner OOF  five fits, each on four inner folds, predicting the fifth
               (the same patient-level inner folds AutoGluon used for bagging)
    eval       one fit on all training rows, predicting the outer fold
    temporal   (final fold-set) the same fit predicting the temporal cohort
Both are single fits of the same kind, so the OOF scores the lock is made on
and the scores of new patients come from the same model type (R11).

Outputs (same layout and columns as s05):
    <OUT>/predictions_tabpfn35/<cfg>/<fold-set>_{oof,eval,temporal}.parquet
    <OUT>/predictions_lc_tabpfn35/<A>_<ST>/outer<k>_<frac>_{oof,eval}.parquet
    <OUT>/runs_tabpfn35/<job>.json                     (written last = finished)
    <OUT>/reports/progress_tabpfn35.log

Sensitivity analysis without a feature (R24): --drop age --tag tabpfn35_noage removes the
named features from every job's feature list (Boruta is not re-run; no ratio contains age)
and writes to predictions_<tag>/, runs_<tag>/ and reports/progress_<tag>.log, so the main
outputs and the lock are untouched.

Usage (Colab, GPU runtime; the TabPFN-3.5 licence must have been accepted at
https://ux.priorlabs.ai and the API key set as TABPFN_TOKEN):
    pip install "tabpfn==9.0.0"
    python s05c_tabpfn.py --out /content/drive/MyDrive/CDS_v4 --data /content/cds_data
    python s05c_tabpfn.py --out ... --data ... --phases main t13 --drop age --tag tabpfn35_noage
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from s05_train import FOLDSETS, FULL, LC_FRACTIONS, META_EVAL, META_TEMP, META_TRAIN, POP, T13

SEED = 42
VERSION = "v3.5"
TAG = "tabpfn35"
LABEL = "TabPFN-3.5"


def job_list(phases: list[str]) -> list[dict]:
    jobs = []
    if "main" in phases:
        jobs += [dict(a=a, fs=fs, sc=sc, st=st, fset=f, frac=None) for a in ("A1", "A2") for fs, sc, st in FULL
                 for f in FOLDSETS]
    if "t13" in phases:
        jobs += [dict(a=a, fs=fs, sc=sc, st=st, fset=f, frac=None) for a in ("A1", "A2") for fs, sc, st in T13
                 for f in FOLDSETS]
    if "lc" in phases:
        jobs += [dict(a=a, fs="FULL", sc="CBC", st=st, fset=f"outer{k}", frac=fr) for a in ("A1", "A2")
                 for st in ("S1", "S2", "MX") for k in range(5) for fr in LC_FRACTIONS]
    for j in jobs:
        j["cfg"] = f"{j['a']}_{j['fs']}_{j['sc']}_{j['st']}"
        j["name"] = (f"{j['cfg']}__{j['fset']}" if j["frac"] is None
                     else f"LC_{j['a']}_{j['st']}__{j['fset']}_{j['frac']:.2f}")
    return jobs


def make_model():
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNClassifier.create_default_for_version(ModelVersion(VERSION), device=device, random_state=SEED), device


def fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, Xte_list: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list]:
    clf, _ = make_model()
    clf.fit(Xtr.to_numpy(dtype=float), ytr)
    out = []
    for X in Xte_list:
        p = clf.predict_proba(X.to_numpy(dtype=float)) if len(X) else np.empty((0, len(clf.classes_)))
        out.append(pd.DataFrame(p, columns=[f"prob_{c}" for c in clf.classes_]))
    return out, list(clf.classes_)


def run(out: Path, data: Path, j: dict, tag: str = TAG, drop: tuple = ()) -> dict | None:
    marker = out / f"runs_{tag}" / f"{j['name']}.json"
    if marker.exists():
        return None
    st, fset, frac = j["st"], j["fset"], j["frac"]
    fdir = data / "features" / f"{j['a']}_{j['fs']}_{j['sc']}" / fset
    if frac is None or frac == 1.0:
        fjson = fdir / f"features_{st}.json"
    else:
        fjson = fdir.parent / "lc" / f"{fset[-1]}_{frac:.2f}" / f"features_{st}.json"
    feats = json.loads(fjson.read_text())
    if drop:
        assert all(d in feats for d in drop), f"{drop} not all in the feature list of {j['name']}"
        assert not any(d in f and f != d for d in drop for f in feats), "a dropped feature enters a ratio"
        feats = [f for f in feats if f not in drop]
    tr = pd.read_parquet(fdir / "train.parquet")
    if frac is not None:
        tr = tr[tr["lc_q"] <= frac]
    if POP[st] is not None:
        tr = tr[tr["cls"].isin(POP[st])]
    tr = tr.reset_index(drop=True)
    assert tr["inner_fold"].notna().all() and tr["inner_fold"].nunique() == 5
    y = (tr["y_s1"] if st == "S1" else tr["cls"]).to_numpy()
    ev = pd.read_parquet(fdir / "eval.parquet")
    assert not set(tr["patient_id"]) & set(ev["patient_id"]), "patient in training and evaluation rows"
    tpath = fdir / "temporal.parquet"
    tm = pd.read_parquet(tpath) if (fset == "final" and frac is None and tpath.exists()) else None
    t0 = time.time()

    oof = None
    for g in sorted(tr["inner_fold"].unique()):                                  # inner OOF
        m = (tr["inner_fold"] == g).to_numpy()
        (p,), _ = fit_predict(tr.loc[~m, feats], y[~m], [tr.loc[m, feats]])
        if oof is None:
            oof = pd.DataFrame(np.nan, index=tr.index, columns=p.columns)
        oof.loc[m, p.columns] = p.to_numpy()
    assert oof.notna().all().all()
    targets = [ev[feats]] + ([tm[feats]] if tm is not None else [])
    preds, classes = fit_predict(tr[feats], y, targets)                             # eval / temporal

    if frac is None:
        pdir, stem = out / f"predictions_{tag}" / j["cfg"], fset
    else:
        pdir, stem = out / f"predictions_lc_{tag}" / f"{j['a']}_{st}", f"{fset}_{frac:.2f}"
    pdir.mkdir(parents=True, exist_ok=True)
    pd.concat([tr[META_TRAIN], oof], axis=1).to_parquet(pdir / f"{stem}_oof.parquet", index=False)
    if len(ev):
        pd.concat([ev[META_EVAL].reset_index(drop=True), preds[0]], axis=1).to_parquet(
            pdir / f"{stem}_eval.parquet", index=False)
    if tm is not None:
        pd.concat([tm[META_TEMP].reset_index(drop=True), preds[1]], axis=1).to_parquet(
            pdir / "final_temporal.parquet", index=False)

    import tabpfn
    import torch
    info = {"job": j["name"], "config": j["cfg"], "fold_set": fset, "fraction": frac,
            "model": f"{LABEL} (TabPFNClassifier.create_default_for_version({VERSION}), default settings)",
            "model_version": VERSION, "tabpfn": getattr(tabpfn, "__version__", "?"), "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "seed": SEED, "classes": [str(c) for c in classes],
            "features": feats, "dropped": list(drop), "tag": tag, "n_train_rows": len(tr), "n_train_patients": int(tr["patient_id"].nunique()),
            "n_eval_rows": len(ev), "seconds": round(time.time() - t0)}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(info, indent=1))                                 # written last
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="CDS_v4 folder (outputs)")
    ap.add_argument("--data", type=Path, default=None, help="folder holding features/ (default: OUT/data)")
    ap.add_argument("--phases", nargs="+", default=["main", "t13", "lc"], choices=["main", "t13", "lc"])
    ap.add_argument("--max-jobs", type=int, default=None, help="testing only")
    ap.add_argument("--drop", nargs="*", default=[], help="features removed from every job (sensitivity analysis)")
    ap.add_argument("--tag", default=TAG, help="output suffix: predictions_<tag>/, runs_<tag>/")
    a = ap.parse_args()
    data = a.data or a.out / "data"
    jobs = job_list(a.phases)
    tag, drop = a.tag, tuple(a.drop)
    assert (tag == TAG) == (not drop), "a run that drops features needs its own --tag"
    todo = [j for j in jobs if not (a.out / f"runs_{tag}" / f"{j['name']}.json").exists()]
    if a.max_jobs:
        todo = todo[:a.max_jobs]
    log = a.out / "reports" / f"progress_{tag}.log"
    log.parent.mkdir(exist_ok=True)

    def note(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {LABEL}{' without ' + '+'.join(drop) if drop else ''} | {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    note(f"START {'+'.join(a.phases)}: {len(jobs)} jobs, {len(jobs) - len(todo)} finished, {len(todo)} to run")
    t_start = time.time()
    for i, j in enumerate(todo, 1):
        try:
            info = run(a.out, data, j, tag, drop)
        except Exception as e:
            note(f"ERROR {j['name']}: {type(e).__name__}: {e}")
            raise
        if info:
            note(f"[{i}/{len(todo)}] {j['name']}: {info['seconds']} s on {info['device']} "
                 f"| elapsed {(time.time() - t_start) / 60:.0f} min")
    done = sum((a.out / f"runs_{tag}" / f"{j['name']}.json").exists() for j in jobs)
    note(f"DONE: {done}/{len(jobs)} jobs finished")


if __name__ == "__main__":
    main()
