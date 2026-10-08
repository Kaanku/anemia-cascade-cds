"""
s05_train.py — Anemia cascade CDS, nested cross-validation (v4)
================================================================

Trains every AutoGluon model of the nested cross-validation and saves its
predictions. Nothing downstream (calibration, thresholds, zones, conformal,
cascade) is done here; s07 and s09 use the saved probabilities.

Jobs (run in this order; a finished job is skipped, so the script can be
re-started after a Colab disconnect):
    main   A1 and A2 × FULL (CBC: S1, S2, MX; CBC_BIO: S1, S2) × 6 fold-sets
    t13    A1 and A2 × T13 CBC (S1, S2, MX) × 6 fold-sets
    lc     learning curve: A1 and A2 × FULL CBC (S1, S2, MX) × 5 outer folds ×
           25/50/75/100 % of the training patients, 120 s per model

A fold-set is outer0 … outer4 (trained on the other four outer folds,
predicts outer fold k) or final (trained on all patients, predicts the temporal
cohort).

Rules (DECISIONS.md T1–T4, N1–N4):
    T1  AutoGluon TabularPredictor 1.5.0, preset best_quality, eval metric
        macro F1, 600 s per model (learning curve: 120 s).
    T2  Bagging folds = the fold-set's patient-level inner folds (`groups`),
        so OOF predictions never see the patient they predict.
    T3  One stacking layer; dynamic stacking off; AutoGluon calibration off.
    T4  Patients of the evaluated outer fold (and the temporal cohort) are never
        passed to fit(); their probabilities are only predicted. Stage 2 and MX
        are predicted for all evaluated rows so the cascade can be evaluated.
    Storage: models are trained on local disk (--work). Only the final models
        are kept, pruned to the models of the best ensemble, and copied to
        <OUT>/models/; outer-fold and learning-curve models are deleted once
        their predictions are saved.

Outputs:
    <OUT>/predictions/<A>_<FS>_<SC>_<ST>/<fold-set>_{oof,eval,temporal}.parquet
    <OUT>/predictions_lc/<A>_<ST>/outer<k>_<frac>_{oof,eval}.parquet
    <OUT>/runs/<job>/model_info.json, leaderboard.csv      (model_info = job finished)
    <OUT>/models/<A>_<FS>_<SC>_<ST>__final/                 final predictors
    <OUT>/reports/training_summary.csv

Usage (Colab):
    python s05_train.py --out /content/drive/MyDrive/CDS_v4 --work /content/work
    python s05_train.py --out ... --phase main        # only one phase
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

AAC = ["IDA", "HA", "HGB_HTZ", "NORMAL"]
POP = {"S1": None, "S2": AAC, "MX": ["IDA", "HGB_HTZ"]}
FOLDSETS = [f"outer{k}" for k in range(5)] + ["final"]
LC_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
FULL = [("FULL", "CBC", st) for st in ("S1", "S2", "MX")] + [("FULL", "CBC_BIO", st) for st in ("S1", "S2")]
T13 = [("T13", "CBC", st) for st in ("S1", "S2", "MX")]
META_TRAIN = ["record_id", "patient_id", "is_index_sample", "cls", "y_s1", "n_bio_measured", "outer_fold", "inner_fold"]
META_EVAL = ["record_id", "patient_id", "is_index_sample", "cls", "y_s1", "n_bio_measured", "outer_fold"]
META_TEMP = ["record_id", "patient_id", "is_index_sample", "cls", "y_s1", "n_bio_measured", "has_analyzer_record"]


def job_list(phase: str) -> list[dict]:
    jobs = []
    if phase in ("main", "all"):
        jobs += [dict(a=a, fs=fs, sc=sc, st=st, fset=f, frac=None, tl=600)
                 for a in ("A1", "A2") for fs, sc, st in FULL for f in FOLDSETS]
    if phase in ("t13", "all"):
        jobs += [dict(a=a, fs=fs, sc=sc, st=st, fset=f, frac=None, tl=600)
                 for a in ("A1", "A2") for fs, sc, st in T13 for f in FOLDSETS]
    if phase in ("lc", "all"):
        jobs += [dict(a=a, fs="FULL", sc="CBC", st=st, fset=f"outer{k}", frac=fr, tl=120)
                 for a in ("A1", "A2") for st in ("S1", "S2", "MX") for k in range(5) for fr in LC_FRACTIONS]
    for j in jobs:
        j["cfg"] = f"{j['a']}_{j['fs']}_{j['sc']}_{j['st']}"
        j["name"] = (f"{j['cfg']}__{j['fset']}" if j["frac"] is None
                     else f"LC_{j['a']}_{j['st']}__{j['fset']}_{j['frac']:.2f}")
    return jobs


def proba(predictor, X: pd.DataFrame) -> pd.DataFrame:
    p = predictor.predict_proba(X).reset_index(drop=True)
    p.columns = [f"prob_{c}" for c in p.columns]
    return p


def run_job(out: Path, work: Path, j: dict, presets: str) -> dict | None:
    from autogluon.tabular import TabularPredictor, __version__ as ag_version

    rdir = out / "runs" / j["name"]
    if (rdir / "model_info.json").exists():
        return None                                                   # finished earlier
    fdir = out / "data" / "features" / f"{j['a']}_{j['fs']}_{j['sc']}" / j["fset"]
    if j["frac"] is None or j["frac"] == 1.0:
        fjson = fdir / f"features_{j['st']}.json"
    else:
        fjson = fdir.parent / "lc" / f"{j['fset'][-1]}_{j['frac']:.2f}" / f"features_{j['st']}.json"
    feats = json.loads(fjson.read_text())
    tr = pd.read_parquet(fdir / "train.parquet")
    if j["frac"] is not None:
        tr = tr[tr["lc_q"] <= j["frac"]]
    if POP[j["st"]] is not None:
        tr = tr[tr["cls"].isin(POP[j["st"]])]
    tr = tr.reset_index(drop=True)
    label = "y_s1" if j["st"] == "S1" else "cls"
    assert tr["inner_fold"].notna().all() and tr["inner_fold"].nunique() == 5
    ev = pd.read_parquet(fdir / "eval.parquet")
    assert not set(tr["patient_id"]) & set(ev["patient_id"]), "patient in training and evaluation rows"

    mdir = work / "models" / j["name"]
    if mdir.exists():
        shutil.rmtree(mdir)                                           # incomplete earlier run
    train_data = tr[feats + [label]].copy()
    train_data["inner_fold"] = tr["inner_fold"].astype(int)
    t0 = time.time()
    predictor = TabularPredictor(label=label, problem_type="multiclass" if j["st"] == "S2" else "binary",
                                 eval_metric="f1_macro", groups="inner_fold", path=str(mdir), verbosity=1)
    predictor.fit(train_data, presets=presets, time_limit=j["tl"], num_stack_levels=0,
                  dynamic_stacking=False, calibrate=False)
    fit_s = time.time() - t0

    if j["frac"] is None:
        pdir = out / "predictions" / j["cfg"]
        stem = j["fset"]
    else:
        pdir = out / "predictions_lc" / f"{j['a']}_{j['st']}"
        stem = f"{j['fset']}_{j['frac']:.2f}"
    pdir.mkdir(parents=True, exist_ok=True)
    oof = predictor.predict_proba_oof().reset_index(drop=True)
    assert len(oof) == len(tr)
    oof.columns = [f"prob_{c}" for c in oof.columns]
    pd.concat([tr[META_TRAIN], oof], axis=1).to_parquet(pdir / f"{stem}_oof.parquet", index=False)
    if len(ev):
        pd.concat([ev[META_EVAL].reset_index(drop=True), proba(predictor, ev[feats])], axis=1).to_parquet(
            pdir / f"{stem}_eval.parquet", index=False)
    tpath = fdir / "temporal.parquet"
    if j["fset"] == "final" and tpath.exists():
        tm = pd.read_parquet(tpath)
        pd.concat([tm[META_TEMP].reset_index(drop=True), proba(predictor, tm[feats])], axis=1).to_parquet(
            pdir / "final_temporal.parquet", index=False)

    lb = predictor.leaderboard(silent=True)
    info = {"job": j["name"], "config": j["cfg"], "fold_set": j["fset"], "fraction": j["frac"],
            "label": label, "classes": list(predictor.class_labels), "features": feats,
            "n_train_rows": len(tr), "n_train_patients": int(tr["patient_id"].nunique()),
            "n_eval_rows": len(ev), "best_model": predictor.model_best, "time_limit_s": j["tl"],
            "fit_seconds": round(fit_s), "presets": presets, "autogluon": ag_version,
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "oof_f1_macro_best": float(lb.loc[lb["model"] == predictor.model_best, "score_val"].iloc[0])}
    rdir.mkdir(parents=True, exist_ok=True)
    lb.to_csv(rdir / "leaderboard.csv", index=False)
    if j["fset"] == "final":                                          # keep the final model, pruned
        predictor.delete_models(models_to_keep="best", dry_run=False)
        predictor.save_space()
        dest = out / "models" / f"{j['cfg']}__final"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(mdir, dest)
        info["saved_model"] = str(dest.relative_to(out))
    shutil.rmtree(mdir, ignore_errors=True)
    (rdir / "model_info.json").write_text(json.dumps(info, indent=1))   # written last = job finished
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="CDS_v4 folder (Drive)")
    ap.add_argument("--work", type=Path, default=Path("/content/work"), help="local scratch for models")
    ap.add_argument("--phase", default="all", choices=["main", "t13", "lc", "all"])
    ap.add_argument("--presets", default="best_quality")
    ap.add_argument("--only", nargs="*", default=None, help="job names (for testing)")
    ap.add_argument("--time-limit", type=int, default=None, help="override every job's limit (testing only)")
    a = ap.parse_args()

    jobs = job_list(a.phase)
    if a.only:
        jobs = [j for j in jobs if j["name"] in set(a.only)]
    if a.time_limit:
        for j in jobs:
            j["tl"] = a.time_limit
    todo = [j for j in jobs if not (a.out / "runs" / j["name"] / "model_info.json").exists()]
    (a.out / "reports").mkdir(exist_ok=True)
    log = a.out / "reports" / "progress.log"

    def note(msg: str) -> None:                       # console + Drive log (followed remotely)
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    note(f"START phase={a.phase}: {len(jobs)} jobs, {len(jobs) - len(todo)} already finished, "
         f"{len(todo)} to run (≈ {sum(j['tl'] for j in todo) / 3600:.1f} h)")
    t_start = time.time()
    for i, j in enumerate(todo, 1):
        try:
            info = run_job(a.out, a.work, j, a.presets)
        except Exception as e:
            note(f"ERROR {j['name']}: {type(e).__name__}: {e}")
            raise
        left = sum(x["tl"] for x in todo[i:]) / 3600
        note(f"[{i}/{len(todo)}] {j['name']}: {info['fit_seconds']} s, OOF macro F1 {info['oof_f1_macro_best']:.3f}"
             f" | elapsed {(time.time() - t_start) / 3600:.1f} h, left ≈ {left:.1f} h")
    note(f"DONE phase={a.phase}")

    rows = []
    for p in sorted((a.out / "runs").glob("*/model_info.json")):
        i = json.loads(p.read_text())
        rows.append({k: i.get(k) for k in ("job", "config", "fold_set", "fraction", "n_train_rows",
                                          "n_train_patients", "best_model", "oof_f1_macro_best", "fit_seconds")})
    (a.out / "reports").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / "reports" / "training_summary.csv", index=False)
    print(f"finished jobs: {len(rows)}; summary saved")


if __name__ == "__main__":
    main()
