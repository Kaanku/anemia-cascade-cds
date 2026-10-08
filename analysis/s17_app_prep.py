"""
s17_app_prep.py — TabPFN-3.5 on the synthetic cohort for the demo app (Colab; DECISIONS R28, R30, R31)
======================================================================================================

The demo app fits TabPFN-3.5 on the synthetic cohort of s16 (never on patients). This script produces
what the app needs and checks it on the real patients:

  1. OOF   5-fold stratified CV (by class, seed 42) on each synthetic matrix with the final feature lists of
           the real pipeline: A1 FULL CBC S1, S2, MX and A1 FULL CBC_BIO S1, S2. Written in the s07 layout
           (record_id, patient_id, is_index_sample, inner_fold, cls, y_s1, prob_*), so the app's thresholds,
           HIGH cut-offs, conformal quantiles and calibration are locked with s07.lock_one, as for the paper.
  2. TSTR  TabPFN-3.5 fitted on the whole synthetic matrix, applied to every real development patient (A1 final
           matrix) and every temporal patient (all stages for everyone, as the app does). For CBC_BIO the real rows' imputed analytes are set back to
           missing and re-imputed with the app's imputer (fitted on the synthetic rows), as the app would.
  3. CPU   wall time on CPU with 2 threads (a small host): model fit and one-patient prediction per
           configuration, default fit mode and fit_with_cache; resident memory.
Model: TabPFNClassifier.create_default_for_version(v3.5), default settings, seed 42 (as s05c).

Outputs: <OUT>/app/oof/<cfg>_oof.parquet, <OUT>/app/tstr/<cfg>_{dev,temporal}.parquet,
         <OUT>/app/app_prep_info.json, <OUT>/reports/progress_app.log   (pack: pack_results.py --set app)
         With --version v3.5-fast and/or --n-estimators k (a lighter model for a small CPU host):
         <OUT>/app[_fast][_n<k>]/..., progress_app[_fast][_n<k>].log
Usage (Colab, GPU runtime, TABPFN_TOKEN set; s04 needs boruta importable):
    python s17_app_prep.py --out /content/drive/MyDrive/CDS_v43
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from s04_features import BIO, CBC_BASE, EXTRA, LabelFreeKNNImputer, add_ratios, apply_imputer, ratio_pairs
from s05_train import POP
from s05c_tabpfn import SEED, VERSION
from s05c_tabpfn import fit_predict as _fit_predict_auto

N_ESTIMATORS = None          # None = the study setting ("auto"); set by --n-estimators for a lighter app
MODEL_VERSION = VERSION      # "v3.5" (the study model) or "v3.5-fast" (--version, small CPU host, R34)


def fit_predict(Xtr, ytr, Xte_list):
    """s05c.fit_predict, optionally with another model version or a fixed ensemble size (app on a small host)."""
    if N_ESTIMATORS is None and MODEL_VERSION == VERSION:
        return _fit_predict_auto(Xtr, ytr, Xte_list)
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {} if N_ESTIMATORS is None else {"n_estimators": N_ESTIMATORS}
    clf = TabPFNClassifier.create_default_for_version(ModelVersion(MODEL_VERSION), device=device, random_state=SEED,
                                                      **kw)
    clf.fit(Xtr.to_numpy(dtype=float), ytr)
    out = [pd.DataFrame(clf.predict_proba(X.to_numpy(dtype=float)) if len(X) else np.empty((0, len(clf.classes_))),
                        columns=[f"prob_{c}" for c in clf.classes_]) for X in Xte_list]
    return out, list(clf.classes_)

CONFIGS = [("CBC", "S1"), ("CBC", "S2"), ("CBC", "MX"), ("CBC_BIO", "S1"), ("CBC_BIO", "S2")]
META = ["record_id", "patient_id", "is_index_sample", "cls", "y_s1", "n_bio_measured"]


def stage(d: pd.DataFrame, st: str) -> pd.DataFrame:
    return (d if POP[st] is None else d[d["cls"].isin(POP[st])]).reset_index(drop=True)


def target(d: pd.DataFrame, st: str) -> np.ndarray:
    return (d["y_s1"] if st == "S1" else d["cls"]).to_numpy()


def app_imputer(out: Path) -> dict:
    """The app's KNN imputer: fitted on the synthetic rows with at least one analyte (as s16 / s04)."""
    syn = pd.read_parquet(out / "data" / "synthetic" / "synthetic_cohort.parquet")
    return LabelFreeKNNImputer(k=5).fit(syn[syn[BIO].notna().any(axis=1)]).state()


def real_bio(path: Path, imp: dict) -> pd.DataFrame:
    tr = pd.read_parquet(path)
    keep = [c for c in META + ["has_analyzer_record"] if c in tr.columns]
    base = tr[keep + CBC_BASE + EXTRA + BIO].copy()
    for c in BIO:
        if f"imputed_{c}" in tr:
            base.loc[tr[f"imputed_{c}"].astype(bool).to_numpy(), c] = np.nan
    return add_ratios(apply_imputer(imp, base), ratio_pairs("CBC_BIO"))[0]


def cpu_timing(X: pd.DataFrame, y: np.ndarray) -> dict:
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(2)
    res = {}
    for mode in ("default", "fit_with_cache"):
        kw = {} if mode == "default" else {"fit_mode": "fit_with_cache"}
        try:
            t0 = time.time()
            if N_ESTIMATORS is not None:
                kw["n_estimators"] = N_ESTIMATORS
            clf = TabPFNClassifier.create_default_for_version(ModelVersion(MODEL_VERSION), device="cpu",
                                                              random_state=SEED, **kw)
            clf.fit(X.to_numpy(float), y)
            t1 = time.time()
            clf.predict_proba(X.iloc[:1].to_numpy(float))
            t2 = time.time()
            clf.predict_proba(X.iloc[1:2].to_numpy(float))
            t3 = time.time()
            res[mode] = {"fit_s": round(t1 - t0, 1), "first_predict_s": round(t2 - t1, 1),
                         "second_predict_s": round(t3 - t2, 1)}
        except Exception as e:                                      # fit_mode may not exist in this release
            res[mode] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=None, help="fixed ensemble size (default: the study's auto)")
    ap.add_argument("--version", default=VERSION, help="TabPFN model version (v3.5 = the study model, or v3.5-fast)")
    a = ap.parse_args()
    global N_ESTIMATORS, MODEL_VERSION
    N_ESTIMATORS, MODEL_VERSION = a.n_estimators, a.version
    sub = ("" if a.version == VERSION else "_fast") + ("" if a.n_estimators is None else f"_n{a.n_estimators}")
    fdir, sdir, adir = a.out / "data" / "features", a.out / "data" / "synthetic", a.out / f"app{sub}"
    (adir / "oof").mkdir(parents=True, exist_ok=True)
    (adir / "tstr").mkdir(parents=True, exist_ok=True)
    log = a.out / "reports" / f"progress_app{sub}.log"

    def note(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | app prep | {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    imp = app_imputer(a.out)
    info = {"configs": {}, "seed": SEED, "model": f"TabPFN-3.5 (create_default_for_version({a.version}))",
            "n_estimators": a.n_estimators or "auto", "version": a.version}
    t_all = time.time()
    for sc, st in CONFIGS:
        cfg = f"A1_FULL_{sc}_{st}"
        t0 = time.time()
        feats = json.loads((fdir / f"A1_FULL_{sc}" / "final" / f"features_{st}.json").read_text())
        syn = stage(pd.read_parquet(sdir / f"A1_FULL_{sc}" / "train.parquet"), st)
        y = target(syn, st)
        folds = np.zeros(len(syn), int)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(syn, syn["cls"])):
            folds[te] = k
        oof = None
        for k in range(5):
            m = folds == k
            (p,), _ = fit_predict(syn.loc[~m, feats], y[~m], [syn.loc[m, feats]])
            if oof is None:
                oof = pd.DataFrame(np.nan, index=syn.index, columns=p.columns)
            oof.loc[m, p.columns] = p.to_numpy()
        assert oof.notna().all().all()
        meta = syn[["record_id", "cls", "y_s1", "n_bio_measured"]].assign(
            patient_id=syn["record_id"], is_index_sample=True, inner_fold=folds)
        pd.concat([meta, oof], axis=1).to_parquet(adir / "oof" / f"{cfg}_oof.parquet", index=False)

        if sc == "CBC":
            dev = pd.read_parquet(fdir / f"A1_FULL_{sc}" / "final" / "train.parquet")
            tm = pd.read_parquet(fdir / f"A1_FULL_{sc}" / "final" / "temporal.parquet")
        else:
            dev = real_bio(fdir / f"A1_FULL_{sc}" / "final" / "train.parquet", imp)
            tm = real_bio(fdir / f"A1_FULL_{sc}" / "final" / "temporal.parquet", imp)
        dev, tm = dev.reset_index(drop=True), tm.reset_index(drop=True)      # every patient: the app runs all stages
        (pd_dev, pd_tm), classes = fit_predict(syn[feats], y, [dev[feats], tm[feats]])
        for name, d, p in (("dev", dev, pd_dev), ("temporal", tm, pd_tm)):
            keep = [c for c in META + ["has_analyzer_record"] if c in d.columns]
            pd.concat([d[keep].reset_index(drop=True), p], axis=1).to_parquet(
                adir / "tstr" / f"{cfg}_{name}.parquet", index=False)
        info["configs"][cfg] = {"n_synthetic": len(syn), "n_features": len(feats), "classes": [str(c) for c in classes],
                                "n_dev": len(dev), "n_temporal": len(tm), "gpu_seconds": round(time.time() - t0)}
        note(f"{cfg}: OOF + TSTR in {time.time() - t0:.0f} s")

    if not a.skip_cpu:
        for sc, st in CONFIGS:
            cfg = f"A1_FULL_{sc}_{st}"
            feats = json.loads((fdir / f"A1_FULL_{sc}" / "final" / f"features_{st}.json").read_text())
            syn = stage(pd.read_parquet(sdir / f"A1_FULL_{sc}" / "train.parquet"), st)
            info["configs"][cfg]["cpu_2_threads"] = cpu_timing(syn[feats], target(syn, st))
            note(f"{cfg}: CPU {info['configs'][cfg]['cpu_2_threads']}")
    try:
        import tabpfn
        import torch
        versions = {"tabpfn": getattr(tabpfn, "__version__", "?"), "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:                                                  # local test with a stand-in model
        versions = {"tabpfn": None, "torch": None, "gpu": None}
    info.update({**versions, "python": platform.python_version(),
                 "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                 "minutes": round((time.time() - t_all) / 60, 1)})
    (adir / "app_prep_info.json").write_text(json.dumps(info, indent=1))
    note(f"DONE in {info['minutes']} min")


if __name__ == "__main__":
    main()
