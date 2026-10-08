"""
s05b_tabpfn.py — Anemia cascade CDS, nested cross-validation (v4)
==================================================================

TabPFN v2 (Hollmann et al., Nature 2025; pretrained weights published on
Hugging Face by Prior Labs) on exactly the same fold-sets, rows and features
as the AutoGluon models of s05, so the two can be compared patient by patient
(DECISIONS N6). TabPFN is not tuned: it is a pretrained model that is fitted
in context on the training rows.

For every configuration (A1/A2 × FULL: CBC S1, S2, MX; CBC_BIO S1, S2) and
fold-set:
    inner OOF  five fits, each on four inner folds, predicting the fifth
               (the same inner folds AutoGluon used for bagging)
    eval       one fit on all training rows, predicting the outer fold
    temporal   (final fold-set) the same fit predicting the temporal cohort

Outputs (same layout and columns as s05):
    <OUT>/predictions_tabpfn/<cfg>/<fold-set>_{oof,eval,temporal}.parquet
    <OUT>/runs_tabpfn/<cfg>__<fold-set>.json        (written last = finished)

Usage (Colab, GPU runtime recommended; CPU works, slower):
    pip install "tabpfn==2.2.1"
    python s05b_tabpfn.py --out /content/drive/MyDrive/CDS_v4
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from s05_train import FOLDSETS, FULL, META_EVAL, META_TEMP, META_TRAIN, POP

SEED = 42


def make_model():
    import torch
    from tabpfn import TabPFNClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNClassifier(device=device, random_state=SEED, ignore_pretraining_limits=False), device


def fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, Xte_list: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list]:
    clf, _ = make_model()
    clf.fit(Xtr.to_numpy(dtype=float), ytr)
    out = []
    for X in Xte_list:
        p = clf.predict_proba(X.to_numpy(dtype=float)) if len(X) else np.empty((0, len(clf.classes_)))
        out.append(pd.DataFrame(p, columns=[f"prob_{c}" for c in clf.classes_]))
    return out, list(clf.classes_)


def run(out: Path, a: str, fs: str, sc: str, st: str, fset: str) -> dict | None:
    cfg = f"{a}_{fs}_{sc}_{st}"
    marker = out / "runs_tabpfn" / f"{cfg}__{fset}.json"
    if marker.exists():
        return None
    fdir = out / "data" / "features" / f"{a}_{fs}_{sc}" / fset
    feats = json.loads((fdir / f"features_{st}.json").read_text())
    tr = pd.read_parquet(fdir / "train.parquet")
    if POP[st] is not None:
        tr = tr[tr["cls"].isin(POP[st])]
    tr = tr.reset_index(drop=True)
    y = (tr["y_s1"] if st == "S1" else tr["cls"]).to_numpy()
    ev = pd.read_parquet(fdir / "eval.parquet")
    tpath = fdir / "temporal.parquet"
    tm = pd.read_parquet(tpath) if (fset == "final" and tpath.exists()) else None
    t0 = time.time()

    oof = None
    for g in sorted(tr["inner_fold"].dropna().unique()):                      # inner OOF
        m = (tr["inner_fold"] == g).to_numpy()
        (p,), classes = fit_predict(tr.loc[~m, feats], y[~m], [tr.loc[m, feats]])
        if oof is None:
            oof = pd.DataFrame(np.nan, index=tr.index, columns=p.columns)
        oof.loc[m, p.columns] = p.to_numpy()
    assert oof.notna().all().all()
    targets = [ev[feats]] + ([tm[feats]] if tm is not None else [])
    preds, classes = fit_predict(tr[feats], y, targets)                         # eval / temporal

    pdir = out / "predictions_tabpfn" / cfg
    pdir.mkdir(parents=True, exist_ok=True)
    pd.concat([tr[META_TRAIN], oof], axis=1).to_parquet(pdir / f"{fset}_oof.parquet", index=False)
    if len(ev):
        pd.concat([ev[META_EVAL].reset_index(drop=True), preds[0]], axis=1).to_parquet(
            pdir / f"{fset}_eval.parquet", index=False)
    if tm is not None:
        pd.concat([tm[META_TEMP].reset_index(drop=True), preds[1]], axis=1).to_parquet(
            pdir / "final_temporal.parquet", index=False)

    import tabpfn
    import torch
    info = {"config": cfg, "fold_set": fset, "model": "TabPFN v2 (TabPFNClassifier, default settings)",
            "tabpfn": getattr(tabpfn, "__version__", "?"), "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu", "python": platform.python_version(),
            "classes": [str(c) for c in classes], "features": feats, "n_train_rows": len(tr),
            "seconds": round(time.time() - t0)}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    jobs = [(an, fs, sc, st, f) for an in ("A1", "A2") for fs, sc, st in FULL for f in FOLDSETS]
    t_start = time.time()
    log = a.out / "reports" / "progress.log"
    log.parent.mkdir(exist_ok=True)

    def note(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | TabPFN | {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    note(f"START: {len(jobs)} fold-sets")
    for i, (an, fs, sc, st, f) in enumerate(jobs, 1):
        try:
            info = run(a.out, an, fs, sc, st, f)
        except Exception as e:                                    # e.g. weights not downloadable
            note(f"ERROR {an}_{fs}_{sc}_{st} {f}: {type(e).__name__}: {e}")
            msg = str(e)
            if "huggingface" in msg.lower() or "401" in msg or "gated" in msg.lower():
                raise SystemExit("TabPFN weights could not be downloaded from Hugging Face. Log in with "
                                 "`from huggingface_hub import login; login()` (after accepting the model "
                                 "licence on its Hugging Face page) and run again.") from e
            raise
        if info:
            note(f"[{i}/{len(jobs)}] {info['config']} {f}: {info['seconds']} s on {info['device']} "
                 f"| elapsed {(time.time() - t_start) / 60:.0f} min")
    note(f"DONE: {len(list((a.out / 'runs_tabpfn').glob('*.json')))} fold-sets finished")


if __name__ == "__main__":
    main()
