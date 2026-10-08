"""
s17b_crossgen.py — cross-generation predictions for locking the demo app on real patients (Colab; R34)
=====================================================================================================

For each half h in {A, B} (s16c): the app's model (TabPFN-3.5 or TabPFN-3.5-fast, fixed ensemble size) is
fitted on the synthetic copy of h and predicts every real patient of the other half, for each configuration
(A1 FULL CBC S1, S2, MX; CBC_BIO S1, S2; final feature lists). Pooled over the two halves, each development
patient has one prediction from a model whose training data were generated without them; s18 locks the app's
operating points on these predictions.

With --data crossgen_k<K> (s16c --folds K) the same is done per fold: the model trained on synthetic_k (generated
without fold k) predicts the real patients of fold k.

Outputs: <OUT>/app_xgen<sub>/<cfg>.parquet (record_id, patient_id, cls, y_s1, half, prob_*), info.json,
         <OUT>/reports/progress_app_xgen<sub>.log;  with --data crossgen_k<K>: app_xgen_k<K><sub>/ (column fold)
Usage (Colab, GPU, TABPFN_TOKEN set):
    python s17b_crossgen.py --out /content/drive/MyDrive/CDS_v43 --version v3.5-fast --n-estimators 2 [--data crossgen_k5]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CONFIGS = [("CBC", "S1"), ("CBC", "S2"), ("CBC", "MX"), ("CBC_BIO", "S1"), ("CBC_BIO", "S2")]
POP = {"S1": None, "S2": ["IDA", "HA", "HGB_HTZ", "NORMAL"], "MX": ["IDA", "HGB_HTZ"]}


def model(version: str, n_estimators: int | None):
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    kw = {} if n_estimators is None else {"n_estimators": n_estimators}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNClassifier.create_default_for_version(ModelVersion(version), device=device, random_state=SEED, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--version", default="v3.5")
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--data", default="crossgen", help="crossgen (halves) or crossgen_k<K> (s16c --folds K)")
    a = ap.parse_args()
    sub = ("" if a.version == "v3.5" else "_fast") + ("" if a.n_estimators is None else f"_n{a.n_estimators}")
    cdir, fdir = a.out / "data" / a.data, a.out / "data" / "features"
    split = json.loads((cdir / "split.json").read_text())
    if "folds" in split:                                   # model on synthetic_k (made without fold k) -> fold k
        pairs = [(str(k), str(k), "fold", k) for k in range(split["folds"])]
        sub = f"_k{split['folds']}{sub}"
    else:                                                  # halves: synthetic_A -> real_B and back
        pairs = [("A", "B", "half", "B"), ("B", "A", "half", "A")]
    od = a.out / f"app_xgen{sub}"
    od.mkdir(parents=True, exist_ok=True)
    log = a.out / "reports" / f"progress_app_xgen{sub}.log"
    t_all = time.time()
    for sc, st in CONFIGS:
        cfg, t0, parts = f"A1_FULL_{sc}_{st}", time.time(), []
        feats = json.loads((fdir / f"A1_FULL_{sc}" / "final" / f"features_{st}.json").read_text())
        for s_tag, r_tag, col, val in pairs:
            syn = pd.read_parquet(cdir / f"synthetic_{s_tag}_{sc}.parquet")
            if POP[st] is not None:
                syn = syn[syn["cls"].isin(POP[st])]
            y = (syn["y_s1"] if st == "S1" else syn["cls"]).to_numpy()
            real = pd.read_parquet(cdir / f"real_{r_tag}_{sc}.parquet").reset_index(drop=True)
            clf = model(a.version, a.n_estimators).fit(syn[feats].to_numpy(float), y)
            p = pd.DataFrame(clf.predict_proba(real[feats].to_numpy(float)),
                             columns=[f"prob_{c}" for c in clf.classes_])
            parts.append(pd.concat([real[["record_id", "patient_id", "cls", "y_s1"]].assign(**{col: val}), p], axis=1))
        pd.concat(parts, ignore_index=True).to_parquet(od / f"{cfg}.parquet", index=False)
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | app xgen{sub} | {cfg}: {time.time() - t0:.0f} s"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")
    import tabpfn
    (od / "info.json").write_text(json.dumps({"version": a.version, "n_estimators": a.n_estimators or "auto",
                                              "data": a.data, "scheme": "folds" if "folds" in split else "halves",
                                              "seed": SEED, "tabpfn": tabpfn.__version__,
                                              "minutes": round((time.time() - t_all) / 60, 1)}, indent=1))


if __name__ == "__main__":
    main()
