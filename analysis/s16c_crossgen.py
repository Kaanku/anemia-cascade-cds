"""
s16c_crossgen.py — cross-generation data for locking the demo app on real patients (DECISIONS R34)
==================================================================================================

Why: TabPFN-3.5 trained on the synthetic cohort discriminates real patients about as well as a model trained on
real data, but it is over-confident on them: the synthetic classes are cleaner than the real ones, so cut-offs
locked on synthetic out-of-fold predictions (s17/s18) do not transfer (temporal cohort: 78 % of patients
finalised at Tier 1 with 79 % accuracy instead of the intended ~90 %). The app's operating points are therefore
locked on real patients the generator has not seen: the 863 development patients are split in two halves
(stratified by class, seed 42); a generator is fitted on each half (s16, unchanged), and the model trained on
the synthetic copy of one half predicts the real patients of the other half. Pooled, every patient gets one
prediction from a model whose training data were generated without them. s18 then applies the study's lock
rules to these predictions (parametric calibration only, so no patient-level map is published).

This script writes, per half h in {A, B}: the synthetic matrices generated from h (CBC and CBC + biochemistry,
built as in s04/s16 with the KNN imputer fitted on the synthetic rows) and the real matrices of the other half,
whose missing analytes are imputed with that same synthetic imputer (as the app would). Model fitting runs on
Colab (s17b_crossgen.py).

With --folds K the cohort is split into K stratified folds instead of halves: for each fold k a generator is
fitted on the other K - 1 folds, and its synthetic copy (same class sizes as those folds) predicts fold k. With
K = 5 the generator and the model see 4/5 of the cohort, close to the app's model (the whole cohort), so the
locked cut-offs are less pessimistic than with halves.

Outputs: <OUT>/data/crossgen/{synthetic,real}_{A,B}_{CBC,CBC_BIO}.parquet, split.json                 (halves)
         <OUT>/data/crossgen_k<K>/{synthetic,real}_<k>_{CBC,CBC_BIO}.parquet, split.json   (--folds K; synthetic_k
         is generated without fold k, real_k is fold k)
Usage:   python s16c_crossgen.py --out /path/CDS_v43 [--folds 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import s16_synthetic as S
from s04_features import BIO, add_ratios, apply_imputer, ratio_pairs, LabelFreeKNNImputer

SEED = 42


def write_pair(od: Path, train: pd.DataFrame, test: pd.DataFrame, tag_syn: str, tag_real: str, seed: int,
               info: dict) -> None:
    """Generator on `train` -> synthetic matrices (tag_syn); `test` prepared as the app prepares a patient (tag_real)."""
    gen = S.Generator(train)
    syn = gen.sample(np.random.default_rng(seed))
    syn["record_id"] = [f"SYN{tag_syn}-{i + 1:05d}" for i in range(len(syn))]
    cbc = add_ratios(syn, ratio_pairs("CBC"))[0]
    sb = syn[syn[BIO].notna().any(axis=1)]
    st = LabelFreeKNNImputer(k=5).fit(sb).state()
    bio = add_ratios(apply_imputer(st, sb), ratio_pairs("CBC_BIO"))[0]
    cbc.to_parquet(od / f"synthetic_{tag_syn}_CBC.parquet", index=False)
    bio.to_parquet(od / f"synthetic_{tag_syn}_CBC_BIO.parquet", index=False)
    rc = add_ratios(test, ratio_pairs("CBC"))[0]
    rb = test[test[BIO].notna().any(axis=1)]
    rb = add_ratios(apply_imputer(st, rb), ratio_pairs("CBC_BIO"))[0]
    keep = ["record_id", "patient_id", "cls", "y_s1"]
    base = S.BASE + BIO
    rc[keep + [c for c in rc.columns if c not in keep and (c in base or "_div_" in c)]].to_parquet(
        od / f"real_{tag_real}_CBC.parquet", index=False)
    rb[keep + [c for c in rb.columns if c not in keep and (c in base or "_div_" in c or c.startswith("imputed_"))]
       ].to_parquet(od / f"real_{tag_real}_CBC_BIO.parquet", index=False)
    info[f"synthetic_{tag_syn}"] = {"n": len(syn), "n_bio": int(len(sb)), "generator_redraws": gen.redraws}
    info[f"real_{tag_real}"] = {"n": len(test), "n_bio": int(len(rb))}
    print(tag_syn, "->", tag_real, len(syn), len(sb), "|", len(test), len(rb), flush=True)


def kfold(real: pd.DataFrame, out: Path, k: int) -> None:
    from sklearn.model_selection import StratifiedKFold
    fold = np.zeros(len(real), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=SEED).split(real, real["cls"])):
        fold[te] = i
    od = out / "data" / f"crossgen_k{k}"
    od.mkdir(parents=True, exist_ok=True)
    info = {"seed": SEED, "folds": k, "n": {str(i): int((fold == i).sum()) for i in range(k)}}
    for i in range(k):
        write_pair(od, real[fold != i].copy(), real[fold == i].copy(), str(i), str(i), SEED + i, info)
    (od / "split.json").write_text(json.dumps(info, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=None, help="K stratified folds instead of halves")
    a = ap.parse_args()
    real = pd.read_parquet(a.out / "data" / "model" / "A1.parquet")
    real["y_s1"] = real["cls"].isin(S.AAC).astype(int)
    if a.folds:
        return kfold(real.reset_index(drop=True), a.out, a.folds)
    ia = real.groupby("cls", group_keys=False).apply(lambda g: g.sample(frac=0.5, random_state=SEED)).index
    halves = {"A": real.loc[ia].copy(), "B": real.drop(ia).copy()}
    od = a.out / "data" / "crossgen"
    od.mkdir(parents=True, exist_ok=True)
    info = {"seed": SEED, "n": {h: len(d) for h, d in halves.items()}}
    for h, other in (("A", "B"), ("B", "A")):
        gen = S.Generator(halves[h])
        syn = gen.sample(np.random.default_rng(SEED + (0 if h == "A" else 1)))
        syn["record_id"] = [f"SYN{h}-{i + 1:05d}" for i in range(len(syn))]
        # synthetic matrices (as s16.matrices) and the imputer fitted on the synthetic rows
        cbc = add_ratios(syn, ratio_pairs("CBC"))[0]
        sb = syn[syn[BIO].notna().any(axis=1)]
        st = LabelFreeKNNImputer(k=5).fit(sb).state()
        bio = add_ratios(apply_imputer(st, sb), ratio_pairs("CBC_BIO"))[0]
        cbc.to_parquet(od / f"synthetic_{h}_CBC.parquet", index=False)
        bio.to_parquet(od / f"synthetic_{h}_CBC_BIO.parquet", index=False)
        # the real patients of the other half, prepared as the app prepares a new patient
        r = halves[other]
        rc = add_ratios(r, ratio_pairs("CBC"))[0]
        rb = r[r[BIO].notna().any(axis=1)]
        rb = add_ratios(apply_imputer(st, rb), ratio_pairs("CBC_BIO"))[0]
        keep = ["record_id", "patient_id", "cls", "y_s1"]
        base = S.BASE + BIO
        rc[keep + [c for c in rc.columns if c not in keep and (c in base or "_div_" in c)]].to_parquet(
            od / f"real_{other}_CBC.parquet", index=False)
        rb[keep + [c for c in rb.columns if c not in keep and (c in base or "_div_" in c or c.startswith("imputed_"))]
           ].to_parquet(od / f"real_{other}_CBC_BIO.parquet", index=False)
        info[f"synthetic_{h}"] = {"n": len(syn), "n_bio": int(len(sb)), "generator_redraws": gen.redraws}
        info[f"real_{other}"] = {"n": len(r), "n_bio": int(len(rb))}
        print(h, "->", other, len(syn), len(sb), "|", len(r), len(rb))
    (od / "split.json").write_text(json.dumps(info, indent=1, default=float))


if __name__ == "__main__":
    main()
