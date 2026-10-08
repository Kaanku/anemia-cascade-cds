"""
s20_secondary.py — the frozen final models on the out-of-scope (secondary) set (post hoc, exploratory)
=====================================================================================================

How does the cascade behave on patients outside its four classes? The secondary set of s01
(59 patients, one first sample each: megaloblastic anaemia 16, diagnosis unclear 15, homozygous or major
haemoglobinopathy 13, hereditary spherocytosis 11, combined IDA + ACD 4) was never used for training,
feature selection, locking or evaluation. Here it is scored by the final models of the main analysis
(A1 FULL, TabPFN-3.5) exactly as the temporal cohort was (DECISIONS R37):

    prep  (local)  features of the secondary patients with the final fold-set's own transforms: canonical
                   ratios; for CBC + biochemistry only patients with at least one analyte measured, gaps
                   filled by the final fold-set's saved label-free imputer (s04 F3/F4, s06)
                   -> data/features/A1_FULL_<SC>/final/secondary.parquet
    fit   (Colab)  TabPFN-3.5 refitted on the final training rows with the final feature lists (s05c
                   fit_predict, default settings, seed 42) predicting the secondary rows and, as a check,
                   the temporal rows -> predictions_tabpfn35/<cfg>/final_secondary.parquet,
                   final_temporal_refit.parquet, runs_tabpfn35/secondary__<cfg>.json
    eval  (local)  refit check against the stored temporal predictions; the locked final choices of
                   lock.json (Stage 1 threshold, Stage 2 HIGH cut-off, LOW < 0.35, APS quantiles) applied
                   unchanged -> reports/extra/secondary_set.csv, secondary_set_patients.csv (internal)

Usage:
    python s20_secondary.py --phase prep --out /path/to/CDS_v43
    python s20_secondary.py --phase fit  --out /content/drive/MyDrive/CDS_v43 --data /content/cds_data
    python s20_secondary.py --phase eval --out /path/to/CDS_v43
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CFGS = ["A1_FULL_CBC_S1", "A1_FULL_CBC_S2", "A1_FULL_CBC_BIO_S1", "A1_FULL_CBC_BIO_S2"]
META = ["record_id", "patient_id", "reason", "n_bio_measured"]


def prep(out: Path) -> None:
    from s04_features import BIO, CBC_BASE, add_ratios, apply_imputer, base_features, ratio_pairs
    sec = pd.read_parquet(out / "data" / "analysis" / "analysis_secondary.parquet")
    sec = sec.copy()
    sec["n_bio_measured"] = sec[BIO].notna().sum(axis=1)
    assert sec[CBC_BASE].notna().all().all(), "thesis CBC inputs must be complete (s01)"
    assert sec["patient_id"].is_unique
    for sc in ("CBC", "CBC_BIO"):
        fdir = out / "data" / "features" / f"A1_FULL_{sc}" / "final"
        x = sec[sec["n_bio_measured"] > 0].copy() if sc == "CBC_BIO" else sec.copy()
        if sc == "CBC_BIO":
            x = apply_imputer(joblib.load(fdir / "imputer.joblib"), x)          # saved final imputer
        x, ratios = add_ratios(x, ratio_pairs(sc))
        feats = set()
        for st in (("S1", "S2") if sc == "CBC_BIO" else ("S1", "S2", "MX")):
            feats |= set(json.loads((fdir / f"features_{st}.json").read_text()))
        tr_cols = pd.read_parquet(fdir / "train.parquet").columns
        missing = [f for f in feats if f not in x.columns]
        assert not missing, f"{sc}: features missing in the secondary set: {missing}"
        keep = META + [c for c in tr_cols if c in x.columns and c not in META and c not in
                       ("is_index_sample", "cls", "y_s1", "outer_fold", "inner_fold", "lc_q")]
        x[keep].to_parquet(fdir / "secondary.parquet", index=False)
        print(f"{sc}: {len(x)} secondary patients -> {fdir / 'secondary.parquet'} ({len(keep)} columns)")


def fit(out: Path, data: Path) -> None:
    import s05c_tabpfn as T
    from s05_train import POP
    for cfg in CFGS:
        marker = out / f"runs_{T.TAG}" / f"secondary__{cfg}.json"
        if marker.exists():
            print(f"{cfg}: finished earlier")
            continue
        st, sc = cfg[-2:], "CBC_BIO" if "_CBC_BIO_" in cfg else "CBC"
        fdir = data / "features" / f"A1_FULL_{sc}" / "final"
        feats = json.loads((fdir / f"features_{st}.json").read_text())
        tr = pd.read_parquet(fdir / "train.parquet")
        if POP[st] is not None:
            tr = tr[tr["cls"].isin(POP[st])]
        tr = tr.reset_index(drop=True)
        y = (tr["y_s1"] if st == "S1" else tr["cls"]).to_numpy()
        sec = pd.read_parquet(fdir / "secondary.parquet")
        tem = pd.read_parquet(fdir / "temporal.parquet")
        t0 = time.time()
        (ps, pt), classes = T.fit_predict(tr[feats], y, [sec[feats], tem[feats]])
        pdir = out / f"predictions_{T.TAG}" / cfg
        pdir.mkdir(parents=True, exist_ok=True)
        pd.concat([sec[META].reset_index(drop=True), ps], axis=1).to_parquet(pdir / "final_secondary.parquet",
                                                                              index=False)
        pd.concat([tem[["record_id", "patient_id"]].reset_index(drop=True), pt], axis=1).to_parquet(
            pdir / "final_temporal_refit.parquet", index=False)
        import tabpfn
        import torch
        info = {"job": f"secondary__{cfg}", "config": cfg, "model": T.LABEL, "model_version": T.VERSION,
                "tabpfn": getattr(tabpfn, "__version__", "?"), "torch": torch.__version__,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "seed": T.SEED, "classes": [str(c) for c in classes], "features": feats,
                "n_train_rows": len(tr), "n_secondary": len(sec), "n_temporal": len(tem),
                "seconds": round(time.time() - t0)}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(info, indent=1))
        print(f"{cfg}: {len(sec)} secondary + {len(tem)} temporal rows, {info['seconds']} s on {info['device']}")


def evaluate(out: Path) -> None:
    import s07_lock as L
    import s09_evaluate as E
    lock = json.loads((out / "lock" / "lock.json").read_text())["engines"]["tabpfn"]
    root = out / "predictions_tabpfn35"
    checks, frames = [], {}
    for cfg in CFGS:
        st = cfg[-2:]
        ref = pd.read_parquet(root / cfg / "final_temporal.parquet").set_index("record_id")
        new = pd.read_parquet(root / cfg / "final_temporal_refit.parquet").set_index("record_id").loc[ref.index]
        pc = [c for c in ref.columns if c.startswith("prob_")]
        d = np.abs(ref[pc].to_numpy(float) - new[pc].to_numpy(float))
        checks.append({"config": cfg, "max_abs_dp": float(d.max()),
                       "argmax_agreement": float((ref[pc].to_numpy().argmax(1) == new[pc].to_numpy().argmax(1)).mean())})
        frames[cfg] = pd.read_parquet(root / cfg / "final_secondary.parquet")
    e = {cfg: lock[cfg]["final"] for cfg in CFGS}
    s1c, s2c = frames["A1_FULL_CBC_S1"], frames["A1_FULL_CBC_S2"]
    m = s1c[META + ["prob_1"]].rename(columns={"prob_1": "s1c"}).merge(
        s2c[["record_id"] + [f"prob_{c}" for c in E.S2]], on="record_id", validate="one_to_one")
    P = m[[f"prob_{c}" for c in E.S2]].to_numpy(float)
    m["s1_aac"] = m["s1c"] >= e["A1_FULL_CBC_S1"]["threshold"]
    m["s2_class"] = np.array(E.S2)[P.argmax(1)]
    top = P.max(1)
    high = e["A1_FULL_CBC_S2"]["high_cutoff"]
    m["s2_zone"] = np.where(top < L.LOW_CUTOFF, "LOW", np.where(top >= high, "HIGH", "MEDIUM"))
    m["tier1_final"] = m["s1_aac"] & (m["s2_zone"] == "HIGH")
    u = np.random.default_rng(E.SEED_APS).uniform(size=len(m))
    sets = E.aps_sets(P, np.full(len(m), e["A1_FULL_CBC_S2"]["conformal_qhat"]["0.1"]), u)
    m["set_size_cbc"] = sets.sum(1)
    b1, b2 = frames["A1_FULL_CBC_BIO_S1"], frames["A1_FULL_CBC_BIO_S2"]
    mb = b1[["record_id", "prob_1"]].rename(columns={"prob_1": "s1b"}).merge(
        b2[["record_id"] + [f"prob_{c}" for c in E.S2]].rename(columns={f"prob_{c}": f"bio_{c}" for c in E.S2}),
        on="record_id", validate="one_to_one")
    Pb = mb[[f"bio_{c}" for c in E.S2]].to_numpy(float)
    mb["t2_s1_aac"] = mb["s1b"] >= e["A1_FULL_CBC_BIO_S1"]["threshold"]
    mb["t2_class"] = np.where(mb["t2_s1_aac"], np.array(E.S2)[Pb.argmax(1)], "OAC")
    tb = Pb.max(1)
    mb["t2_zone"] = np.where(~mb["t2_s1_aac"], "-", np.where(tb < L.LOW_CUTOFF, "LOW", np.where(
        tb >= e["A1_FULL_CBC_BIO_S2"]["high_cutoff"], "HIGH", "MEDIUM")))
    m = m.merge(mb[["record_id", "t2_s1_aac", "t2_class", "t2_zone"]], on="record_id", how="left")
    # cascade: Tier 1 HIGH is final; otherwise Tier 2 when biochemistry is measured
    m["cascade_end"] = np.where(m["tier1_final"], "Tier 1 HIGH (" + m["s2_class"] + ")",
                                np.where(m["t2_class"].isna(), "escalated, no biochemistry",
                                         np.where(m["t2_class"] == "OAC", "Tier 2 OAC",
                                                  "Tier 2 " + m["t2_zone"].fillna(""))))
    rows = []
    for reason, g in [("all", m)] + list(m.groupby("reason")):
        rows.append({"category": reason, "n": len(g), "stage1_oac": int((~g["s1_aac"]).sum()),
                     "stage1_aac": int(g["s1_aac"].sum()),
                     "s2_high": int((g["s1_aac"] & (g["s2_zone"] == "HIGH")).sum()),
                     "s2_medium": int((g["s1_aac"] & (g["s2_zone"] == "MEDIUM")).sum()),
                     "s2_low": int((g["s1_aac"] & (g["s2_zone"] == "LOW")).sum()),
                     "tier1_finalised": int(g["tier1_final"].sum()),
                     "tier1_classes": "; ".join(f"{k} {v}" for k, v in g.loc[g["tier1_final"], "s2_class"].value_counts().items()),
                     "with_biochemistry": int(g["t2_class"].notna().sum()),
                     "tier2_oac": int((g["t2_class"] == "OAC").sum()),
                     "tier2_high": int((g["t2_zone"] == "HIGH").sum()),
                     "tier2_medium_low": int(g["t2_zone"].isin(["MEDIUM", "LOW"]).sum()),
                     "median_set_size_cbc": float(g["set_size_cbc"].median())})
    od = out / "reports" / "extra"
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(od / "secondary_set.csv", index=False)
    pd.DataFrame(checks).to_csv(od / "secondary_refit_check.csv", index=False)
    m.drop(columns=["patient_id"]).to_csv(od / "secondary_set_patients.csv", index=False)   # internal only
    pd.set_option("display.width", 250)
    print(pd.DataFrame(checks).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["prep", "fit", "eval"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None)
    a = ap.parse_args()
    if a.phase == "prep":
        prep(a.out)
    elif a.phase == "fit":
        fit(a.out, a.data or a.out / "data")
    else:
        evaluate(a.out)
