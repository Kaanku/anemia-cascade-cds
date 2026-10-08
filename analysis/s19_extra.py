"""
s19_extra.py — descriptive analyses added for the manuscript (TabPFN-3.5, A1 FULL, first samples)
================================================================================================

Reads the same outer-fold and temporal predictions as s09, with the identical lock (the opening is
logged in lock/unlock_log.json like every s09 run). Nothing is fitted or re-locked; every operating
choice comes from lock.json. Three tables:

    conformal_by_class   class-conditional coverage and set size of the locked APS sets
                         (s09 reports marginal coverage only; randomisation as in s09, seed 123)
    cascade_by_target    the cascade with the Tier 1 HIGH cut-off taken from each locked operating
                         point (80, 85, 90, 95 % target accuracy); the 90 % row is the pre-specified
                         cascade and must equal s09's cascade block
    subgroups            Stage 1, Stage 2 and end-to-end CBC-only performance by sex and age group
                         (nested CV; the temporal cohort has no sex and no OAC)
    stage2_paired_by_class  Stage 2 recall per class of the CBC and the CBC + biochemistry models on the
                         same true-AAC patients with measured biochemistry (exact McNemar per class)

95 % CIs: percentile bootstrap over patients (2,000 resamples, seed 42) as in s09; Stage 1 AUC with
DeLong CIs.

Usage:
    python s19_extra.py --out /path/to/CDS_v43 [--boot 2000]
Outputs:
    <OUT>/reports/extra/{conformal_by_class,cascade_by_target,subgroups}.csv and extra.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import s07_lock as L
import s09_evaluate as E

ENGINE = "tabpfn"
CFG = {"c1": "A1_FULL_CBC_S1", "c2": "A1_FULL_CBC_S2", "b1": "A1_FULL_CBC_BIO_S1", "b2": "A1_FULL_CBC_BIO_S2"}
TARGETS = [f"{t:.2f}" for t in L.TARGETS]
AGE_BINS = [(-np.inf, 18, "<18"), (18, 40, "18-39"), (40, 65, "40-64"), (65, np.inf, ">=65")]


def first(d):
    return d[d["is_index_sample"].astype(bool)]


def conformal_by_class(d: pd.DataFrame, bio_ids=None, **meta) -> list[dict]:
    """Same sets as s09.Evaluator.conformal (sorted by record_id, one uniform per row, seed 123).
    bio_ids (CBC model only): also summarise the same sets in the patients with measured biochemistry,
    so that the CBC and CBC + biochemistry sets can be compared on the same patients."""
    d = d[d["cls"].isin(E.AAC)].sort_values("record_id")
    P, y = E.P2(d), E.y2(d)
    u = np.random.default_rng(E.SEED_APS).uniform(size=len(d))
    rows = []
    for a in L.ALPHAS:
        sets = E.aps_sets(P, d[f"qhat_{a}"].to_numpy(), u)
        cover = sets[np.arange(len(y)), y]
        size = sets.sum(1)
        rows.append({**meta, "alpha": a, "class": "all", "n": len(y), "coverage": cover.mean(),
                     "mean_set_size": size.mean(), "singleton_share": (size == 1).mean()})
        for i, c in enumerate(E.S2):
            m = y == i
            rows.append({**meta, "alpha": a, "class": c, "n": int(m.sum()), "coverage": cover[m].mean(),
                         "mean_set_size": size[m].mean(), "singleton_share": (size[m] == 1).mean()})
        ida, htz = E.S2.index("IDA"), E.S2.index("HGB_HTZ")
        both = sets[:, ida] & sets[:, htz]
        rows.append({**meta, "alpha": a, "class": "sets with IDA and HGB_HTZ", "n": len(y),
                     "coverage": np.nan, "mean_set_size": np.nan, "singleton_share": both.mean()})
        if bio_ids is not None:
            k = d["record_id"].isin(bio_ids).to_numpy()
            rows.append({**meta, "alpha": a, "class": "all, patients with biochemistry", "n": int(k.sum()),
                         "coverage": cover[k].mean(), "mean_set_size": size[k].mean(),
                         "singleton_share": (size[k] == 1).mean()})
            rows.append({**meta, "alpha": a, "class": "sets with IDA and HGB_HTZ, patients with biochemistry",
                         "n": int(k.sum()), "coverage": np.nan, "mean_set_size": np.nan,
                         "singleton_share": both[k].mean()})
    return rows


def cascade_frame(c1, c2, b1, b2) -> pd.DataFrame:
    keep = ["record_id", "patient_id", "cls"]
    ops = [f"op_{t}" for t in TARGETS]
    return (c1[keep + ["s", "thr"]].rename(columns={"s": "s1c", "thr": "t1c"})
            .merge(c2[["record_id"] + ops + [f"prob_{c}" for c in E.S2]], on="record_id")
            .merge(b1[["record_id", "s", "thr"]].rename(columns={"s": "s1b", "thr": "t1b"}), on="record_id")
            .merge(b2[["record_id"] + [f"prob_{c}" for c in E.S2]].rename(
                columns={f"prob_{c}": f"bio_{c}" for c in E.S2}), on="record_id"))


def cascade_by_target(m: pd.DataFrame, b: int, **meta) -> list[dict]:
    y5 = m["cls"].map({c: i for i, c in enumerate(E.C5)}).to_numpy()
    Pc = m[[f"prob_{c}" for c in E.S2]].to_numpy(float)
    Pb = m[[f"bio_{c}" for c in E.S2]].to_numpy(float)

    def fn(y5, s1c, t1c, Pc, high, s1b, t1b, Pb):
        tier1 = (s1c >= t1c) & (Pc.max(1) >= high)
        pred_bio = np.where(s1b >= t1b, Pb.argmax(1), 4)
        pred = np.where(tier1, Pc.argmax(1), pred_bio)
        return {"tier1_share": float(tier1.mean()),
                "tier1_accuracy": float((pred[tier1] == y5[tier1]).mean()) if tier1.any() else np.nan,
                "cascade_accuracy": float((pred == y5).mean()),
                "cbc_bio_accuracy": float((pred_bio == y5).mean()),
                "accuracy_difference": float((pred == y5).mean() - (pred_bio == y5).mean())}

    rows = []
    for tgt in TARGETS:
        high = m[f"op_{tgt}"].to_numpy(float)                     # inf where no cut-off reached the target
        arrays = dict(y5=y5, s1c=m["s1c"].to_numpy(), t1c=m["t1c"].to_numpy(), Pc=Pc, high=high,
                      s1b=m["s1b"].to_numpy(), t1b=m["t1b"].to_numpy(), Pb=Pb)
        res = E.with_ci(fn, arrays, m["patient_id"].to_numpy(), b)
        tier1 = (arrays["s1c"] >= arrays["t1c"]) & (Pc.max(1) >= high)
        pred_bio = np.where(arrays["s1b"] >= arrays["t1b"], Pb.argmax(1), 4)
        pred = np.where(tier1, Pc.argmax(1), pred_bio)
        b_, c_, p = E.mcnemar(pred == y5, pred_bio == y5)
        row = {**meta, "target": tgt, "n": len(m), "n_tier1": int(tier1.sum()),
               "folds_with_cutoff": int(np.isfinite(m.groupby("fold_set")[f"op_{tgt}"].first()).sum())
               if "fold_set" in m else int(np.isfinite(high).any()),
               "cascade_only_correct": b_, "cbc_bio_only_correct": c_, "p_mcnemar": p}
        for k, (v, lo, hi) in res.items():
            row[k], row[f"{k}_low"], row[f"{k}_high"] = v, lo, hi
        rows.append(row)
    return rows


def age_group(age: pd.Series) -> pd.Series:
    out = pd.Series(index=age.index, dtype=object)
    for lo, hi, name in AGE_BINS:
        out[(age >= lo) & (age < hi)] = name
    return out


def subgroups(c1, c2, demo: pd.DataFrame, b: int) -> list[dict]:
    d1 = c1.merge(demo, on="record_id", validate="one_to_one")
    d2 = c2.merge(demo, on="record_id", validate="one_to_one")
    e2e = (d1[["record_id", "patient_id", "cls", "s", "thr", "sex", "age_group"]]
           .merge(d2[["record_id"] + [f"prob_{c}" for c in E.S2]], on="record_id", validate="one_to_one"))
    rows = []
    groups = [("sex", "female", "K"), ("sex", "male", "E")] + [("age", g, g) for _, _, g in AGE_BINS]
    for var, label, val in groups:
        col = "sex" if var == "sex" else "age_group"
        x1, x2, xe = d1[d1[col] == val], d2[(d2[col] == val) & d2["cls"].isin(E.AAC)], e2e[e2e[col] == val]
        r = {"variable": var, "group": label, "n_patients": len(x1), "n_oac": int((x1["cls"] == "OAC").sum()),
             "n_aac": int((x1["cls"] != "OAC").sum())}
        y, s = x1["y_s1"].to_numpy(int), x1["s"].to_numpy()
        if 0 < y.sum() < len(y):
            r["s1_auc"], r["s1_auc_low"], r["s1_auc_high"] = E.delong_ci(y, s)
        st1 = E.with_ci(lambda y, s, t: E.binary_metrics(y, s, t),
                        dict(y=y, s=s, t=x1["thr"].to_numpy()), x1["patient_id"].to_numpy(), b)
        for k in ("sensitivity", "specificity"):
            r[f"s1_{k}"], r[f"s1_{k}_low"], r[f"s1_{k}_high"] = st1[k]
        if len(x2) >= 10:
            st2 = E.with_ci(lambda y, P: E.multiclass_metrics(y, P), dict(y=E.y2(x2), P=E.P2(x2)),
                            x2["patient_id"].to_numpy(), b)
            r["n_s2"] = len(x2)
            for k in ("macro_auc", "accuracy"):
                r[f"s2_{k}"], r[f"s2_{k}_low"], r[f"s2_{k}_high"] = st2[k]
        y5 = xe["cls"].map({c: i for i, c in enumerate(E.C5)}).to_numpy()
        ee = E.with_ci(lambda y5, s, t, P: E.label_metrics(y5, np.where(s >= t, P.argmax(1), 4), E.C5),
                       dict(y5=y5, s=xe["s"].to_numpy(), t=xe["thr"].to_numpy(), P=E.P2(xe)),
                       xe["patient_id"].to_numpy(), b)
        r["e2e_accuracy"], r["e2e_accuracy_low"], r["e2e_accuracy_high"] = ee["accuracy"]
        rows.append(r)
    return rows


def build(out: Path, b: int) -> None:
    lock, cals = E.guard_and_log(out, b)
    D = E.Data(out, lock, cals)
    nested = {k: first(D.pooled(ENGINE, c)) for k, c in CFG.items()}
    temporal = {k: first(D.temporal(ENGINE, c)) for k, c in CFG.items()}
    temporal = {k: v[v["has_analyzer_record"].astype(bool)] for k, v in temporal.items()}     # N8 (FULL)

    conf = []
    for set_name, fr in (("nested CV", nested), ("temporal", temporal)):
        conf += conformal_by_class(fr["c2"], bio_ids=set(fr["b2"]["record_id"]), set=set_name, scenario="CBC")
        conf += conformal_by_class(fr["b2"], set=set_name, scenario="CBC_BIO")

    casc = []
    for set_name, fr in (("nested CV", nested), ("temporal", temporal)):
        bio_ids = set(fr["b1"]["record_id"])
        sel = {k: v[v["record_id"].isin(bio_ids)] for k, v in fr.items()}
        m = cascade_frame(sel["c1"], sel["c2"], sel["b1"], sel["b2"])
        if set_name == "nested CV":
            m = m.merge(sel["c2"][["record_id", "fold_set"]], on="record_id")
        casc += cascade_by_target(m, b, set=set_name)

    paired = []                                        # Stage 2, CBC vs CBC + biochemistry on the same patients
    for set_name, fr in (("nested CV", nested), ("temporal", temporal)):
        c2, b2 = fr["c2"][fr["c2"]["cls"].isin(E.AAC)], fr["b2"][fr["b2"]["cls"].isin(E.AAC)]
        m = c2.merge(b2, on=["record_id", "cls"], suffixes=("_c", "_b"), validate="one_to_one")
        y = m["cls"].map({c: i for i, c in enumerate(E.S2)}).to_numpy()
        pc = m[[f"prob_{c}_c" for c in E.S2]].to_numpy(float).argmax(1)
        pb = m[[f"prob_{c}_b" for c in E.S2]].to_numpy(float).argmax(1)
        for i, c in enumerate(["all"] + E.S2):
            k = np.ones(len(y), bool) if c == "all" else (y == i - 1)
            b_, c_, p = E.mcnemar(pc[k] == y[k], pb[k] == y[k])
            paired.append({"set": set_name, "class": c, "n": int(k.sum()), "recall_cbc": float((pc[k] == y[k]).mean()),
                           "recall_cbc_bio": float((pb[k] == y[k]).mean()), "only_cbc_correct": b_,
                           "only_bio_correct": c_, "p_mcnemar": p})

    coh = pd.read_parquet(out / "data" / "cohort" / "cohort_primary.parquet")
    demo = pd.DataFrame({"record_id": coh["record_id"], "sex": coh["sex"], "age_group": age_group(coh["age"])})
    sub = subgroups(nested["c1"], nested["c2"], demo, b)

    od = out / "reports" / "extra"
    od.mkdir(parents=True, exist_ok=True)
    tabs = {"conformal_by_class": pd.DataFrame(conf), "cascade_by_target": pd.DataFrame(casc),
            "subgroups": pd.DataFrame(sub), "stage2_paired_by_class": pd.DataFrame(paired)}
    with pd.ExcelWriter(od / "extra.xlsx") as xw:
        for name, t in tabs.items():
            t.to_csv(od / f"{name}.csv", index=False)
            t.to_excel(xw, sheet_name=name, index=False)

    # consistency with s09 (the 90 % cascade and the marginal coverage must match metrics.parquet)
    met = pd.read_parquet(out / "reports" / "figures_data" / "metrics.parquet")
    met = met[(met.engine == ENGINE) & (met.analysis == "A1") & (met.feature_set == "FULL")
              & (met.unit == "first sample")]
    c90 = tabs["cascade_by_target"].set_index(["set", "target"])
    for s in ("nested CV", "temporal"):
        ref = met[(met.block == "cascade") & (met.set == s)].set_index("metric")["value"]
        for k in ("tier1_share", "tier1_accuracy", "cascade_accuracy", "cbc_bio_accuracy"):
            assert abs(c90.loc[(s, "0.90"), k] - ref[k]) < 1e-9, (s, k)
        cb = tabs["conformal_by_class"]
        for sc in ("CBC", "CBC_BIO"):
            for a in L.ALPHAS:
                v = cb[(cb.set == s) & (cb.scenario == sc) & (cb.alpha == a) & (cb["class"] == "all")]["coverage"].iloc[0]
                r = met[(met.block == "conformal") & (met.set == s) & (met.scenario == sc)
                        & (met.population == f"alpha {a}") & (met.metric == "coverage")]["value"].iloc[0]
                assert abs(v - r) < 1e-9, (s, sc, a)
    print("checks passed: 90 % cascade and marginal coverage equal s09")
    pd.set_option("display.width", 250)
    for name, t in tabs.items():
        print(f"\n== {name}\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=E.B_DEFAULT)
    a = ap.parse_args()
    build(a.out, a.boot)
