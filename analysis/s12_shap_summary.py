"""
s12_shap_summary.py — summaries of the TabPFN-3.5 Shapley values (CDS v4.2, R21)
=================================================================================

Reads the per-patient Shapley values written by s11 (shap_tabpfn35/<cfg>/<fold-set>.parquet)
and writes aggregate tables only (no patient rows leave this script, except the
temporal case examples, which carry the pseudonymous record id and are for internal
review).

Global importance (nested CV, outer folds pooled, one row per patient):
    each outer-fold model explained its own outer-fold patients; a feature that a
    fold's Boruta step did not select contributes 0 for that fold's patients.
    mean |phi| per feature with a percentile bootstrap CI over patients (B = 2000,
    seed 42), mean phi, the share of patients for whom the feature pushes the
    prediction up, the Spearman correlation between the feature value and phi
    (direction of the effect), and the number of folds that selected the feature.
Grouped importance: Shapley values are additive, so the phi of the features of a
    clinical domain are summed per patient before taking |.|; ratios selected by
    Boruta form their own group.
Stability: Spearman correlation of the per-fold importance vectors (pairs of folds).
Stage 2: everything per explained class, plus "all classes" = mean over the four
    classes of mean |phi|.

Outputs: reports/shap/<cfg>_{global,groups,stability}.csv, reports/shap/shap_summary.xlsx,
         reports/shap/<cfg>_beeswarm.parquet (top features: value percentile, phi; aggregate
         figure data), reports/shap/<cfg>_temporal_cases.csv (top 5 contributions per patient)

Usage:
    python s12_shap_summary.py --out /path/to/CDS_v4 [--boot 2000]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUTER = [f"outer{k}" for k in range(5)]
SEED = 42
TOP_BEESWARM = 15

GROUPS = [            # first match wins; ratios ("_div_") are checked before this list
    ("Age", ["age"]),
    ("Red cell mass (HGB, RBC, HCT)", ["hgb_g_d_l", "rbc_10_6_u_l", "hct_pct"]),
    ("Red cell indices (MCV, MCH, MCHC, RDW)", ["mcv_f_l", "mch_pg", "mchc_g_dl", "rdw_sd_fl", "rdw_cv_pct"]),
    ("Haemoglobin content (RET-He, RBC-He, Delta-He)", ["ret_he_pg", "rbc_he_pg", "delta_he_pg", "ret_y_ch",
                                                       "ret_rbc_y_ch"]),
    ("Micro / macro RBC", ["micro_r_pct", "macro_r_pct", "micro_macro_ratio"]),
    ("Reticulocytes (RET, IRF, LFR/MFR/HFR)", ["ret_number_10_6_l", "ret_pct", "irf_pct", "lfr_pct", "mfr_pct",
                                              "hfr_pct", "irf_y_ch"]),
    ("Fragmented RBC / NRBC", ["frc_perc", "nrbc_pct", "nrbc_number_10_3_u_l"]),
    ("White cells", ["wbc_10_3_u_l", "neut_number_10_3_u_l", "lymph_number_10_3_u_l", "mono_number_10_3_u_l",
                     "eo_number_10_3_u_l", "baso_number_10_3_u_l", "ig_number_10_3_u_l"]),
    ("Platelets", ["plt_10_3_u_l", "mpv_fl", "pdw_fl", "p_lcr_pct"]),
]
BIO_GROUP = "Biochemistry"


BIO_KEYS = ("ferritin", "iron", "uibc", "tibc", "transferrin", "b12", "folate", "ldh", "bilirubin", "haptoglobin",
            "crp", "tsat")


def is_bio(f: str) -> bool:
    return any(k in f for k in BIO_KEYS)


def group_of(f: str) -> str:
    if "_div_" in f:                    # ratios selected by Boruta; split by whether a biochemistry term enters
        return "Ratios with biochemistry" if is_bio(f) else "Ratios (CBC only)"
    for g, members in GROUPS:
        if f in members:
            return g
    return BIO_GROUP if is_bio(f) else "Other"


def load(out: Path, cfg: str) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    sdir = out / "shap_tabpfn35" / cfg
    parts = []
    for f in OUTER:
        p = sdir / f"{f}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p).assign(fold_set=f))
    cv = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    tp = sdir / "final.parquet"
    tm = pd.read_parquet(tp).assign(fold_set="final") if tp.exists() else None
    feats = sorted({c[4:] for c in cv.columns if c.startswith("phi_")})
    return cv, tm, feats


def boot_ci(v: np.ndarray, b: int, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(v), size=(b, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(cv: pd.DataFrame, feats: list[str], b: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows, grows, srows = [], [], []
    for cls, d in cv.groupby("explained_class"):
        d = d.reset_index(drop=True)
        assert d["record_id"].is_unique, "one row per patient and class expected"
        PHI = d[[f"phi_{f}" for f in feats]].fillna(0.0).to_numpy()       # not selected in a fold -> 0
        X = d[[f"x_{f}" for f in feats]].to_numpy(dtype=float)
        sel = {f: int(d.groupby("fold_set")[f"phi_{f}"].apply(lambda s: s.notna().any()).sum()) for f in feats}
        for j, f in enumerate(feats):
            a = np.abs(PHI[:, j])
            lo, hi = boot_ci(a, b, rng)
            ok = np.isfinite(X[:, j]) & d[f"phi_{f}"].notna().to_numpy()
            rho = spearmanr(X[ok, j], PHI[ok, j]).statistic if ok.sum() > 10 and np.std(X[ok, j]) > 0 else np.nan
            rows.append({"explained_class": cls, "feature": f, "group": group_of(f), "folds_selected": sel[f],
                         "mean_abs_phi": a.mean(), "ci_low": lo, "ci_high": hi, "mean_phi": PHI[:, j].mean(),
                         "share_positive": float((PHI[:, j] > 0).mean()), "spearman_value_phi": rho,
                         "n_patients": len(d)})
        groups = sorted({group_of(f) for f in feats})
        for g in groups:
            cols = [j for j, f in enumerate(feats) if group_of(f) == g]
            a = np.abs(PHI[:, cols].sum(axis=1))
            lo, hi = boot_ci(a, b, rng)
            grows.append({"explained_class": cls, "group": g, "n_features": len(cols), "mean_abs_phi": a.mean(),
                          "ci_low": lo, "ci_high": hi})
        per_fold = {f: np.abs(d.loc[d["fold_set"] == f, [f"phi_{x}" for x in feats]].fillna(0.0)).mean().to_numpy()
                    for f in OUTER if (d["fold_set"] == f).any()}
        for a_, b_ in itertools.combinations(per_fold, 2):
            srows.append({"explained_class": cls, "fold_a": a_, "fold_b": b_,
                          "spearman": spearmanr(per_fold[a_], per_fold[b_]).statistic})
    G = pd.DataFrame(rows)
    G["rank"] = G.groupby("explained_class")["mean_abs_phi"].rank(ascending=False, method="first").astype(int)
    if G["explained_class"].nunique() > 1:                                          # Stage 2: all classes
        allc = (G.groupby("feature").agg(group=("group", "first"), folds_selected=("folds_selected", "first"),
                                         mean_abs_phi=("mean_abs_phi", "mean"), n_patients=("n_patients", "max"))
                .reset_index().assign(explained_class="all classes (mean)"))
        allc["rank"] = allc["mean_abs_phi"].rank(ascending=False, method="first").astype(int)
        G = pd.concat([G, allc], ignore_index=True)
    return (G.sort_values(["explained_class", "rank"]).reset_index(drop=True),
            pd.DataFrame(grows).sort_values(["explained_class", "mean_abs_phi"], ascending=[True, False]),
            pd.DataFrame(srows))


def beeswarm_data(cv: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    """Aggregate-safe figure data: for the top features, phi with the value as a within-feature percentile."""
    out = []
    for cls, d in cv.groupby("explained_class"):
        top = G[(G["explained_class"] == cls)].nsmallest(TOP_BEESWARM, "rank")["feature"]
        for f in top:
            x, p = d[f"x_{f}"], d[f"phi_{f}"]
            ok = p.notna()
            out.append(pd.DataFrame({"explained_class": cls, "feature": f,
                                     "value_pct": x[ok].rank(pct=True).round(3).to_numpy(),
                                     "phi": p[ok].round(5).to_numpy()}).sample(frac=1, random_state=SEED))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def temporal_global(tm: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    """Temporal cohort (final model): mean |phi| per feature and class, with its rank among the final model's
    features, next to the nested-CV importance of the same features; rho = Spearman of the two importances."""
    feats = sorted({c[4:] for c in tm.columns if c.startswith("phi_")})
    rows = []
    for cls, d in tm.groupby("explained_class"):
        cvm = G[G["explained_class"] == cls].set_index("feature")["mean_abs_phi"]
        for f in feats:
            rows.append({"explained_class": cls, "feature": f, "group": group_of(f), "n_patients": len(d),
                         "mean_abs_phi_temporal": float(d[f"phi_{f}"].abs().mean()),
                         "mean_abs_phi_cv": float(cvm.get(f, np.nan))})
    T = pd.DataFrame(rows)
    T["rank_temporal"] = T.groupby("explained_class")["mean_abs_phi_temporal"].rank(ascending=False, method="first").astype(int)
    T["rho_cv_temporal"] = T.groupby("explained_class").apply(
        lambda d: spearmanr(d["mean_abs_phi_temporal"], d["mean_abs_phi_cv"], nan_policy="omit").statistic,
        include_groups=False).reindex(T["explained_class"]).to_numpy()
    return T.sort_values(["explained_class", "rank_temporal"]).reset_index(drop=True)


def temporal_cases(tm: pd.DataFrame, feats_tm: list[str], k: int = 5) -> pd.DataFrame:
    rows = []
    for _, r in tm.iterrows():
        phi = {f: r[f"phi_{f}"] for f in feats_tm if pd.notna(r.get(f"phi_{f}"))}
        top = sorted(phi.items(), key=lambda t: -abs(t[1]))[:k]
        rows.append({"record_id": r["record_id"], "cls": r["cls"], "explained_class": r["explained_class"],
                     "base_value": round(r["base_value"], 4), "prediction": round(r["prediction"], 4),
                     **{f"top{i + 1}": f"{f} ({v:+.3f}; x={r[f'x_{f}']:.3g})" for i, (f, v) in enumerate(top)}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rdir = a.out / "reports" / "shap"
    rdir.mkdir(parents=True, exist_ok=True)
    sheets = {}
    for cdir in sorted((a.out / "shap_tabpfn35").glob("A1_*")):
        cfg = cdir.name
        cv, tm, feats = load(a.out, cfg)
        if cv.empty:
            continue
        folds = sorted(cv["fold_set"].unique())
        G, GR, ST = summarise(cv, feats, a.boot)
        G.to_csv(rdir / f"{cfg}_global.csv", index=False)
        GR.to_csv(rdir / f"{cfg}_groups.csv", index=False)
        ST.to_csv(rdir / f"{cfg}_stability.csv", index=False)
        beeswarm_data(cv, G).to_parquet(rdir / f"{cfg}_beeswarm.parquet", index=False)
        if tm is not None:
            TG = temporal_global(tm, G)
            TG.to_csv(rdir / f"{cfg}_temporal_global.csv", index=False)
            sheets[f"{cfg[3:]}_temporal"[:31]] = TG
            print(f"   temporal (final model, {tm['record_id'].nunique()} patients): rho CV vs temporal importance "
                  + ", ".join(f"{c} {r:.2f}" for c, r in TG.groupby("explained_class")["rho_cv_temporal"].first().items()))
            temporal_cases(tm, sorted({c[4:] for c in tm.columns if c.startswith("phi_")})).to_csv(
                rdir / f"{cfg}_temporal_cases.csv", index=False)
        sheets[f"{cfg[3:]}_global"[:31]] = G
        sheets[f"{cfg[3:]}_groups"[:31]] = GR
        top = G[G["rank"] <= 10].groupby("explained_class")["feature"].apply(lambda s: ", ".join(s))
        print(f"{cfg}: folds {folds}, {cv['record_id'].nunique()} patients, {len(feats)} features, "
              f"fold-pair rank stability median rho {ST['spearman'].median():.2f}")
        for c, t in top.items():
            print(f"   top-10 [{c}]: {t}")
    with pd.ExcelWriter(rdir / "shap_summary.xlsx") as xw:
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k, index=False)


if __name__ == "__main__":
    main()
