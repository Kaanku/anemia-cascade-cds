"""
s16_synthetic.py — synthetic training cohort for the public demo app (CDS v4.3, R30)
====================================================================================

TabPFN keeps its training rows as in-context examples, so a public app cannot carry real patient
data. The app is trained on a synthetic cohort generated here, class by class, with a Gaussian
copula fitted to the development cohort (A1, first sample per patient, 863 patients):

  * 32 free variables per class: age, RBC, MCV, MCHC, RDW-SD, RDW-CV, RET#, IRF, HFR, RET-He, Delta-He,
    MicroR, MacroR, NRBC%, NEUT#, LYMPH#, MONO#, EO#, BASO#, IG#, PLT, MPV, PDW, P-LCR, FRC (fraction),
    RET-Y, RET-RBC-Y, IRF-Y and the four analytes (ferritin, iron, UIBC, LDH). MCHC and Delta-He are
    sampled (not HGB and RBC-He) because both are narrow differences/ratios of strongly correlated
    measurements: derived from their parts they come out too wide (first run: MCHC IQR doubled).
  * Marginals: the class's empirical quantile function (linear interpolation between order statistics at
    plotting positions (i - 0.5)/n; analytes from measured values only). Dependence: correlation of the
    normal scores (pairwise complete), projected to the nearest positive definite matrix.
  * The 10 analyzer identities are recomputed, never sampled: HCT = RBC x MCV / 10, HGB = MCHC x HCT / 100,
    MCH = MCV x MCHC / 100, RET% = 100 x RET# / RBC, MFR = IRF - HFR, LFR = 100 - IRF,
    RBC-He = RET-He - Delta-He, MicroR / MacroR, WBC = NEUT# + LYMPH# + MONO# + EO# + BASO#,
    NRBC# = NRBC% x WBC / 100. Draws that break a physical bound (MFR < 0, IRF > 100) or put a derived
    index outside the class's observed range (5 % margin) are redrawn.
  * Values are rounded to the analyzer's reporting resolution. Missingness patterns (platelet indices,
    differential, which analytes were measured) are copied from randomly chosen real patients of the same
    class (the pattern only, never values). Class sizes and biochemistry-measured counts equal the real ones.
  * Matrices as in the pipeline (s04): CBC + biochemistry rows = at least one analyte measured, KNN
    imputation of the remaining analytes fitted on those rows, ratio features.

Checks (reports/synthetic/):
  fidelity  per class and variable: Kolmogorov-Smirnov distance and standardised mean difference; mean
            absolute difference of the Spearman correlation matrices; a real-vs-synthetic discriminator
            (gradient boosting, 5-fold CV AUC; 0.5 = indistinguishable).
  privacy   holdout design, 5 repeats: the generator is fitted on a random half A (stratified by class);
            distance to the closest record (DCR) in A for synthetic rows and for the real holdout half B;
            share of synthetic rows closer to A than to B (0.5 = no memorisation); exact matches. Final
            generator (all 863): DCR to the nearest real patient against the real nearest-neighbour distances.
  utility   train on synthetic, test on real (TSTR) vs train on real (TRTR) with gradient boosting as a fast
            proxy (TabPFN in the app notebook): holdout halves (same 5 repeats) and the temporal cohort.
Distances: Euclidean on normal scores of the 32 free variables (reference = the real rows they are compared
with), missing coordinates handled as in sklearn's nan_euclidean_distances.

Outputs: data/synthetic/synthetic_cohort.parquet, data/synthetic/A1_FULL_CBC{,_BIO}/train.parquet (+ imputer),
         reports/synthetic/{fidelity,correlation,privacy,utility}.csv, synthetic.xlsx, summary.json
Usage:   python s16_synthetic.py --out /path/CDS_v43 [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from s04_features import (AAC, BIO, CBC_BASE, EXTRA, MX_CLASSES, LabelFreeKNNImputer, add_ratios,
                          apply_imputer, ratio_pairs)

SEED = 42
SHRINK = 0.0   # weight of the pooled within-class correlation; 0.3 and 0.5 were tried (R30): the holdout
               # closer-to-A share moved within repeat noise while fidelity worsened, so none is used
CLASSES = ["IDA", "HA", "HGB_HTZ", "NORMAL", "OAC"]
DIFF5 = ["neut_number_10_3_u_l", "lymph_number_10_3_u_l", "mono_number_10_3_u_l", "eo_number_10_3_u_l",
         "baso_number_10_3_u_l"]
FREE_CBC = ["age", "rbc_10_6_u_l", "mcv_f_l", "mchc_g_dl", "rdw_sd_fl", "rdw_cv_pct", "ret_number_10_6_l",
            "irf_pct", "hfr_pct", "ret_he_pg", "delta_he_pg", "micro_r_pct", "macro_r_pct", "nrbc_pct", *DIFF5,
            "ig_number_10_3_u_l", "plt_10_3_u_l", "mpv_fl", "pdw_fl", "p_lcr_pct", "frc_perc", "ret_y_ch",
            "ret_rbc_y_ch", "irf_y_ch"]
FREE = FREE_CBC + BIO
DERIVED = ["hct_pct", "hgb_g_d_l", "mch_pg", "ret_pct", "mfr_pct", "lfr_pct", "rbc_he_pg",
           "micro_macro_ratio", "wbc_10_3_u_l", "nrbc_number_10_3_u_l"]
RANGE_CHECKED = ["hct_pct", "hgb_g_d_l", "mch_pg", "ret_pct", "rbc_he_pg", "wbc_10_3_u_l"]
BASE = CBC_BASE + EXTRA                       # the 38 CBC model inputs (s04); + BIO for CBC_BIO
assert sorted(FREE_CBC + DERIVED) == sorted(BASE), "free + derived must cover the CBC base features"
NOT_ROUNDED = {"micro_macro_ratio"}           # computed from the rounded MicroR / MacroR (s01)


# ───────────────────────────────────────────────────────────────── generator
def resolution(s: pd.Series) -> int | None:
    v = s.dropna().to_numpy(float)
    for d in range(7):
        if np.allclose(v * 10 ** d, np.round(v * 10 ** d), atol=1e-6):
            return d
    return None


def normal_scores(x: pd.Series) -> pd.Series:
    n = x.notna().sum()
    return pd.Series(stats.norm.ppf((x.rank(method="average") - 0.5) / n), index=x.index)


def nearest_pd(r: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    r = (r + r.T) / 2
    w, v = np.linalg.eigh(r)
    r = v @ np.diag(np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(r))
    return r / np.outer(d, d)


def score_corr(x: pd.DataFrame) -> np.ndarray:
    z = pd.DataFrame({c: normal_scores(x[c]) for c in x.columns})
    r = z.corr(min_periods=10).fillna(0).to_numpy().copy()
    np.fill_diagonal(r, 1.0)
    return r


class ClassCopula:
    def __init__(self, x: pd.DataFrame, r: np.ndarray):
        self.cols = list(x.columns)
        self.sorted = {c: np.sort(x[c].dropna().to_numpy(float)) for c in self.cols}
        self.r = nearest_pd(r)
        self.chol = np.linalg.cholesky(self.r)

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        u = stats.norm.cdf(rng.standard_normal((n, len(self.cols))) @ self.chol.T)
        out = {}
        for j, c in enumerate(self.cols):
            s = self.sorted[c]
            out[c] = np.interp(u[:, j], (np.arange(len(s)) + 0.5) / len(s), s)
        return pd.DataFrame(out)


def derive(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["hct_pct"] = d["rbc_10_6_u_l"] * d["mcv_f_l"] / 10
    d["hgb_g_d_l"] = d["mchc_g_dl"] * d["hct_pct"] / 100
    d["mch_pg"] = d["mcv_f_l"] * d["mchc_g_dl"] / 100
    d["ret_pct"] = 100 * d["ret_number_10_6_l"] / d["rbc_10_6_u_l"]
    d["mfr_pct"] = d["irf_pct"] - d["hfr_pct"]
    d["lfr_pct"] = 100 - d["irf_pct"]
    d["rbc_he_pg"] = d["ret_he_pg"] - d["delta_he_pg"]
    d["micro_macro_ratio"] = d["micro_r_pct"] / d["macro_r_pct"]
    d["wbc_10_3_u_l"] = d[DIFF5].sum(axis=1)
    d["nrbc_number_10_3_u_l"] = d["nrbc_pct"] * d["wbc_10_3_u_l"] / 100
    return d


class Generator:
    """Class-wise Gaussian copula on the free variables of a real cohort table (data/model/A1 layout)."""

    def __init__(self, real: pd.DataFrame):
        self.res = {c: resolution(real[c]) for c in FREE + DERIVED if c not in NOT_ROUNDED}
        r = {k: score_corr(real.loc[real["cls"] == k, FREE]) for k in CLASSES}
        w = real["cls"].value_counts()
        pooled = sum(w[k] * r[k] for k in CLASSES) / w[CLASSES].sum()
        self.cop = {k: ClassCopula(real.loc[real["cls"] == k, FREE], (1 - SHRINK) * r[k] + SHRINK * pooled)
                    for k in CLASSES}
        self.lim = {k: {c: (g[c].min(), g[c].max()) for c in RANGE_CHECKED}
                    for k, g in real.groupby("cls")}
        self.cbc_masks = {k: g[FREE_CBC].isna().to_numpy() for k, g in real.groupby("cls")}
        bio = real[BIO].notna()
        self.bio_masks = {k: bio.loc[g.index][bio.loc[g.index].any(axis=1)].to_numpy()
                          for k, g in real.groupby("cls")}
        self.n = real["cls"].value_counts().to_dict()
        self.n_bio = real.loc[bio.any(axis=1), "cls"].value_counts().to_dict()
        self.redraws = {}

    def _round(self, d: pd.DataFrame, cols) -> pd.DataFrame:
        for c in cols:
            if c in self.res and self.res[c] is not None:
                d[c] = d[c].round(self.res[c])
        return d

    def _valid(self, d: pd.DataFrame, k: str) -> np.ndarray:
        ok = (d["mfr_pct"] >= 0) & (d["irf_pct"] <= 100) & (d["macro_r_pct"] > 0)
        for c, (lo, hi) in self.lim[k].items():
            m = 0.05 * (hi - lo)
            ok &= d[c].between(lo - m, hi + m)
        ratio_cols = ["hgb_g_d_l", "rbc_10_6_u_l", "ret_number_10_6_l", "mcv_f_l", "mchc_g_dl", "rdw_sd_fl",
                      "ret_he_pg", "irf_pct", "micro_macro_ratio"] + BIO
        ok &= (d[ratio_cols] > 0).all(axis=1)
        return ok.to_numpy()

    def sample_class(self, k: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
        rows, drawn, bad = [], 0, 0
        while sum(len(r) for r in rows) < n:
            d = self.cop[k].sample(max(2 * n, 50), rng)
            d = self._round(d, FREE)
            d = self._round(derive(d), DERIVED)
            ok = self._valid(d, k)
            drawn, bad = drawn + len(d), bad + int((~ok).sum())
            rows.append(d[ok])
        d = pd.concat(rows, ignore_index=True).iloc[:n].copy()
        self.redraws[k] = bad / drawn
        mask = self.cbc_masks[k][rng.integers(0, len(self.cbc_masks[k]), n)]   # CBC gaps (pattern only)
        d[FREE_CBC] = d[FREE_CBC].mask(mask)
        n_bio = round(self.n_bio.get(k, 0) * n / self.n[k])                      # the real share (= real count at n_k)
        bio_rows = rng.choice(n, n_bio, replace=False)                            # analytes ordered
        bmask = np.ones((n, len(BIO)), bool)
        bmask[bio_rows] = ~self.bio_masks[k][rng.integers(0, len(self.bio_masks[k]), len(bio_rows))]
        d[BIO] = d[BIO].mask(bmask)
        d.insert(0, "cls", k)
        return d

    def sample(self, rng: np.random.Generator, n: dict | None = None) -> pd.DataFrame:
        n = n or self.n
        d = pd.concat([self.sample_class(k, n[k], rng) for k in CLASSES if n.get(k, 0)], ignore_index=True)
        d = d.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
        d.insert(0, "record_id", [f"SYN-{i + 1:05d}" for i in range(len(d))])
        d["y_s1"] = d["cls"].isin(AAC).astype(int)
        d["n_bio_measured"] = d[BIO].notna().sum(axis=1)
        return d


# ───────────────────────────────────────────────────────────────── matrices (as in s04)
def matrices(train: pd.DataFrame, test: pd.DataFrame | None = None) -> dict:
    """CBC and CBC_BIO model matrices; the KNN imputer is fitted on the training rows only."""
    out = {}
    tr, _ = add_ratios(train, ratio_pairs("CBC"))
    te = add_ratios(test, ratio_pairs("CBC"))[0] if test is not None else None
    out["CBC"] = (tr, te)
    trb = train[train[BIO].notna().any(axis=1)]
    st = LabelFreeKNNImputer(k=5).fit(trb).state()
    trb = add_ratios(apply_imputer(st, trb), ratio_pairs("CBC_BIO"))[0]
    teb = None
    if test is not None:
        tb = test[test[BIO].notna().any(axis=1)]
        teb = add_ratios(apply_imputer(st, tb), ratio_pairs("CBC_BIO"))[0] if len(tb) else None
    out["CBC_BIO"] = (trb, teb)
    out["imputer"] = st
    return out


# ───────────────────────────────────────────────────────────────── checks
class NormalScoreSpace:
    """Maps the free variables to normal scores of a reference set (mid-distribution ECDF, interpolated)."""

    def __init__(self, ref: pd.DataFrame):
        self.maps = {}
        for c in FREE:
            v = ref[c].dropna().to_numpy(float)
            u, cnt = np.unique(v, return_counts=True)
            mid = (np.cumsum(cnt) - cnt / 2) / len(v)
            self.maps[c] = (u, mid, len(v))

    def __call__(self, d: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in FREE:
            u, mid, n = self.maps[c]
            p = np.interp(d[c].to_numpy(float), u, mid, left=0.5 / n, right=1 - 0.5 / n)
            p = np.clip(p, 0.5 / n, 1 - 0.5 / n)
            cols.append(np.where(d[c].isna(), np.nan, stats.norm.ppf(p)))
        return np.column_stack(cols)


def min_dist(a: np.ndarray, b: np.ndarray, same: bool = False) -> np.ndarray:
    d = nan_euclidean_distances(a, b)
    if same:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def exact_matches(s: pd.DataFrame, r: pd.DataFrame) -> int:
    key = lambda d: d[FREE_CBC].round(6).fillna(-1).astype(str).agg("|".join, axis=1)
    return int(key(s).isin(set(key(r))).sum())


def fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, crow = [], []
    cols = FREE + DERIVED
    for k in CLASSES:
        a, b = real[real["cls"] == k], syn[syn["cls"] == k]
        for c in cols:
            x, y = a[c].dropna(), b[c].dropna()
            sd = np.sqrt((x.var() + y.var()) / 2)
            rows.append({"class": k, "variable": c, "kind": "derived" if c in DERIVED else "free",
                         "n_real": len(x), "n_syn": len(y), "ks": stats.ks_2samp(x, y).statistic,
                         "smd": (y.mean() - x.mean()) / sd if sd > 0 else 0.0,
                         "median_real": x.median(), "median_syn": y.median(),
                         "iqr_real": x.quantile(.75) - x.quantile(.25), "iqr_syn": y.quantile(.75) - y.quantile(.25)})
        ra, rb = a[cols].corr("spearman", min_periods=10), b[cols].corr("spearman", min_periods=10)
        iu = np.triu_indices(len(cols), 1)
        diff = np.abs(ra.to_numpy() - rb.to_numpy())[iu]
        crow.append({"class": k, "pairs": int(np.isfinite(diff).sum()), "mean_abs_diff": np.nanmean(diff),
                     "p90_abs_diff": np.nanpercentile(diff, 90), "max_abs_diff": np.nanmax(diff)})
    return pd.DataFrame(rows), pd.DataFrame(crow)


def discriminator(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    cols = FREE + DERIVED
    x = pd.concat([real[cols], syn[cols]], ignore_index=True)
    x = pd.concat([x, pd.get_dummies(pd.concat([real["cls"], syn["cls"]], ignore_index=True)).astype(float)], axis=1)
    y = np.r_[np.zeros(len(real)), np.ones(len(syn))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED),
                          x, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return {"discriminator_auc": roc_auc_score(y, p)}


def task_list(fdir: Path) -> list[tuple[str, str, list[str]]]:
    t = []
    for sc, stages in (("CBC", ("S1", "S2", "MX")), ("CBC_BIO", ("S1", "S2"))):
        for st in stages:
            t.append((sc, st, json.loads((fdir / f"A1_FULL_{sc}" / "final" / f"features_{st}.json").read_text())))
    return t


def stage_rows(d: pd.DataFrame, st: str) -> pd.DataFrame:
    return d if st == "S1" else d[d["cls"].isin(AAC if st == "S2" else MX_CLASSES)]


def fit_eval(tr: pd.DataFrame, te: pd.DataFrame, st: str, feats: list[str]) -> dict:
    tr, te = stage_rows(tr, st), stage_rows(te, st)
    y_tr = tr["y_s1"] if st == "S1" else tr["cls"]
    y_te = te["y_s1"] if st == "S1" else te["cls"]
    if y_te.nunique() < 2:
        return {}
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(tr[feats], y_tr)
    p = m.predict_proba(te[feats])
    if st == "S2":
        lab = list(m.classes_)
        present = [c for c in lab if (y_te == c).any()]
        auc = np.mean([roc_auc_score(y_te == c, p[:, lab.index(c)]) for c in present])
        yhat = np.array(lab)[p.argmax(1)]
        return {"auc": auc, "accuracy": accuracy_score(y_te, yhat), "macro_f1": f1_score(y_te, yhat, average="macro"),
                "n_test": len(te)}
    pos = 1 if st == "S1" else "IDA"
    j = list(m.classes_).index(pos)
    return {"auc": roc_auc_score(y_te == pos, p[:, j]), "n_test": len(te)}


def utility(train_real, train_syn, test, tasks, label) -> list[dict]:
    rr, ss = matrices(train_real, test), matrices(train_syn, test)
    rows = []
    for sc, st, feats in tasks:
        for src, m in (("real", rr), ("synthetic", ss)):
            tr, te = m[sc]
            if te is None:
                continue
            r = fit_eval(tr, te, st, feats)
            if r:
                rows.append({"design": label, "scenario": sc, "stage": st, "train": src, **r})
    return rows


# ───────────────────────────────────────────────────────────────── main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    real = pd.read_parquet(a.out / "data" / "model" / "A1.parquet")
    assert len(real) == 863 and real["patient_id"].is_unique
    temporal = pd.read_parquet(a.out / "data" / "temporal" / "temporal_cohort.parquet")
    temporal = temporal[temporal["has_analyzer_record"].astype(bool)].copy()
    real["y_s1"] = real["cls"].isin(AAC).astype(int)
    temporal["y_s1"] = temporal["cls"].isin(AAC).astype(int)
    tasks = task_list(a.out / "data" / "features")
    rd, sd = a.out / "reports" / "synthetic", a.out / "data" / "synthetic"
    rd.mkdir(parents=True, exist_ok=True)

    # holdout repeats: privacy + utility
    priv, util = [], []
    for r in range(a.repeats):
        rng = np.random.default_rng(SEED + r)
        ia = real.groupby("cls", group_keys=False).apply(lambda g: g.sample(frac=0.5, random_state=SEED + r)).index
        A, B = real.loc[ia].copy(), real.drop(ia).copy()
        S = Generator(A).sample(rng)
        sp = NormalScoreSpace(A)
        zs, za, zb = sp(S), sp(A), sp(B)
        d_sa, d_sb, d_ba = min_dist(zs, za), min_dist(zs, zb), min_dist(zb, za)
        p5 = np.percentile(d_ba, 5)
        priv.append({"design": "holdout", "repeat": r, "n_A": len(A), "n_B": len(B), "n_syn": len(S),
                     "share_syn_closer_to_A": float(np.mean(d_sa < d_sb)),
                     "median_dcr_syn": float(np.median(d_sa)), "median_dcr_holdout": float(np.median(d_ba)),
                     "p5_dcr_syn": float(np.percentile(d_sa, 5)), "p5_dcr_holdout": float(p5),
                     "share_syn_below_holdout_p5": float(np.mean(d_sa < p5)),
                     "min_dcr_syn": float(d_sa.min()), "exact_matches": exact_matches(S, A)})
        util += [dict(x, repeat=r) for x in utility(A, S, B, tasks, "holdout")]
        print(f"repeat {r}: closer-to-A {priv[-1]['share_syn_closer_to_A']:.3f}, "
              f"DCR median syn {priv[-1]['median_dcr_syn']:.2f} vs holdout {priv[-1]['median_dcr_holdout']:.2f}")

    # final generator on all 863 patients
    rng = np.random.default_rng(SEED)
    gen = Generator(real)
    S = gen.sample(rng)
    sp = NormalScoreSpace(real)
    zs, zr = sp(S), sp(real)
    d_sr, d_rr = min_dist(zs, zr), min_dist(zr, zr, same=True)
    p5 = np.percentile(d_rr, 5)
    priv.append({"design": "final", "repeat": -1, "n_A": len(real), "n_B": 0, "n_syn": len(S),
                 "share_syn_closer_to_A": np.nan, "median_dcr_syn": float(np.median(d_sr)),
                 "median_dcr_holdout": float(np.median(d_rr)), "p5_dcr_syn": float(np.percentile(d_sr, 5)),
                 "p5_dcr_holdout": float(p5), "share_syn_below_holdout_p5": float(np.mean(d_sr < p5)),
                 "min_dcr_syn": float(d_sr.min()), "exact_matches": exact_matches(S, real)})
    util += [dict(x, repeat=-1) for x in utility(real, S, temporal, tasks, "temporal")]
    fid, cor = fidelity(real, S)
    disc = discriminator(real, S)

    # outputs
    sd.mkdir(parents=True, exist_ok=True)
    S.to_parquet(sd / "synthetic_cohort.parquet", index=False)
    m = matrices(S)
    for sc in ("CBC", "CBC_BIO"):
        (sd / f"A1_FULL_{sc}").mkdir(exist_ok=True)
        m[sc][0].to_parquet(sd / f"A1_FULL_{sc}" / "train.parquet", index=False)
    joblib.dump(m["imputer"], sd / "A1_FULL_CBC_BIO" / "imputer.joblib")
    P, U = pd.DataFrame(priv), pd.DataFrame(util)
    for name, df in (("fidelity", fid), ("correlation", cor), ("privacy", P), ("utility", U)):
        df.to_csv(rd / f"{name}.csv", index=False)
    with pd.ExcelWriter(rd / "synthetic.xlsx") as xw:
        for name, df in (("fidelity", fid), ("correlation", cor), ("privacy", P), ("utility", U)):
            df.to_excel(xw, sheet_name=name, index=False)
    us = (U.groupby(["design", "scenario", "stage", "train"])[["auc", "accuracy", "macro_f1"]].mean()
          .round(4).reset_index())
    summary = {
        "n_synthetic": len(S), "class_counts": S["cls"].value_counts().to_dict(),
        "n_bio_measured": int((S["n_bio_measured"] > 0).sum()), "redraw_share": gen.redraws,
        "fidelity": {"median_ks_free": float(fid.loc[fid.kind == "free", "ks"].median()),
                     "median_ks_derived": float(fid.loc[fid.kind == "derived", "ks"].median()),
                     "max_ks": float(fid["ks"].max()),
                     "worst": fid.nlargest(5, "ks")[["class", "variable", "ks"]].to_dict("records"),
                     "corr_mean_abs_diff": cor.set_index("class")["mean_abs_diff"].round(4).to_dict(), **disc},
        "privacy": P.drop(columns=["design"]).groupby(P["design"]).mean().round(4).to_dict("index"),
        "utility": us.to_dict("records"),
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
