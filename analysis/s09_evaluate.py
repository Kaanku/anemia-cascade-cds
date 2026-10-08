"""
s09_evaluate.py — Anemia cascade CDS, nested cross-validation (v4)
===================================================================

Evaluates the outer-fold predictions of the nested cross-validation and the
final models on the temporal cohort. Every outer fold is scored with the
choices frozen for it in lock/lock.json (s07, made on that fold's inner OOF
predictions only); the five folds are then pooled.

Reported (DECISIONS P1–P4, N1–N8, X1–X4):
    P1  Stage 1, Stage 2 and end-to-end performance
    P2  accuracy–coverage operating points (80/85/90/95 %), HIGH/MEDIUM/LOW zones
    P3  CBC vs CBC + biochemistry on patients with measured biochemistry; cascade
    P4  IDA vs HGB HTZ: MX model vs seven classic indices, all and MCV < 80 fL
    L1  calibration of displayed probabilities; L5 conformal prediction sets
    D10' A1 vs A2 (paired, first samples) · N6 FULL vs T13 and AutoGluon vs TabPFN
    N4  learning curves (per engine, R15) · N8 temporal validation of the final models

Engines (R12–R15): TabPFN-3.5 is the main model; AutoGluon is the pre-specified
comparator. Lock v2 covers both on all 16 configurations.

Unit (D12): the patient — each patient's first sample. A2 models are also
evaluated on all their samples (secondary). 95 % CIs: percentile bootstrap over
patients (2,000 resamples, seed 42; all samples of a patient move together);
single AUCs with DeLong CIs on first samples and with the patient-cluster
bootstrap on all samples (R9); paired AUC comparisons by the DeLong test;
paired accuracy comparisons by the exact McNemar test; Holm correction across
the seven index comparisons. Tests assume independent rows, so on all samples
(repeat samples of a patient) they are not computed (R9).

Also reported (R9): how far the inner-OOF confidence of each configuration
differs from its outer-fold confidence (confidence_shift), because the
probability cut-offs locked on OOF only carry over when the two agree.

Safeguards: refuses to run unless lock.json has no missing fold-set and every
OOF file still has the hash recorded at locking; every opening of outer-fold or
temporal predictions is logged in lock/unlock_log.json and a re-run is only
allowed with the identical lock.json.

Outputs:
    <OUT>/reports/evaluation.xlsx          long table + wide tables per block
    <OUT>/reports/figures_data/*.parquet   the same tables plus curve data (ROC on a fixed
                                           FPR grid, accuracy–coverage, reliability before and
                                           after calibration) for the HTML report (s10)

Usage:
    python s09_evaluate.py --out "/path/to/Kaan/CDS_v4" [--boot 2000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, rankdata

import s07_lock as L
from s08_indices import INDICES

AAC = ["IDA", "HA", "HGB_HTZ", "NORMAL"]
S2 = L.S2_CLASSES                               # HA, HGB_HTZ, IDA, NORMAL
C5 = S2 + ["OAC"]
OUTER = [f"outer{k}" for k in range(5)]
B_DEFAULT, SEED_BOOT, SEED_APS = 2000, 42, 123
Z975 = norm.ppf(0.975)


# ═══════════════════════════════════════════════════════════ statistics
class Boot:
    """Patient-level bootstrap index generator (all rows of a patient together)."""

    def __init__(self, patients: np.ndarray, b: int, seed: int = SEED_BOOT):
        codes, uniq = pd.factorize(pd.Series(patients))
        n_pat = len(uniq)
        counts = np.bincount(codes, minlength=n_pat)
        self.one = bool((counts == 1).all())
        order = np.argsort(codes, kind="stable")
        if self.one:
            self.pos = np.empty(n_pat, dtype=int)
            self.pos[codes] = np.arange(len(codes))
        else:
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            self.rows = [order[s:s + c] for s, c in zip(starts, counts)]
        self.draws = np.random.default_rng(seed).integers(0, n_pat, size=(b, n_pat))

    def __iter__(self):
        for d in self.draws:
            yield self.pos[d] if self.one else np.concatenate([self.rows[p] for p in d])


def with_ci(fn, arrays: dict, patients: np.ndarray, b: int) -> dict:
    point = fn(**arrays)
    reps = {k: [] for k in point}
    for idx in Boot(patients, b):
        r = fn(**{k: v[idx] for k, v in arrays.items()})
        for k in point:
            reps[k].append(r.get(k, np.nan))
    out = {}
    for k, v in point.items():
        a = np.asarray(reps[k], dtype=float)
        lo, hi = (np.nanpercentile(a, [2.5, 97.5]) if np.isfinite(a).any() else (np.nan, np.nan))
        out[k] = (float(v) if v is not None else np.nan, float(lo), float(hi))
    return out


def auc_binary(y, s) -> float:
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x, kind="mergesort")
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong (Sun & Xu 2014). scores: (k, n). Returns AUCs (k,) and covariance (k, k)."""
    y = np.asarray(y).astype(bool)
    X, Y = scores[:, y], scores[:, ~y]
    m, n = X.shape[1], Y.shape[1]
    tx = np.array([_midrank(r) for r in X])
    ty = np.array([_midrank(r) for r in Y])
    tz = np.array([_midrank(np.concatenate([a, b])) for a, b in zip(X, Y)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(y, s) -> tuple[float, float, float]:
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return (np.nan, np.nan, np.nan)
    a, c = delong(y, np.asarray(s, float)[None, :])
    se = float(np.sqrt(max(c[0, 0], 0)))
    return float(a[0]), max(0.0, float(a[0]) - Z975 * se), min(1.0, float(a[0]) + Z975 * se)


def delong_test(y, s1, s2) -> tuple[float, float, float, float]:
    """Paired DeLong test. Returns (AUC1, AUC2, difference, p); NaN when one class is absent."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():                     # e.g. Stage 1 in the temporal cohort (no OAC)
        return (np.nan,) * 4
    a, c = delong(y, np.vstack([np.asarray(s1, float), np.asarray(s2, float)]))
    var = c[0, 0] + c[1, 1] - 2 * c[0, 1]
    z = (a[0] - a[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(a[0]), float(a[1]), float(a[0] - a[1]), float(2 * norm.sf(abs(z)))


def mcnemar(c1: np.ndarray, c2: np.ndarray) -> tuple[int, int, float]:
    b, c = int(np.sum(c1 & ~c2)), int(np.sum(~c1 & c2))
    return b, c, (1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue))


def holm(p) -> np.ndarray:
    """Holm-adjusted p-values over the finite entries; NaN stays NaN (test not computed)."""
    p = np.asarray(p, float)
    adj = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    run = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok], kind="stable")]):
        run = max(run, (len(ok) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ═══════════════════════════════════════════════════════════ metrics
def _div(a, b):
    return float(a) / float(b) if b else np.nan


def binary_metrics(y, s, t) -> dict:
    """Threshold t may be an array (fold-specific thresholds)."""
    y = np.asarray(y).astype(bool)
    yh = np.asarray(s) >= np.asarray(t)
    tp, tn = int(np.sum(yh & y)), int(np.sum(~yh & ~y))
    fp, fn = int(np.sum(yh & ~y)), int(np.sum(~yh & y))
    f1p, f1n = _div(2 * tp, 2 * tp + fp + fn), _div(2 * tn, 2 * tn + fn + fp)
    return {"accuracy": _div(tp + tn, len(y)), "sensitivity": _div(tp, tp + fn), "specificity": _div(tn, tn + fp),
            "ppv": _div(tp, tp + fp), "npv": _div(tn, tn + fn), "macro_f1": float(np.nanmean([f1p, f1n]))}


def label_metrics(y, yh, names) -> dict:
    k = len(names)
    cm = np.bincount(np.asarray(y) * k + np.asarray(yh), minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    sup, prd = cm.sum(1), cm.sum(0)
    rec = np.where(sup > 0, tp / np.maximum(sup, 1), np.nan)
    prec = np.where(prd > 0, tp / np.maximum(prd, 1), 0.0)
    den = prec + np.nan_to_num(rec)
    f1 = np.where(sup > 0, np.where(den > 0, 2 * prec * np.nan_to_num(rec) / np.where(den > 0, den, 1), 0.0), np.nan)
    out = {"accuracy": float(tp.sum() / len(y)), "macro_f1": float(np.nanmean(f1)),
           "balanced_accuracy": float(np.nanmean(rec))}
    for i, n in enumerate(names):
        out[f"recall_{n}"], out[f"precision_{n}"], out[f"f1_{n}"] = float(rec[i]), float(prec[i]), float(f1[i])
    return out


def multiclass_metrics(y, P) -> dict:
    out = label_metrics(y, P.argmax(1), S2)
    aucs = [auc_binary(y == i, P[:, i]) for i in range(len(S2))]
    out["macro_auc"] = float(np.nanmean(aucs))
    for i, n in enumerate(S2):
        out[f"auc_{n}"] = aucs[i]
    return out


def ece_brier(P, y, binary: bool) -> dict:
    return {k: v for k, v in L.calib_metrics(P, y, binary).items()}


ROC_GRID = np.round(np.linspace(0, 1, 101), 2)


def roc_points(y, s) -> pd.DataFrame:
    """ROC curve on a fixed false-positive-rate grid (upper envelope), for the report figures."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    if y.all() or (~y).all():
        return pd.DataFrame(columns=["fpr", "tpr"])
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    last = np.r_[ss[1:] != ss[:-1], True]                   # one point per distinct score
    tpr = np.r_[0.0, np.cumsum(ys)[last] / ys.sum()]
    fpr = np.r_[0.0, np.cumsum(~ys)[last] / (~ys).sum()]
    env = np.array([tpr[fpr <= f + 1e-12].max() for f in ROC_GRID])
    return pd.DataFrame({"fpr": ROC_GRID, "tpr": np.round(env, 4)})


def reliability(conf, hit, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_predicted": float(conf[m].mean()) if m.any() else np.nan,
                     "observed": float(hit[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def aps_sets(P: np.ndarray, qhat: np.ndarray, u: np.ndarray) -> np.ndarray:
    order = np.argsort(-P, axis=1, kind="stable")
    Ps = np.take_along_axis(P, order, axis=1)
    score = np.cumsum(Ps, axis=1) - Ps + u[:, None] * Ps
    inc = score <= qhat[:, None]
    inc[:, 0] = True                                          # top class always included
    sets = np.zeros_like(P, dtype=bool)
    np.put_along_axis(sets, order, inc, axis=1)
    return sets


# ═══════════════════════════════════════════════════════════ data
class Data:
    def __init__(self, out: Path, lock: dict, cals: dict):
        self.out, self.lock, self.cals = out, lock, cals

    def entry(self, engine, cfg, fset) -> dict:
        return self.lock["engines"][engine][cfg][fset]

    def oof(self, engine: str, cfg: str) -> pd.DataFrame:
        """Inner-OOF predictions of the five outer fold-sets (the rows the lock was made on)."""
        root = L.ENGINES[engine][0]
        return pd.concat([pd.read_parquet(self.out / root / cfg / f"{f}_oof.parquet").assign(fold_set=f)
                          for f in OUTER], ignore_index=True)

    def pooled(self, engine: str, cfg: str) -> pd.DataFrame:
        """Outer-fold predictions of one configuration with each fold's frozen choices attached."""
        root = L.ENGINES[engine][0]
        st = cfg[-2:]
        frames = []
        for f in OUTER:
            d = pd.read_parquet(self.out / root / cfg / f"{f}_eval.parquet")
            frames.append(self._attach(d, engine, cfg, f, st))
        return pd.concat(frames, ignore_index=True)

    def temporal(self, engine: str, cfg: str) -> pd.DataFrame | None:
        p = self.out / L.ENGINES[engine][0] / cfg / "final_temporal.parquet"
        if not p.exists():
            return None
        return self._attach(pd.read_parquet(p), engine, cfg, "final", cfg[-2:])

    def _attach(self, d: pd.DataFrame, engine, cfg, f, st) -> pd.DataFrame:
        e = self.entry(engine, cfg, f)
        cal = self.cals[(engine, cfg, f)]["model"]
        d = d.copy()
        d["fold_set"] = f
        if st == "S1":
            d["s"] = d["prob_1"].astype(float)
            d["s_cal"] = cal.transform(d["s"].to_numpy())
        elif st == "MX":
            d["s"] = d[f"prob_{L.MX_POSITIVE}"].astype(float)
            d["s_cal"] = cal.transform(d["s"].to_numpy())
        else:
            P = d[[f"prob_{c}" for c in S2]].to_numpy(float)
            Pc = cal.transform(P)
            for i, c in enumerate(S2):
                d[f"cal_{c}"] = Pc[:, i]
            d["high"] = np.inf if e["high_cutoff"] is None else e["high_cutoff"]
            for a in L.ALPHAS:
                d[f"qhat_{a}"] = e["conformal_qhat"][str(a)]
        if st in ("S1", "MX"):
            d["thr"] = e["threshold"]
            d["thr_youden"] = e["threshold_youden"]
        if st in ("S2", "MX"):
            for tgt, op in e["operating_points"].items():
                d[f"op_{tgt}"] = np.inf if op is None else op["cutoff"]
        return d


def P2(d: pd.DataFrame) -> np.ndarray:
    return d[[f"prob_{c}" for c in S2]].to_numpy(float)


def y2(d: pd.DataFrame) -> np.ndarray:
    return d["cls"].map({c: i for i, c in enumerate(S2)}).to_numpy()


# ═══════════════════════════════════════════════════════════ evaluation
class Evaluator:
    def __init__(self, data: Data, b: int):
        self.D, self.b = data, b
        self.rows: list[dict] = []
        self.tables: dict[str, list[pd.DataFrame]] = {}

    def add(self, res: dict, **meta):
        for k, v in res.items():
            self.rows.append({**meta, "metric": k, "value": v[0], "ci_low": v[1], "ci_high": v[2]})

    def table(self, name: str, df: pd.DataFrame, **meta):
        for k, v in meta.items():
            df.insert(0, k, v)
        self.tables.setdefault(name, []).append(df)

    def auc_ci(self, y, s, pat, unit: str) -> tuple[float, float, float]:
        """DeLong on first samples; on all samples the rows of a patient are not independent,
        so the CI comes from the patient-cluster bootstrap instead (R9)."""
        if unit == "all samples":
            return with_ci(lambda y, s: {"auc": auc_binary(y, s)}, dict(y=np.asarray(y), s=np.asarray(s, float)),
                           np.asarray(pat), self.b)["auc"]
        return delong_ci(y, s)

    # ── P1 ───────────────────────────────────────────────────────────
    def stage1(self, d, **meta):
        y, s, pat = d["y_s1"].to_numpy(int), d["s"].to_numpy(), d["patient_id"].to_numpy()
        fn = lambda y, s, t, ty: {**binary_metrics(y, s, t),
                                  **{f"youden_{k}": v for k, v in binary_metrics(y, s, ty).items()}}
        res = with_ci(fn, dict(y=y, s=s, t=d["thr"].to_numpy(), ty=d["thr_youden"].to_numpy()), pat, self.b)
        res["auc"] = self.auc_ci(y, s, pat, meta["unit"])
        self.add(res, block="stage1", population="all patients", n=len(d), **meta)
        self.table("roc", roc_points(y, s).assign(curve="AAC vs OAC", subgroup="all"), **meta)

    def stage2(self, d, **meta):
        d = d[d["cls"].isin(AAC)]
        res = with_ci(lambda y, P: multiclass_metrics(y, P), dict(y=y2(d), P=P2(d)),
                      d["patient_id"].to_numpy(), self.b)
        self.add(res, block="stage2", population="true AAC", n=len(d), **meta)
        for i, c in enumerate(S2):                                                  # one-vs-rest curves
            self.table("roc", roc_points(y2(d) == i, P2(d)[:, i]).assign(curve=c, subgroup="all"), **meta)
        cm = pd.crosstab(pd.Categorical(d["cls"], S2), pd.Categorical(np.array(S2)[P2(d).argmax(1)], S2),
                         dropna=False)
        self.table("confusion_stage2", cm.reset_index().rename(columns={"row_0": "true"}), **meta)

    def end_to_end(self, d1, d2, **meta):
        m = d1[["record_id", "patient_id", "cls", "s", "thr"]].merge(
            d2[["record_id"] + [f"prob_{c}" for c in S2]], on="record_id", validate="one_to_one")
        y5 = m["cls"].map({c: i for i, c in enumerate(C5)}).to_numpy()
        fn = lambda y5, s, t, P: label_metrics(y5, np.where(s >= t, P.argmax(1), 4), C5)
        res = with_ci(fn, dict(y5=y5, s=m["s"].to_numpy(), t=m["thr"].to_numpy(), P=P2(m)),
                      m["patient_id"].to_numpy(), self.b)
        self.add(res, block="end_to_end", population="all patients", n=len(m), **meta)
        return m

    # ── P2 ───────────────────────────────────────────────────────────
    def selective(self, d, st, **meta):
        if st == "S2":
            d = d[d["cls"].isin(AAC)]
            P = P2(d)
            conf, hit = P.max(1), P.argmax(1) == y2(d)
        else:
            d = d[d["cls"].isin(["IDA", "HGB_HTZ"])]
            s = d["s"].to_numpy()
            conf, hit = np.maximum(s, 1 - s), (s >= 0.5) == (d["cls"] == L.MX_POSITIVE).to_numpy()
        pat = d["patient_id"].to_numpy()
        for tgt in [f"{t:.2f}" for t in L.TARGETS]:
            cut = d[f"op_{tgt}"].to_numpy()
            fn = lambda conf, hit, cut: {"coverage": float(np.mean(conf >= cut)),
                                         "accuracy_retained": (float(hit[conf >= cut].mean())
                                                               if (conf >= cut).any() else np.nan)}
            res = with_ci(fn, dict(conf=conf, hit=hit, cut=cut), pat, self.b)
            res["n_retained"] = (float((conf >= cut).sum()),) * 3
            res["folds_with_cutoff"] = (float(np.isfinite(d.groupby("fold_set")[f"op_{tgt}"].first()).sum()),) * 3
            self.add(res, block=f"selective_{st}", population=f"target {tgt}", n=len(d), **meta)
        order = np.argsort(-conf, kind="stable")                       # threshold-free curve
        cum = np.cumsum(hit[order])
        n = len(conf)
        step = max(int(np.ceil(0.02 * n)), 5)          # 2 % grid, never fewer than 5 patients per step (R9)
        ks = np.unique(np.r_[np.arange(step, n, step), n])
        curve = pd.DataFrame({"coverage": ks / n, "accuracy": cum[ks - 1] / ks, "n_decided": ks})
        self.table(f"coverage_curve_{st}", curve, **meta)
        if st == "S2":
            hi = d["high"].to_numpy()
            zone = np.where(conf < L.LOW_CUTOFF, "LOW", np.where(conf >= hi, "HIGH", "MEDIUM"))
            z = pd.DataFrame({"zone": zone, "hit": hit}).groupby("zone")["hit"].agg(["size", "mean"])
            z = z.rename(columns={"size": "n", "mean": "accuracy"}).reindex(["HIGH", "MEDIUM", "LOW"]).reset_index()
            z["n"] = z["n"].fillna(0).astype(int)                       # empty zone = 0 patients, not NaN
            z["share"] = z["n"] / len(d)
            self.table("zones_S2", z, **meta)

    # ── L1 / L5 ──────────────────────────────────────────────────────
    def calibration(self, d, st, **meta):
        if st == "S2":
            d = d[d["cls"].isin(AAC)]
            y = y2(d)
            raw, cal = P2(d), d[[f"cal_{c}" for c in S2]].to_numpy(float)
            conf, hit = cal.max(1), cal.argmax(1) == y
        else:
            if st == "MX":
                d = d[d["cls"].isin(["IDA", "HGB_HTZ"])]
                y = (d["cls"] == L.MX_POSITIVE).to_numpy(int)
            else:
                y = d["y_s1"].to_numpy(int)
            raw, cal = d["s"].to_numpy(), d["s_cal"].to_numpy()
            conf, hit = cal, y.astype(float)
        binary = st != "S2"
        res = {f"uncalibrated_{k}": (v, np.nan, np.nan) for k, v in ece_brier(raw, y, binary).items()}
        res.update({f"calibrated_{k}": (v, np.nan, np.nan) for k, v in ece_brier(cal, y, binary).items()})
        self.add(res, block=f"calibration_{st}", population="evaluated", n=len(d), **meta)
        if binary:
            conf_raw, hit_raw = raw, y.astype(float)
        else:
            conf_raw, hit_raw = raw.max(1), raw.argmax(1) == y
        self.table(f"reliability_{st}", pd.concat(
            [reliability(conf, hit).assign(version="calibrated"),
             reliability(conf_raw, hit_raw).assign(version="uncalibrated")], ignore_index=True), **meta)

    def conformal(self, d, **meta):
        d = d[d["cls"].isin(AAC)].sort_values("record_id")
        P, y = P2(d), y2(d)
        u = np.random.default_rng(SEED_APS).uniform(size=len(d))
        for a in L.ALPHAS:
            sets = aps_sets(P, d[f"qhat_{a}"].to_numpy(), u)
            cover = sets[np.arange(len(y)), y]
            size = sets.sum(1)
            res = {"coverage": (float(cover.mean()), np.nan, np.nan), "mean_set_size": (float(size.mean()), np.nan, np.nan),
                   "singleton_share": (float((size == 1).mean()), np.nan, np.nan),
                   "singleton_accuracy": (float(cover[size == 1].mean()) if (size == 1).any() else np.nan, np.nan, np.nan)}
            self.add(res, block="conformal", population=f"alpha {a}", n=len(d), **meta)

    # ── R9: does the OOF confidence the lock was made on match the outer-fold confidence? ──
    def confidence_shift(self, cfg: str, **meta):
        engine, st = meta["engine"], cfg[-2:]
        rows = []
        for part, d in (("inner OOF", self.D.oof(engine, cfg)), ("outer fold", self.D.pooled(engine, cfg))):
            d = d[d["is_index_sample"].astype(bool)]
            if st == "S2":
                d = d[d["cls"].isin(AAC)]
                conf = P2(d).max(1)
            else:
                if st == "MX":
                    d = d[d["cls"].isin(["IDA", "HGB_HTZ"])]
                s = d[f"prob_{L.MX_POSITIVE}" if st == "MX" else "prob_1"].to_numpy(float)
                conf = np.maximum(s, 1 - s)
            row = {"part": part, "n": len(d), "p50": float(np.median(conf)), "p90": float(np.quantile(conf, 0.9)),
                   "share_ge_0.95": float(np.mean(conf >= 0.95))}
            if st == "S2":                                     # share at or above each fold's HIGH cut-off
                cut = d["fold_set"].map({f: (self.D.entry(engine, cfg, f)["high_cutoff"] or np.inf) for f in OUTER})
                row["share_high"] = float(np.mean(conf >= cut.to_numpy(float)))
            rows.append(row)
        self.table("confidence_shift", pd.DataFrame(rows), **meta)

    # ── P4 ───────────────────────────────────────────────────────────
    def mx_vs_indices(self, dm, idx, **meta):
        d = dm[dm["cls"].isin(["IDA", "HGB_HTZ"])].merge(idx, on="record_id", how="left", validate="one_to_one")
        for sub, m in [("all", np.ones(len(d), bool)), ("MCV<80", (d["mcv_f_l"] < 80).to_numpy())]:
            x = d[m]
            y = (x["cls"] == L.MX_POSITIVE).to_numpy(int)
            if len(x) < 10 or y.min() == y.max():
                continue
            pat = x["patient_id"].to_numpy()
            independent = meta["unit"] != "all samples"                 # tests need independent rows (R9)
            res = with_ci(lambda y, s, t: binary_metrics(y, s, t),
                          dict(y=y, s=x["s"].to_numpy(), t=x["thr"].to_numpy()), pat, self.b)
            res["auc"] = self.auc_ci(y, x["s"].to_numpy(), pat, meta["unit"])
            self.add(res, block="mx", population=f"{sub} (model)", n=len(x), **meta)
            self.table("roc", roc_points(y, x["s"].to_numpy()).assign(curve="model", subgroup=sub), **meta)
            comp = []
            for k, (name, _, cut, _) in INDICES.items():
                v = x[f"idx_{k}"].to_numpy(float)
                ok = np.isfinite(v)
                if ok.sum() < 10:
                    continue
                yy, ss, vv = y[ok], x["s"].to_numpy()[ok], v[ok]
                self.table("roc", roc_points(yy, -vv).assign(curve=name, subgroup=sub), **meta)
                r = with_ci(lambda y, s: binary_metrics(y, s, 0.5),
                            dict(y=yy, s=(vv < cut).astype(float)), pat[ok], self.b)
                r["auc"] = self.auc_ci(yy, -vv, pat[ok], meta["unit"])
                self.add(r, block="mx", population=f"{sub} ({name})", n=int(ok.sum()), **meta)
                a1, a2, diff, p = delong_test(yy, ss, -vv)
                c_model = ((ss >= x["thr"].to_numpy()[ok]).astype(int) == yy)
                c_index = ((vv < cut).astype(int) == yy)
                b_, c_, pm = mcnemar(c_model, c_index)
                if not independent:
                    p = pm = np.nan
                comp.append({"subgroup": sub, "index": name, "n": int(ok.sum()), "auc_model": a1, "auc_index": a2,
                             "auc_difference": diff, "p_delong": p, "acc_model": c_model.mean(),
                             "acc_index": c_index.mean(), "model_only_correct": b_, "index_only_correct": c_,
                             "p_mcnemar": pm})
            if comp:
                c = pd.DataFrame(comp)
                c["p_delong_holm"] = holm(c["p_delong"])
                c["p_mcnemar_holm"] = holm(c["p_mcnemar"])
                self.table("mx_vs_indices", c, **meta)

    # ── P3: cascade (L6) ─────────────────────────────────────────────
    def cascade(self, c1, c2, b1, b2, **meta):
        """c* = CBC S1/S2 frames, b* = CBC_BIO S1/S2 frames (same analysis, first samples)."""
        keep = ["record_id", "patient_id", "cls"]
        m = (c1[keep + ["s", "thr"]].rename(columns={"s": "s1c", "thr": "t1c"})
             .merge(c2[["record_id", "high"] + [f"prob_{c}" for c in S2]], on="record_id")
             .merge(b1[["record_id", "s", "thr"]].rename(columns={"s": "s1b", "thr": "t1b"}), on="record_id")
             .merge(b2[["record_id"] + [f"prob_{c}" for c in S2]].rename(
                 columns={f"prob_{c}": f"bio_{c}" for c in S2}), on="record_id"))
        y5 = m["cls"].map({c: i for i, c in enumerate(C5)}).to_numpy()
        Pc = m[[f"prob_{c}" for c in S2]].to_numpy(float)
        Pb = m[[f"bio_{c}" for c in S2]].to_numpy(float)

        def fn(y5, s1c, t1c, Pc, high, s1b, t1b, Pb):
            tier1 = (s1c >= t1c) & (Pc.max(1) >= high)
            pred_bio = np.where(s1b >= t1b, Pb.argmax(1), 4)
            pred = np.where(tier1, Pc.argmax(1), pred_bio)
            pred_cbc = np.where(s1c >= t1c, Pc.argmax(1), 4)
            return {"tier1_share": float(tier1.mean()),
                    "tier1_accuracy": float((pred[tier1] == y5[tier1]).mean()) if tier1.any() else np.nan,
                    "cascade_accuracy": float((pred == y5).mean()),
                    "cbc_bio_accuracy": float((pred_bio == y5).mean()),
                    "cbc_only_accuracy": float((pred_cbc == y5).mean()),
                    "cascade_macro_f1": label_metrics(y5, pred, C5)["macro_f1"],
                    "cbc_bio_macro_f1": label_metrics(y5, pred_bio, C5)["macro_f1"]}
        arrays = dict(y5=y5, s1c=m["s1c"].to_numpy(), t1c=m["t1c"].to_numpy(), Pc=Pc, high=m["high"].to_numpy(),
                      s1b=m["s1b"].to_numpy(), t1b=m["t1b"].to_numpy(), Pb=Pb)
        res = with_ci(fn, arrays, m["patient_id"].to_numpy(), self.b)
        tier1 = (arrays["s1c"] >= arrays["t1c"]) & (Pc.max(1) >= arrays["high"])
        pred = np.where(tier1, Pc.argmax(1), np.where(arrays["s1b"] >= arrays["t1b"], Pb.argmax(1), 4))
        pred_bio = np.where(arrays["s1b"] >= arrays["t1b"], Pb.argmax(1), 4)
        b_, c_, p = mcnemar(pred == y5, pred_bio == y5)
        res["p_mcnemar_cascade_vs_cbc_bio"] = (p, np.nan, np.nan)
        self.add(res, block="cascade", population="measured biochemistry", n=len(m), **meta)

    # ── paired comparisons ───────────────────────────────────────────
    def compare(self, name, da, db, st, **meta):
        """Paired comparison of two models on the same rows (first samples)."""
        ka = da.set_index("record_id")
        kb = db.set_index("record_id")
        ids = ka.index.intersection(kb.index)
        a, b = ka.loc[ids], kb.loc[ids]
        row = {"comparison": name, "stage": st, "n": len(ids)}
        if st in ("S1", "MX"):
            if st == "MX":
                m = a["cls"].isin(["IDA", "HGB_HTZ"]).to_numpy()
                a, b = a[m], b[m]
                y = (a["cls"] == L.MX_POSITIVE).to_numpy(int)
            else:
                y = a["y_s1"].to_numpy(int)
            row.update(zip(["auc_a", "auc_b", "auc_diff", "p_delong"], delong_test(y, a["s"], b["s"])))
            ca = (a["s"].to_numpy() >= a["thr"].to_numpy()).astype(int) == y
            cb = (b["s"].to_numpy() >= b["thr"].to_numpy()).astype(int) == y
        else:
            m = a["cls"].isin(AAC).to_numpy()
            a, b = a[m], b[m]
            y = y2(a)
            row["macro_auc_a"] = multiclass_metrics(y, P2(a))["macro_auc"]
            row["macro_auc_b"] = multiclass_metrics(y, P2(b))["macro_auc"]
            ca, cb = P2(a).argmax(1) == y, P2(b).argmax(1) == y
        row["n"] = len(y)
        row["acc_a"], row["acc_b"] = float(ca.mean()), float(cb.mean())
        row["a_only_correct"], row["b_only_correct"], row["p_mcnemar"] = mcnemar(ca, cb)
        self.table("comparisons", pd.DataFrame([row]), **meta)


# ═══════════════════════════════════════════════════════════ learning curve
def learning_curve(out: Path) -> pd.DataFrame:
    """N4, per engine (R15): AutoGluon predictions_lc/, TabPFN-3.5 predictions_lc_tabpfn35/."""
    rows = []
    for engine, root in L.LC_ROOTS.items():
        rows += _learning_curve_engine(out / root, engine)
    return pd.DataFrame(rows)


def _learning_curve_engine(root: Path, engine: str) -> list[dict]:
    rows = []
    for p in sorted(root.glob("*/*_eval.parquet")):
        a, st = p.parent.name.split("_")
        fset, frac = p.name.replace("_eval.parquet", "").split("_")
        for part, f in [("outer fold", p), ("inner OOF", p.with_name(p.name.replace("_eval", "_oof")))]:
            d = pd.read_parquet(f)
            d = d[d["is_index_sample"].astype(bool)]
            if st == "S1":
                val = auc_binary(d["y_s1"], d["prob_1"])
                metric = "AUC"
            elif st == "MX":
                d = d[d["cls"].isin(["IDA", "HGB_HTZ"])]
                val = auc_binary(d["cls"] == L.MX_POSITIVE, d[f"prob_{L.MX_POSITIVE}"])
                metric = "AUC"
            else:
                d = d[d["cls"].isin(AAC)]
                val = multiclass_metrics(y2(d), P2(d))["macro_auc"]
                metric = "macro AUC"
            rows.append({"engine": engine, "analysis": a, "stage": st, "outer_fold": fset, "fraction": float(frac),
                         "part": part, "metric": metric, "value": val, "n": len(d)})
    return rows


# ═══════════════════════════════════════════════════════════ main
def guard_and_log(out: Path, b: int) -> tuple[dict, dict]:
    lp = out / "lock" / "lock.json"
    lock = json.loads(lp.read_text())
    assert not lock["missing"], f"lock is not final: {len(lock['missing'])} fold-sets missing"
    for rel, h in lock["files_read_sha256"].items():
        assert L.sha256(out / rel) == h, f"OOF file changed after locking: {rel}"
    log_p = out / "lock" / "unlock_log.json"
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "lock_sha256": L.sha256(lp), "bootstrap": b}
    log = json.loads(log_p.read_text()) if log_p.exists() else {"runs": []}
    if log["runs"]:
        assert log["runs"][0]["lock_sha256"] == entry["lock_sha256"], "lock.json changed after evaluation"
    log["runs"].append(entry)
    log_p.write_text(json.dumps(log, indent=1))
    return lock, joblib.load(out / "lock" / "calibrators.joblib")


def build(out: Path, b: int) -> None:
    lock, cals = guard_and_log(out, b)
    D = Data(out, lock, cals)
    E = Evaluator(D, b)
    idx_dev = pd.read_parquet(out / "data" / "indices" / "indices_dev.parquet")
    idx_tem = pd.read_parquet(out / "data" / "indices" / "indices_temporal.parquet")
    idx_cols = ["record_id", "mcv_f_l"] + [f"idx_{k}" for k in INDICES]
    engines = [e for e in L.ENGINES if lock["engines"].get(e)]
    t0 = time.time()

    first = lambda d: d[d["is_index_sample"].astype(bool)]
    cache: dict = {}
    for engine in engines:
        for cfg in lock["engines"][engine]:
            cache[(engine, cfg)] = D.pooled(engine, cfg)
            cache[(engine, cfg, "temporal")] = D.temporal(engine, cfg)

    def meta(engine, cfg, unit, set_name="nested CV"):
        a, fs = cfg.split("_")[0], cfg.split("_")[1]
        sc = "CBC_BIO" if "_CBC_BIO_" in cfg else "CBC"
        return dict(set=set_name, unit=unit, engine=engine, analysis=a, feature_set=fs, scenario=sc, stage=cfg[-2:])

    for (engine, *rest), d in list(cache.items()):
        if d is None:
            continue
        cfg = rest[0]
        is_temp = len(rest) > 1
        st = cfg[-2:]
        a, fs = cfg.split("_")[0], cfg.split("_")[1]
        if is_temp and fs == "FULL":
            d = d[d["has_analyzer_record"].astype(bool)]                       # N8
        units = [("first sample", first(d))] + ([("all samples", d)] if a == "A2" and not is_temp else [])
        for unit, x in units:
            m = meta(engine, cfg, unit, "temporal" if is_temp else "nested CV")
            if st == "S1":
                E.stage1(x, **m)
                E.calibration(x, "S1", **m)
            elif st == "S2":
                E.stage2(x, **m)
                E.selective(x, "S2", **m)
                E.calibration(x, "S2", **m)
                E.conformal(x, **m)
            else:
                E.selective(x, "MX", **m)
                E.calibration(x, "MX", **m)
                E.mx_vs_indices(x, (idx_tem if is_temp else idx_dev)[idx_cols], **m)
        print(f"  {engine} {cfg}{' temporal' if is_temp else ''} ({time.time() - t0:.0f} s)", flush=True)

    for engine in engines:                                                          # R9
        for cfg in lock["engines"][engine]:
            E.confidence_shift(cfg, **meta(engine, cfg, "first sample"))

    # end-to-end, cascade and paired comparisons (first samples)
    for engine in engines:
        cfgs = lock["engines"][engine]
        for a in ("A1", "A2"):
            for fs, sc in (("FULL", "CBC"), ("FULL", "CBC_BIO"), ("T13", "CBC")):
                c1, c2 = f"{a}_{fs}_{sc}_S1", f"{a}_{fs}_{sc}_S2"
                if c1 not in cfgs:
                    continue
                for tag in ("", "temporal"):
                    d1 = cache[(engine, c1, "temporal")] if tag else cache[(engine, c1)]
                    d2 = cache[(engine, c2, "temporal")] if tag else cache[(engine, c2)]
                    if d1 is None or d2 is None:
                        continue
                    if tag and fs == "FULL":
                        d1, d2 = d1[d1["has_analyzer_record"].astype(bool)], d2[d2["has_analyzer_record"].astype(bool)]
                    E.end_to_end(first(d1), first(d2), **meta(engine, c1, "first sample", tag or "nested CV"))
            cb = [f"{a}_FULL_CBC_S1", f"{a}_FULL_CBC_S2", f"{a}_FULL_CBC_BIO_S1", f"{a}_FULL_CBC_BIO_S2"]
            if all(c in cfgs for c in cb):
                for tag in ("", "temporal"):
                    fr = [cache[(engine, c, "temporal")] if tag else cache[(engine, c)] for c in cb]
                    if any(f is None for f in fr):
                        continue
                    if tag:
                        fr = [f[f["has_analyzer_record"].astype(bool)] for f in fr]
                    bio_ids = set(first(fr[2])["record_id"])
                    fr = [first(f)[first(f)["record_id"].isin(bio_ids)] for f in fr]
                    E.cascade(*fr, **meta(engine, f"{a}_FULL_CBC_S1", "first sample", tag or "nested CV"))
                    for st, (x, y) in {"S1": (fr[0], fr[2]), "S2": (fr[1], fr[3])}.items():   # P3
                        E.compare("CBC vs CBC+BIO", x, y, st, set=tag or "nested CV", engine=engine, analysis=a)
        for st in ("S1", "S2", "MX"):
            for fs, sc in (("FULL", "CBC"), ("T13", "CBC"), ("FULL", "CBC_BIO")):
                c1, c2 = f"A1_{fs}_{sc}_{st}", f"A2_{fs}_{sc}_{st}"
                if c1 in cfgs and c2 in cfgs:                                              # D10'
                    E.compare("A1 vs A2", first(cache[(engine, c1)]), first(cache[(engine, c2)]), st,
                              set="nested CV", engine=engine, feature_set=fs, scenario=sc)
            for a in ("A1", "A2"):
                c1, c2 = f"{a}_FULL_CBC_{st}", f"{a}_T13_CBC_{st}"
                if c1 in cfgs and c2 in cfgs:                                              # N6
                    E.compare("FULL vs T13", first(cache[(engine, c1)]), first(cache[(engine, c2)]), st,
                              set="nested CV", engine=engine, analysis=a)
    if "tabpfn" in engines and "autogluon" in engines:                               # (preview: TabPFN only)
        for cfg in lock["engines"]["tabpfn"]:                                             # N6
            E.compare("AutoGluon vs TabPFN", first(cache[("autogluon", cfg)]), first(cache[("tabpfn", cfg)]),
                      cfg[-2:], set="nested CV", config=cfg)

    lc = learning_curve(out)
    long = pd.DataFrame(E.rows)
    rdir = out / "reports"
    rdir.mkdir(exist_ok=True)
    with pd.ExcelWriter(rdir / "evaluation.xlsx") as xw:
        long.to_excel(xw, sheet_name="all_metrics", index=False)
        for name, parts in E.tables.items():
            pd.concat(parts, ignore_index=True).to_excel(xw, sheet_name=name[:31], index=False)
        if len(lc):
            lc.to_excel(xw, sheet_name="learning_curve", index=False)
    fd = rdir / "figures_data"
    fd.mkdir(exist_ok=True)
    long.to_parquet(fd / "metrics.parquet", index=False)
    for name, parts in E.tables.items():
        pd.concat(parts, ignore_index=True).to_parquet(fd / f"{name}.parquet", index=False)
    if len(lc):
        lc.to_parquet(fd / "learning_curve.parquet", index=False)
    print(f"evaluation written ({time.time() - t0:.0f} s): {len(long)} metric rows, {len(E.tables)} tables")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    a = ap.parse_args()
    build(a.out, a.boot)
