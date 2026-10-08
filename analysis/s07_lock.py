"""
s07_lock.py — Anemia cascade CDS, nested cross-validation (v4)
===============================================================

Fixes every post-training choice of every fold-set on that fold-set's own
inner OUT-OF-FOLD (OOF) predictions and writes them to a frozen lock file.
It only reads *_oof.parquet files (enforced in read_oof); the outer-fold and
temporal predictions are opened by s09 only.

Lock v2 (DECISIONS R15): engines AutoGluon (predictions/) and TabPFN-3.5
(predictions_tabpfn35/), all 16 configurations each. Lock v1 is archived in
lock/v1/; s07 refuses to overwrite a lock that s09 has already opened.

For each engine (AutoGluon, TabPFN), configuration and fold-set (outer0 …
outer4, final), on the inner-OOF scores of the patients' FIRST samples (D12;
for A1 every row is a first sample):
    L3/X2  threshold = maximum OOF macro F1 on 0.05–0.95 (ties towards 0.50);
           HGB HTZ is the positive class for MX
    N7     Youden threshold on the same grid (secondary operating point)
    P2/L4  accuracy–coverage operating points (80/85/90/95 %, n >= 20);
           HIGH zone = the 90 % point; LOW = top-class probability < 0.35
    L5     randomised APS conformal quantiles (alpha 0.05/0.10/0.20, seed 42)
    L1     calibration for the displayed probabilities: chosen by
           cross-fitted ECE over the inner folds, refitted on all OOF rows
Operating decisions use the uncalibrated scores (L2').

Outputs:
    <OUT>/lock/lock.json            {engine: {config: {fold-set: choices}}} + hashes
    <OUT>/lock/calibrators.joblib   {(engine, config, fold-set): calibrator}
    <OUT>/reports/lock_report.xlsx

Usage:
    python s07_lock.py --out "/path/to/Kaan/CDS_v4"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

FOLDSETS = [f"outer{k}" for k in range(5)] + ["final"]
CONFIGS = ([f"{a}_FULL_CBC_{st}" for a in ("A1", "A2") for st in ("S1", "S2", "MX")]
           + [f"{a}_FULL_CBC_BIO_{st}" for a in ("A1", "A2") for st in ("S1", "S2")]
           + [f"{a}_T13_CBC_{st}" for a in ("A1", "A2") for st in ("S1", "S2", "MX")])
# Lock v2 (DECISIONS R14, R15): the TabPFN engine is TabPFN-3.5 on all 16 configurations.
# The TabPFN v2 outputs (predictions_tabpfn/) are archived and no longer locked.
ENGINES = {"autogluon": ("predictions", CONFIGS),
           "tabpfn": ("predictions_tabpfn35", CONFIGS)}
LC_ROOTS = {"autogluon": "predictions_lc", "tabpfn": "predictions_lc_tabpfn35"}
ENGINE_LABELS = {"autogluon": "AutoGluon 1.5.0", "tabpfn": "TabPFN-3.5 (tabpfn 9.0.0)"}
LOCK_VERSION = 3          # v4.3 data; v2 = v4 data (archived)

SEED_CAL = 42
ALPHAS = (0.05, 0.10, 0.20)
S2_CLASSES = ["HA", "HGB_HTZ", "IDA", "NORMAL"]       # AutoGluon's (sorted) class order
MX_POSITIVE, MX_NEGATIVE = "HGB_HTZ", "IDA"
LOW_CUTOFF = 0.35
TARGETS = (0.80, 0.85, 0.90, 0.95)
HIGH_TARGET, MIN_N = 0.90, 20
GRID_THRESHOLD = np.round(np.arange(0.05, 0.951, 0.01), 2)
GRID_TOP = {"S2": np.round(np.arange(0.25, 0.991, 0.01), 2),
            "MX": np.round(np.arange(0.50, 0.991, 0.01), 2)}
ECE_MARGIN = 0.005
EPS = 1e-6


# ─────────────────────────────────────────────────────────────── metrics
def ece_mce(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
    return float(ece), float(mce)


def calib_metrics(P: np.ndarray, y: np.ndarray, binary: bool) -> dict:
    """P: (n,) P(positive) for binary, (n,K) for multiclass; y: 0/1 or class index."""
    if binary:
        ece, mce = ece_mce(P, y.astype(float))
        brier = float(np.mean((P - y) ** 2))
    else:
        conf, pred = P.max(axis=1), P.argmax(axis=1)
        ece, mce = ece_mce(conf, (pred == y).astype(float))
        Y = np.eye(P.shape[1])[y]
        brier = float(np.mean(((P - Y) ** 2).sum(axis=1)) / P.shape[1])
    return {"ECE": ece, "MCE": mce, "Brier": brier}


# ─────────────────────────────────────────────────────────── calibrators
class Identity:
    def fit(self, P, y): return self
    def transform(self, P): return P


class PlattBinary:
    def fit(self, p, y):
        z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1)).reshape(-1, 1)
        self.lr = LogisticRegression(C=1e4, max_iter=1000).fit(z, y)
        return self

    def transform(self, p):
        z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1)).reshape(-1, 1)
        return self.lr.predict_proba(z)[:, 1]


class IsotonicBinary:
    def fit(self, p, y):
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p):
        return self.iso.predict(p)


class Temperature:
    def fit(self, P, y):
        L = np.log(np.clip(P, EPS, 1))
        def nll(logT):
            Z = L / np.exp(logT)
            Z = Z - Z.max(axis=1, keepdims=True)
            lp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
            return -lp[np.arange(len(y)), y].mean()
        self.T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        return self

    def transform(self, P):
        Z = np.log(np.clip(P, EPS, 1)) / self.T
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)


class IsotonicOvR:
    def fit(self, P, y):
        self.isos = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(P[:, k], (y == k).astype(float))
                     for k in range(P.shape[1])]
        return self

    def transform(self, P):
        Q = np.column_stack([iso.predict(P[:, k]) for k, iso in enumerate(self.isos)])
        s = Q.sum(axis=1, keepdims=True)
        bad = s[:, 0] <= 0
        Q[~bad] = Q[~bad] / s[~bad]
        Q[bad] = P[bad]                                   # fall back to uncalibrated row
        return Q


METHODS = {True: {"uncalibrated": Identity, "platt": PlattBinary, "isotonic": IsotonicBinary},
           False: {"uncalibrated": Identity, "temperature": Temperature, "isotonic": IsotonicOvR}}


def cross_fit(method, P, y, folds):
    out = np.zeros_like(P, dtype=float)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        out[te] = method().fit(P[tr], y[tr]).transform(P[te])
    return out


# ──────────────────────────────────────────────────────────── decisions
def best_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    """L3 / X2: maximum OOF macro F1 on the grid; ties broken towards 0.50."""
    f1 = np.array([f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
                   for t in GRID_THRESHOLD])
    cands = GRID_THRESHOLD[np.isclose(f1, f1.max())]
    t = float(cands[np.argmin(np.abs(cands - 0.5))])
    return t, pd.DataFrame({"threshold": GRID_THRESHOLD, "oof_macro_f1": f1})


def binary_summary(p: np.ndarray, y: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {"macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "sensitivity": round(tp / max(int((y == 1).sum()), 1), 4),
            "specificity": round(tn / max(int((y == 0).sum()), 1), 4),
            "accuracy": round(float((pred == y).mean()), 4)}


def coverage_table(conf: np.ndarray, correct: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in grid:
        m = conf >= t
        rows.append({"cutoff": float(t), "n_at_or_above": int(m.sum()), "share": float(m.mean()),
                     "accuracy": float(correct[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def operating_points(tab: pd.DataFrame) -> dict:
    """P2: lowest cut-off whose retained OOF accuracy reaches each target (n >= MIN_N)."""
    out = {}
    for target in TARGETS:
        ok = tab[(tab["accuracy"] >= target - 1e-12) & (tab["n_at_or_above"] >= MIN_N)]
        if len(ok):
            r = ok.sort_values("cutoff").iloc[0]
            out[f"{target:.2f}"] = {"cutoff": float(r["cutoff"]), "n": int(r["n_at_or_above"]),
                                    "share": round(float(r["share"]), 4),
                                    "accuracy": round(float(r["accuracy"]), 4)}
        else:
            out[f"{target:.2f}"] = None
    return out


def aps_scores(P: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(-P, axis=1, kind="stable")
    Ps = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(Ps, axis=1) - Ps                        # mass strictly before each rank
    rank_y = np.argmax(order == y[:, None], axis=1)
    U = rng.uniform(size=len(y))
    i = np.arange(len(y))
    return cum[i, rank_y] + U * Ps[i, rank_y]


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_oof(path: Path) -> pd.DataFrame:
    assert path.name.endswith("_oof.parquet"), f"s07 may only read OOF files: {path.name}"
    return pd.read_parquet(path)


def get_scores(oof: pd.DataFrame, stage: str) -> tuple[np.ndarray, np.ndarray]:
    if stage == "S1":
        return oof["prob_1"].to_numpy(float), oof["y_s1"].astype(int).to_numpy()
    if stage == "S2":
        assert [c[5:] for c in oof.columns if c.startswith("prob_")] == S2_CLASSES
        P = oof[[f"prob_{c}" for c in S2_CLASSES]].to_numpy(float)
        return P, oof["cls"].map({c: i for i, c in enumerate(S2_CLASSES)}).to_numpy()
    assert set(oof["cls"]) == {MX_POSITIVE, MX_NEGATIVE}
    return oof[f"prob_{MX_POSITIVE}"].to_numpy(float), (oof["cls"] == MX_POSITIVE).astype(int).to_numpy()


# ──────────────────────────────────────────────────────────────── main
def youden_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """N7: maximum sensitivity + specificity - 1 on the grid; ties towards 0.50."""
    j = np.array([((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1)
                  + ((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1) - 1 for t in GRID_THRESHOLD])
    cands = GRID_THRESHOLD[np.isclose(j, j.max())]
    return float(cands[np.argmin(np.abs(cands - 0.5))])


def lock_one(oof_all: pd.DataFrame, stage: str) -> tuple[dict, object, dict]:
    oof = oof_all[oof_all["is_index_sample"].astype(bool)].reset_index(drop=True)
    assert oof["patient_id"].is_unique
    folds = oof["inner_fold"].astype(int).to_numpy()
    binary = stage != "S2"
    P, y = get_scores(oof, stage)
    entry = {"n_oof_rows": int(len(oof_all)), "n_oof_first_samples": int(len(oof))}
    tabs = {}
    if stage in ("S1", "MX"):
        t, curve = best_threshold(P, y)
        entry["threshold"] = t
        entry["oof_at_threshold"] = binary_summary(P, y, t)
        ty = youden_threshold(P, y)
        entry["threshold_youden"] = ty
        entry["oof_at_youden"] = binary_summary(P, y, ty)
        tabs["threshold_curve"] = curve
        if stage == "MX":
            entry["positive_class"] = MX_POSITIVE
    if stage in ("S2", "MX"):
        if stage == "S2":
            conf, correct = P.max(axis=1), P.argmax(axis=1) == y
        else:
            conf, correct = np.maximum(P, 1 - P), (P >= 0.5).astype(int) == y
        tab = coverage_table(conf, correct, GRID_TOP[stage])
        tabs["coverage"] = tab
        entry["oof_accuracy_all"] = round(float(correct.mean()), 4)
        entry["operating_points"] = operating_points(tab)
    if stage == "S2":
        hp = entry["operating_points"][f"{HIGH_TARGET:.2f}"]
        entry["high_cutoff"] = None if hp is None else hp["cutoff"]
        entry["low_cutoff"] = LOW_CUTOFF
        sc = aps_scores(P, y, np.random.default_rng(SEED_CAL))
        entry["conformal_qhat"] = {str(a): conformal_qhat(sc, a) for a in ALPHAS}
        entry["conformal_n_cal"] = int(len(sc))
    res = {}
    for name, M in METHODS[binary].items():
        res[name] = calib_metrics(cross_fit(M, P, y, folds), y, binary)
    best = min(res, key=lambda m: res[m]["ECE"])
    chosen = best if res["uncalibrated"]["ECE"] - res[best]["ECE"] > ECE_MARGIN else "uncalibrated"
    cal = {"method": chosen, "model": METHODS[binary][chosen]().fit(P, y), "binary": binary,
           "classes": S2_CLASSES if stage == "S2" else None,
           "positive": {"S1": "AAC", "MX": MX_POSITIVE}.get(stage)}
    entry["calibration"] = chosen
    entry["cv_ECE"] = {m: round(r["ECE"], 4) for m, r in res.items()}
    return entry, cal, tabs


def build(out: Path) -> None:
    ldir = out / "lock"
    ldir.mkdir(parents=True, exist_ok=True)
    # A lock that has been used for evaluation is frozen: archive it (lock/v1/, R15) before re-locking.
    assert not (ldir / "unlock_log.json").exists(), \
        "lock/unlock_log.json exists: this lock was already used by s09; archive lock/ before re-locking"
    lock = {"version": LOCK_VERSION, "engine_labels": ENGINE_LABELS,
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "rules": "DECISIONS.md L1, L2', L3-L6, P2, X2, N2, N7",
            "unit": "first samples (one per patient); uncalibrated inner-OOF scores of each fold-set",
            "files_read_sha256": {}, "missing": [], "engines": {}}
    calibrators, summ, curves = {}, [], {}
    for engine, (pred_root, configs) in ENGINES.items():
        lock["engines"][engine] = {}
        for cfg in configs:
            stage = cfg[-2:]
            lock["engines"][engine][cfg] = {}
            for fset in FOLDSETS:
                f = out / pred_root / cfg / f"{fset}_oof.parquet"
                if not f.exists():
                    lock["missing"].append(f"{engine}/{cfg}/{fset}")
                    continue
                lock["files_read_sha256"][f"{pred_root}/{cfg}/{f.name}"] = sha256(f)
                entry, cal, tabs = lock_one(read_oof(f), stage)
                lock["engines"][engine][cfg][fset] = entry
                calibrators[(engine, cfg, fset)] = cal
                for k, t in tabs.items():
                    curves[(engine, cfg, fset, k)] = t
                summ.append({"engine": engine, "config": cfg, "fold_set": fset, "n": entry["n_oof_first_samples"],
                             "threshold": entry.get("threshold"), "youden": entry.get("threshold_youden"),
                             "high_cutoff": entry.get("high_cutoff"), "calibration": entry["calibration"],
                             **({f"oof_{k}": v for k, v in entry["oof_at_threshold"].items()}
                                if "oof_at_threshold" in entry else {}),
                             "oof_top_accuracy": entry.get("oof_accuracy_all")})
    joblib.dump(calibrators, ldir / "calibrators.joblib")
    (ldir / "lock.json").write_text(json.dumps(lock, indent=1))
    with pd.ExcelWriter(out / "reports" / "lock_report.xlsx") as xw:
        pd.DataFrame(summ).to_excel(xw, sheet_name="summary", index=False)
    print(pd.DataFrame(summ).groupby(["engine", "config"]).agg(
        folds=("fold_set", "size"), threshold=("threshold", "mean"), high=("high_cutoff", "mean")).to_string())
    if lock["missing"]:
        print(f"\nNOT FINAL - {len(lock['missing'])} fold-sets missing, e.g. {lock['missing'][:3]}")


if __name__ == "__main__":
    import s07_lock      # run build() from the importable module so calibrators pickle as s07_lock.<Class>
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    s07_lock.build(ap.parse_args().out)
