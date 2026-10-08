"""
s04_features.py — Anemia cascade CDS, nested cross-validation (v4)
===================================================================

Feature engineering, label-free imputation and Boruta selection for every
fold-set of the nested cross-validation (s03). Every data-dependent step is
fitted on the TRAINING patients of the fold-set only; the patients it is
evaluated on are only transformed.

    analysis     A1 (first sample per patient) | A2 (all samples)
    feature set  FULL (full blood count + extended reticulocyte panel) |
                 T13  (the 13 thesis parameters; CBC scenario only)
    scenario     CBC | CBC_BIO (+ iron, UIBC, ferritin, LDH)
    stage        S1: AAC vs OAC | S2: IDA / HA / HGB_HTZ / NORMAL |
                 MX: IDA vs HGB_HTZ (CBC scenario only)
    fold-set     outer0 … outer4 (train = the other four outer folds,
                 evaluated on outer fold k) | final (train = all patients)

Rules (DECISIONS.md F1–F7, N1–N6):
    F2  Candidate ratios: a/b for every pair of strictly positive thesis
        parameters in a fixed canonical order (36; 78 with biochemistry), for
        both feature sets. The added parameters enter as single features.
    F3  CBC_BIO population: samples with at least one analyte measured (M1).
    F4  Remaining analyte gaps: KNN (k = 5) on standardised thesis CBC
        parameters and log ferritin/LDH, fitted on the fold-set's training
        rows, without the class label. No missing-indicator features.
    F5  Boruta (random forest, 500 trees, depth 7, balanced class weights,
        100 iterations, seed 42) on the fold-set's training rows, per stage;
        base features are always kept, confirmed ratios are added. Missing
        values of the added parameters (analyzer flag '----') are filled with
        the training median for the Boruta step only; the models see them as
        missing.
    N4  Learning curve: the same selection repeated on nested subsets
        (25/50/75 %) of each outer fold's training patients (s03 lc_q_k).

Inputs:
    <OUT>/data/model/A1.parquet, A2.parquet                    (from s03)

Outputs:
    <OUT>/data/features/<A>_<FS>_<SC>/<fold-set>/train.parquet, eval.parquet
    <OUT>/data/features/<A>_<FS>_<SC>/<fold-set>/features_<stage>.json, boruta_<stage>.csv
    <OUT>/data/features/<A>_<FS>_<SC>/<fold-set>/imputer.joblib           (CBC_BIO)
    <OUT>/data/features/<A>_FULL_CBC/lc/<k>_<f>/features_<stage>.json     (learning curve)
    <OUT>/data/features/manifest.json, <OUT>/reports/feature_report.xlsx

Usage:
    python s04_features.py --out "/path/to/Kaan/CDS_v4" [--workers 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from s01_build_cohort import EXTRA_RAW

SEED = 42
AAC = ["IDA", "HA", "HGB_HTZ", "NORMAL"]
MX_CLASSES = ["IDA", "HGB_HTZ"]
CBC_BASE = ["age", "hgb_g_d_l", "rbc_10_6_u_l", "ret_number_10_6_l", "mcv_f_l", "mchc_g_dl",
            "rdw_sd_fl", "ret_he_pg", "irf_pct", "micro_macro_ratio", "nrbc_pct",
            "delta_he_pg", "frc_perc"]
EXTRA = list(EXTRA_RAW.values())
BIO = ["ferritin", "iron", "ldh", "uibc"]
RATIO_CBC = ["hgb_g_d_l", "rbc_10_6_u_l", "ret_number_10_6_l", "mcv_f_l", "mchc_g_dl",
             "rdw_sd_fl", "ret_he_pg", "irf_pct", "micro_macro_ratio"]
RATIO_BIO = ["ferritin", "iron", "ldh", "uibc"]
LOG_FOR_KNN = ["ferritin", "ldh"]
N_OUTER = 5
FOLDSETS = [f"outer{k}" for k in range(N_OUTER)] + ["final"]
LC_FRACTIONS = (0.25, 0.50, 0.75)
META = ["record_id", "patient_id", "is_index_sample", "cls", "y_s1", "n_bio_measured",
        "outer_fold", "inner_fold", "lc_q"]
POP = {"S1": None, "S2": AAC, "MX": MX_CLASSES}
# (analysis, feature set, scenario) -> stages
MATRICES = {(a, fs, sc): (["S1", "S2", "MX"] if sc == "CBC" else ["S1", "S2"])
            for a in ("A1", "A2") for fs, sc in (("FULL", "CBC"), ("FULL", "CBC_BIO"), ("T13", "CBC"))}


# ───────────────────────────────────────────────────────────────── helpers
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_features(fs: str, sc: str) -> list[str]:
    return CBC_BASE + (EXTRA if fs == "FULL" else []) + (BIO if sc == "CBC_BIO" else [])


def ratio_pairs(scenario: str) -> list[tuple[str, str]]:
    order = RATIO_CBC + (RATIO_BIO if scenario == "CBC_BIO" else [])
    return [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]


def add_ratios(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> tuple[pd.DataFrame, list[str]]:
    new = {f"{a}_div_{b}": df[a] / df[b] for a, b in pairs}
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    names = list(new)
    assert np.isfinite(out[names].to_numpy()).all(), "Non-finite ratio values"
    return out, names


class LabelFreeKNNImputer:
    """KNN imputation of the four analytes, fitted on training rows only.
    Distances use standardised thesis CBC parameters and log ferritin/LDH.
    The class label is never an input."""

    def __init__(self, k: int = 5):
        self.k = k
        self.cols = CBC_BASE + BIO

    def _space(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols].astype(float).copy()
        for c in LOG_FOR_KNN:
            x[c] = np.log(x[c])
        return x.to_numpy()

    def fit(self, df: pd.DataFrame) -> "LabelFreeKNNImputer":
        x = self._space(df)
        self.scaler_ = StandardScaler().fit(x)
        self.knn_ = KNNImputer(n_neighbors=self.k).fit(self.scaler_.transform(x))
        return self

    def state(self) -> dict:
        return {"cols": self.cols, "log_cols": LOG_FOR_KNN, "bio": BIO,
                "scaler": self.scaler_, "knn": self.knn_, "k": self.k}


def apply_imputer(state: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a saved imputer state (imputer.joblib) to new rows (evaluation, temporal, app)."""
    x = df[state["cols"]].astype(float).copy()
    for c in state["log_cols"]:
        x[c] = np.log(x[c])
    z = state["knn"].transform(state["scaler"].transform(x.to_numpy()))
    x = pd.DataFrame(state["scaler"].inverse_transform(z), columns=state["cols"], index=df.index)
    for c in state["log_cols"]:
        x[c] = np.exp(x[c])
    out = df.copy()
    for c in state["bio"]:
        out[f"imputed_{c}"] = out[c].isna()
        out[c] = out[c].fillna(x[c])
    assert out[state["bio"]].notna().all().all()
    return out


def boruta_select(X: pd.DataFrame, y: pd.Series, candidates: list[str]) -> pd.DataFrame:
    # n_jobs=1: single-threaded so that the selection is bit-for-bit reproducible
    Xf = X.fillna(X.median(numeric_only=True))            # F5: Boruta step only
    rf = RandomForestClassifier(n_estimators=500, max_depth=7, class_weight="balanced_subsample",
                                n_jobs=1, random_state=SEED)
    bor = BorutaPy(rf, n_estimators="auto", max_iter=100, random_state=SEED, verbose=0)
    bor.fit(Xf.to_numpy(dtype=float), y.to_numpy())
    rep = pd.DataFrame({"feature": list(X.columns), "rank": bor.ranking_,
                        "decision": np.select([bor.support_, bor.support_weak_],
                                              ["Confirmed", "Tentative"], "Rejected"),
                        "is_ratio": [c in candidates for c in X.columns]})
    return rep.sort_values(["rank", "feature"]).reset_index(drop=True)


def fold_rows(data: pd.DataFrame, foldset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and evaluation rows of a fold-set, with generic inner_fold / lc_q columns."""
    d = data.copy()
    if foldset == "final":
        tr, ev = d.copy(), d.iloc[0:0].copy()
        tr["inner_fold"] = tr["outer_fold"].astype(int)
        tr["lc_q"] = np.nan
        ev["inner_fold"] = pd.array([], dtype="Int64")
        ev["lc_q"] = np.array([], dtype=float)
    else:
        k = int(foldset[-1])
        tr, ev = d[d["outer_fold"] != k].copy(), d[d["outer_fold"] == k].copy()
        tr["inner_fold"] = tr[f"inner_fold_{k}"].astype(int)
        tr["lc_q"] = tr[f"lc_q_{k}"].astype(float)
        ev["inner_fold"] = pd.array([pd.NA] * len(ev), dtype="Int64")
        ev["lc_q"] = np.nan
    tr["inner_fold"] = tr["inner_fold"].astype("Int64")
    return tr, ev


# ──────────────────────────────────────────────────────────────── building
def build_matrices(out: Path) -> list[dict]:
    """Writes train/eval matrices per fold-set; returns the Boruta tasks."""
    mdir, fdir = out / "data" / "model", out / "data" / "features"
    tasks = []
    for (a, fs, sc), stages in MATRICES.items():
        data = pd.read_parquet(mdir / f"{a}.parquet")
        data["n_bio_measured"] = data[BIO].notna().sum(axis=1)
        data["y_s1"] = data["cls"].isin(AAC).astype(int)
        assert data[CBC_BASE].notna().all().all(), "thesis CBC inputs must be complete (s01)"
        if sc == "CBC_BIO":
            data = data[data["n_bio_measured"] > 0]                                    # F3
        base = base_features(fs, sc)
        for foldset in FOLDSETS:
            cdir = fdir / f"{a}_{fs}_{sc}" / foldset
            cdir.mkdir(parents=True, exist_ok=True)
            tr, ev = fold_rows(data, foldset)
            if sc == "CBC_BIO":                                                        # F4
                st = LabelFreeKNNImputer(k=5).fit(tr).state()
                tr = apply_imputer(st, tr)
                ev = apply_imputer(st, ev) if len(ev) else ev.assign(**{f"imputed_{c}": False for c in BIO})
                joblib.dump(st, cdir / "imputer.joblib")
            tr, ratios = add_ratios(tr, ratio_pairs(sc))                               # F2
            ev, _ = add_ratios(ev, ratio_pairs(sc))
            keep = META + base + ratios + [c for c in tr.columns if c.startswith("imputed_")]
            tr[keep].to_parquet(cdir / "train.parquet", index=False)
            ev[keep].to_parquet(cdir / "eval.parquet", index=False)
            for stage in stages:
                tasks.append({"dir": str(cdir), "stage": stage, "base": base, "ratios": ratios,
                              "subset": None, "out_dir": str(cdir), "key": f"{a}_{fs}_{sc}/{foldset}/{stage}"})
                if fs == "FULL" and sc == "CBC" and foldset != "final":                # N4
                    k = int(foldset[-1])
                    for f in LC_FRACTIONS:
                        tasks.append({"dir": str(cdir), "stage": stage, "base": base, "ratios": ratios,
                                      "subset": f, "out_dir": str(fdir / f"{a}_{fs}_{sc}" / "lc" / f"{k}_{f:.2f}"),
                                      "key": f"{a}_{fs}_{sc}/lc/{k}_{f:.2f}/{stage}"})
    return tasks


def run_task(t: dict) -> dict:
    t0 = time.time()
    tr = pd.read_parquet(Path(t["dir"]) / "train.parquet")
    if t["subset"] is not None:
        tr = tr[tr["lc_q"] <= t["subset"]]
    pop = POP[t["stage"]]
    rows = tr if pop is None else tr[tr["cls"].isin(pop)]
    y = rows["y_s1"] if t["stage"] == "S1" else rows["cls"]
    rep = boruta_select(rows[t["base"] + t["ratios"]], y, t["ratios"])
    od = Path(t["out_dir"])
    od.mkdir(parents=True, exist_ok=True)
    rep.to_csv(od / f"boruta_{t['stage']}.csv", index=False)
    sel = set(rep.loc[rep["is_ratio"] & (rep["decision"] == "Confirmed"), "feature"])
    feats = t["base"] + [r for r in t["ratios"] if r in sel]                     # canonical order
    (od / f"features_{t['stage']}.json").write_text(json.dumps(feats, indent=1))
    return {"key": t["key"], "train_rows": len(rows), "train_patients": int(rows["patient_id"].nunique()),
            "n_features": len(feats), "ratios_confirmed": len(sel), "seconds": round(time.time() - t0)}


def build(out: Path, workers: int, resume: bool = False) -> None:
    t0 = time.time()
    (out / "reports").mkdir(parents=True, exist_ok=True)
    tasks = build_matrices(out)
    if resume:                                    # a run is finished when its features json exists
        done = [t for t in tasks if (Path(t["out_dir"]) / f"features_{t['stage']}.json").exists()]
        tasks = [t for t in tasks if t not in done]
        print(f"resume: {len(done)} Boruta runs already finished", flush=True)
    print(f"matrices written ({time.time() - t0:.0f} s); {len(tasks)} Boruta runs", flush=True)
    with Pool(workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
            res.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done ({time.time() - t0:.0f} s)", flush=True)
    if resume:                                    # report covers every run, earlier ones re-read from disk
        for t in done:
            od = Path(t["out_dir"])
            rep_t = pd.read_csv(od / f"boruta_{t['stage']}.csv")
            feats = json.loads((od / f"features_{t['stage']}.json").read_text())
            res.append({"key": t["key"], "train_rows": np.nan, "train_patients": np.nan, "n_features": len(feats),
                        "ratios_confirmed": int((rep_t["is_ratio"] & (rep_t["decision"] == "Confirmed")).sum()),
                        "seconds": np.nan})
    rep = pd.DataFrame(res).sort_values("key").reset_index(drop=True)
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M"), "seed": SEED, "python": platform.python_version(),
                "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                "inputs": {f"{a}.parquet": sha256(out / "data" / "model" / f"{a}.parquet") for a in ("A1", "A2")},
                "n_boruta_runs": len(rep)}
    (out / "data" / "features" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with pd.ExcelWriter(out / "reports" / "feature_report.xlsx") as xw:
        rep.to_excel(xw, sheet_name="boruta_runs", index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="CDS_v4 folder")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="skip Boruta runs whose features json exists")
    a = ap.parse_args()
    build(a.out, a.workers, a.resume)
