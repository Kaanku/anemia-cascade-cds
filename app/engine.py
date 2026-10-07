"""
engine.py — the two-tier anaemia cascade of the study, fitted on SYNTHETIC data (research demo)
================================================================================================

What it reproduces from the study pipeline (CDS v4.3):
  * inputs and features: the 38 full blood count / reticulocyte inputs, the four analytes and the
    ratio features exactly as in s04 (same order, same formulas); the final feature list of every
    model is the one Boruta selected on the real development cohort (names only, data/features.json);
  * models: TabPFN-3.5 (Prior Labs), seed 42, one per configuration (A1 FULL: CBC S1, CBC S2,
    CBC + biochemistry S1, CBC + biochemistry S2); the model version and ensemble size are those recorded
    in data/lock.json (_meta), here TabPFN-3.5-fast with a small fixed ensemble so that a 2-core host
    answers in seconds (the study used TabPFN-3.5 with its default ensemble);
  * decisions: Stage 1 threshold, HIGH cut-off, LOW = top-class probability < 0.35, APS conformal quantiles
    and the display calibration (parametric only), locked with the study's rules (s07) on cross-generation
    predictions on the real development patients: 5 folds, each fold predicted by the same kind of model
    trained on synthetic data generated from the other four (DECISIONS R34; data/lock.json). One rule differs:
    the HIGH cut-off is the 85 % accuracy point, because models trained on synthetic data do not reach the
    study's 90 % point on real patients (the target is recorded in the lock);
  * cascade (s09/s15): Tier 1 = CBC only; finalised when Stage 1 says AAC and the Stage 2 top-class
    probability reaches the HIGH cut-off; otherwise the biochemistry panel is added (Tier 2);
  * reflex recommendations: the 14 rules of the study (data/reflex_rules.json).
What is different: the training rows are synthetic (data/synthetic_cohort.parquet, class-wise Gaussian
copula, DECISIONS R30), so the probabilities are those of a model trained on synthetic data, not the
study's model. Not a medical device; research use only.

Backends: "tabpfn" (default; needs the TabPFN-3.5 weights, TABPFN_TOKEN) or "proxy"
(gradient boosting; only for testing the interface, labelled as such in the app).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA = HERE / "data" if (HERE / "data" / "lock.json").exists() else HERE      # or a flat layout
SEED = 42
CBC_BASE = ["age", "hgb_g_d_l", "rbc_10_6_u_l", "ret_number_10_6_l", "mcv_f_l", "mchc_g_dl",
            "rdw_sd_fl", "ret_he_pg", "irf_pct", "micro_macro_ratio", "nrbc_pct", "delta_he_pg", "frc_perc"]
EXTRA = ["hct_pct", "mch_pg", "rdw_cv_pct", "nrbc_number_10_3_u_l", "wbc_10_3_u_l", "neut_number_10_3_u_l",
         "lymph_number_10_3_u_l", "mono_number_10_3_u_l", "eo_number_10_3_u_l", "baso_number_10_3_u_l",
         "ig_number_10_3_u_l", "plt_10_3_u_l", "mpv_fl", "pdw_fl", "p_lcr_pct", "ret_pct", "lfr_pct",
         "mfr_pct", "hfr_pct", "rbc_he_pg", "micro_r_pct", "macro_r_pct", "ret_y_ch", "ret_rbc_y_ch", "irf_y_ch"]
BIO = ["ferritin", "iron", "ldh", "uibc"]
RATIO_CBC = ["hgb_g_d_l", "rbc_10_6_u_l", "ret_number_10_6_l", "mcv_f_l", "mchc_g_dl", "rdw_sd_fl",
             "ret_he_pg", "irf_pct", "micro_macro_ratio"]
LOG_FOR_KNN = ["ferritin", "ldh"]
AAC = ["IDA", "HA", "HGB_HTZ", "NORMAL"]
S2_CLASSES = ["HA", "HGB_HTZ", "IDA", "NORMAL"]          # order of the stored probabilities (s07)
LOW_CUTOFF = 0.35
ALPHA = "0.1"
CONFIGS = {"cbc_s1": "A1_FULL_CBC_S1", "cbc_s2": "A1_FULL_CBC_S2",
           "bio_s1": "A1_FULL_CBC_BIO_S1", "bio_s2": "A1_FULL_CBC_BIO_S2"}
CLASS_LABEL = {"IDA": "Iron deficiency anaemia", "HA": "Haemolytic anaemia",
               "HGB_HTZ": "Heterozygous haemoglobinopathy", "NORMAL": "Normal red cell profile",
               "OAC": "Anaemia of another cause"}
RULE_CLASS = {"IDA": "IDA", "HA": "HA", "HGB_HTZ": "HGB HTZ", "NORMAL": "Normal"}


# ───────────────────────────────────────────────────────────── features (as s04)
def ratio_pairs(scenario: str) -> list[tuple[str, str]]:
    order = RATIO_CBC + (BIO if scenario == "CBC_BIO" else [])
    return [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]


def add_ratios(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    new = {f"{a}_div_{b}": df[a] / df[b] for a, b in ratio_pairs(scenario)}
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


class Imputer:
    """Label-free KNN imputation of the analytes (k = 5) on standardised thesis CBC inputs and
    log ferritin / LDH, fitted on the synthetic rows with at least one analyte (as s04)."""

    cols = CBC_BASE + BIO

    def __init__(self, rows: pd.DataFrame, k: int = 5):
        x = self._space(rows)
        self.scaler = StandardScaler().fit(x)
        self.knn = KNNImputer(n_neighbors=k).fit(self.scaler.transform(x))

    def _space(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols].astype(float).copy()
        for c in LOG_FOR_KNN:
            x[c] = np.log(x[c])
        return x.to_numpy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        z = self.knn.transform(self.scaler.transform(self._space(df)))
        x = pd.DataFrame(self.scaler.inverse_transform(z), columns=self.cols, index=df.index)
        for c in LOG_FOR_KNN:
            x[c] = np.exp(x[c])
        out = df.copy()
        for c in BIO:
            out[f"imputed_{c}"] = out[c].isna()
            out[c] = out[c].fillna(x[c])
        return out


# ───────────────────────────────────────────────────────────── calibration (as s07, from JSON)
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrate(P: np.ndarray, cal: dict) -> np.ndarray:
    m = cal["method"]
    if m == "uncalibrated":
        return P
    if m == "platt":
        return 1 / (1 + np.exp(-(cal["coef"] * _logit(P) + cal["intercept"])))
    if m == "temperature":
        z = np.log(np.clip(P, 1e-6, 1)) / cal["T"]
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)
    if m == "isotonic" and "x" in cal:
        return np.interp(P, cal["x"], cal["y"])
    if m == "isotonic":
        Q = np.column_stack([np.interp(P[:, k], c["x"], c["y"]) for k, c in enumerate(cal["per_class"])])
        s = Q.sum(axis=1, keepdims=True)
        return np.where(s > 0, Q / np.where(s > 0, s, 1), P)
    raise ValueError(f"unknown calibration {m}")


def zone(top: float, high: float | None) -> str:
    if high is not None and top >= high:
        return "HIGH"
    return "LOW" if top < LOW_CUTOFF else "MEDIUM"


def aps_set(P: np.ndarray, qhat: float) -> list[int]:
    """Conservative (non-randomised) APS set: classes in decreasing probability while the mass before them
    is at most qhat; the top class is always included."""
    order = np.argsort(-P, kind="stable")
    before = np.cumsum(P[order]) - P[order]
    keep = [int(k) for k, b in zip(order, before) if b <= qhat]
    return keep or [int(order[0])]


# ───────────────────────────────────────────────────────────── models
class ProxyModel:
    """Interface test only: gradient boosting in place of TabPFN-3.5."""

    def __init__(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)

    def fit(self, X, y):
        self.m.fit(X, y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def make_model(backend: str, n_estimators: int | None = None, version: str = "v3.5"):
    """TabPFN-3.5 as locked (data/lock.json _meta: model_version, n_estimators), on CPU."""
    if backend == "proxy":
        return ProxyModel()
    os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")      # the four stages share one model in memory
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    kw = {"fit_mode": os.environ.get("CDS_FIT_MODE")} if os.environ.get("CDS_FIT_MODE") else {}
    if n_estimators:                                           # must match the lock (data/lock.json _meta)
        kw["n_estimators"] = int(n_estimators)
    return TabPFNClassifier.create_default_for_version(ModelVersion(version), device="cpu", random_state=SEED, **kw)


class Cascade:
    def __init__(self, backend: str | None = None):
        self.backend = backend or os.environ.get("CDS_DEMO_BACKEND", "tabpfn")
        t0 = time.time()
        syn = pd.read_parquet(DATA / "synthetic_cohort.parquet")
        self.features = json.loads((DATA / "features.json").read_text())
        self.lock = json.loads((DATA / "lock.json").read_text())
        meta = self.lock.get("_meta", {})
        self.lock_engine, self.n_estimators = meta.get("engine", "?"), meta.get("n_estimators")
        self.model_version, self.locked_on = meta.get("model_version", "v3.5"), meta.get("locked_on", "")
        self.high_target = float(meta.get("high_target", 0.90))       # study: 0.90; the demo: 0.85 (R34)
        if self.backend == "tabpfn" and self.lock_engine != "tabpfn-3.5":
            raise RuntimeError(f"lock.json was made with '{self.lock_engine}', not TabPFN-3.5")
        self.model_label = ("TabPFN-3.5" + ("-fast" if self.model_version.endswith("fast") else "")
                            + (f", {self.n_estimators} ensemble members" if self.n_estimators else ""))
        self.rules = {r["rule"]: r for r in json.loads((DATA / "reflex_rules.json").read_text())}
        self.ranges = {c: (float(syn[c].min()), float(syn[c].max())) for c in CBC_BASE + EXTRA + BIO}
        bio_rows = syn[syn[BIO].notna().any(axis=1)]
        self.imputer = Imputer(bio_rows)
        mats = {"CBC": add_ratios(syn, "CBC"), "CBC_BIO": add_ratios(self.imputer.transform(bio_rows), "CBC_BIO")}
        self.models = {}
        for key, cfg in CONFIGS.items():
            sc, st = ("CBC_BIO" if key.startswith("bio") else "CBC"), cfg[-2:]          # "S1" / "S2"
            d = mats[sc] if st == "S1" else mats[sc][mats[sc]["cls"].isin(AAC)]
            y = (d["y_s1"] if st == "S1" else d["cls"]).to_numpy()
            m = make_model(self.backend, self.n_estimators, self.model_version).fit(
                d[self.features[cfg]].to_numpy(float), y)
            classes = [str(c) for c in m.classes_]
            assert classes == (["0", "1"] if st == "S1" else S2_CLASSES), (cfg, classes)
            self.models[key] = m
        self.n_train = {"synthetic_patients": len(syn), "with_biochemistry": len(bio_rows)}
        self.load_seconds = round(time.time() - t0, 1)

    # ── input handling ────────────────────────────────────────────
    @staticmethod
    def complete(x: dict) -> dict:
        """Derived inputs: MicroR/MacroR, NRBC# from NRBC% and WBC (when not given)."""
        x = dict(x)
        if x.get("micro_r_pct") is not None and x.get("macro_r_pct"):
            x["micro_macro_ratio"] = x["micro_r_pct"] / x["macro_r_pct"]
        if x.get("nrbc_number_10_3_u_l") is None and x.get("nrbc_pct") is not None and x.get("wbc_10_3_u_l") is not None:
            x["nrbc_number_10_3_u_l"] = x["nrbc_pct"] * x["wbc_10_3_u_l"] / 100
        return x

    def check(self, x: dict) -> dict:
        missing = [c for c in CBC_BASE if x.get(c) is None or not np.isfinite(x[c])]
        absent = [c for c in EXTRA if x.get(c) is None]
        outside = [c for c in CBC_BASE + EXTRA + BIO if x.get(c) is not None
                   and not (self.ranges[c][0] <= x[c] <= self.ranges[c][1])]
        bad_ratio = [c for c in RATIO_CBC if x.get(c) is not None and x[c] <= 0]
        return {"missing_required": missing, "absent_optional": absent, "outside_training_range": outside,
                "non_positive": bad_ratio}

    def _row(self, x: dict) -> pd.DataFrame:
        return pd.DataFrame([{c: (np.nan if x.get(c) is None else float(x[c])) for c in CBC_BASE + EXTRA + BIO}])

    def _predict(self, key: str, d: pd.DataFrame) -> np.ndarray:
        return self.models[key].predict_proba(d[self.features[CONFIGS[key]]].to_numpy(float))[0]

    def _stage1(self, key: str, d: pd.DataFrame) -> dict:
        e = self.lock[CONFIGS[key]]
        p = float(self._predict(key, d)[1])
        return {"score": p, "threshold": e["threshold"], "aac": p >= e["threshold"],
                "p_display": float(calibrate(np.array([p]), e["calibration"])[0])}

    def _stage2(self, key: str, d: pd.DataFrame) -> dict:
        e = self.lock[CONFIGS[key]]
        P = self._predict(key, d)
        Pd = calibrate(P[None, :], e["calibration"])[0]
        top = int(P.argmax())
        return {"probs": dict(zip(S2_CLASSES, map(float, P))), "probs_display": dict(zip(S2_CLASSES, map(float, Pd))),
                "top": S2_CLASSES[top], "top_score": float(P[top]), "high_cutoff": e["high_cutoff"],
                "zone": zone(float(P[top]), e["high_cutoff"]),
                "conformal_set": [S2_CLASSES[k] for k in aps_set(P, e["conformal_qhat"][ALPHA])]}

    # ── the cascade ───────────────────────────────────────────────
    def start(self, x: dict) -> dict:
        """Input checks and Tier 1. Tier 2 is pending ('final' absent) when Tier 1 escalates and at least one
        analyte was entered; finish() runs it. The split lets the interface show Tier 1 while Tier 2 runs."""
        t0 = time.time()
        x = self.complete(x)
        chk = self.check(x)
        if chk["missing_required"] or chk["non_positive"]:
            return {"ok": False, "check": chk}
        cbc = add_ratios(self._row(x), "CBC")
        s1, s2 = self._stage1("cbc_s1", cbc), self._stage2("cbc_s2", cbc)
        res = {"ok": True, "check": chk, "tier1": {"stage1": s1, "stage2": s2}, "inputs": x}
        if s1["aac"] and s2["zone"] == "HIGH":
            res["final"] = {"tier": 1, "rule": {"IDA": "T1-1", "HA": "T1-2", "HGB_HTZ": "T1-3", "NORMAL": "T1-4"}[s2["top"]],
                            "class": s2["top"]}
        else:
            res["escalation_rule"] = "T1-7" if not s1["aac"] else ("T1-6" if s2["zone"] == "LOW" else "T1-5")
            if not any(x.get(c) is not None for c in BIO):
                res["final"] = {"tier": 1, "rule": res["escalation_rule"], "class": None, "needs_biochemistry": True}
        res["seconds"] = round(time.time() - t0, 1)
        return res

    def finish(self, res: dict) -> dict:
        """Tier 2 when pending, then the reflex recommendation(s)."""
        if not res["ok"]:
            return res
        t0 = time.time()
        if "final" not in res:
            x = res["inputs"]
            bio = add_ratios(self.imputer.transform(self._row(x)), "CBC_BIO")
            b1, b2 = self._stage1("bio_s1", bio), self._stage2("bio_s2", bio)
            res["tier2"] = {"stage1": b1, "stage2": b2, "n_analytes": sum(x.get(c) is not None for c in BIO),
                            "imputed": [c for c in BIO if x.get(c) is None]}
            if not b1["aac"]:
                res["final"] = {"tier": 2, "rule": "T2-5", "class": "OAC"}
            elif b2["zone"] == "HIGH":
                res["final"] = {"tier": 2, "rule": {"IDA": "T2-1", "HA": "T2-2", "HGB_HTZ": "T2-3",
                                                    "NORMAL": "T2-4"}[b2["top"]], "class": b2["top"]}
            else:
                res["final"] = {"tier": 2, "rule": "T2-6" if b2["zone"] == "MEDIUM" else "T2-7", "class": b2["top"]}
        res["final"]["recommendation"] = self.rules[res["final"]["rule"]]
        if "escalation_rule" in res:
            res["escalation"] = self.rules[res["escalation_rule"]]
        res["seconds"] = round(res["seconds"] + time.time() - t0, 1)
        return res

    def run(self, x: dict) -> dict:
        return self.finish(self.start(x))
