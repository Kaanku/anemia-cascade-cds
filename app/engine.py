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

import itertools
import json
import math
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
DERIVED = {"micro_macro_ratio": ("micro_r_pct", "macro_r_pct"),          # inputs the app computes (complete())
           "nrbc_number_10_3_u_l": ("nrbc_pct", "wbc_10_3_u_l")}
SHAP_BUDGET = int(os.environ.get("CDS_SHAP_BUDGET", "512"))             # coalitions per explanation


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


# ───────────────────────────────────────────────────────────── explanation (KernelSHAP)
def kernel_design(p: int, budget: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Coalitions (n × p, 0/1) and regression weights of KernelSHAP (Lundberg and Lee, 2017), drawn as in the
    reference implementation (shap.KernelExplainer): coalition sizes are enumerated completely from the outside
    in (1 and p − 1, then 2 and p − 2, …) while the budget covers a size's share of the Shapley kernel; the
    remaining sizes are sampled in complementary pairs with probability proportional to the kernel, and the
    weights of the sampled coalitions are rescaled to the kernel mass left. With budget ≥ 2^p − 2 every
    coalition is used and the result is the exact Shapley value."""
    if p < 2:
        return np.zeros((0, p), np.int8), np.zeros(0)
    if 2 ** p - 2 <= budget:                                              # exact: every proper coalition
        Z = ((np.arange(1, 2 ** p - 1)[:, None] >> np.arange(p)) & 1).astype(np.int8)
        s = Z.sum(1)
        return Z, np.array([(p - 1) / (math.comb(p, int(k)) * k * (p - k)) for k in s])
    rng = np.random.default_rng(seed)
    n_sizes, n_paired = math.ceil((p - 1) / 2), (p - 1) // 2
    sizes = np.arange(1, n_sizes + 1)
    kw = (p - 1) / (sizes * (p - sizes))
    kw[:n_paired] *= 2                                                    # a size and its complement
    kw /= kw.sum()
    Z, w, left, rem, n_full = [], [], budget, kw.copy(), 0
    for i, s in enumerate(sizes):                                         # complete sizes, outside in
        paired = i < n_paired
        n_sub = math.comb(p, int(s)) * (2 if paired else 1)
        if left * rem[i] / n_sub < 1 - 1e-8:
            break
        n_full, left = n_full + 1, left - n_sub
        if rem[i] < 1:
            rem = rem / (1 - rem[i])
        wi = kw[i] / math.comb(p, int(s)) / (2 if paired else 1)
        for idx in itertools.combinations(range(p), int(s)):
            z = np.zeros(p, np.int8)
            z[list(idx)] = 1
            Z.append(z)
            w.append(wi)
            if paired:
                Z.append(1 - z)
                w.append(wi)
    n_fixed = len(Z)
    if n_full < n_sizes and left > 0:                                    # sample the rest in pairs
        prob = kw[n_full:].copy()
        prob[: max(0, n_paired - n_full)] /= 2
        prob /= prob.sum()
        seen: dict[bytes, int] = {}
        for d in rng.choice(len(prob), size=4 * left, p=prob):
            if left <= 0:
                break
            s = int(d) + n_full + 1
            z = np.zeros(p, np.int8)
            z[rng.permutation(p)[:s]] = 1
            t = z.tobytes()
            if t in seen:                                                 # repeated draw: add weight
                w[seen[t]] += 1.0
                if s <= n_paired:
                    w[seen[t] + 1] += 1.0
                continue
            seen[t] = len(Z)
            Z.append(z)
            w.append(1.0)
            left -= 1
            if left > 0 and s <= n_paired:
                Z.append(1 - z)
                w.append(1.0)
                left -= 1
        w = np.asarray(w, float)
        w[n_fixed:] *= kw[n_full:].sum() / w[n_fixed:].sum()
    return np.asarray(Z, np.int8), np.asarray(w, float)


def shapley_regression(Z: np.ndarray, w: np.ndarray, Y: np.ndarray, f_ref: np.ndarray,
                       f_x: np.ndarray) -> np.ndarray:
    """Constrained weighted least squares of KernelSHAP: minimise Σ w (Y − f_ref − Zφ)² subject to
    Σφ = f_x − f_ref (efficiency), with the last player eliminated. Y: n × k outputs of the coalitions;
    returns φ, p × k."""
    total = np.asarray(f_x, float) - np.asarray(f_ref, float)
    p = Z.shape[1]
    if p == 0:
        return np.zeros((0, len(total)))
    if p == 1:
        return total[None, :]
    Zf = Z.astype(float)
    A = Zf[:, :-1] - Zf[:, -1:]
    b = Y - f_ref - Zf[:, -1:] * total
    sw = np.sqrt(w)[:, None]
    head = np.linalg.lstsq(A * sw, b * sw, rcond=None)[0]
    return np.vstack([head, total - head.sum(0)])


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
        self.models, self.reference = {}, {}
        for key, cfg in CONFIGS.items():
            sc, st = ("CBC_BIO" if key.startswith("bio") else "CBC"), cfg[-2:]          # "S1" / "S2"
            d = mats[sc] if st == "S1" else mats[sc][mats[sc]["cls"].isin(AAC)]
            y = (d["y_s1"] if st == "S1" else d["cls"]).to_numpy()
            m = make_model(self.backend, self.n_estimators, self.model_version).fit(
                d[self.features[cfg]].to_numpy(float), y)
            classes = [str(c) for c in m.classes_]
            assert classes == (["0", "1"] if st == "S1" else S2_CLASSES), (cfg, classes)
            self.models[key] = m
            ref = d[self.inputs(key)].astype(float).median()       # SHAP reference: the median training record
            self.reference[key] = self._derive(ref.to_frame().T).iloc[0]
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

    @staticmethod
    def inputs(key: str) -> list[str]:
        """The inputs a model sees before the ratio features: the CBC inputs, plus the analytes at Tier 2."""
        return CBC_BASE + EXTRA + (BIO if key.startswith("bio") else [])

    @staticmethod
    def _derive(df: pd.DataFrame) -> pd.DataFrame:
        """Inputs the app computes from others (MicroR/MacroR, NRBC#), recomputed from their sources."""
        df = df.copy()
        df["micro_macro_ratio"] = df["micro_r_pct"] / df["macro_r_pct"]
        df["nrbc_number_10_3_u_l"] = df["nrbc_pct"] * df["wbc_10_3_u_l"] / 100
        return df

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

    # ── explanation ───────────────────────────────────────────────
    def explain(self, res: dict, key: str, budget: int | None = None, seed: int = SEED) -> dict:
        """Shapley values of one model's raw output for the patient of a cascade result (KernelSHAP).

        Players are the values the model receives: the CBC inputs and, at Tier 2, the four analytes (imputed ones
        flagged). Ratio features follow their two values, and MicroR/MacroR and NRBC# follow their sources. A value
        outside a coalition takes the reference value, the median of the model's synthetic training records.
        Fields left empty stay empty, and a value equal to the reference cannot contribute, so neither is a
        player. All coalitions go to the model in one batched call (CDS_SHAP_CHUNK rows at a time if set). The
        values add up to f(patient) − f(reference); Stage 1 explains the probability of a classifiable cause,
        Stage 2 the four class probabilities."""
        t0 = time.time()
        budget = int(budget or SHAP_BUDGET)
        cols = self.inputs(key)
        row = self._row(res["inputs"])
        if key.startswith("bio"):
            row = self.imputer.transform(row)
        xv = row[cols].iloc[0].astype(float)
        ref = self.reference[key].reindex(cols).astype(float)
        empty = [c for c in cols if not np.isfinite(xv[c])]
        cand = [c for c in cols if c not in DERIVED and np.isfinite(xv[c])]
        players = [c for c in cand if not np.isclose(xv[c], ref[c], rtol=0, atol=1e-12)]
        p = len(players)
        Z, w = kernel_design(p, budget, seed)
        M = np.vstack([np.zeros((1, p), np.int8), np.ones((1, p), np.int8), Z])   # reference, patient, coalitions
        R = np.repeat(ref.to_numpy(float)[None, :], len(M), axis=0)
        R[:, [cols.index(c) for c in empty]] = np.nan
        idx = [cols.index(c) for c in players]
        R[:, idx] = np.where(M == 1, xv.to_numpy(float)[idx][None, :], R[:, idx])
        rows = self._derive(pd.DataFrame(R, columns=cols))
        for c, src in DERIVED.items():                       # the patient's own value when its sources are his
            same = np.logical_and.reduce([np.isclose(rows[s].to_numpy(float), xv[s], equal_nan=True) for s in src])
            rows.loc[same, c] = xv[c]
        F = add_ratios(rows, "CBC_BIO" if key.startswith("bio") else "CBC")[self.features[CONFIGS[key]]].to_numpy(float)
        chunk = int(os.environ.get("CDS_SHAP_CHUNK", "0")) or len(F)
        P = np.vstack([self.models[key].predict_proba(F[i:i + chunk]) for i in range(0, len(F), chunk)])
        Y = P[:, [1]] if key.endswith("s1") else P
        phi = shapley_regression(Z, w, Y[2:], Y[0], Y[1])
        tier = res.get("tier1" if key.startswith("cbc") else "tier2", {})
        if key.endswith("s1"):
            done = [tier["stage1"]["score"]] if tier else None
        else:
            done = [tier["stage2"]["probs"][c] for c in S2_CLASSES] if tier else None
        return {"model": key, "classes": ["AAC"] if key.endswith("s1") else list(S2_CLASSES),
                "players": players, "value": [float(xv[c]) for c in players],
                "reference": [float(ref[c]) for c in players],
                "imputed": [bool(row[f"imputed_{c}"].iloc[0]) if f"imputed_{c}" in row else False for c in players],
                "phi": phi.tolist(), "f_x": Y[1].tolist(), "f_ref": Y[0].tolist(),
                "check_vs_cascade": None if done is None else float(np.max(np.abs(np.asarray(done) - Y[1]))),
                "n_coalitions": int(len(Z)), "exact": bool(p < 2 or len(Z) == 2 ** p - 2), "budget": budget,
                "empty": [c for c in empty if c not in DERIVED], "at_reference": [c for c in cand if c not in players],
                "seconds": round(time.time() - t0, 1)}
