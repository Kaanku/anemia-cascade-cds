"""
cascade_engine.py — Two-stage anemia cascade inference.

Pure-logic module (no Streamlit). Loads the deployment AutoGluon predictors,
derives ratio features from raw input, runs the Stage 1 -> Stage 2 cascade,
assigns confidence zones, builds conformal prediction sets, and looks up
reflex-test recommendations.

Internal class codes (DAS/IAS, DEA) are preserved exactly as the models were
trained. Display labels (OAC/AAC, IDA, HGB HTZ) are applied only at the
presentation boundary via DISPLAY_*.
"""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

# ── Display mappings (presentation only — never feed back into models) ──
S1_DISPLAY = {"IAS": "AAC", "DAS": "OAC"}
S2_DISPLAY = {"DEA": "IDA", "HA": "HA", "HGB HTZ": "HGB HTZ", "Normal": "Normal"}
# class code (index) -> internal label
S1_CLASSES = {0: "DAS", 1: "IAS"}
S2_CLASSES = {0: "DEA", 1: "HA", 2: "HGB HTZ", 3: "Normal"}
# raw label in demo data -> internal S2 code label
DIAGNOSIS_TO_S2 = {"IDA": "DEA", "HA": "HA", "HGB_HTZ": "HGB HTZ", "NORMAL": "Normal"}

SCENARIOS = {
    "CBC_Only": {"s1": "s1_cbc", "s2": "s2_cbc"},
    "CBC_BIO":  {"s1": "s1_bio", "s2": "s2_bio"},
}


def display_s1(label: str) -> str:
    return S1_DISPLAY.get(label, label)


def display_s2(label: str) -> str:
    return S2_DISPLAY.get(label, label)


# ── Feature derivation (mirrors m4_e2e.compute_missing_ratios) ──
def compute_missing_ratios(df: pd.DataFrame, feat_names: list[str]) -> pd.DataFrame:
    """Compute any '<num>_div_<den>' feature the model expects from raw columns."""
    df = df.copy()
    for f in feat_names:
        if f not in df.columns and "_div_" in f:
            num, den = f.split("_div_")
            if num in df.columns and den in df.columns:
                df[f] = df[num] / df[den].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


class CascadeEngine:
    """Loads models + assets once; runs cascade inference per patient frame."""

    def __init__(self, models_dir: str | Path, assets_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.assets_dir = Path(assets_dir)
        self._predictors: dict[str, object] = {}
        self._feature_names: dict[str, list[str]] = {}

        self.thresholds = json.load(open(self.assets_dir / "threshold_config.json"))["scenarios"]
        self.qhat = json.load(open(self.assets_dir / "conformal_qhat.json"))
        self.reflex = json.load(open(self.assets_dir / "reflex_rules.json"))["rules"]
        self.cal_registry = joblib.load(
            self.assets_dir / "calibrators" / "calibration_registry.joblib"
        )
        self._calibrators: dict[str, object] = {}

    # ── lazy model loading (keeps memory low until a scenario is used) ──
    def _predictor(self, key: str):
        if key not in self._predictors:
            from autogluon.tabular import TabularPredictor
            self._predictors[key] = TabularPredictor.load(
                str(self.models_dir / key), require_version_match=False
            )
            self._feature_names[key] = joblib.load(
                self.models_dir / key / "feature_names.joblib"
            )
        return self._predictors[key], self._feature_names[key]

    def _calibrator(self, name: str):
        """Load an isotonic calibrator file lazily; None if uncalibrated."""
        if name not in self._calibrators:
            fp = self.assets_dir / "calibrators" / f"{name}_calibrator.joblib"
            obj = joblib.load(fp) if fp.exists() else None
            # calibrator file may be wrapped: {'calibrator': <estimator>, 'meta': {...}}
            if isinstance(obj, dict) and "calibrator" in obj:
                obj = obj["calibrator"]
            self._calibrators[name] = obj
        return self._calibrators[name]

    # ── probability helpers ──
    def _predict_proba(self, key: str, X: pd.DataFrame) -> pd.DataFrame:
        pred, feats = self._predictor(key)
        Xd = compute_missing_ratios(X, feats)
        missing = [f for f in feats if f not in Xd.columns]
        for f in missing:
            Xd[f] = np.nan
        return pred.predict_proba(Xd[feats])

    def _apply_calibration_s1(self, scenario: str, prob_ias: np.ndarray) -> np.ndarray:
        reg = self.cal_registry.get(f"stage1_{scenario}", {})
        if reg.get("method") == "isotonic":
            cal = self._calibrator(f"stage1_{scenario.lower()}")
            if cal is not None:
                return cal.predict(prob_ias)
        return prob_ias

    # ── confidence zone ──
    @staticmethod
    def _zone(confidence: float, low: float, high: float) -> str:
        if confidence >= high:
            return "HIGH"
        if confidence < low:
            return "LOW"
        return "MEDIUM"

    # ── conformal set ──
    def _conformal_set(self, scenario: str, stage: int, probs: np.ndarray,
                       alpha: float = 0.10) -> list[str]:
        key = f"{scenario.lower()}_S{stage}_a{alpha:.2f}"
        q = self.qhat.get(key)
        names = S1_CLASSES if stage == 1 else S2_CLASSES
        if q is None:
            return []
        order = np.argsort(-probs)
        cumsum, pset = 0.0, []
        for cls in order:
            pset.append(names[int(cls)])
            cumsum += probs[cls]
            if cumsum >= q:
                break
        return pset

    # ── reflex lookup (wildcard '*' supported) ──
    def _reflex(self, pred_label: str, zone: str, tier: int) -> dict | None:
        for r in self.reflex:
            if r["tier"] != tier:
                continue
            if r["zone"] != zone:
                continue
            if r["prediction"] not in (pred_label, "*"):
                continue
            return r
        return None

    # ── main entry: single-patient cascade ──
    def run(self, raw_row: dict, scenario: str, alpha: float = 0.10) -> dict:
        """Run the full cascade for one patient (dict of raw measured values)."""
        sc = self.thresholds[scenario]
        keys = SCENARIOS[scenario]
        X = pd.DataFrame([raw_row])

        # ── Tier 1: Stage 1 binary (AAC vs OAC) ──
        s1_proba = self._predict_proba(keys["s1"], X)
        # AutoGluon may label columns as strings ('IAS') or integers (1=IAS,0=DAS)
        if "IAS" in s1_proba.columns:
            p_ias = float(s1_proba["IAS"].values[0])
        elif 1 in s1_proba.columns:
            p_ias = float(s1_proba[1].values[0])
        else:
            p_ias = float(s1_proba.iloc[:, -1].values[0])
        p_ias = float(self._apply_calibration_s1(scenario, np.array([p_ias]))[0])
        p_das = 1.0 - p_ias
        s1_threshold = sc["stage1"]["threshold"]
        s1_pred = "IAS" if p_ias >= s1_threshold else "DAS"
        s1_probs_vec = np.array([p_das, p_ias])
        s1_conf = float(max(p_das, p_ias))
        s1_cset = self._conformal_set(scenario, 1, s1_probs_vec, alpha)

        result = {
            "scenario": scenario,
            "alpha": alpha,
            "stage1": {
                "pred_internal": s1_pred,
                "pred_display": display_s1(s1_pred),
                "p_AAC": p_ias,
                "p_OAC": p_das,
                "threshold": s1_threshold,
                "conformal_set": [display_s1(c) for c in s1_cset],
            },
            "escalated": False,
            "stage2": None,
            "final": {},
        }

        # OAC -> excluded from Stage 2 (cascade stops, OAC is the answer)
        if s1_pred == "DAS":
            reflex = self._reflex("*", "LOW", 1)  # OAC needs further non-anemia workup
            result["final"] = {
                "label_internal": "DAS",
                "label_display": "OAC",
                "zone": "Excluded",
                "tier": 1,
                "confidence": s1_conf,
                "reflex": reflex,
            }
            return result

        # ── Tier 2: Stage 2 four-class (subtype) ──
        s2_proba = self._predict_proba(keys["s2"], X)
        # S2 model columns are integer class indices 0..3 (0=DEA,1=HA,2=HGB HTZ,3=Normal)
        probs = np.zeros(4)
        for i in range(4):
            if i in s2_proba.columns:
                probs[i] = float(s2_proba[i].values[0])
            elif str(i) in s2_proba.columns:
                probs[i] = float(s2_proba[str(i)].values[0])
            elif S2_CLASSES[i] in s2_proba.columns:
                probs[i] = float(s2_proba[S2_CLASSES[i]].values[0])
        # normalize defensively
        if probs.sum() > 0:
            probs = probs / probs.sum()
        s2_idx = int(np.argmax(probs))
        s2_internal = S2_CLASSES[s2_idx]
        s2_conf = float(probs[s2_idx])
        zlow, zhigh = sc["stage2"]["zone_low"], sc["stage2"]["zone_high"]
        zone = self._zone(s2_conf, zlow, zhigh)
        s2_cset = self._conformal_set(scenario, 2, probs, alpha)
        tier = 2
        reflex = self._reflex(s2_internal, zone, 1) or self._reflex(s2_internal, zone, 2) \
                 or self._reflex("*", zone, 1)

        result["escalated"] = True
        result["stage2"] = {
            "pred_internal": s2_internal,
            "pred_display": display_s2(s2_internal),
            "probs": {display_s2(S2_CLASSES[i]): float(probs[i]) for i in range(4)},
            "confidence": s2_conf,
            "zone": zone,
            "zone_low": zlow,
            "zone_high": zhigh,
            "conformal_set": [display_s2(c) for c in s2_cset],
        }
        result["final"] = {
            "label_internal": s2_internal,
            "label_display": display_s2(s2_internal),
            "zone": zone,
            "tier": tier,
            "confidence": s2_conf,
            "reflex": reflex,
        }
        return result
