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

# Feature display names for SHAP panel (mirrors m5_shap.FEATURE_DISPLAY)
FEATURE_DISPLAY = {
    "mcv_f_l": "MCV", "ret_he_pg": "Ret-He", "rbc_10_6_u_l": "RBC",
    "ret_number_10_6_l": "RET#", "rdw_sd_fl": "RDW-SD", "delta_he_pg": "Delta-He",
    "frc_perc": "FRC", "mchc_g_dl": "MCHC", "irf_pct": "IRF", "nrbc_pct": "NRBC",
    "hgb_g_d_l": "HGB", "micro_macro_ratio": "MicroR/MacroR", "yas": "Age",
    "ferritin": "Ferritin", "demir": "Iron", "ldh": "LD", "uibc": "UIBC",
    "hgb_g_d_l_div_mcv_f_l": "HGB/MCV", "hgb_g_d_l_div_ret_he_pg": "HGB/Ret-He",
    "rbc_10_6_u_l_div_mcv_f_l": "RBC/MCV", "rbc_10_6_u_l_div_ret_he_pg": "RBC/Ret-He",
    "irf_pct_div_micro_macro_ratio": "IRF/(MicroR/MacroR)",
    "hgb_g_d_l_div_rdw_sd_fl": "HGB/RDW-SD", "rbc_10_6_u_l_div_rdw_sd_fl": "RBC/RDW-SD",
    "hgb_g_d_l_div_mchc_g_dl": "HGB/MCHC", "mcv_f_l_div_rdw_sd_fl": "MCV/RDW-SD",
    "rdw_sd_fl_div_micro_macro_ratio": "RDW-SD/(MicroR/MacroR)",
    "ret_he_pg_div_micro_macro_ratio": "Ret-He/(MicroR/MacroR)",
    "ret_number_10_6_l_div_ret_he_pg": "RET#/Ret-He",
    "mchc_g_dl_div_rdw_sd_fl": "MCHC/RDW-SD",
    "mcv_f_l_div_micro_macro_ratio": "MCV/(MicroR/MacroR)",
    "hgb_g_d_l_div_rbc_10_6_u_l": "HGB/RBC", "ret_number_10_6_l_div_mcv_f_l": "RET#/MCV",
    "ret_number_10_6_l_div_mchc_g_dl": "RET#/MCHC",
    "rbc_10_6_u_l_div_mchc_g_dl": "RBC/MCHC", "rdw_sd_fl_div_ferritin": "RDW-SD/Ferritin",
    "mchc_g_dl_div_ferritin": "MCHC/Ferritin", "mcv_f_l_div_ferritin": "MCV/Ferritin",
    "rbc_10_6_u_l_div_ferritin": "RBC/Ferritin", "ret_he_pg_div_ferritin": "Ret-He/Ferritin",
    "hgb_g_d_l_div_ferritin": "HGB/Ferritin", "ret_he_pg_div_demir": "Ret-He/Iron",
    "mcv_f_l_div_demir": "MCV/Iron", "mchc_g_dl_div_demir": "MCHC/Iron",
    "rdw_sd_fl_div_demir": "RDW-SD/Iron", "rbc_10_6_u_l_div_demir": "RBC/Iron",
    "hgb_g_d_l_div_demir": "HGB/Iron", "ferritin_div_uibc": "Ferritin/UIBC",
    "demir_div_uibc": "Iron/UIBC", "ldh_div_uibc": "LD/UIBC", "demir_div_ldh": "Iron/LD",
}


def feat_display(name: str) -> str:
    return FEATURE_DISPLAY.get(name, name)


# Clinical narrative templates (condensed from m8_cds_report.NARRATIVE_TEMPLATES)
def clinical_narrative(result: dict) -> str:
    fin = result["final"]
    zone = fin["zone"]
    label = fin["label_display"]
    escalated = result["escalated"]
    s1_zone = "HIGH" if not escalated else result.get("stage2", {}).get("zone", "")

    if zone == "Excluded":
        return (f"[{label}] — Classified as Other Anemia Cause at Stage 1. "
                "The four-subtype model is not applied; further non-anemia "
                "workup (e.g. chronic disease, B12/folate, renal causes) is "
                "indicated.")
    if zone == "LOW":
        return (f"[{label}] — Classified in the low-confidence zone. The model "
                "prediction alone is insufficient; expert hematology "
                "consultation is recommended.")
    if escalated and result["final"]["tier"] == 2 and zone == "HIGH":
        return (f"[{label}] — Initially classified with moderate confidence "
                "(CBC only); confidence improved to HIGH after biochemistry was "
                "added. The cascade demonstrates the selective biochemistry "
                "strategy.")
    if zone == "MEDIUM":
        return (f"[{label}] — Moderate confidence after the available panel. "
                "Expert review and additional clinical information are "
                "recommended.")
    # HIGH at tier 1 — class-specific
    by_class = {
        "IDA": "CBC profile shows strong concordance with iron deficiency "
               "anemia. Ferritin confirmation is recommended.",
        "HA": "CBC parameters show strong concordance with hemolytic anemia; "
              "reticulocyte and hemolysis markers support this classification.",
        "HGB HTZ": "CBC profile is consistent with heterozygous "
                   "hemoglobinopathy. Hemoglobin electrophoresis is recommended "
                   "for definitive diagnosis.",
        "Normal": "CBC parameters are within normal limits; no anemia subtype "
                  "detected. Additional biochemical workup is unlikely to be "
                  "needed.",
    }
    tail = by_class.get(label, "Classified with high confidence.")
    return f"[{label}] — {tail}"


def display_s1(label: str) -> str:
    return S1_DISPLAY.get(label, label)


def display_s2(label: str) -> str:
    return S2_DISPLAY.get(label, label)


def display_diagnosis(raw_label: str) -> str:
    """Convert a raw data label (IDA/HGB_HTZ/NORMAL/...) to its display form."""
    if raw_label in DIAGNOSIS_TO_S2:
        internal = DIAGNOSIS_TO_S2[raw_label]
        return S2_DISPLAY.get(internal, internal)
    return raw_label


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
                str(self.models_dir / key),
                require_version_match=False,
                require_py_version_match=False,
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
        # reflex rules use underscore form (HGB_HTZ); internal codes use space (HGB HTZ)
        candidates = {pred_label, pred_label.replace(" ", "_"), "*"}
        for r in self.reflex:
            if r["tier"] != tier:
                continue
            if r["zone"] != zone:
                continue
            if r["prediction"] in candidates:
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
            # Prefer the explicit OAC (DAS/Excluded) rule; fall back gracefully.
            reflex = self._reflex("DAS", "Excluded", 1) or self._reflex("*", "LOW", 1)
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

    # ── SHAP explanation for one patient (KernelExplainer, on demand) ──
    def explain(self, raw_row: dict, scenario: str, stage: int,
                class_index: int, top_k: int = 10,
                n_background: int = 20) -> pd.DataFrame:
        """Per-patient SHAP top-k features for the predicted class.

        Uses KernelExplainer (mirrors m5_shap) with a small background set for
        speed. Returns a DataFrame: feature_display, shap_value (signed),
        sorted by absolute impact. Computed on demand — not in the main path.
        """
        import shap
        from functools import partial

        keys = SCENARIOS[scenario]
        model_key = keys["s1"] if stage == 1 else keys["s2"]
        pred, feats = self._predictor(model_key)

        # background: a small sample of demo training rows, ratio-completed
        bg_path = self.assets_dir / "demo_data" / "train_real_anon.csv"
        bg_raw = pd.read_csv(bg_path)
        bg = compute_missing_ratios(bg_raw, feats)
        for f in feats:
            if f not in bg.columns:
                bg[f] = np.nan
        bg = bg[feats].dropna(how="all")
        if len(bg) > n_background:
            bg = bg.sample(n_background, random_state=42)

        # patient to explain
        X = compute_missing_ratios(pd.DataFrame([raw_row]), feats)
        for f in feats:
            if f not in X.columns:
                X[f] = np.nan
        X = X[feats]

        cols = feats

        def predict_fn(arr, _ci):
            Xdf = pd.DataFrame(arr, columns=cols)
            proba = pred.predict_proba(Xdf)
            vals = proba.values if isinstance(proba, pd.DataFrame) else proba
            if stage == 1:
                # binary: explain P(IAS) (column for class 1)
                if vals.ndim == 2 and vals.shape[1] == 2:
                    return vals[:, 1]
                return vals.ravel()
            return vals[:, _ci]

        explainer = shap.KernelExplainer(
            partial(predict_fn, _ci=class_index), bg.values
        )
        sv = explainer.shap_values(X.values, silent=True, nsamples=100)
        sv = np.array(sv).reshape(-1)  # one patient

        df = pd.DataFrame({
            "feature": feats,
            "shap_value": sv,
            "abs": np.abs(sv),
        }).sort_values("abs", ascending=False).head(top_k)
        df["feature_display"] = df["feature"].map(feat_display)
        return df[["feature_display", "shap_value"]].reset_index(drop=True)
