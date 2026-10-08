"""
m5_shap.py — SHAP Explainability Functions
Anemia CDS Project — Module 5
"""

import numpy as np
import pandas as pd
import shap
import pickle
import joblib
from pathlib import Path

FEATURE_DISPLAY = {'mcv_f_l': 'MCV', 'ret_he_pg': 'Ret-He', 'rbc_10_6_u_l': 'RBC', 'ret_number_10_6_l': 'RET#', 'rdw_sd_fl': 'RDW-SD', 'delta_he_pg': 'Delta-He', 'frc_perc': 'FRC', 'mchc_g_dl': 'MCHC', 'irf_pct': 'IRF', 'nrbc_pct': 'NRBC', 'hgb_g_d_l': 'HGB', 'micro_macro_ratio': 'MicroR/MacroR', 'yas': 'Age', 'ferritin': 'Ferritin', 'demir': 'Iron', 'ldh': 'LD', 'uibc': 'UIBC', 'hgb_g_d_l_div_mcv_f_l': 'HGB/MCV', 'hgb_g_d_l_div_ret_he_pg': 'HGB/Ret-He', 'rbc_10_6_u_l_div_mcv_f_l': 'RBC/MCV', 'rbc_10_6_u_l_div_ret_he_pg': 'RBC/Ret-He', 'irf_pct_div_micro_macro_ratio': 'IRF/(MicroR/MacroR)', 'hgb_g_d_l_div_rdw_sd_fl': 'HGB/RDW-SD', 'rbc_10_6_u_l_div_rdw_sd_fl': 'RBC/RDW-SD', 'hgb_g_d_l_div_mchc_g_dl': 'HGB/MCHC', 'mcv_f_l_div_rdw_sd_fl': 'MCV/RDW-SD', 'rdw_sd_fl_div_micro_macro_ratio': 'RDW-SD/(MicroR/MacroR)', 'ret_he_pg_div_micro_macro_ratio': 'Ret-He/(MicroR/MacroR)', 'ret_number_10_6_l_div_ret_he_pg': 'RET#/Ret-He', 'mchc_g_dl_div_rdw_sd_fl': 'MCHC/RDW-SD', 'mcv_f_l_div_micro_macro_ratio': 'MCV/(MicroR/MacroR)', 'hgb_g_d_l_div_rbc_10_6_u_l': 'HGB/RBC', 'ret_number_10_6_l_div_mcv_f_l': 'RET#/MCV', 'ret_number_10_6_l_div_mchc_g_dl': 'RET#/MCHC', 'rbc_10_6_u_l_div_mchc_g_dl': 'RBC/MCHC', 'rdw_sd_fl_div_ferritin': 'RDW-SD/Ferritin', 'mchc_g_dl_div_ferritin': 'MCHC/Ferritin', 'mcv_f_l_div_ferritin': 'MCV/Ferritin', 'rbc_10_6_u_l_div_ferritin': 'RBC/Ferritin', 'ret_he_pg_div_ferritin': 'Ret-He/Ferritin', 'hgb_g_d_l_div_ferritin': 'HGB/Ferritin', 'ret_he_pg_div_demir': 'Ret-He/Iron', 'mcv_f_l_div_demir': 'MCV/Iron', 'mchc_g_dl_div_demir': 'MCHC/Iron', 'rdw_sd_fl_div_demir': 'RDW-SD/Iron', 'rbc_10_6_u_l_div_demir': 'RBC/Iron', 'hgb_g_d_l_div_demir': 'HGB/Iron', 'ferritin_div_uibc': 'Ferritin/UIBC', 'demir_div_uibc': 'Iron/UIBC', 'ldh_div_uibc': 'LD/UIBC', 'demir_div_ldh': 'Iron/LD'}

BIO_FEATURES = ['ferritin', 'demir', 'ldh', 'uibc', 'rdw_sd_fl_div_ferritin', 'mchc_g_dl_div_ferritin', 'mcv_f_l_div_ferritin', 'rbc_10_6_u_l_div_ferritin', 'ret_he_pg_div_ferritin', 'hgb_g_d_l_div_ferritin', 'ret_he_pg_div_demir', 'mcv_f_l_div_demir', 'mchc_g_dl_div_demir', 'rdw_sd_fl_div_demir', 'rbc_10_6_u_l_div_demir', 'hgb_g_d_l_div_demir', 'ferritin_div_uibc', 'demir_div_uibc', 'ldh_div_uibc', 'demir_div_ldh']


def feat_display(name):
    return FEATURE_DISPLAY.get(name, name)


def normalize_shap(shap_vals):
    """Convert KernelSHAP output to a standard format.
    (n_samples, n_features, n_classes) -> list of (n_samples, n_features)"""
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        return [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
    return shap_vals


def load_model_for_shap(frozen_models_dir, stage, scenario):
    from autogluon.tabular import TabularPredictor  # lazy import
    pattern = f"*Stage{stage}_{scenario}*"
    model_dir = list(Path(frozen_models_dir).glob(pattern))[0]
    model = TabularPredictor.load(str(model_dir / "AutoGluon_Model"))
    features = joblib.load(model_dir / "feature_names.joblib")
    class_map = None
    cm_path = model_dir / "class_mapping.joblib"
    if cm_path.exists():
        class_map = joblib.load(cm_path)
    return model, features, class_map


def compute_shap_values(model, X_background, X_explain, stage,
                        n_background=50, cache_path=None):
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return normalize_shap(pickle.load(f))
    if len(X_background) > n_background:
        bg_idx = np.random.RandomState(42).choice(
            len(X_background), n_background, replace=False)
        bg = X_background.iloc[bg_idx]
    else:
        bg = X_background.copy()
    cols = list(X_background.columns)
    if stage == "1":
        def predict_fn(X_arr):
            X_df = pd.DataFrame(X_arr, columns=cols)
            proba = model.predict_proba(X_df)
            if isinstance(proba, pd.DataFrame):
                return proba.values[:, 1] if proba.shape[1] == 2 else proba.values
            return proba
    else:
        def predict_fn(X_arr):
            X_df = pd.DataFrame(X_arr, columns=cols)
            proba = model.predict_proba(X_df)
            return proba.values if isinstance(proba, pd.DataFrame) else proba
    explainer = shap.KernelExplainer(predict_fn, bg.values)
    shap_values = explainer.shap_values(X_explain.values, silent=True)
    shap_values = normalize_shap(shap_values)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(shap_values, f)
    return shap_values


def get_shap_importance(shap_vals, feature_names, class_names=None):
    shap_vals = normalize_shap(shap_vals)
    if isinstance(shap_vals, list):
        global_imp = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": global_imp})
        if class_names is not None:
            for i, cn in enumerate(class_names):
                df_imp[f"shap_{cn}"] = np.abs(shap_vals[i]).mean(axis=0)
    else:
        global_imp = np.abs(shap_vals).mean(axis=0)
        df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": global_imp})
    df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_imp["rank"] = range(1, len(df_imp) + 1)
    df_imp["feature_display"] = df_imp["feature"].map(feat_display)
    return df_imp


def load_cached_shap(cache_dir, stage, scenario):
    cache_file = Path(cache_dir) / f"shap_S{stage}_{scenario}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return normalize_shap(pickle.load(f))
    return None