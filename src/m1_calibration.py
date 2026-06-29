"""
m1_calibration.py — Probability Calibration Module
===================================================
Created in Chat 1. Usage in later modules:
    from m1_calibration import load_calibrated_probs, apply_calibration
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# ═══════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_calibrated_probs(paths, stage, scenario, split):
    """Load calibrated probabilities.

    Parameters
    ----------
    paths : CDSPaths
    stage : int (1 or 2)
    scenario : str ("CBC_Only" or "CBC_BIO")
    split : str ("oof" or "test")

    Returns
    -------
    pd.DataFrame
    """
    fname = f"stage{stage}_{scenario.lower()}_{split}_calibrated.parquet"
    return pd.read_parquet(paths.probabilities / fname)


def load_calibrator(paths, stage, scenario):
    """Load the saved calibrator. Returns None if uncalibrated.

    Returns
    -------
    dict with keys 'calibrator' and 'meta', or None
    """
    cal_dir = paths.module_dir("m1_calibration") / "calibrators"
    fpath = cal_dir / f"stage{stage}_{scenario.lower()}_calibrator.joblib"
    if fpath.exists():
        return joblib.load(fpath)
    return None


def load_calibration_registry(paths):
    """Load the calibration registry."""
    cal_dir = paths.module_dir("m1_calibration") / "calibrators"
    return joblib.load(cal_dir / "calibration_registry.joblib")


# ═══════════════════════════════════════════════════════════════════════
# TRANSFORM
# ═══════════════════════════════════════════════════════════════════════

def apply_calibration_binary(calibrator_obj, prob_positive):
    """Apply calibration to a binary probability.

    Parameters
    ----------
    calibrator_obj : dict from load_calibrator, or None
    prob_positive : np.array, positive class probability

    Returns
    -------
    np.array — calibrated probabilities
    """
    if calibrator_obj is None:
        return prob_positive

    method = calibrator_obj["meta"]["method"]
    cal = calibrator_obj["calibrator"]

    if method == "uncalibrated":
        return prob_positive
    elif method == "isotonic":
        return cal.predict(prob_positive)
    elif method == "platt":
        logits = np.log(np.clip(prob_positive, 1e-8, 1-1e-8) /
                        (1 - np.clip(prob_positive, 1e-8, 1-1e-8)))
        return cal.predict_proba(logits.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def apply_calibration_multiclass(calibrator_obj, prob_matrix, class_names):
    """Apply calibration to a multiclass probability matrix + normalize.

    Parameters
    ----------
    calibrator_obj : dict from load_calibrator, or None
    prob_matrix : np.array (n_samples, n_classes)
    class_names : list of str

    Returns
    -------
    np.array — calibrated & normalized probabilities
    """
    if calibrator_obj is None:
        return prob_matrix

    method = calibrator_obj["meta"]["method"]

    if method == "uncalibrated":
        return prob_matrix
    elif method == "isotonic":
        cals = calibrator_obj["calibrator"]
        cal_probs = np.zeros_like(prob_matrix)
        for i, cls in enumerate(class_names):
            cal_probs[:, i] = cals[cls].predict(prob_matrix[:, i])
        row_sums = cal_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return cal_probs / row_sums
    else:
        raise ValueError(f"Unknown calibration method: {method}")


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def calc_ece_mce(y_true, y_prob, n_bins=10):
    """Expected & Maximum Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (y_prob >= lo) & (y_prob <= hi) if i == 0 else (y_prob > lo) & (y_prob <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        gap = abs(y_true[mask].mean() - y_prob[mask].mean())
        ece += (n_bin / len(y_true)) * gap
        mce = max(mce, gap)
    return ece, mce