"""
M6 Conformal Prediction — Reusable Functions
=============================================
APS (Adaptive Prediction Sets) for anemia classification.
Manual implementation validated against MAPIE v1.3.0.

Usage:
    from m6_conformal import (
        fit_conformal, predict_sets, compute_coverage,
        load_prediction_sets
    )
"""

import numpy as np
import pandas as pd
import os


# ── Stage/class configuration ──────────────────────────────

S1_PROB_COLS = ['prob_DAS_cal', 'prob_IAS_cal']
S2_PROB_COLS = ['prob_DEA_cal', 'prob_HA_cal', 'prob_HGB_HTZ_cal', 'prob_Normal_cal']
S1_CLASS_NAMES = {0: 'DAS', 1: 'IAS'}
S2_CLASS_NAMES = {0: 'DEA', 1: 'HA', 2: 'HGB HTZ', 3: 'Normal'}


# ── Core APS functions ─────────────────────────────────────

def compute_aps_scores(probs, labels, randomized=True, rng=None):
    """
    Compute APS nonconformity scores on calibration set.

    Parameters
    ----------
    probs : np.ndarray, shape (n, K) — calibrated probabilities
    labels : np.ndarray, shape (n,) — true class indices
    randomized : bool — if True, use randomized scores (tighter sets)
    rng : np.random.Generator — random state (only if randomized=True)

    Returns
    -------
    scores : np.ndarray, shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n, K = probs.shape
    scores = np.zeros(n)
    U = rng.uniform(0, 1, size=n) if randomized else None

    for i in range(n):
        sorted_idx = np.argsort(-probs[i])
        cumsum = 0.0
        for cls in sorted_idx:
            if randomized and cls == labels[i]:
                scores[i] = cumsum + U[i] * probs[i, cls]
                break
            elif not randomized:
                cumsum += probs[i, cls]
                if cls == labels[i]:
                    scores[i] = cumsum
                    break
            else:
                cumsum += probs[i, cls]

    return scores


def fit_conformal(probs_cal, labels_cal, alpha, randomized=True, rng=None):
    """
    Fit conformal predictor: compute scores and quantile threshold.

    Parameters
    ----------
    probs_cal : np.ndarray, shape (n, K)
    labels_cal : np.ndarray, shape (n,)
    alpha : float — significance level
    randomized : bool
    rng : np.random.Generator

    Returns
    -------
    dict with 'q_hat', 'scores', 'alpha', 'randomized'
    """
    scores = compute_aps_scores(probs_cal, labels_cal, randomized=randomized, rng=rng)
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(scores, q_level, method='higher')

    return {
        'q_hat': q_hat,
        'scores': scores,
        'alpha': alpha,
        'randomized': randomized,
        'n_cal': n,
    }


def predict_sets(probs, q_hat, randomized=True, rng=None):
    """
    Generate prediction sets for new samples.

    Parameters
    ----------
    probs : np.ndarray, shape (n, K)
    q_hat : float — conformal threshold
    randomized : bool
    rng : np.random.Generator

    Returns
    -------
    pred_sets : list of lists
    set_sizes : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng(123)

    n, K = probs.shape
    pred_sets = []
    set_sizes = np.zeros(n, dtype=int)

    if randomized:
        U = rng.uniform(0, 1, size=(n, K))

    for i in range(n):
        sorted_idx = np.argsort(-probs[i])
        cumsum = 0.0
        pset = []
        for rank, cls in enumerate(sorted_idx):
            pset.append(int(cls))
            cumsum += probs[i, cls]
            if cumsum >= q_hat:
                break
        pred_sets.append(pset)
        set_sizes[i] = len(pset)

    return pred_sets, set_sizes


def compute_coverage(pred_sets, labels, class_names):
    """
    Compute coverage metrics.

    Returns
    -------
    dict with empirical_coverage, avg_set_size, singleton_rate,
         class_coverage, size_distribution
    """
    n = len(labels)
    set_sizes = np.array([len(ps) for ps in pred_sets])
    covered = np.array([labels[i] in pred_sets[i] for i in range(n)])

    class_coverage = {}
    for cls_idx, cls_name in class_names.items():
        mask = labels == cls_idx
        if mask.sum() > 0:
            class_coverage[cls_name] = covered[mask].mean()

    return {
        'empirical_coverage': covered.mean(),
        'avg_set_size': set_sizes.mean(),
        'singleton_rate': (set_sizes == 1).mean(),
        'empty_rate': (set_sizes == 0).mean(),
        'class_coverage': class_coverage,
        'n_samples': n,
        'covered': covered,
        'set_sizes': set_sizes,
    }


def load_prediction_sets(PATHS, scenario, stage, alpha=0.10):
    """
    Load saved prediction set parquet.

    Parameters
    ----------
    PATHS : CDSPaths instance
    scenario : str — 'CBC_Only' or 'CBC_BIO'
    stage : str — '1' or '2'
    alpha : float — significance level

    Returns
    -------
    pd.DataFrame with in_* boolean columns, set_size, etc.
    """
    analysis_dir = PATHS.module_dir('m6_conformal', 'analysis')
    fname = f'prediction_sets_{scenario}_S{stage}_alpha{alpha:.2f}.parquet'
    fpath = os.path.join(analysis_dir, fname)
    return pd.read_parquet(fpath)
