"""
M7 Cascade & Reflex Engine
===========================
Tier 1 (CBC_Only) → Tier 2 (CBC_BIO) cascade simulation,
efficiency curve, and reflex test rule engine.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


S2_LABELS = {0: 'DEA', 1: 'HA', 2: 'HGB_HTZ', 3: 'Normal'}


def load_cascade_simulation(paths):
    """Load the saved cascade simulation."""
    fpath = paths.module_dir('m7_cascade', 'analysis') / 'cascade_simulation.parquet'
    return pd.read_parquet(fpath)


def load_efficiency_curve(paths):
    """Load the efficiency curve data."""
    fpath = paths.module_dir('m7_cascade', 'analysis') / 'efficiency_curve_data.parquet'
    return pd.read_parquet(fpath)


def load_reflex_rules(paths):
    """Load the reflex test rules."""
    fpath = paths.module_dir('m7_cascade', 'rules') / 'reflex_rules.json'
    with open(fpath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_reflex_recommendations(paths):
    """Load per-patient reflex recommendations."""
    fpath = paths.module_dir('m7_cascade', 'analysis') / 'reflex_recommendations.parquet'
    return pd.read_parquet(fpath)


def get_reflex_recommendation(pred_label, zone, tier, rules):
    """Find the most appropriate reflex test rule for a patient.

    Parameters
    ----------
    pred_label : str — predicted class (DEA, HA, HGB_HTZ, Normal, DAS)
    zone : str — confidence zone (HIGH, MEDIUM, LOW, Excluded)
    tier : int — cascade tier (1 or 2)
    rules : dict — reflex rule set (output of load_reflex_rules)

    Returns
    -------
    dict or None — matching rule
    """
    for rule in rules['rules']:
        if rule['tier'] != tier:
            continue
        zone_match = (rule['zone'] == zone)
        pred_match = (rule['prediction'] == pred_label or rule['prediction'] == '*')
        if zone_match and pred_match:
            return rule
    return None


def compute_e2e_pred(idx, s1_df, s2_df, ops, s1_to_s2):
    """Compute the E2E prediction for a single patient.

    Returns
    -------
    tuple: (pred_code, pred_label, confidence)
    """
    prob_ias = s1_df.loc[idx, 'prob_IAS_cal']
    s1_pred = int(prob_ias >= ops['s1_threshold'])

    if s1_pred == 0:
        return 99, 'DAS', 1 - prob_ias

    s2_idx = s1_to_s2.get(idx)
    if s2_idx is not None and s2_idx in s2_df.index:
        prob_cols = ['prob_DEA_cal', 'prob_HA_cal', 'prob_HGB_HTZ_cal', 'prob_Normal_cal']
        probs = s2_df.loc[s2_idx, prob_cols].values
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return pred, S2_LABELS[pred], conf
    else:
        return -1, 'FP_NO_S2', prob_ias


def build_s1_to_s2_mapping(s1_df, s2_df, feature_cols):
    """Feature-based index matching between S1 IAS patients and S2.

    Parameters
    ----------
    s1_df : DataFrame — full S1 data (not filtered to IAS)
    s2_df : DataFrame — S2 data (IAS patients only)
    feature_cols : list — common feature columns used for matching

    Returns
    -------
    dict: {s1_index: s2_index}
    """
    s1_ias = s1_df[s1_df['target'] == 1]
    s1_key = s1_ias[feature_cols].round(8).astype(str).agg('|'.join, axis=1)
    s2_key = s2_df[feature_cols].round(8).astype(str).agg('|'.join, axis=1)
    s2_key_to_idx = dict(zip(s2_key.values, s2_key.index))

    mapping = {}
    for s1_idx, key_val in s1_key.items():
        s2_idx = s2_key_to_idx.get(key_val)
        if s2_idx is not None:
            mapping[s1_idx] = s2_idx
    return mapping
