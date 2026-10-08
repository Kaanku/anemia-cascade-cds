"""
M2 — Threshold Optimization & Confidence Zones
================================================
Functions for runtime threshold and zone assignment in the CDS pipeline.

Usage:
    from m2_thresholds import load_threshold_config, apply_stage1_threshold, assign_confidence_zone

Last updated: Chat 2
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ──
S2_CAL_COLS = ['prob_DEA_cal', 'prob_HA_cal', 'prob_HGB_HTZ_cal', 'prob_Normal_cal']
S2_CLASSES  = ['DEA', 'HA', 'HGB_HTZ', 'Normal']


def load_threshold_config(paths):
    """Load the threshold_config.json file."""
    config_path = paths.module_dir('m2_thresholds', 'configs') / 'threshold_config.json'
    with open(config_path, 'r') as f:
        return json.load(f)


def apply_stage1_threshold(df, threshold, prob_col='prob_IAS_cal'):
    """
    Apply the Stage 1 binary threshold.

    Returns:
        pd.Series: 'IAS' or 'DAS' string labels [internal codes]
    """
    return pd.Series(
        np.where(df[prob_col].values >= threshold, 'IAS', 'DAS'),
        index=df.index, name='s1_prediction'
    )


def assign_confidence_zone(df, zone_low, zone_high, prob_cols=None):
    """
    Stage 2 confidence zone assignment.

    Args:
        df: DataFrame containing Stage 2 probabilities
        zone_low: LOW/MEDIUM boundary
        zone_high: MEDIUM/HIGH boundary
        prob_cols: Probability columns (default: S2_CAL_COLS)

    Returns:
        pd.DataFrame: pred_class, confidence, zone, pred_label columns
    """
    if prob_cols is None:
        prob_cols = S2_CAL_COLS

    probs = df[prob_cols].values
    pred_idx = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    zones = np.where(confidence < zone_low, 'LOW',
            np.where(confidence >= zone_high, 'HIGH', 'MEDIUM'))

    pred_labels = [S2_CLASSES[i] for i in pred_idx]

    return pd.DataFrame({
        'pred_class': pred_idx,
        'pred_label': pred_labels,
        'confidence': confidence,
        'zone': zones
    }, index=df.index)


def get_operating_point(config, scenario):
    """Return the operating point for a given scenario."""
    sc = config['scenarios'][scenario]
    return {
        's1_threshold': sc['stage1']['threshold'],
        's2_zone_low':  sc['stage2']['zone_low'],
        's2_zone_high': sc['stage2']['zone_high']
    }


def apply_full_pipeline(df_s1, df_s2, config, scenario):
    """
    Full CDS pipeline: S1 threshold + S2 zone assignment.

    Returns:
        s1_pred: Stage 1 predictions
        s2_result: Stage 2 zone assignments (only for those passing as IAS)
    """
    op = get_operating_point(config, scenario)

    # Stage 1
    s1_pred = apply_stage1_threshold(df_s1, op['s1_threshold'])

    # Stage 2 (only those passing as IAS)
    s2_result = assign_confidence_zone(df_s2, op['s2_zone_low'], op['s2_zone_high'])

    return s1_pred, s2_result
