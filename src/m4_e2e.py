"""
M4 — End-to-End Pipeline Analysis
Reusable functions for E2E simulation, confusion matrix, lost patient analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support

from m1_calibration import load_calibrated_probs
from m2_thresholds import load_threshold_config, get_operating_point


E2E_CLASSES = ['DAS', 'DEA', 'HA', 'HGB HTZ', 'Normal']
E2E_ZONES   = ['Excluded', 'HIGH', 'MEDIUM', 'LOW']


def simulate_e2e_pipeline(paths, scenario, split, th_config):
    """
    E2E pipeline simulation — S1-S2 merge via positional alignment.
    S1 filter → routing to S2 → 5-class E2E label.

    Returns: DataFrame with columns:
        idx_s1, s1_true, diagnosis, s1_prob, s1_pred,
        s2_true, s2_pred, s2_confidence, s2_zone,
        e2e_true, e2e_pred, e2e_zone, e2e_correct, is_lost,
        scenario, split
    """
    op = get_operating_point(th_config, scenario)
    s1_th  = op['s1_threshold']
    s2_low = op['s2_zone_low']
    s2_high = op['s2_zone_high']

    # Load S1
    s1_raw = load_calibrated_probs(paths, '1', scenario, split)
    prob_col_s1 = 'prob_IAS_cal' if 'prob_IAS_cal' in s1_raw.columns else 'prob_IAS'

    e2e = pd.DataFrame({
        'idx_s1':     s1_raw.index,
        's1_true':    s1_raw['target'].map({1: 'IAS', 0: 'DAS'}),
        'diagnosis':  s1_raw['diagnosis'],
        's1_prob':    s1_raw[prob_col_s1],
        's1_pred':    np.where(s1_raw[prob_col_s1] >= s1_th, 'IAS', 'DAS'),
    })

    # Load S2
    s2_raw = load_calibrated_probs(paths, '2', scenario, split)
    s2_info = pd.DataFrame({
        's2_true':       s2_raw['target_label'].values,
        's2_pred':       s2_raw['pred_label'].values,
        's2_confidence': s2_raw['confidence'].values,
    })
    s2_info['s2_zone'] = np.where(
        s2_info['s2_confidence'] >= s2_high, 'HIGH',
        np.where(s2_info['s2_confidence'] < s2_low, 'LOW', 'MEDIUM'))

    # Positional alignment: attach S2 info to S1-IAS rows
    ias_mask = (e2e['s1_true'] == 'IAS')
    for col in s2_info.columns:
        e2e[col] = np.nan
        e2e.loc[ias_mask, col] = s2_info[col].values

    # 5-class E2E labels
    e2e['e2e_true'] = np.where(e2e['s1_true'] == 'DAS', 'DAS', e2e['s2_true'])
    e2e['e2e_pred'] = np.where(e2e['s1_pred'] == 'DAS', 'DAS', e2e['s2_pred'])
    e2e['e2e_zone'] = np.where(e2e['s1_pred'] == 'DAS', 'Excluded', e2e['s2_zone'])
    e2e['e2e_correct'] = (e2e['e2e_true'] == e2e['e2e_pred'])
    e2e['is_lost'] = (e2e['s1_true'] == 'IAS') & (e2e['s1_pred'] == 'DAS')
    e2e['scenario'] = scenario
    e2e['split'] = split

    return e2e


def e2e_confusion_matrix(df, labels=None):
    """Return 5×5 raw + row-normalized confusion matrix."""
    if labels is None:
        labels = E2E_CLASSES
    cm_raw = confusion_matrix(df['e2e_true'], df['e2e_pred'], labels=labels)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_raw.astype(float), row_sums,
                        where=row_sums != 0, out=np.zeros_like(cm_raw, dtype=float))
    return cm_raw, cm_norm


def e2e_metrics(df, labels=None):
    """E2E accuracy, macro F1, per-class precision/recall/f1."""
    if labels is None:
        labels = E2E_CLASSES
    y_true, y_pred = df['e2e_true'], df['e2e_pred']
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, labels=labels, average='macro')
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    per_class = {cls: {'precision': p[i], 'recall': r[i], 'f1': f1[i], 'support': int(sup[i])}
                 for i, cls in enumerate(labels)}
    return {'accuracy': acc, 'f1_macro': f1m, 'per_class': per_class}


def lost_patient_analysis(df):
    """Summary statistics for lost patients (is_lost=True)."""
    lost = df[df['is_lost']].copy()
    n_ias = (df['s1_true'] == 'IAS').sum()

    summary = {
        'n_lost': len(lost),
        'n_ias': n_ias,
        'lost_rate': len(lost) / n_ias if n_ias > 0 else 0,
        's2_true_distribution': lost['s2_true'].value_counts().to_dict() if len(lost) > 0 else {},
        's1_prob_stats': {
            'mean': lost['s1_prob'].mean(),
            'median': lost['s1_prob'].median(),
            'min': lost['s1_prob'].min(),
            'max': lost['s1_prob'].max(),
        } if len(lost) > 0 else {},
    }
    return summary


def e2e_zone_summary(df):
    """Per-zone E2E statistics."""
    rows = []
    for zone in E2E_ZONES:
        z_df = df[df['e2e_zone'] == zone]
        rows.append({
            'zone': zone, 'n': len(z_df),
            'pct': len(z_df) / len(df) * 100,
            'accuracy': z_df['e2e_correct'].mean() if len(z_df) > 0 else np.nan
        })
    return pd.DataFrame(rows)


def compute_missing_ratios(df, feat_names):
    """Compute S2 ratio features from S1 base features."""
    df = df.copy()
    for f in feat_names:
        if f not in df.columns and '_div_' in f:
            num, den = f.split('_div_')
            if num in df.columns and den in df.columns:
                df[f] = df[num] / df[den].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ═══════════════════════════════════════════════════
# Cascade / Reflex Test Functions (M4 Extension)
# ═══════════════════════════════════════════════════

def simulate_cascade(t1_df, t2_df, escalation_zones=None):
    """
    Tier 1 → escalation → Tier 2 cascade simulation.
    escalation_zones: list of zones to escalate (default: MEDIUM, LOW, Excluded)
    Returns: DataFrame with cascade results.
    """
    if escalation_zones is None:
        escalation_zones = ['MEDIUM', 'LOW', 'Excluded']

    cascade = pd.DataFrame({
        'patient_idx':  np.arange(len(t1_df)),
        'true_label':   t1_df['e2e_true'].values,
        'tier1_pred':   t1_df['e2e_pred'].values,
        'tier1_zone':   t1_df['e2e_zone'].values,
        'tier1_correct': t1_df['e2e_correct'].values,
        'tier2_pred':   t2_df['e2e_pred'].values,
        'tier2_zone':   t2_df['e2e_zone'].values,
        'tier2_correct': t2_df['e2e_correct'].values,
    })
    cascade['escalated'] = cascade['tier1_zone'].isin(escalation_zones)
    cascade['final_pred'] = np.where(cascade['escalated'],
                                      cascade['tier2_pred'], cascade['tier1_pred'])
    cascade['final_zone'] = np.where(cascade['escalated'],
                                      cascade['tier2_zone'], cascade['tier1_zone'])
    cascade['final_correct'] = (cascade['true_label'] == cascade['final_pred'])
    return cascade


# ═══════════════════════════════════════════════════
# Cascade / Reflex Test Functions (M4 Extension)
# ═══════════════════════════════════════════════════

def simulate_cascade(t1_df, t2_df, escalation_zones=None):
    """
    Tier 1 → escalation → Tier 2 cascade simulation.
    escalation_zones: list of zones to escalate (default: MEDIUM, LOW, Excluded)
    Returns: DataFrame with cascade results.
    """
    if escalation_zones is None:
        escalation_zones = ['MEDIUM', 'LOW', 'Excluded']

    cascade = pd.DataFrame({
        'patient_idx':  np.arange(len(t1_df)),
        'true_label':   t1_df['e2e_true'].values,
        'tier1_pred':   t1_df['e2e_pred'].values,
        'tier1_zone':   t1_df['e2e_zone'].values,
        'tier1_correct': t1_df['e2e_correct'].values,
        'tier2_pred':   t2_df['e2e_pred'].values,
        'tier2_zone':   t2_df['e2e_zone'].values,
        'tier2_correct': t2_df['e2e_correct'].values,
    })
    cascade['escalated'] = cascade['tier1_zone'].isin(escalation_zones)
    cascade['final_pred'] = np.where(cascade['escalated'],
                                      cascade['tier2_pred'], cascade['tier1_pred'])
    cascade['final_zone'] = np.where(cascade['escalated'],
                                      cascade['tier2_zone'], cascade['tier1_zone'])
    cascade['final_correct'] = (cascade['true_label'] == cascade['final_pred'])
    return cascade
