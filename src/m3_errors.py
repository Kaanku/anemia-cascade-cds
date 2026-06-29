"""
M3 Error Analysis -- Error analysis functions.
CDS Pipeline Module 3.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score)

# --- Constants ---
S1_LABELS = {0: 'DAS', 1: 'IAS'}
S2_LABELS = {0: 'DEA', 1: 'HA', 2: 'HGB_HTZ', 3: 'Normal'}
S1_NAMES  = ['DAS', 'IAS']
S2_NAMES  = ['DEA', 'HA', 'HGB_HTZ', 'Normal']

CLINICAL_SEVERITY = {
    ('DEA', 'HA'):       (2, 'Iron therapy omitted; hemolytic anemia workup initiated'),
    ('DEA', 'HGB_HTZ'):  (3, 'Iron therapy omitted; referral for genetic testing'),
    ('DEA', 'Normal'):   (3, 'IDA missed — delayed initiation of iron therapy'),
    ('HA', 'DEA'):       (2, 'Unnecessary iron therapy; hemolysis workup delayed'),
    ('HA', 'HGB_HTZ'):   (2, 'Hemolysis treatment omitted; referral for genetic testing'),
    ('HA', 'Normal'):    (3, 'Hemolysis missed — potential crisis risk'),
    ('HGB_HTZ', 'DEA'):  (3, 'Genetic counseling omitted; unnecessary iron therapy initiated'),
    ('HGB_HTZ', 'HA'):   (2, 'Genetic counseling omitted; hemolysis workup initiated'),
    ('HGB_HTZ', 'Normal'): (2, 'Carrier status overlooked — family planning affected'),
    ('Normal', 'DEA'):   (2, 'Unnecessary iron therapy initiated'),
    ('Normal', 'HA'):    (2, 'Unnecessary hemolysis workup'),
    ('Normal', 'HGB_HTZ'): (1, 'Unnecessary genetic testing — low clinical risk'),
    ('DAS', 'IAS'):  (1, 'Unnecessary further testing — low risk but increased cost'),
    ('IAS', 'DAS'):  (3, 'Anemia-associated disease missed — treatment delay'),
}

SEVERITY_LABELS = {1: 'Low', 2: 'Medium', 3: 'High'}


def get_misclassified(df, stage=2):
    """Return misclassified samples."""
    return df[df['correct'] == 0].copy()


def confusion_summary(df, stage=2):
    """Return confusion matrix + normalized version."""
    labels = S1_NAMES if stage == 1 else S2_NAMES
    label_ids = list(range(len(labels)))
    cm = confusion_matrix(df['y_true'], df['y_pred'], labels=label_ids)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return {
        'raw': pd.DataFrame(cm, index=labels, columns=labels),
        'normalized': pd.DataFrame(np.round(cm_norm, 3),
                                   index=labels, columns=labels)
    }


def per_class_metrics(df, stage=2):
    """Compute per-class precision, recall, F1."""
    labels = S1_NAMES if stage == 1 else S2_NAMES
    label_ids = list(range(len(labels)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        df['y_true'], df['y_pred'], labels=label_ids,
        average=None, zero_division=0)
    return pd.DataFrame({
        'Class': labels,
        'Precision': np.round(prec, 3),
        'Recall': np.round(rec, 3),
        'F1': np.round(f1, 3),
        'Support': sup.astype(int)
    })


def clinical_risk_score(df, stage=2):
    """Compute clinical risk scores for errors."""
    df_err = df[df['correct'] == 0].copy()
    if len(df_err) == 0:
        return pd.DataFrame()

    labels = S1_NAMES if stage == 1 else S2_NAMES
    rows = []
    for (true_cls, pred_cls), (sev, desc) in CLINICAL_SEVERITY.items():
        if true_cls in labels and pred_cls in labels:
            count = len(df_err[(df_err['y_true_label'] == true_cls) &
                               (df_err['y_pred_label'] == pred_cls)])
            if count > 0:
                rows.append({
                    'Actual': true_cls, 'Predicted': pred_cls,
                    'Error_Count': count, 'Severity': sev,
                    'Severity_Label': SEVERITY_LABELS[sev],
                    'Clinical_Impact': desc,
                    'Weighted_Risk': count * sev
                })

    return pd.DataFrame(rows).sort_values('Weighted_Risk', ascending=False)


def error_pair_frequencies(df, stage=2):
    """Error pair frequencies + confidence statistics."""
    df_err = df[df['correct'] == 0]
    if len(df_err) == 0:
        return pd.DataFrame()

    pairs = df_err.groupby(['y_true_label', 'y_pred_label']).agg(
        count=('correct', 'size'),
        mean_confidence=('confidence', 'mean'),
        median_confidence=('confidence', 'median'),
    ).reset_index()
    pairs.columns = ['True_Class', 'Pred_Class', 'Count',
                     'Mean_Conf', 'Median_Conf']
    return pairs.sort_values('Count', ascending=False)
