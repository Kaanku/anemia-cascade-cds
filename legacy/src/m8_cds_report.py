"""
m8_cds_report.py — Per-patient CDS Report Module (Anemia CDS)
=============================================================
Auto-generated from CDS_08 notebook on 2026-06-28
via inspect.getsource — source of truth is the notebook.

Display labels: DEA→IDA, HGB_HTZ→HGB HTZ, LDH→LD (see SHORT_DISPLAY / FEATURE_DISPLAY).
Internal prediction keys (DEA/DAS) are preserved for cascade_df matching.

Usage:
    from m8_cds_report import generate_cds_report, select_example_patients
Dependencies: cds_config (CLASS_COLORS, ZONE_COLORS, PALETTE, PATHS), plot_style (save_fig)
"""

import json, textwrap, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from cds_config import CLASS_COLORS, ZONE_COLORS, PALETTE, PATHS
    from plot_style import save_fig, despine_all
except ImportError as e:
    raise ImportError("cds_config and plot_style are required (run from the src/ directory): " + str(e))


# === Constants (auto-extracted from notebook) ===
CONFORMAL_ALPHA = 0.1

FIG_WIDTH = 8.5

FIG_HEIGHT = 13.0

FIG_DPI = 150

COMP_WIDTH = 12.0

COMP_HEIGHT = 7.0

S2_CLASS_ORDER = ['DEA', 'HA', 'HGB HTZ', 'Normal']

S2_CLASS_NAMES = ['DEA', 'HA', 'HGB_HTZ', 'Normal']

SHORT_DISPLAY = {'DEA': 'IDA', 'HGB_HTZ': 'HGB HTZ'}

CLASS_LABELS = {'DEA': 'Iron Deficiency Anemia',
 'HA': 'Hemolytic Anemia',
 'HGB_HTZ': 'Heterozygous Hemoglobinopathy',
 'HGB HTZ': 'Heterozygous Hemoglobinopathy',
 'Normal': 'Normal',
 'DAS': 'Other (DAS)',
 'FP_NO_S2': 'No Biochemistry Data',
 'Excluded': 'Excluded at S1'}

FEATURE_DISPLAY = {'mcv_f_l': 'MCV',
 'ret_he_pg': 'Ret-He',
 'rbc_10_6_u_l': 'RBC',
 'ret_number_10_6_l': 'RET#',
 'rdw_sd_fl': 'RDW-SD',
 'delta_he_pg': 'Delta-He',
 'frc_perc': 'FRC',
 'mchc_g_dl': 'MCHC',
 'irf_pct': 'IRF',
 'nrbc_pct': 'NRBC',
 'hgb_g_d_l': 'HGB',
 'micro_macro_ratio': 'MicroR/MacroR',
 'yas': 'Age',
 'ferritin': 'Ferritin',
 'demir': 'Iron',
 'ldh': 'LD',
 'uibc': 'UIBC',
 'hgb_g_d_l_div_mcv_f_l': 'HGB/MCV',
 'hgb_g_d_l_div_ret_he_pg': 'HGB/Ret-He',
 'rbc_10_6_u_l_div_mcv_f_l': 'RBC/MCV',
 'rbc_10_6_u_l_div_ret_he_pg': 'RBC/Ret-He',
 'irf_pct_div_micro_macro_ratio': 'IRF/(MicroR/MacroR)',
 'hgb_g_d_l_div_rdw_sd_fl': 'HGB/RDW-SD',
 'rbc_10_6_u_l_div_rdw_sd_fl': 'RBC/RDW-SD',
 'hgb_g_d_l_div_mchc_g_dl': 'HGB/MCHC',
 'mcv_f_l_div_rdw_sd_fl': 'MCV/RDW-SD',
 'rdw_sd_fl_div_micro_macro_ratio': 'RDW-SD/(MicroR/MacroR)',
 'ret_he_pg_div_micro_macro_ratio': 'Ret-He/(MicroR/MacroR)',
 'ret_number_10_6_l_div_ret_he_pg': 'RET#/Ret-He',
 'mchc_g_dl_div_rdw_sd_fl': 'MCHC/RDW-SD',
 'mcv_f_l_div_micro_macro_ratio': 'MCV/(MicroR/MacroR)',
 'hgb_g_d_l_div_rbc_10_6_u_l': 'HGB/RBC',
 'ret_number_10_6_l_div_mcv_f_l': 'RET#/MCV',
 'ret_number_10_6_l_div_mchc_g_dl': 'RET#/MCHC',
 'rbc_10_6_u_l_div_mchc_g_dl': 'RBC/MCHC',
 'rdw_sd_fl_div_ferritin': 'RDW-SD/Ferritin',
 'mchc_g_dl_div_ferritin': 'MCHC/Ferritin',
 'mcv_f_l_div_ferritin': 'MCV/Ferritin',
 'rbc_10_6_u_l_div_ferritin': 'RBC/Ferritin',
 'ret_he_pg_div_ferritin': 'Ret-He/Ferritin',
 'hgb_g_d_l_div_ferritin': 'HGB/Ferritin',
 'ret_he_pg_div_demir': 'Ret-He/Iron',
 'mcv_f_l_div_demir': 'MCV/Iron',
 'mchc_g_dl_div_demir': 'MCHC/Iron',
 'rdw_sd_fl_div_demir': 'RDW-SD/Iron',
 'rbc_10_6_u_l_div_demir': 'RBC/Iron',
 'hgb_g_d_l_div_demir': 'HGB/Iron',
 'ferritin_div_uibc': 'Ferritin/UIBC',
 'demir_div_uibc': 'Iron/UIBC',
 'ldh_div_uibc': 'LD/UIBC',
 'demir_div_ldh': 'Iron/LD'}

NARRATIVE_TEMPLATES = {('HIGH', 'DEA', 'Tier1'): "This patient's CBC profile shows strong concordance with iron "
                           'deficiency anemia (IDA). Classified in the high-confidence zone, this '
                           'result can guide treatment planning without additional biochemical '
                           'testing. Ferritin confirmation is recommended.',
 ('HIGH', 'HA', 'Tier1'): 'CBC parameters show strong concordance with hemolytic anemia (HA). '
                          'Reticulocyte parameters and hemolysis markers support this '
                          'classification.',
 ('HIGH', 'HGB_HTZ', 'Tier1'): 'CBC profile is consistent with heterozygous hemoglobinopathy. '
                               'Hemoglobin electrophoresis is recommended for definitive '
                               'diagnosis.',
 ('HIGH', 'Normal', 'Tier1'): 'CBC parameters are within normal limits; no anemia subtype '
                              'detected. Additional biochemical workup is unlikely to be needed.',
 ('MEDIUM', '*', 'Tier1_to_Tier2_HIGH'): 'Initially classified with moderate confidence (CBC '
                                         "only), this patient's confidence improved to HIGH after "
                                         'biochemistry was added. The cascade system demonstrates '
                                         'the effectiveness of selective biochemistry strategy.',
 ('MEDIUM', '*', 'still_MEDIUM'): 'Moderate confidence was obtained even after both CBC and '
                                  'biochemistry evaluation. Expert review and additional clinical '
                                  'information are recommended.',
 ('LOW', '*', '*'): 'Classified in the low-confidence zone, this result indicates the model '
                    'prediction alone is insufficient. Expert hematology consultation is '
                    'recommended.',
 ('Excluded', '*', 'escalated'): 'Initially excluded from anemia-related classification at S1, '
                                 'this patient was redirected to biochemical evaluation by the '
                                 'cascade system.',
 ('HIGH', '*', 'Tier2'): 'Classified with high confidence after biochemistry was added. The full '
                         'biochemistry panel provided a definitive result.',
 ('HIGH', 'HGB HTZ', 'Tier1'): 'CBC profile is consistent with heterozygous hemoglobinopathy. '
                               'Hemoglobin electrophoresis is recommended for definitive '
                               'diagnosis.'}


# === Functions (auto-extracted from notebook, verbatim) ===

def disp_label(lbl):
    """Map internal short code to display form: DEA→IDA, HGB_HTZ→HGB HTZ."""
    return SHORT_DISPLAY.get(str(lbl), str(lbl).replace('_', ' '))


def get_patient_probs(patient_row, probs_dict, conf_data_dict):
    """Return probability vectors and conformal sets for a patient."""
    pid = patient_row['patient_idx']

    result = {
        'tier1_probs': None, 'tier2_probs': None,
        'tier1_conf_set': None, 'tier2_conf_set': None,
    }

    # S2 probability/conformal files are IAS-only (165 rows), indexed in the
    # same order as cascade_to_s2. Map cascade patient_idx → S2 row position.
    # Non-IAS patients (DAS / Excluded) are absent from S2 → leave as None.
    s2_row = cascade_to_s2.get(int(pid))

    # Tier 1 = CBC_Only S2 probs
    df_cbc = probs_dict.get('CBC_Only')
    if df_cbc is not None and s2_row is not None and s2_row < len(df_cbc):
        prob_cols = ['prob_DEA_cal', 'prob_HA_cal', 'prob_HGB_HTZ_cal', 'prob_Normal_cal']
        existing = [c for c in prob_cols if c in df_cbc.columns]
        if existing:
            result['tier1_probs'] = df_cbc.iloc[s2_row][existing].values.astype(float)

    # Tier 2 = CBC_BIO S2 probs
    df_bio = probs_dict.get('CBC_BIO')
    if df_bio is not None and s2_row is not None and s2_row < len(df_bio):
        prob_cols = ['prob_DEA_cal', 'prob_HA_cal', 'prob_HGB_HTZ_cal', 'prob_Normal_cal']
        existing = [c for c in prob_cols if c in df_bio.columns]
        if existing:
            result['tier2_probs'] = df_bio.iloc[s2_row][existing].values.astype(float)

    # Conformal sets (S2 files are IAS-only, same ordering as cascade_to_s2)
    for tier_key, (scen, stage) in [('tier1_conf_set', ('CBC_Only', 'S2')),
                                      ('tier2_conf_set', ('CBC_BIO', 'S2'))]:
        df_conf = conf_data_dict.get((scen, stage))
        if df_conf is not None and s2_row is not None and s2_row < len(df_conf):
            set_cols = [c for c in df_conf.columns if c.startswith('in_')]
            if set_cols:
                row = df_conf.iloc[s2_row]
                members = [c.replace('in_', '') for c in set_cols if row[c] == True]
                result[tier_key] = members

    # Fall back to conformal info carried in the cascade dataframe
    if result['tier1_conf_set'] is None:
        raw = patient_row.get('tier1_conformal_set')
        if raw is not None and raw == raw:  # NaN check
            if isinstance(raw, str):
                try:
                    result['tier1_conf_set'] = eval(raw) if raw.startswith('{') or raw.startswith('[') else [raw]
                except:
                    result['tier1_conf_set'] = [raw]
            elif isinstance(raw, (list, set)):
                result['tier1_conf_set'] = list(raw)

    if result['tier2_conf_set'] is None:
        raw = patient_row.get('tier2_conformal_set')
        if raw is not None and raw == raw:
            if isinstance(raw, str):
                try:
                    result['tier2_conf_set'] = eval(raw) if raw.startswith('{') or raw.startswith('[') else [raw]
                except:
                    result['tier2_conf_set'] = [raw]
            elif isinstance(raw, (list, set)):
                result['tier2_conf_set'] = list(raw)

    return result


def get_narrative(patient_row):
    """Select the matching narrative template for a patient."""
    zone = patient_row['final_zone']
    pred = patient_row['final_pred_label']
    tier = patient_row['final_tier']
    tier = f"Tier {tier}" if isinstance(tier, (int, np.integer)) else str(tier)
    t1_zone = patient_row.get('tier1_zone', '')
    t2_zone = patient_row.get('tier2_zone', '')
    escalated = patient_row.get('escalated', False)

    # 1. Exact match
    key = (zone, pred, tier)
    if key in NARRATIVE_TEMPLATES:
        return NARRATIVE_TEMPLATES[key]

    # 2. Tier1 HIGH — class-specific
    if tier == 'Tier1' and zone == 'HIGH':
        key = (zone, pred, 'Tier1')
        if key in NARRATIVE_TEMPLATES:
            return NARRATIVE_TEMPLATES[key]

    # 3. Escalation: MEDIUM → Tier2 HIGH
    if escalated and t1_zone == 'MEDIUM' and t2_zone == 'HIGH':
        key = ('MEDIUM', '*', 'Tier1_to_Tier2_HIGH')
        if key in NARRATIVE_TEMPLATES:
            return NARRATIVE_TEMPLATES[key]

    # 4. Still MEDIUM after Tier2
    if t2_zone == 'MEDIUM' and t1_zone == 'MEDIUM':
        key = ('MEDIUM', '*', 'still_MEDIUM')
        if key in NARRATIVE_TEMPLATES:
            return NARRATIVE_TEMPLATES[key]

    # 5. Excluded → escalated
    if t1_zone == 'Excluded' and escalated:
        key = ('Excluded', '*', 'escalated')
        if key in NARRATIVE_TEMPLATES:
            return NARRATIVE_TEMPLATES[key]

    # 6. LOW zone
    if zone == 'LOW':
        key = ('LOW', '*', '*')
        if key in NARRATIVE_TEMPLATES:
            return NARRATIVE_TEMPLATES[key]

    # 7. Tier2 HIGH (any class)
    if tier == 'Tier2' and zone == 'HIGH':
        key = ('HIGH', '*', 'Tier2')
        if key in NARRATIVE_TEMPLATES:
            return NARRATIVE_TEMPLATES[key]

    # 8. Generic fallback
    return f"Prediction: {disp_label(pred)}, Confidence zone: {zone}, Source: {tier}."


def draw_title_band(ax, patient_row):
    """Patient title band: ID, prediction badge, zone badge, tier."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis('off')

    pid = patient_row['patient_idx']
    pred = patient_row['final_pred_label']
    zone = patient_row['final_zone']
    tier = patient_row['final_tier']
    tier = f"Tier {tier}" if isinstance(tier, (int, np.integer)) else str(tier)
    true_lbl = patient_row['true_label_name']

    pred_color = CLASS_COLORS.get(pred, CLASS_COLORS.get(pred.replace('_', ' '), '#999'))
    zone_color = ZONE_COLORS.get(zone, '#999')

    # Patient ID
    ax.text(2, 5, f"Patient #{pid:03d}", fontsize=16, fontweight='bold', va='center')

    # Prediction badge
    bbox_pred = FancyBboxPatch((28, 2.5), 25, 5.5, boxstyle="round,pad=0.3",
                                facecolor=pred_color, edgecolor='none', alpha=0.9)
    ax.add_patch(bbox_pred)
    ax.text(40.5, 5.2, CLASS_LABELS.get(pred, pred), fontsize=11, fontweight='bold',
            va='center', ha='center', color='white')

    # Zone badge
    bbox_zone = FancyBboxPatch((56, 2.5), 12, 5.5, boxstyle="round,pad=0.3",
                                facecolor=zone_color, edgecolor='none', alpha=0.85)
    ax.add_patch(bbox_zone)
    ax.text(62, 5.2, zone, fontsize=10, fontweight='bold',
            va='center', ha='center', color='white')

    # Tier
    tier_display = f"Tier {tier}" if str(tier).isdigit() else str(tier)
    ax.text(72, 5.2, tier_display, fontsize=10, va='center', color=PALETTE['base1'])

    # True label (small reference)
    ax.text(85, 5.2, f"(True: {disp_label(true_lbl)})", fontsize=8, va='center',
            color='#333', fontstyle='italic')

    # Bottom line
    ax.plot([2, 98], [0.5, 0.5], color=PALETTE['base2'], linewidth=1.5, clip_on=False)


def draw_cascade_diagram(ax, patient_row):
    """Cascade flow diagram."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis('off')

    t1_zone = patient_row.get('tier1_zone', '?')
    t1_pred = patient_row.get('tier1_pred_label', '?')
    t2_zone = patient_row.get('tier2_zone', '')
    t2_pred = patient_row.get('tier2_pred_label', '')
    escalated = patient_row.get('escalated', False)
    esc_reason = patient_row.get('escalation_reason', '')
    final_pred = patient_row['final_pred_label']
    final_zone = patient_row['final_zone']

    box_h, box_w = 4.5, 18
    y_mid = 2.8
    bstyle = "round,pad=0.2"

    def add_box(x, y, line1, line2, color, tc='white'):
        bp = FancyBboxPatch((x, y), box_w, box_h, boxstyle=bstyle,
                            facecolor=color, edgecolor='#444', linewidth=0.6, alpha=0.9)
        ax.add_patch(bp)
        ax.text(x + box_w/2, y + box_h*0.62, line1, fontsize=8.5, fontweight='bold',
                ha='center', va='center', color=tc)
        if line2:
            ax.text(x + box_w/2, y + box_h*0.28, line2, fontsize=7.5,
                    ha='center', va='center', color=tc, alpha=0.9)

    def arrow(x1, x2, y):
        ax.annotate('', xy=(x2, y + box_h/2), xytext=(x1 + box_w, y + box_h/2),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.3))

    final_color = CLASS_COLORS.get(final_pred, CLASS_COLORS.get(final_pred.replace('_', ' '), '#555'))

    if not escalated:
        # Short path: CBC → Tier1 → Final
        add_box(5, y_mid, 'CBC', 'Hemogram', PALETTE['base1'])
        arrow(5, 28, y_mid)
        t1c = ZONE_COLORS.get(t1_zone, '#999')
        add_box(28, y_mid, f'Tier 1: {disp_label(t1_pred)}', t1_zone, t1c)
        arrow(28, 51, y_mid)
        ax.text(48.5, y_mid + box_h + 0.8, 'No escalation', fontsize=7.5,
                ha='center', color=PALETTE['accent1'], fontweight='bold')
        add_box(51, y_mid, f'Final: {disp_label(final_pred)}', final_zone, final_color)

    elif t1_zone == 'Excluded':
        # Excluded → Escalate → Tier2 → Final
        add_box(2, y_mid, 'CBC', 'Hemogram', PALETTE['base1'])
        arrow(2, 22, y_mid)
        add_box(22, y_mid, 'S1: Excluded', 'OAC predicted', '#888')
        arrow(22, 42, y_mid)
        ax.text(40, y_mid + box_h + 0.8, 'Escalate', fontsize=7.5,
                ha='center', color=PALETTE['accent2'], fontweight='bold')
        t2c = ZONE_COLORS.get(t2_zone, '#999')
        add_box(42, y_mid, f'Tier 2: {disp_label(t2_pred)}', t2_zone, t2c)
        arrow(42, 63, y_mid)
        add_box(63, y_mid, f'Final: {disp_label(final_pred)}', final_zone, final_color)

    else:
        # Long path: CBC → Tier1 → Escalate → Tier2 → Final
        bw = 16  # slightly narrower boxes
        # CBC box drawn with bw (not global box_w=18) for uniform spacing
        bp0 = FancyBboxPatch((1, y_mid), bw, box_h, boxstyle=bstyle,
                             facecolor=PALETTE['base1'], edgecolor='#444',
                             linewidth=0.6, alpha=0.9)
        ax.add_patch(bp0)
        ax.text(1 + bw/2, y_mid + box_h*0.62, 'CBC', fontsize=8,
                fontweight='bold', ha='center', va='center', color='white')
        ax.text(1 + bw/2, y_mid + box_h*0.28, 'Hemogram', fontsize=7.5,
                ha='center', va='center', color='white', alpha=0.9)
        arrow_x = 1
        ax2 = 1 + bw + 2
        # override box_w locally
        bp1 = FancyBboxPatch((ax2, y_mid), bw, box_h, boxstyle=bstyle,
                              facecolor=ZONE_COLORS.get(t1_zone, '#999'),
                              edgecolor='#444', linewidth=0.6, alpha=0.9)
        ax.add_patch(bp1)
        ax.text(ax2 + bw/2, y_mid + box_h*0.62, f'Tier 1: {disp_label(t1_pred)}', fontsize=8,
                fontweight='bold', ha='center', va='center', color='white')
        ax.text(ax2 + bw/2, y_mid + box_h*0.28, t1_zone, fontsize=7.5,
                ha='center', va='center', color='white', alpha=0.9)
        ax.annotate('', xy=(ax2, y_mid + box_h/2),
                    xytext=(1 + bw, y_mid + box_h/2),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))

        ax3 = ax2 + bw + 2
        reason_short = esc_reason.replace('zone_', '').replace('fp_no_s2_data', 'no BIO')[:15]
        ax.text((ax2 + bw + ax3)/2, y_mid + box_h + 0.8, reason_short, fontsize=7,
                ha='center', color=PALETTE['accent2'], fontweight='bold')
        ax.annotate('', xy=(ax3, y_mid + box_h/2),
                    xytext=(ax2 + bw, y_mid + box_h/2),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))

        t2c = ZONE_COLORS.get(t2_zone, '#999')
        bp2 = FancyBboxPatch((ax3, y_mid), bw, box_h, boxstyle=bstyle,
                              facecolor=t2c, edgecolor='#444', linewidth=0.6, alpha=0.9)
        ax.add_patch(bp2)
        ax.text(ax3 + bw/2, y_mid + box_h*0.62, f'Tier 2: {disp_label(t2_pred)}', fontsize=8,
                fontweight='bold', ha='center', va='center', color='white')
        ax.text(ax3 + bw/2, y_mid + box_h*0.28, t2_zone, fontsize=7.5,
                ha='center', va='center', color='white', alpha=0.9)

        ax4 = ax3 + bw + 2
        ax.annotate('', xy=(ax4, y_mid + box_h/2),
                    xytext=(ax3 + bw, y_mid + box_h/2),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))
        bp3 = FancyBboxPatch((ax4, y_mid), bw, box_h, boxstyle=bstyle,
                              facecolor=final_color, edgecolor='#444', linewidth=0.6, alpha=0.9)
        ax.add_patch(bp3)
        ax.text(ax4 + bw/2, y_mid + box_h*0.62, f'Final: {disp_label(final_pred)}', fontsize=8,
                fontweight='bold', ha='center', va='center', color='white')
        ax.text(ax4 + bw/2, y_mid + box_h*0.28, final_zone, fontsize=7.5,
                ha='center', va='center', color='white', alpha=0.9)


def draw_prob_barchart(ax, probs_vec, pred_label, title=''):
    """4-class calibrated probability horizontal bar chart."""
    if probs_vec is None or len(probs_vec) < 4:
        ax.text(0.5, 0.5, 'Probability data\nnot available', ha='center', va='center',
                fontsize=10, color='#999', transform=ax.transAxes)
        ax.axis('off')
        return

    classes = S2_CLASS_ORDER
    colors = [CLASS_COLORS.get(c, '#999') for c in classes]
    pred_norm = pred_label.replace('_', ' ')
    alphas = [1.0 if c == pred_norm else 0.45 for c in classes]

    bars = ax.barh(range(4), probs_vec, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.55)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    ax.set_yticks(range(4))
    _disp = {'DEA': 'IDA', 'HGB HTZ': 'HGB HTZ'}
    ax.set_yticklabels([_disp.get(c, c) for c in classes], fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Calibrated Probability', fontsize=8)
    ax.invert_yaxis()

    for i, v in enumerate(probs_vec):
        ax.text(v + 0.015, i, f'{v:.3f}', va='center', fontsize=8,
                fontweight='bold' if alphas[i] == 1.0 else 'normal')

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    despine_all(ax)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(left=False)


def draw_conformal_set(ax, conf_set, set_size_override=None, title=''):
    """Conformal prediction set display."""
    ax.axis('off')

    if conf_set is None:
        ax.text(0.5, 0.5, 'Conformal set\nnot available', ha='center', va='center',
                fontsize=10, color='#999', transform=ax.transAxes)
        return

    set_size = len(conf_set) if set_size_override is None else int(set_size_override)

    if set_size == 1:
        size_color, size_label = ZONE_COLORS['HIGH'], 'Certain'
    elif set_size == 2:
        size_color, size_label = ZONE_COLORS['MEDIUM'], 'Additional test suggested'
    else:
        size_color, size_label = ZONE_COLORS['LOW'], 'Uncertain — expert review'

    if title:
        ax.text(0.5, 0.95, title, fontsize=9, fontweight='bold', ha='center',
                va='top', transform=ax.transAxes)

    # Set members
    _disp = {'DEA': 'IDA'}
    set_str = '{ ' + ',  '.join(disp_label(s) for s in conf_set) + ' }'
    ax.text(0.5, 0.55, set_str, fontsize=14, fontweight='bold', ha='center',
            va='center', color='#333', transform=ax.transAxes)

    # Size badge
    badge = FancyBboxPatch((0.15, 0.08), 0.7, 0.22,
                            boxstyle="round,pad=0.03", facecolor=size_color,
                            edgecolor='none', alpha=0.15, transform=ax.transAxes)
    ax.add_patch(badge)
    ax.text(0.5, 0.19, f'Set size: {set_size} — {size_label}',
            fontsize=8.5, ha='center', va='center', color=size_color,
            fontweight='bold', transform=ax.transAxes)


def draw_shap_waterfall(ax, patient_row, top_n=10):
    """
    SHAP horizontal bar for the predicted class — top N features.
    Uses shap_values, shap_features, cascade_to_s2, FEATURE_DISPLAY from outer scope.
    """
    pid = patient_row['patient_idx']
    tier = patient_row['final_tier']
    tier = f"Tier {tier}" if isinstance(tier, (int, np.integer)) else str(tier)
    tier = tier.replace(' ', '')
    pred_label = patient_row['final_pred_label']

    # Determine scenario
    scenario = 'CBC_BIO' if tier == 'Tier2' else 'CBC_Only'

    # Map class label → index
    pred_norm = pred_label.replace(' ', '_')
    if pred_norm not in S2_CLASS_NAMES:
        _draw_shap_placeholder(ax, f"No SHAP for class '{pred_label}'")
        return
    class_idx = S2_CLASS_NAMES.index(pred_norm)

    # Check SHAP data availability
    shap_key = ('test', '2', scenario)
    if shap_key not in shap_values:
        _draw_shap_placeholder(ax, f"SHAP test S2 {scenario} not available")
        return

    # Map cascade patient_idx → S2 row index
    if pid not in cascade_to_s2:
        _draw_shap_placeholder(ax, f"Patient {pid} not in S2 (DAS/Excluded)")
        return
    s2_idx = cascade_to_s2[pid]

    # Extract SHAP values for this patient & predicted class
    shap_list = shap_values[shap_key]   # list of 4 arrays, each (165, n_feat)
    shap_arr = shap_list[class_idx]
    if s2_idx >= shap_arr.shape[0]:
        _draw_shap_placeholder(ax, f"S2 index {s2_idx} out of range")
        return

    patient_shap = shap_arr[s2_idx]     # (n_features,)
    feat_names = shap_features.get(('2', scenario), [f'f{i}' for i in range(len(patient_shap))])

    # Top N by absolute value
    abs_vals = np.abs(patient_shap)
    top_idx = np.argsort(abs_vals)[-top_n:][::-1]

    values = patient_shap[top_idx]
    names = [FEATURE_DISPLAY.get(feat_names[i], feat_names[i]) for i in top_idx]

    # Draw horizontal bars
    colors = [PALETTE['highlight'] if v > 0 else PALETTE['accent1'] for v in values]
    y_pos = np.arange(len(values))

    ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5,
            height=0.6, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8, fontfamily='sans-serif')
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value (impact on prediction)', fontsize=8, fontfamily='sans-serif')
    _pred_disp = {'DEA': 'IDA'}.get(pred_label, pred_label)
    ax.set_title(f'SHAP — Top {top_n} Features for {disp_label(pred_label)} ({scenario.replace("_", " ")})',
                 fontsize=9, fontweight='bold', fontfamily='sans-serif', pad=8)

    # Value labels
    for i, v in enumerate(values):
        ha = 'left' if v >= 0 else 'right'
        offset = abs(values).max() * 0.03 * (1 if v >= 0 else -1)
        ax.text(v + offset, i, f'{v:+.3f}', va='center', ha=ha, fontsize=7, fontfamily='sans-serif')

    # Zero line
    ax.axvline(x=0, color='#333', linewidth=0.8)
    despine_all(ax)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(left=False)


def draw_reflex_recommendation(ax, patient_row, df_reflex_data):
    """Reflex test recommendation panel."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    reflex = get_reflex_info(patient_row, df_reflex_data)

    if reflex is None:
        ax.text(5, 1.1, 'No reflex recommendation available', ha='center', va='center',
                fontsize=9, color='#999', fontfamily='sans-serif')
        return

    test = reflex.get('reflex_test', '—')
    urgency = reflex.get('urgency', '—')
    rationale = reflex.get('rationale', '—')

    # Title
    ax.text(0.3, 1.95, 'Reflex Test Recommendation', fontsize=10,
            fontweight='bold', fontfamily='sans-serif', color=PALETTE['base1'])

    # Urgency badge
    urg_color = URGENCY_COLORS.get(urgency, '#999')
    badge = FancyBboxPatch((7.0, 1.7), 2.5, 0.4, boxstyle="round,pad=0.08",
                            facecolor=urg_color, edgecolor='none', alpha=0.85)
    ax.add_patch(badge)
    ax.text(8.25, 1.9, urgency.upper() if urgency else '—', fontsize=9,
            fontweight='bold', ha='center', va='center', color='white', fontfamily='sans-serif')

    # Test
    ax.text(0.5, 1.35, 'Recommended Test:', fontsize=8, fontweight='bold',
            fontfamily='sans-serif', color='#555')
    ax.text(3.2, 1.35, str(test), fontsize=9, fontfamily='sans-serif', color='#333')

    # Rationale
    ax.text(0.5, 0.85, 'Rationale:', fontsize=8, fontweight='bold',
            fontfamily='sans-serif', color='#555')
    wrapped = textwrap.fill(str(rationale), width=65)
    ax.text(3.2, 0.85, wrapped, fontsize=8, fontfamily='sans-serif', color='#555',
            va='top', linespacing=1.4)


def generate_cds_report(patient_row, probs_dict, conf_data_dict, df_reflex_data,
                        output_dir=None, show=True):
    """Generate full CDS report figure for a single patient."""

    pdata = get_patient_probs(patient_row, probs_dict, conf_data_dict)
    pid = patient_row['patient_idx']
    tier = patient_row['final_tier']
    # Normalize tier to string format
    tier = f"Tier {tier}" if isinstance(tier, (int, np.integer)) else str(tier)
    tier = tier.replace(' ', '')
    escalated = patient_row.get('escalated', False)

    # Active probs & conformal
    if tier == 'Tier2' and pdata['tier2_probs'] is not None:
        active_probs = pdata['tier2_probs']
        active_conf = pdata['tier2_conf_set']
    else:
        active_probs = pdata['tier1_probs']
        active_conf = pdata['tier1_conf_set']

    conf_size = patient_row.get('tier2_conformal_size' if tier == 'Tier2' else 'tier1_conformal_size', None)
    if conf_size is not None and conf_size == conf_size:
        conf_size = int(conf_size)
    else:
        conf_size = len(active_conf) if active_conf else None

    # ---- Layout ----
    fig = plt.figure(figsize=(FIG_WIDTH, 11.5), dpi=FIG_DPI, facecolor='white')

    gs = gridspec.GridSpec(
        6, 2, figure=fig,
        height_ratios=[0.5, 0.7, 1.5, 1.5, 1.0, 1.8],
        width_ratios=[1.1, 0.9],
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.94, top=0.96, bottom=0.04
    )

    # Row 0: Title (full width)
    ax_title = fig.add_subplot(gs[0, :])
    draw_title_band(ax_title, patient_row)

    # Row 1: Cascade (full width)
    ax_cascade = fig.add_subplot(gs[1, :])
    draw_cascade_diagram(ax_cascade, patient_row)

    # Row 2 left: Probability bars
    ax_prob = fig.add_subplot(gs[2, 0])
    prob_title = f"{'Tier 2 (CBC+BIO)' if tier == 'Tier2' else 'Tier 1 (CBC Only)'} Probabilities"
    draw_prob_barchart(ax_prob, active_probs, patient_row['final_pred_label'], title=prob_title)

    # Row 2 right: Conformal set
    ax_conf = fig.add_subplot(gs[2, 1])
    draw_conformal_set(ax_conf, active_conf, set_size_override=conf_size,
                       title=f'Conformal Set (α={CONFORMAL_ALPHA})')

    # Row 3: SHAP waterfall (full width)
    ax_shap = fig.add_subplot(gs[3, :])
    draw_shap_waterfall(ax_shap, patient_row)

    # Row 4: Reflex (full width)
    ax_reflex = fig.add_subplot(gs[4, :])
    draw_reflex_recommendation(ax_reflex, patient_row, df_reflex_data)

    # Row 5: Narrative + Footer combined (full width)
    ax_bottom = fig.add_subplot(gs[5, :])
    ax_bottom.set_xlim(0, 100)
    ax_bottom.set_ylim(0, 20)
    ax_bottom.axis('off')

    # Narrative
    narr = get_narrative(patient_row)
    pred = patient_row['final_pred_label']
    pred_en = CLASS_LABELS.get(pred, pred)
    display_text = narr if pred_en in narr else f"[{pred_en}] — {narr}"
    wrapped = textwrap.fill(display_text, width=95)

    ax_bottom.text(2, 18, 'Clinical Narrative', fontsize=10, fontweight='bold',
                   color=PALETTE['base1'])
    ax_bottom.plot([2, 98], [17, 17], color='#eee', linewidth=0.8, clip_on=False)
    ax_bottom.text(3, 15.5, wrapped, fontsize=8.5, color='#333',
                   va='top', linespacing=1.5)

    # Footer
    ax_bottom.plot([2, 98], [3, 3], color='#ddd', linewidth=0.6, clip_on=False)
    tier_display = f"Tier {tier}" if str(tier).isdigit() else str(tier)
    ax_bottom.text(2, 1.5, f"CDS Pipeline v1.0  |  Source: {tier_display}  |  α={CONFORMAL_ALPHA}",
                   fontsize=7, color='#333')
    ax_bottom.text(2, 0,
                   "This report is for clinical decision support only. "
                   "Final diagnosis authority rests with the treating physician.",
                   fontsize=6.5, color='#555', fontstyle='italic')

    if output_dir:
        save_fig(fig, str(output_dir), f"cds_report_patient_{pid:03d}")

    if show:
        plt.show()

    return fig, {
        'patient_idx': pid,
        'final_pred': patient_row['final_pred_label'],
        'final_zone': patient_row['final_zone'],
        'final_tier': tier,
        'true_label': patient_row['true_label_name'],
        'correct': patient_row['correct'],
        'escalated': escalated,
        'conf_set_size': conf_size,
    }


def generate_comparison_panel(patient_row, probs_dict, conf_data_dict,
                               output_dir=None, show=True):
    """
    Side-by-side Tier1 vs Tier2 comparison for an escalated patient.
    Meaningful only for escalated=True patients.
    """
    pdata = get_patient_probs(patient_row, probs_dict, conf_data_dict)
    pid = patient_row['patient_idx']
    pred = patient_row['final_pred_label']

    fig, axes = plt.subplots(2, 2, figsize=(COMP_WIDTH, COMP_HEIGHT),
                              facecolor='white', dpi=FIG_DPI)

    # Top left: Tier 1 prob bars
    draw_prob_barchart(axes[0, 0], pdata['tier1_probs'], patient_row.get('tier1_pred_label', pred),
                       title='CBC Only (Tier 1)')

    # Top right: Tier 2 prob bars
    draw_prob_barchart(axes[0, 1], pdata['tier2_probs'], patient_row.get('tier2_pred_label', pred),
                       title='CBC + BIO (Tier 2)')

    # Bottom left: Tier 1 conformal
    draw_conformal_set(axes[1, 0], pdata['tier1_conf_set'],
                       title=f'Tier 1 Conformal (α={CONFORMAL_ALPHA})')

    # Bottom right: Tier 2 conformal
    draw_conformal_set(axes[1, 1], pdata['tier2_conf_set'],
                       title=f'Tier 2 Conformal (α={CONFORMAL_ALPHA})')

    fig.suptitle(f"Patient #{pid:03d} — Escalation Comparison: {disp_label(patient_row.get('tier1_pred_label','?'))} "
                 f"({patient_row.get('tier1_zone','?')}) → {disp_label(patient_row.get('tier2_pred_label','?'))} "
                 f"({patient_row.get('tier2_zone','?')})",
                 fontsize=12, fontweight='bold', fontfamily='sans-serif', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_dir:
        save_fig(fig, str(output_dir), f"comparison_patient_{pid:03d}")

    if show:
        plt.show()

    return fig


def generate_html_report(patient_row, probs_dict, conf_data_dict,
                          df_reflex_data, output_dir=None):
    """Generate a minimal HTML report for a single patient."""
    pid = patient_row['patient_idx']
    pred = patient_row['final_pred_label']
    zone = patient_row['final_zone']
    tier = patient_row['final_tier']
    tier = f"Tier {tier}" if isinstance(tier, (int, np.integer)) else str(tier)
    true_lbl = patient_row['true_label_name']
    escalated = patient_row.get('escalated', False)

    pdata = get_patient_probs(patient_row, probs_dict, conf_data_dict)
    narr = get_narrative(patient_row)
    reflex = get_reflex_info(patient_row, df_reflex_data)

    pred_color = CLASS_COLORS.get(pred, CLASS_COLORS.get(pred.replace('_', ' '), '#999'))
    zone_color = ZONE_COLORS.get(zone, '#999')

    # Active probabilities & conformal
    active_probs = pdata['tier2_probs'] if tier == 'Tier2' and pdata['tier2_probs'] is not None else pdata['tier1_probs']
    active_conf = pdata['tier2_conf_set'] if tier == 'Tier2' else pdata['tier1_conf_set']

    # --- Probability bars HTML ---
    prob_bars_html = ''
    if active_probs is not None:
        for cls, p in zip(S2_CLASS_ORDER, active_probs):
            c = CLASS_COLORS.get(cls, '#999')
            w = max(p * 100, 2)
            highlight = 'font-weight:bold;' if cls == pred.replace('_', ' ') else 'opacity:0.6;'
            prob_bars_html += f'''
            <div style="display:flex;align-items:center;margin:4px 0;{highlight}">
              <span style="width:80px;font-size:13px;">{cls}</span>
              <div style="flex:1;background:#eee;border-radius:4px;height:22px;margin:0 8px;">
                <div style="width:{w:.1f}%;background:{c};height:100%;border-radius:4px;"></div>
              </div>
              <span style="font-size:12px;min-width:50px;">{p:.3f}</span>
            </div>'''

    # --- Conformal set HTML ---
    conf_html = ''
    if active_conf:
        size = len(active_conf)
        if size == 1:
            badge_color, badge_text = ZONE_COLORS['HIGH'], 'Certain'
        elif size == 2:
            badge_color, badge_text = ZONE_COLORS['MEDIUM'], 'Additional test suggested'
        else:
            badge_color, badge_text = ZONE_COLORS['LOW'], 'Uncertain'
        members = ', '.join(str(s) for s in active_conf)
        conf_html = f'''
        <div style="text-align:center;padding:12px;background:#f9f9f9;border-radius:8px;margin:8px 0;">
          <div style="font-size:18px;font-weight:bold;margin-bottom:6px;">{{ {members} }}</div>
          <span style="background:{badge_color};color:white;padding:3px 10px;border-radius:12px;font-size:12px;">
            Set size: {size} — {badge_text}
          </span>
        </div>'''

    # --- SHAP HTML ---
    shap_html = ''
    scenario = 'CBC_BIO' if tier == 'Tier2' else 'CBC_Only'
    shap_key = ('test', '2', scenario)
    pred_norm = pred.replace(' ', '_')
    if (shap_key in shap_values and pid in cascade_to_s2
            and pred_norm in S2_CLASS_NAMES):
        class_idx = S2_CLASS_NAMES.index(pred_norm)
        s2_idx = cascade_to_s2[pid]
        shap_arr = shap_values[shap_key][class_idx]
        if s2_idx < shap_arr.shape[0]:
            patient_shap = shap_arr[s2_idx]
            feat_names = shap_features.get(('2', scenario), [])
            abs_vals = np.abs(patient_shap)
            top_idx = np.argsort(abs_vals)[-10:][::-1]
            max_abs = abs_vals[top_idx[0]] if len(top_idx) > 0 else 1.0
            for i in top_idx:
                v = patient_shap[i]
                fn = FEATURE_DISPLAY.get(feat_names[i], feat_names[i]) if i < len(feat_names) else f'f{i}'
                bar_color = PALETTE['highlight'] if v > 0 else PALETTE['accent1']
                w = min(abs(v) / max_abs * 45, 45)  # scale to max 45%
                direction = 'left:50%;' if v >= 0 else f'right:50%;'
                shap_html += f'''
                <div style="display:flex;align-items:center;margin:2px 0;font-size:12px;">
                  <span style="width:130px;text-align:right;padding-right:8px;font-size:11px;">{fn}</span>
                  <div style="flex:1;position:relative;height:18px;">
                    <div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:#ddd;"></div>
                    <div style="position:absolute;{direction}top:2px;height:14px;width:{w:.1f}%;background:{bar_color};border-radius:2px;opacity:0.8;"></div>
                  </div>
                  <span style="width:65px;font-size:11px;text-align:right;font-family:monospace;">{v:+.4f}</span>
                </div>'''

    # --- Reflex HTML ---
    reflex_html = ''
    if reflex:
        test = reflex.get('reflex_test', '')
        urgency = reflex.get('urgency', '')
        rationale = reflex.get('rationale', '')
        urg_color = {'urgent': PALETTE['highlight'], 'routine': PALETTE['accent1'], 'none': PALETTE['base2']}.get(urgency, '#999')
        reflex_html = f'''
        <div style="margin:8px 0;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <strong style="font-size:13px;">Test:</strong>
            <span style="font-size:13px;">{test}</span>
            <span style="background:{urg_color};color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold;">{urgency.upper()}</span>
          </div>
          <div style="font-size:12px;color:#666;"><strong>Rationale:</strong> {rationale}</div>
        </div>'''

    # --- Cascade path ---
    t1z = patient_row.get('tier1_zone', '?')
    t1p = patient_row.get('tier1_pred_label', '?')
    t2z = patient_row.get('tier2_zone', '')
    t2p = patient_row.get('tier2_pred_label', '')

    if not escalated:
        cascade_html = f'CBC → Tier 1: {t1p} ({t1z}) → <strong>Final: {pred} ({zone})</strong>'
    elif t1z == 'Excluded':
        cascade_html = f'CBC → S1: Excluded → Escalate → Tier 2: {t2p} ({t2z}) → <strong>Final: {pred} ({zone})</strong>'
    else:
        cascade_html = f'CBC → Tier 1: {t1p} ({t1z}) → Escalate → Tier 2: {t2p} ({t2z}) → <strong>Final: {pred} ({zone})</strong>'

    # --- Assemble HTML ---
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CDS Report — Patient #{pid:03d}</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family:Arial,Helvetica,sans-serif; background:#fff; color:#333; max-width:700px; margin:0 auto; padding:16px; }}
    .header {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:12px 0; border-bottom:2px solid #eee; margin-bottom:16px; }}
    .badge {{ padding:4px 12px; border-radius:16px; color:white; font-weight:bold; font-size:14px; }}
    .section {{ margin:16px 0; }}
    .section-title {{ font-size:14px; font-weight:bold; color:#5D6D7E; margin-bottom:8px; border-bottom:1px solid #eee; padding-bottom:4px; }}
    .cascade {{ background:#f5f5f5; padding:10px 14px; border-radius:6px; font-size:13px; }}
    .narrative {{ background:#fafafa; padding:12px; border-radius:6px; font-size:13px; line-height:1.6; }}
    .narrative .tr {{ color:#777; font-style:italic; margin-top:8px; }}
    .footer {{ margin-top:24px; padding-top:12px; border-top:1px solid #eee; font-size:11px; color:#bbb; }}
  </style>
</head>
<body>
  <div class="header">
    <span style="font-size:18px;font-weight:bold;">Patient #{pid:03d}</span>
    <span class="badge" style="background:{pred_color};">{CLASS_LABELS.get(pred, pred)}</span>
    <span class="badge" style="background:{zone_color};">{zone}</span>
    <span style="font-size:13px;color:#999;">{tier} | True: {true_lbl}</span>
  </div>

  <div class="section">
    <div class="section-title">Cascade Path</div>
    <div class="cascade">{cascade_html}</div>
  </div>

  <div class="section">
    <div class="section-title">{'Tier 2' if tier=='Tier2' else 'Tier 1'} Probabilities</div>
    {prob_bars_html if prob_bars_html else '<p style="color:#999;">Not available</p>'}
  </div>

  <div class="section">
    <div class="section-title">Conformal Prediction Set (α={CONFORMAL_ALPHA})</div>
    {conf_html if conf_html else '<p style="color:#999;">Not available</p>'}
  </div>

  <div class="section">
    <div class="section-title">SHAP Feature Contributions — Top 10</div>
    {shap_html if shap_html else '<div style="background:#f5f5f5;padding:20px;border-radius:6px;text-align:center;color:#aaa;border:1px dashed #ccc;">SHAP data not available for this patient</div>'}
  </div>

  <div class="section">
    <div class="section-title">Reflex Test Recommendation</div>
    {reflex_html if reflex_html else '<p style="color:#999;">No recommendation available</p>'}
  </div>

  <div class="section">
    <div class="section-title">Clinical Narrative</div>
    <div class="narrative">{narr}</div>
  </div>

  <div class="footer">
    CDS Pipeline v1.0 &nbsp;|&nbsp; Source: {tier} &nbsp;|&nbsp; α={CONFORMAL_ALPHA}<br>
    This report is for clinical decision support only. Final diagnosis authority rests with the treating physician.
  </div>
</body>
</html>'''

    if output_dir:
        out_path = Path(output_dir) / f'cds_report_patient_{pid:03d}.html'
        out_path.write_text(html, encoding='utf-8')
        print(f"  ✅ HTML: {out_path.name}")

    return html


def select_example_patients(df):
    """
    Systematic patient selection for 12 distinct scenarios.
    Picks the most suitable patient for each scenario in a data-driven way.
    """
    selected = {}
    dc = df.copy()

    # Helper: pick the first patient matching the condition (prefer correct=True)
    def pick(mask, label, prefer_correct=True):
        pool = dc[mask]
        if pool.empty:
            print(f"  ⚠️  {label}: no matching patient")
            return None
        if prefer_correct and (pool['correct'] == True).any():
            pool = pool[pool['correct'] == True]
        row = pool.iloc[0]
        idx = row.name if hasattr(row, 'name') else row['patient_idx']
        selected[label] = row
        print(f"  ✅ {label}: patient_idx={row['patient_idx']}, "
              f"true={row['true_label_name']}, final={row['final_pred_label']}, "
              f"zone={row['final_zone']}, tier={row['final_tier']}, correct={row['correct']}")
        return row

    # Scenarios 1-4: Tier 1 HIGH (no escalation), one per class
    for cls in ['DEA', 'HA', 'HGB_HTZ', 'Normal']:
        true_col_val = cls.replace('_', ' ') if cls == 'HGB_HTZ' else cls
        # Check the true_label_name format
        mask_options = [
            (dc['final_tier'] == 1) & (dc['final_zone'] == 'HIGH') & (dc['true_label_name'] == true_col_val),
            (dc['final_tier'] == 1) & (dc['final_zone'] == 'HIGH') & (dc['true_label_name'].str.contains(cls[:3], case=False, na=False)),
            (dc['escalated'] == False) & (dc['final_zone'] == 'HIGH') & (dc['final_pred_label'] == cls),
        ]
        found = False
        for mask in mask_options:
            if mask.any():
                pick(mask, f"S{list(selected).__len__()+1}_{cls}_HIGH_T1")
                found = True
                break
        if not found:
            pick(dc['final_zone'] == 'HIGH', f"S{list(selected).__len__()+1}_{cls}_HIGH_T1_fallback")

    # Scenarios 5-8: MEDIUM → Tier2 HIGH (resolved with BIO), one per class
    for cls in ['DEA', 'HA', 'HGB_HTZ', 'Normal']:
        mask = ((dc['tier1_zone'] == 'MEDIUM') &
                (dc['escalated'] == True) &
                (dc['tier2_zone'] == 'HIGH') &
                (dc['final_pred_label'] == cls))
        if not mask.any():
            # Fallback: any escalation + HIGH
            mask = ((dc['escalated'] == True) &
                    (dc['tier2_zone'] == 'HIGH') &
                    (dc['final_pred_label'] == cls))
        pick(mask, f"S{len(selected)+1}_{cls}_MED_T1T2_HIGH")

    # Scenario 9: Tier1 MEDIUM → Tier2 MEDIUM (still uncertain)
    mask9 = ((dc['tier1_zone'] == 'MEDIUM') &
             (dc['tier2_zone'] == 'MEDIUM'))
    pick(mask9, f"S{len(selected)+1}_STILL_MEDIUM")

    # Scenario 10: Excluded at Tier1 → escalated to Tier2 (missed as DAS)
    mask10 = ((dc['tier1_zone'] == 'Excluded') &
              (dc['escalated'] == True))
    pick(mask10, f"S{len(selected)+1}_EXCLUDED_ESCALATED")

    # Scenario 11: LOW zone (Tier 2, expert review needed)
    mask11 = dc['final_zone'] == 'LOW'
    if not mask11.any():
        # If no LOW, pick the MEDIUM patient with the lowest confidence
        dc_med = dc[dc['final_zone'] == 'MEDIUM'].copy()
        conf_col = 'tier2_confidence' if 'tier2_confidence' in dc.columns else 'tier1_confidence'
        dc_med = dc_med.sort_values(conf_col, ascending=True)
        if not dc_med.empty:
            lowest_idx = dc_med.index[0]
            pick(dc.index == lowest_idx, f"S{len(selected)+1}_LOW_or_lowest")
        else:
            print(f"  ⚠️  No LOW zone patient found")
    else:
        pick(mask11, f"S{len(selected)+1}_LOW_EXPERT")


    # Scenario 12: Tier 2 HIGH (full BIO definitive result) — must be a patient
    # not already chosen for another scenario, to avoid duplicate showcase figures
    already_idx = {int(r['patient_idx']) for r in selected.values()}
    base12 = (dc['final_tier'] == 2) & (dc['final_zone'] == 'HIGH') & \
             (~dc['patient_idx'].astype(int).isin(already_idx))
    mask12 = base12 & (dc['final_pred_label'] == 'DEA')
    if not mask12.any():
        mask12 = base12              # any unused Tier2 HIGH patient
    if not mask12.any():
        # All Tier2 HIGH patients already used → fall back to any Tier2 HIGH
        mask12 = (dc['final_tier'] == 2) & (dc['final_zone'] == 'HIGH')
    pick(mask12, f"S{len(selected)+1}_T2_HIGH_FINAL")

    return selected


