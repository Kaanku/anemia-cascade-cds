"""
streamlit_app.py — research demo of the two-tier anaemia cascade (Streamlit). Run: streamlit run app/streamlit_app.py
The models are TabPFN-3.5(-fast) fitted on SYNTHETIC data; see engine.py and README.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))      # engine.py next to this file, whatever the cwd
from engine import BIO, CLASS_LABEL, DATA, S2_CLASSES, Cascade  # noqa: E402

st.set_page_config(page_title="Anaemia cascade demo", page_icon="🩸", layout="wide")
try:                                                    # licence key from the host's secrets (never in the code)
    if "TABPFN_TOKEN" in st.secrets and not os.environ.get("TABPFN_TOKEN"):
        os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]
except Exception:                                       # no secrets file (local run with the key in the environment)
    pass

# (feature key, label, unit, group, step, factor input→feature, required)
FIELDS = [
    ("age", "Age", "years", "core", 1.0, 1, True),
    ("hgb_g_d_l", "HGB", "g/dL", "core", 0.1, 1, True),
    ("rbc_10_6_u_l", "RBC", "10⁶/µL", "core", 0.01, 1, True),
    ("mcv_f_l", "MCV", "fL", "core", 0.1, 1, True),
    ("mchc_g_dl", "MCHC", "g/dL", "core", 0.1, 1, True),
    ("rdw_sd_fl", "RDW-SD", "fL", "core", 0.1, 1, True),
    ("ret_number_10_6_l", "RET#", "10⁹/L", "core", 0.1, 1000, True),
    ("irf_pct", "IRF", "%", "core", 0.1, 1, True),
    ("ret_he_pg", "RET-He", "pg", "core", 0.1, 1, True),
    ("delta_he_pg", "Delta-He", "pg", "core", 0.1, 1, True),
    ("micro_r_pct", "MicroR", "%", "core", 0.1, 1, True),
    ("macro_r_pct", "MacroR", "%", "core", 0.1, 1, True),
    ("nrbc_pct", "NRBC%", "/100 WBC", "core", 0.1, 1, True),
    ("frc_perc", "FRC% (research)", "%", "core", 0.01, 100, True),
    ("hct_pct", "HCT", "%", "red", 0.1, 1, False),
    ("mch_pg", "MCH", "pg", "red", 0.1, 1, False),
    ("rdw_cv_pct", "RDW-CV", "%", "red", 0.1, 1, False),
    ("ret_pct", "RET%", "%", "red", 0.01, 1, False),
    ("lfr_pct", "LFR", "%", "red", 0.1, 1, False),
    ("mfr_pct", "MFR", "%", "red", 0.1, 1, False),
    ("hfr_pct", "HFR", "%", "red", 0.1, 1, False),
    ("rbc_he_pg", "RBC-He", "pg", "red", 0.1, 1, False),
    ("wbc_10_3_u_l", "WBC", "10³/µL", "white", 0.01, 1, False),
    ("neut_number_10_3_u_l", "NEUT#", "10³/µL", "white", 0.01, 1, False),
    ("lymph_number_10_3_u_l", "LYMPH#", "10³/µL", "white", 0.01, 1, False),
    ("mono_number_10_3_u_l", "MONO#", "10³/µL", "white", 0.01, 1, False),
    ("eo_number_10_3_u_l", "EO#", "10³/µL", "white", 0.01, 1, False),
    ("baso_number_10_3_u_l", "BASO#", "10³/µL", "white", 0.01, 1, False),
    ("ig_number_10_3_u_l", "IG#", "10³/µL", "white", 0.01, 1, False),
    ("plt_10_3_u_l", "PLT", "10³/µL", "white", 1.0, 1, False),
    ("mpv_fl", "MPV", "fL", "white", 0.1, 1, False),
    ("pdw_fl", "PDW", "fL", "white", 0.1, 1, False),
    ("p_lcr_pct", "P-LCR", "%", "white", 0.1, 1, False),
    ("ret_y_ch", "RET-Y (research)", "ch", "research", 0.1, 1, False),
    ("ret_rbc_y_ch", "RET-RBC-Y (research)", "ch", "research", 0.1, 1, False),
    ("irf_y_ch", "IRF-Y (research)", "ch", "research", 0.1, 1, False),
    ("ferritin", "Ferritin", "ng/mL", "bio", 0.1, 1, False),
    ("iron", "Iron", "µg/dL", "bio", 1.0, 1, False),
    ("uibc", "UIBC", "µg/dL", "bio", 1.0, 1, False),
    ("ldh", "LDH", "U/L", "bio", 1.0, 1, False),
]
GROUPS = {"core": "Thesis parameters (required)", "red": "Red cell and reticulocyte indices",
          "white": "White cells and platelets", "research": "Research parameters",
          "bio": "Biochemistry panel (Tier 2; optional)"}
ZONE_TEXT = {"HIGH": "high confidence", "MEDIUM": "medium confidence", "LOW": "low confidence"}
URGENCY = {"none": "—", "routine": "routine", "priority": "priority", "urgent": "urgent"}
SHORT = {"IDA": "IDA", "HA": "HA", "HGB_HTZ": "HGB HTZ", "NORMAL": "Normal", "OAC": "OAC"}


@st.cache_resource(show_spinner="Fitting the four models on the synthetic training set (first start only)…")
def engine() -> Cascade:
    return Cascade()


@st.cache_data
def examples() -> dict:
    return json.loads((DATA / "examples.json").read_text())


def load_example():
    name = st.session_state.get("example")
    ex = examples().get(name) if name in examples() else None
    for f in FIELDS:
        key, factor = f[0], f[5]
        st.session_state[f"in_{key}"] = None if ex is None or ex.get(key) is None else round(ex[key] * factor, 4)


def read_inputs() -> dict:
    x = {}
    for key, label, unit, group, step, factor, req in FIELDS:
        v = st.session_state.get(f"in_{key}")
        x[key] = None if v is None else float(v) / factor
    return x


def tstr_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = [("Tier 1 · Stage 1 AUC", "A1_FULL_CBC_S1", "auc"), ("Tier 1 · Stage 1 sensitivity (classifiable)", "A1_FULL_CBC_S1", "sensitivity_aac"),
            ("Tier 1 · Stage 2 macro AUC", "A1_FULL_CBC_S2", "auc"), ("Tier 1 · Stage 2 accuracy", "A1_FULL_CBC_S2", "accuracy"),
            ("Tier 2 · Stage 1 AUC", "A1_FULL_CBC_BIO_S1", "auc"), ("Tier 2 · Stage 2 macro AUC", "A1_FULL_CBC_BIO_S2", "auc"),
            ("Tier 2 · Stage 2 accuracy", "A1_FULL_CBC_BIO_S2", "accuracy"), ("Cascade · accuracy", "cascade", "accuracy"),
            ("Cascade · share finalised at Tier 1", "cascade", "tier1_share"),
            ("Cascade · accuracy of Tier 1 decisions", "cascade", "tier1_accuracy")]
    out = []
    for label, cfg, col in rows:
        r = {"Measure": label}
        for which, name in (("dev", "Development"), ("temporal", "Temporal")):
            v = t.loc[(t["set"] == which) & (t["config"] == cfg), col] if col in t else pd.Series(dtype=float)
            r[name] = "—" if v.empty or pd.isna(v.iloc[0]) else f"{v.iloc[0]:.3f}"
        out.append(r)
    return pd.DataFrame(out)


def stage_panel(title: str, tier: dict):
    s1, s2 = tier["stage1"], tier["stage2"]
    st.markdown(f"#### {title}")
    st.metric("Stage 1 · probability of a classifiable cause", f"{s1['p_display']:.2f}",
              help="Probability shown after the locked display calibration. The decision uses the raw "
                   f"score {s1['score']:.2f} against the locked threshold {s1['threshold']:.2f}.")
    st.caption(("Classifiable cause (one of the four classes)" if s1["aac"] else "Another cause (OAC)")
               + f" · raw score {s1['score']:.2f}, threshold {s1['threshold']:.2f}")
    st.markdown("Stage 2 · class probabilities")
    for c in sorted(s2["probs_display"], key=s2["probs_display"].get, reverse=True):
        p = s2["probs_display"][c]
        mark = " · in the 90 % conformal set" if c in s2["conformal_set"] else ""
        st.progress(min(max(p, 0.0), 1.0), text=f"{CLASS_LABEL[c]} — {p:.2f}{mark}")
    hc = s2["high_cutoff"]
    high = (f"HIGH ≥ {hc:.2f}, the {eng.high_target * 100:.0f} % accuracy point" if hc is not None
            else f"no HIGH zone: no cut-off reached {eng.high_target * 100:.0f} % accuracy")
    st.caption(f"Stage 2 top class {SHORT[s2['top']]} · {ZONE_TEXT[s2['zone']]} "
               f"({high}, LOW < 0.35 on the raw top-class score {s2['top_score']:.2f})")


def show_pending(res: dict):
    """Tier 1 while Tier 2 is still running."""
    st.info("**Tier 1 did not finalise this case** · Tier 2 (with the biochemistry panel) is running…")
    stage_panel("Tier 1 · full blood count only", res["tier1"])


def show_result(res: dict):
    chk = res["check"]
    if not res["ok"]:
        names = {f[0]: f[1] for f in FIELDS}
        if chk["missing_required"]:
            st.error("Missing required inputs: " + ", ".join(names.get(c, c) for c in chk["missing_required"]
                                                             if c != "micro_macro_ratio")
                     + (" (MicroR and MacroR)" if "micro_macro_ratio" in chk["missing_required"] else ""))
        if chk["non_positive"]:
            st.error("These inputs must be positive: " + ", ".join(names.get(c, c) for c in chk["non_positive"]))
        return
    fin = res["final"]
    rec = fin["recommendation"]
    head = (f"Tier {fin['tier']} · rule {fin['rule']}"
            + (f" · {CLASS_LABEL[fin['class']]}" if fin.get("class") else ""))
    box = st.success if fin["tier"] == 1 and not fin.get("needs_biochemistry") else st.info
    box(f"**{head}**  \n**Recommended next step:** {rec['recommended_test']}  \n"
        f"Urgency: {URGENCY.get(rec['urgency'], rec['urgency'])} · {rec['rationale']}")
    if fin.get("needs_biochemistry"):
        st.caption("Tier 1 did not finalise this case. Enter the biochemistry panel (at least one analyte; "
                   "missing analytes are imputed as in the study) to run Tier 2.")
    stage_panel("Tier 1 · full blood count only", res["tier1"])
    if "tier2" in res:
        t2 = res["tier2"]
        stage_panel("Tier 2 · full blood count + biochemistry", t2)
        if t2["imputed"]:
            st.caption("Imputed (KNN, as in the study): " + ", ".join(t2["imputed"]))
    if chk["outside_training_range"] or chk["absent_optional"]:
        names = {f[0]: f[1] for f in FIELDS}
        with st.expander("Input notes"):
            if chk["outside_training_range"]:
                st.write("Outside the range of the training data (less reliable): "
                         + ", ".join(names.get(c, c) for c in chk["outside_training_range"]))
            if chk["absent_optional"]:
                st.write("Not entered (treated as missing): "
                         + ", ".join(names.get(c, c) for c in chk["absent_optional"] if c in names))
    st.caption(f"Computed in {res['seconds']} s on CPU.")



# ───────────────────────────────────────────────────────────── page
st.title("Two-tier anaemia cascade · research demo")
st.warning("**Research demonstration, not a medical device.** The models were trained on a synthetic data set "
           "drawn from the study's class-wise distributions; no patient data are contained in or used by this app. "
           "Outputs are not the study model's and must not be used for patient care.", icon="⚠️")

try:
    eng = engine()
except Exception as e:                                                     # weights or token missing
    st.error(f"The models could not be loaded: {type(e).__name__}: {e}")
    st.stop()
if eng.backend != "tabpfn":
    st.error(f"Interface test backend '{eng.backend}' (not TabPFN) with a '{eng.lock_engine}' lock: "
             "probabilities are not meaningful.")

with st.sidebar:
    st.header("Inputs")
    st.selectbox("Load a synthetic example", ["—"] + list(examples()), key="example", on_change=load_example,
                 format_func=lambda k: k if k == "—" else f"{CLASS_LABEL[k]} ({examples()[k]['record_id']})",
                 help="Examples are synthetic records, chosen near the middle of their class.")
    st.button("Clear all inputs", on_click=lambda: [st.session_state.update({f"in_{f[0]}": None}) for f in FIELDS])
    st.caption("Sysmex XN units as exported. Leave optional fields empty if not measured; "
               "MicroR/MacroR and NRBC# are computed by the app.")

left, right = st.columns([3, 2], gap="large")
with left, st.form("inputs"):
    for g, title in GROUPS.items():
        with st.expander(title, expanded=g in ("core", "bio")):
            fs = [f for f in FIELDS if f[3] == g]
            for i, (key, label, unit, group, step, factor, req) in enumerate(fs):
                if i % 3 == 0:                                   # one row per three fields (keeps the order on phones)
                    cols = st.columns(3)
                cols[i % 3].number_input(f"{label} ({unit})", value=None, step=step, key=f"in_{key}",
                                         format="%.4f" if step < 0.1 else ("%.2f" if step < 1 else "%.1f"),
                                         placeholder="required" if req else "optional")
    run = st.form_submit_button("Run the cascade", type="primary", width="stretch")

with right:
    st.subheader("Result")
    result_box = st.empty()
    if run:                                             # Tier 1 first, shown while Tier 2 runs (CPU: ~20 s each)
        with result_box.container(), st.spinner("Tier 1 · full blood count… (about 20 s on the free server)"):
            res = eng.start(read_inputs())
        if res["ok"] and "final" not in res:
            with result_box.container():
                show_pending(res)
                with st.spinner("Tier 2 · adding the biochemistry panel… (about 20 s)"):
                    res = eng.finish(res)
        else:
            res = eng.finish(res)
        st.session_state["result"] = res
    res = st.session_state.get("result")
    with result_box.container():
        if res is None:
            st.info("Load a synthetic example from the sidebar or type a full blood count, then run the cascade.")
        else:
            show_result(res)
with st.expander("About this demo"):
    st.markdown(
        "**Cascade.** Tier 1 uses the full blood count only. Stage 1 separates anaemias with a classifiable cause "
        "(iron deficiency, haemolytic anaemia, heterozygous haemoglobinopathy, normal red cell profile) from other "
        "causes; Stage 2 assigns one of the four. A case is finalised at Tier 1 when Stage 1 says classifiable and "
        "the Stage 2 top-class probability reaches the HIGH cut-off; otherwise the biochemistry panel (ferritin, "
        "iron, UIBC, LDH) is added and Tier 2 decides. Recommendations follow the study's 14 reflex rules.\n\n"
        f"**Models.** {eng.model_label} (Prior Labs), one model per stage and tier, with the feature lists selected "
        "in the study (the study itself used TabPFN-3.5 with its default ensemble; the lighter version keeps this "
        "demo responsive on a small server). Thresholds, confidence zones, 90 % conformal sets (APS, conservative "
        "version) and the display calibration were locked with the study's rules on real development patients, "
        "each predicted by a model of the same kind trained on synthetic data generated without that patient "
        "(5 folds; the synthetic copy of four folds predicted the fifth). Only these cut-offs and two-parameter "
        "calibrations are stored; no patient-level value is.\n\n"
        f"**One rule differs from the study.** The HIGH cut-off is the {eng.high_target * 100:.0f} % accuracy point "
        "(study: 90 %). Models trained on synthetic data are less accurate on real patients than the study model, "
        "and none of them reached 90 % accuracy at any cut-off; with the study's rule this demo would never "
        "finalise a case at Tier 1.\n\n"
        f"**Training data.** {eng.n_train['synthetic_patients']} synthetic records "
        f"({eng.n_train['with_biochemistry']} with biochemistry), generated class by class with a Gaussian copula; "
        "no real record is contained. Analyzer identities (HCT, HGB, MCH, RET%, LFR/MFR, RBC-He, WBC) are kept.\n\n"
        "**Licence.** TabPFN-3.5 weights are under the non-commercial TabPFN-3.5 licence; this demo is for research use.")
    tp = DATA / "tstr_summary.csv"
    if tp.exists():
        st.markdown("**On the real patients.** Development (863 patients): the cross-generation predictions above "
                    "(the cut-offs were chosen on them, so the cascade figures are slightly optimistic). Temporal "
                    "(97 patients): this app's models, trained on the whole synthetic set, on an independent later "
                    "cohort (no other-cause patients, so Stage 1 AUC is not defined there). Cascade figures are for "
                    "patients with the biochemistry panel measured.")
        st.dataframe(tstr_table(pd.read_csv(tp)), hide_index=True, width="stretch")
