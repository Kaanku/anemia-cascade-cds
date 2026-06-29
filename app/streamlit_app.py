"""
Anemia Cascade CDS — Streamlit demo app.

Two-stage clinical decision support for anemia subtype classification.
Run locally or on Streamlit Community Cloud:

    streamlit run app/streamlit_app.py

Models (deploy_models/) and assets (app/assets/) are loaded from the repo.
No real patient data ships with the app — the demo data is anonymized or
synthetic. This tool is for research demonstration only and is not a
medical device.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
from cascade_engine import CascadeEngine, display_s2  # noqa: E402

MODELS_DIR = ROOT / "deploy_models"
ASSETS_DIR = APP_DIR / "assets"
DEMO_DIR = ASSETS_DIR / "demo_data"

# ── Raw input fields (label, unit, default, min, max, step) ──
CBC_FIELDS = [
    ("yas", "Age", "years", 45, 0, 120, 1),
    ("hgb_g_d_l", "Hemoglobin (HGB)", "g/dL", 11.2, 2.0, 22.0, 0.1),
    ("rbc_10_6_u_l", "RBC count", "10⁶/µL", 4.35, 1.0, 8.0, 0.01),
    ("mcv_f_l", "MCV", "fL", 78.5, 40.0, 130.0, 0.1),
    ("mchc_g_dl", "MCHC", "g/dL", 32.1, 24.0, 40.0, 0.1),
    ("rdw_sd_fl", "RDW-SD", "fL", 48.0, 25.0, 110.0, 0.1),
    ("ret_he_pg", "RET-He", "pg", 28.4, 10.0, 50.0, 0.1),
    ("delta_he_pg", "Delta-He", "pg", -2.1, -15.0, 15.0, 0.1),
    ("ret_number_10_6_l", "Reticulocyte # (RET#)", "10⁶/L", 0.045, 0.0, 0.5, 0.001),
    ("irf_pct", "IRF", "%", 9.8, 0.0, 60.0, 0.1),
    ("frc_perc", "FRC", "%", 0.6, 0.0, 10.0, 0.1),
    ("nrbc_pct", "NRBC", "%", 0.0, 0.0, 10.0, 0.1),
    ("micro_macro_ratio", "Micro/Macro ratio", "ratio", 1.85, 0.0, 100.0, 0.01),
]
BIO_FIELDS = [
    ("ferritin", "Ferritin", "ng/mL", 18.0, 0.0, 2000.0, 1.0),
    ("demir", "Serum iron", "µg/dL", 42.0, 0.0, 500.0, 1.0),
    ("ldh", "LD", "U/L", 210.0, 0.0, 2000.0, 1.0),
    ("uibc", "UIBC", "µg/dL", 310.0, 0.0, 600.0, 1.0),
]

ZONE_STYLE = {
    "HIGH":     ("#1B998B", "Automatable — high confidence"),
    "MEDIUM":   ("#E8A33D", "Technician review — moderate confidence"),
    "LOW":      ("#C0392B", "Expert review — low confidence"),
    "Excluded": ("#6B7280", "Not an associated anemia cause"),
}
URGENCY_STYLE = {"urgent": "#C0392B", "routine": "#1B6FB0", "none": "#6B7280"}


@st.cache_resource(show_spinner="Loading models…")
def get_engine() -> CascadeEngine:
    return CascadeEngine(MODELS_DIR, ASSETS_DIR)


@st.cache_data
def load_demo(split: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(DEMO_DIR / f"{split}_{kind}.csv")


def zone_badge(zone: str) -> str:
    color, label = ZONE_STYLE.get(zone, ("#6B7280", zone))
    return (
        f"<span style='background:{color};color:#fff;padding:3px 12px;"
        f"border-radius:14px;font-weight:600;font-size:0.85rem'>{zone}</span> "
        f"<span style='color:#555;font-size:0.85rem'>{label}</span>"
    )


def render_result(res: dict):
    s1 = res["stage1"]
    fin = res["final"]

    # ── Stage 1 ──
    st.markdown("#### Stage 1 — Associated vs Other Anemia Cause")
    c1, c2 = st.columns([1, 1])
    c1.metric("Stage 1 call", s1["pred_display"],
              help=f"Decision threshold {s1['threshold']} on P(AAC)")
    c2.metric("P(AAC)", f"{s1['p_AAC']:.1%}")
    st.progress(min(max(s1["p_AAC"], 0.0), 1.0))
    if s1["conformal_set"]:
        st.caption(f"Conformal set (α={res['alpha']}): "
                   + ", ".join(s1["conformal_set"]))

    if not res["escalated"]:
        st.markdown("---")
        st.markdown("#### Result")
        st.markdown(zone_badge("Excluded"), unsafe_allow_html=True)
        st.info("Classified as **OAC** (Other Anemia Cause) at Stage 1. "
                "The four-subtype Stage 2 model is not applied; further "
                "non-anemia workup is indicated.")
        render_reflex(fin.get("reflex"))
        return

    # ── Stage 2 ──
    st.markdown("---")
    st.markdown("#### Stage 2 — Anemia subtype")
    s2 = res["stage2"]
    prob_df = (
        pd.DataFrame({"Subtype": list(s2["probs"].keys()),
                      "Probability": list(s2["probs"].values())})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    cc1, cc2 = st.columns([1, 1.3])
    with cc1:
        st.metric("Stage 2 call", s2["pred_display"])
        st.markdown(zone_badge(s2["zone"]), unsafe_allow_html=True)
        st.caption(
            f"Confidence {s2['confidence']:.1%} "
            f"(zones: LOW <{s2['zone_low']} ≤ MEDIUM < {s2['zone_high']} ≤ HIGH)"
        )
        if s2["conformal_set"]:
            st.caption(f"Conformal set (α={res['alpha']}): "
                       + ", ".join(s2["conformal_set"]))
    with cc2:
        st.bar_chart(prob_df.set_index("Subtype"), height=210)

    st.markdown("---")
    st.markdown("#### Result")
    color, _ = ZONE_STYLE.get(fin["zone"], ("#6B7280", ""))
    st.markdown(
        f"<div style='font-size:1.3rem;font-weight:700'>{fin['label_display']}"
        f"&nbsp;&nbsp;{zone_badge(fin['zone'])}</div>",
        unsafe_allow_html=True,
    )
    render_reflex(fin.get("reflex"))


def render_reflex(reflex: dict | None):
    if not reflex:
        st.caption("No reflex recommendation matched.")
        return
    ucolor = URGENCY_STYLE.get(reflex.get("urgency", ""), "#6B7280")
    st.markdown("##### Reflex recommendation")
    st.markdown(
        f"<div style='border-left:4px solid {ucolor};padding:8px 14px;"
        f"background:#fafafa'>"
        f"<b>{reflex['test']}</b><br>"
        f"<span style='color:{ucolor};font-weight:600;text-transform:uppercase;"
        f"font-size:0.8rem'>{reflex.get('urgency','')}</span>"
        f"<span style='color:#666'> · {reflex.get('rationale','')}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Anemia Cascade CDS", page_icon="🩸",
                       layout="wide")
    st.title("Anemia Cascade CDS")
    st.caption(
        "Two-stage clinical decision support for anemia subtype "
        "classification · research demonstration, not a medical device."
    )

    with st.sidebar:
        st.header("Configuration")
        scenario = st.radio(
            "Scenario",
            ["CBC_BIO", "CBC_Only"],
            format_func=lambda s: "CBC + biochemistry" if s == "CBC_BIO" else "CBC only",
            help="CBC+biochemistry adds ferritin, iron, LD and UIBC.",
        )
        alpha = st.select_slider(
            "Conformal α (1 − coverage)",
            options=[0.05, 0.10, 0.20], value=0.10,
            help="Smaller α → wider prediction sets, higher coverage.",
        )
        st.markdown("---")
        mode = st.radio("Input", ["Demo patient", "Manual entry"])

    engine = get_engine()
    needs_bio = scenario == "CBC_BIO"
    fields = CBC_FIELDS + (BIO_FIELDS if needs_bio else [])

    raw = {}
    if mode == "Demo patient":
        cda, cdb, cdc = st.columns(3)
        split = cda.selectbox("Cohort", ["test", "train", "temporal"], index=0)
        kind = cdb.selectbox(
            "Data type", ["real_anon", "synthetic"],
            format_func=lambda k: "Anonymized real" if k == "real_anon" else "Synthetic",
            help="Anonymized real patients (edge cases) or privacy-safe synthetic.",
        )
        df = load_demo(split, kind)
        idx = cdc.selectbox(
            "Patient", df.index,
            format_func=lambda i: f"{df.loc[i,'sample_id']} · true: "
            f"{display_s2(df.loc[i,'DIAGNOSIS']) if df.loc[i,'DIAGNOSIS'] in ('IDA','HA','HGB_HTZ','NORMAL') else 'OAC'}",
        )
        row = df.loc[idx]
        raw = {f[0]: float(row[f[0]]) for f in fields if f[0] in row}
        with st.expander("Raw values for this patient"):
            st.dataframe(
                pd.DataFrame({"value": {f[1]: row.get(f[0]) for f in fields}}),
                use_container_width=True,
            )
        true_label = row.get("DIAGNOSIS")
    else:
        st.markdown("Enter raw measured values:")
        true_label = None
        cols = st.columns(3)
        for n, (key, label, unit, dflt, lo, hi, step) in enumerate(fields):
            with cols[n % 3]:
                raw[key] = st.number_input(
                    f"{label} ({unit})", value=float(dflt),
                    min_value=float(lo), max_value=float(hi), step=float(step),
                )

    st.markdown("")
    if st.button("Run cascade", type="primary", use_container_width=True):
        with st.spinner("Running cascade…"):
            res = engine.run(raw, scenario=scenario, alpha=alpha)
        render_result(res)
        if true_label is not None:
            shown = (display_s2(true_label)
                     if true_label in ("IDA", "HA", "HGB_HTZ", "NORMAL") else "OAC")
            st.caption(f"Reference label for this demo patient: **{shown}** "
                       "(not used by the model).")

    st.markdown("---")
    st.caption(
        "Internal class codes are preserved as trained; AAC/OAC/IDA/HGB HTZ are "
        "display labels. Demo data contains no identifiable patient information."
    )


if __name__ == "__main__":
    main()
