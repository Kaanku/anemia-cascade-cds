"""make_figures.py — figures of the v4.3 manuscript, drawn only from the study outputs.

Main: Fig1_flow, Fig2_design, Fig3_selective_cascade, Fig4_ida_vs_hgbhtz
Supplement: FigS1_roc, FigS2_reliability, FigS4_learning_curve (FigS3 = Shapley values, from s12 after the SHAP run)
Each figure is written as PNG (600 dpi) and PDF (vector) to ms/figures/.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

O = Path(os.environ.get("CDS_OUT", Path(__file__).resolve().parent.parent))   # the study folder
R = O / "reports"
FIG = O / "ms" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "out",
                     "ytick.direction": "out", "savefig.dpi": 600, "pdf.fonttype": 42})
OK = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "purple": "#CC79A7", "orange": "#E69F00",
      "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000", "grey": "#7F7F7F"}
CLS_COL = {"IDA": OK["verm"], "HA": OK["purple"], "HGB_HTZ": OK["blue"], "NORMAL": OK["green"]}
CLS_LAB = {"IDA": "IDA", "HA": "HA", "HGB_HTZ": "HGB HTZ", "NORMAL": "Normal"}

m = pd.read_parquet(R / "figures_data/metrics.parquet")
SEL = dict(engine="tabpfn", analysis="A1", feature_set="FULL", unit="first sample")


def sel(df, **kw):
    for k, v in {**SEL, **kw}.items():
        df = df[df[k] == v]
    return df


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("written", name)


# ═══════════════════════════════════════════════════════════ diagram helpers
def box(ax, x, y, w, h, text, fc="white", ec="black", lw=0.7, fs=7, weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw))
    tx = x if ha == "center" else x - w / 2 + 0.012
    ax.text(tx, y, text, ha=ha, va="center", fontsize=fs, weight=weight, linespacing=1.25)


def arrow(ax, p, q, lw=0.7, style="-|>", color="black"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0))


# ═══════════════════════════════════════════════════════════ Fig. 1 flow
US = {"haematological": "hematological", "Haematological": "Hematological", "haemolysis": "hemolysis",
      "haemoglobinopathy": "hemoglobinopathy", "Combined IDA + ACD": "Combined IDA and chronic disease anemia"}


def us(t):
    for a, b in US.items():
        t = t.replace(a, b)
    return t


def fig1():
    flow = pd.read_excel(R / "flow_counts.xlsx")
    get = lambda pat: int(flow[flow.description.str.contains(pat, regex=False)]["n"].iloc[0])
    reasons = flow[(flow.step == 1) & flow.description.str.startswith("  –")].copy()
    reasons["description"] = reasons["description"].str.replace("  – ", "", regex=False).map(us)
    n0 = get("Records in LIS export")
    n_ex1 = get("Excluded at chart review (total)")
    n_ex2 = get("Excluded: non-anemic diagnosis after chart review")
    n_ex3 = get("Excluded: missing or non-numeric analyzer result")
    n_rep = get("Excluded from one-sample set")
    n_pri = get("Primary cohort (total patients)")
    n_sec = get("Secondary set (total patients)")
    n_a2 = get("All-samples set: samples from primary-cohort patients")
    n_a2p = get("All-samples set: patients with >1 sample")
    pri = {c: get(f"Primary cohort – {c}") for c in ["OAC", "IDA", "HGB_HTZ", "NORMAL", "HA"]}
    sec = flow[(flow.step == 7) & ~flow.description.str.contains("total")].copy()
    sec["description"] = sec["description"].str.replace("Secondary set – ", "", regex=False).map(us)
    tem = pd.read_excel(R / "temporal_build_report.xlsx", sheet_name="flow")
    tem = tem.assign(item=tem["item"].astype(str).str.strip()).set_index("item")["n"]
    n_t0 = int(tem.iloc[0]); n_tex = int(tem["Excluded: missing or non-numeric analyzer result"])
    n_t = int(tem["Temporal cohort"])
    tcl = {c: int(tem[c]) for c in ["HA", "IDA", "NORMAL", "HGB_HTZ"]}
    n_t_an = int(pd.read_parquet(O / "data/temporal/temporal_cohort.parquet")["has_analyzer_record"].sum())
    n1 = n0 - n_ex1; n2 = n1 - n_ex2; n3 = n2 - n_ex3; n4 = n3 - n_rep
    assert n4 == n_pri + n_sec, (n4, n_pri, n_sec)

    fig, ax = plt.subplots(figsize=(6.85, 7.6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis("off")
    LX, LW, LH = 0.235, 0.44, 0.044        # main column
    EX, EW = 0.745, 0.50                    # exclusion column (centre, width)
    fs, fe = 7.0, 6.7
    ax.text(0.015, 0.995, "Development cohort", fontsize=8.5, weight="bold", va="top")
    ys = [0.925, 0.745, 0.640, 0.540, 0.440]
    dts = pd.to_datetime(pd.read_excel(R / "record_audit.xlsx")["sample_date"])
    span = f"{dts.min().day} {dts.min():%b %Y} – {dts.max().day} {dts.max():%b %Y}"
    texts = [f"Records with reticulocyte-channel analysis\n{span}: n = {n0:,}",
             f"Records with an anemia-related diagnosis\nafter chart review: n = {n1:,}",
             f"Records with a diagnosis in scope or\nin the secondary set: n = {n2:,}",
             f"Analytically valid records: n = {n3:,}",
             f"Patients, first valid sample each: n = {n4:,}"]
    for i, (yy, t) in enumerate(zip(ys, texts)):
        box(ax, LX, yy, LW, LH, t, fs=fs, weight="bold" if i == 0 else "normal")
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (LX, a - LH / 2), (LX, b + LH / 2))
    lines = "\n".join(f"  {r.description}: {int(r.n)}" for r in reasons.itertuples())
    ex = [(0.865, 0.165, f"Excluded at chart review: n = {n_ex1:,}\n" + lines),
          (0.693, 0.034, f"Excluded: non-anemic diagnosis after chart review: n = {n_ex2}"),
          (0.590, 0.034, f"Excluded: missing or non-numeric analyzer result: n = {n_ex3}"),
          (0.490, 0.050, f"Later samples of patients with repeat testing: n = {n_rep}\n(used only in the all-samples analysis)")]
    joins = [(ys[0] + ys[1]) / 2 + 0.03, (ys[1] + ys[2]) / 2, (ys[2] + ys[3]) / 2, (ys[3] + ys[4]) / 2]
    for (yy, hh, t), yj in zip(ex, joins):
        box(ax, EX, yy, EW, hh, t, fs=fe, ha="left")
        ax.plot([LX, EX - EW / 2 - 0.012], [yj, yj], color="black", lw=0.6)
        arrow(ax, (EX - EW / 2 - 0.02, yj), (EX - EW / 2, yj))
    # split into primary cohort and secondary set
    yS = 0.245
    pri_txt = (f"Primary cohort: n = {n_pri} patients\n"
               f"OAC {pri['OAC']} · IDA {pri['IDA']} · HGB HTZ {pri['HGB_HTZ']}\n"
               f"Normal {pri['NORMAL']} · HA {pri['HA']}\n\n"
               f"Nested cross-validation (5 outer folds)\n"
               f"All-samples analysis: {n_a2:,} samples\n({n_a2p} patients with >1 sample)")
    box(ax, 0.145, yS, 0.305, 0.16, pri_txt, fc="#EAF2FA", fs=6.8)
    sec_txt = (f"Secondary set (out of scope): n = {n_sec}\n" +
               "\n".join(f"{r.description.replace('Combined IDA and chronic disease anemia', 'Combined IDA and chronic disease').replace('Homozygous or major hemoglobinopathy', 'Homozygous/major hemoglobinopathy')}: {int(r.n)}" for r in sec.itertuples()) +
               "\nNot used for model development")
    box(ax, 0.48, yS, 0.33, 0.16, sec_txt, fc="#F2F2F2", fs=6.8)
    ax.plot([LX, LX], [ys[4] - LH / 2, 0.375], color="black", lw=0.6)
    ax.plot([0.145, 0.48], [0.375, 0.375], color="black", lw=0.6)
    arrow(ax, (0.145, 0.375), (0.145, yS + 0.08)); arrow(ax, (0.48, 0.375), (0.48, yS + 0.08))
    # temporal cohort
    TX, TW = 0.835, 0.30
    ax.text(TX - TW / 2, 0.395, "Temporal validation\n(same center)", fontsize=8.0, weight="bold", va="bottom")
    box(ax, TX, 0.330, TW, 0.075, f"Consecutive patients\n1 Feb – 7 Apr 2026, four target\nclasses, complete biochemistry\npanel: n = {n_t0}", fc="#FDF1E7", fs=6.7)
    box(ax, TX, 0.230, TW, 0.040, f"Excluded: RDW-SD not\nreported: n = {n_tex}", fs=6.7)
    box(ax, TX, 0.115, TW, 0.095, f"Temporal cohort: n = {n_t}\nHA {tcl['HA']} · IDA {tcl['IDA']} · Normal {tcl['NORMAL']}\nHGB HTZ {tcl['HGB_HTZ']}\n"
        f"With analyzer record: {n_t_an}\n(models with the full feature set)", fc="#FDF1E7", fs=6.7)
    arrow(ax, (TX, 0.2925), (TX, 0.250)); arrow(ax, (TX, 0.210), (TX, 0.1625))
    save(fig, "Fig1_flow")


# ═══════════════════════════════════════════════════════════ Fig. 2 design
def fig2():
    fig, ax = plt.subplots(figsize=(6.85, 5.0))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0.08, 1.0); ax.axis("off")
    f = 6.8
    c1, c2, cg, cy = "#EAF2FA", "#FDF1E7", "#E8F5EE", "#FFF7E6"
    # ── A: cascade (x 0.00–0.47)
    ax.text(0.0, 0.995, "A  Two-tier cascade", fontsize=8.5, weight="bold", va="top")
    A = 0.235
    box(ax, A, 0.905, 0.40, 0.060, "Tier 1: CBC and reticulocyte panel\n(all patients)", fc=c1, weight="bold", fs=f)
    box(ax, A, 0.805, 0.40, 0.060, "Stage 1: four target classes (AAC)\nvs other anemia causes (OAC)", fs=f)
    box(ax, A, 0.700, 0.40, 0.060, "Stage 2: IDA · HA · HGB HTZ · Normal\nconfidence zone + 90% conformal set", fs=f)
    box(ax, 0.110, 0.560, 0.20, 0.095, "Stage 1 AAC and\nStage 2 HIGH zone\n→ Tier 1 result\n+ reflex step", fc=cg, fs=f)
    box(ax, 0.345, 0.560, 0.24, 0.095, "Stage 1 OAC, or\nMEDIUM or LOW zone\n→ order biochemistry\n(ferritin, iron, UIBC, LDH)", fs=f)
    box(ax, A, 0.415, 0.40, 0.060, "Tier 2: CBC + biochemistry\n(escalated patients)", fc=c2, weight="bold", fs=f)
    box(ax, A, 0.315, 0.40, 0.060, "Stage 1 and Stage 2 re-run\nwith biochemistry", fs=f)
    box(ax, A, 0.195, 0.40, 0.080, "Tier 2 result: class, calibrated probabilities,\nzone, 90% conformal set and reflex step\n(HIGH-zone step, other causes or review)", fc=cy, fs=f)
    arrow(ax, (A, 0.875), (A, 0.835)); arrow(ax, (A, 0.775), (A, 0.730))
    arrow(ax, (A - 0.08, 0.670), (0.110, 0.6075)); arrow(ax, (A + 0.08, 0.670), (0.345, 0.6075))
    arrow(ax, (0.345, 0.5125), (A + 0.06, 0.445)); arrow(ax, (A, 0.385), (A, 0.345)); arrow(ax, (A, 0.285), (A, 0.235))
    # ── B: nested CV and locking (x 0.53–1.00)
    ax.text(0.53, 0.995, "B  Nested cross-validation and locking", fontsize=8.5, weight="bold", va="top")
    B = 0.765
    ax.text(B, 0.930, "863 patients in 5 outer folds (stratified by class and repeat samples)", ha="center", va="center", fontsize=6.4)
    fx0, fw = 0.555, 0.084
    for k in range(5):
        ax.add_patch(plt.Rectangle((fx0 + k * fw, 0.865), fw - 0.006, 0.040,
                                   fc=(OK["orange"] if k == 4 else "#D9E6F2"), ec="black", lw=0.5))
        ax.text(fx0 + k * fw + (fw - 0.006) / 2, 0.885, f"Fold {k + 1}", ha="center", va="center", fontsize=6.4)
    ax.text(fx0 + 2 * fw - 0.003, 0.850, "training patients", ha="center", va="top", fontsize=6.4)
    ax.text(fx0 + 4.5 * fw - 0.003, 0.850, "held out", ha="center", va="top", fontsize=6.4)
    steps = [(0.770, "Imputation, ratio features and Boruta\nselection on the training patients"),
             (0.675, "Inner 5-fold CV → out-of-fold (OOF)\npredictions for the training patients"),
             (0.565, "Lock on OOF first samples: Stage 1 threshold,\nzone cut-offs, conformal quantiles and\ndisplay calibration (hashed lock file)"),
             (0.455, "Refit on all training patients →\npredict the held-out fold once")]
    for yy, t in steps:
        box(ax, B - 0.05, yy, 0.375, 0.085 if t.count("\n") == 2 else 0.062, t, fs=f)
    arrow(ax, (B - 0.05, 0.828), (B - 0.05, 0.801))
    for a_, b_ in ((0.739, 0.706), (0.644, 0.6075), (0.5225, 0.486)):
        arrow(ax, (B - 0.05, a_), (B - 0.05, b_))
    xr = 0.965
    ax.plot([B + 0.1375, xr], [0.455, 0.455], color="black", lw=0.7)
    arrow(ax, (xr, 0.455), (xr, 0.833))

    ax.text(B, 0.385, "Repeated for each of the five outer folds; pooled\nouter-fold predictions give the nested CV estimates",
            ha="center", va="center", fontsize=6.4, style="italic")
    box(ax, B, 0.255, 0.44, 0.090, "Final models: all 863 patients, choices locked on\ntheir own OOF predictions → temporal cohort\n(97 patients with an analyzer record)", fc=c2, fs=f)
    box(ax, B, 0.130, 0.44, 0.065, "Outer-fold and temporal predictions opened only\nafter the lock file was complete", fc="#F2F2F2", fs=f)
    save(fig, "Fig2_design")


# ═══════════════════════════════════════════════════════════ Fig. 3 selective + cascade
def fig3():
    from matplotlib.lines import Line2D
    cc = sel(pd.read_parquet(R / "figures_data/coverage_curve_S2.parquet"))
    fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.95))
    ax = axs[0]
    styles = [("nested CV", "CBC", OK["blue"], "-", "CBC, nested CV"),
              ("nested CV", "CBC_BIO", OK["verm"], "-", "CBC + biochemistry, nested CV"),
              ("temporal", "CBC", OK["blue"], "--", "CBC, temporal"),
              ("temporal", "CBC_BIO", OK["verm"], "--", "CBC + biochemistry, temporal")]
    handles = []
    for sset, scen, col, ls, lab in styles:
        d = cc[(cc.set == sset) & (cc.scenario == scen)].sort_values("coverage")
        h, = ax.plot(100 * d.coverage, 100 * d.accuracy, color=col, ls=ls, lw=1.0, label=lab)
        handles.append(h)
    s2 = sel(m, block="selective_S2")
    for sset, scen, col, mk in (("nested CV", "CBC", OK["blue"], "o"), ("nested CV", "CBC_BIO", OK["verm"], "o"),
                                ("temporal", "CBC", OK["blue"], "s"), ("temporal", "CBC_BIO", OK["verm"], "s")):
        for tgt in ("0.80", "0.85", "0.90"):
            x = s2[(s2.set == sset) & (s2.scenario == scen) & (s2.population == f"target {tgt}")].set_index("metric")["value"]
            if np.isfinite(x.get("accuracy_retained", np.nan)) and x["coverage"] > 0:
                ax.plot(100 * x["coverage"], 100 * x["accuracy_retained"], marker=mk, ms=3.8, mfc="white", mec=col,
                        mew=0.9, ls="none")
    handles += [Line2D([], [], marker="o", ls="none", mfc="white", mec="black", mew=0.8, ms=3.8,
                       label="Locked 80/85/90% points, nested CV"),
                Line2D([], [], marker="s", ls="none", mfc="white", mec="black", mew=0.8, ms=3.8,
                       label="Locked 80/85/90% points, temporal")]
    ax.axhline(90, color=OK["grey"], lw=0.6, ls=":")
    ax.text(99, 90.6, "90% target", ha="right", va="bottom", fontsize=6.3, color=OK["grey"])
    ax.set_xlabel("Patients retained (coverage), %"); ax.set_ylabel("Accuracy of retained patients, %")
    ax.set_xlim(0, 100); ax.set_ylim(45, 101)
    ax.legend(handles=handles, loc="lower left", frameon=False, handlelength=2.0, fontsize=6.0)
    ax.set_title("A  Stage 2 accuracy–coverage", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axs[1]
    cas = pd.read_csv(R / "extra/cascade_by_target.csv")
    for sset, col, mk, dy in (("nested CV", OK["blue"], "o", -11), ("temporal", OK["verm"], "s", 6)):
        d = cas[cas.set == sset].sort_values("tier1_share")
        ax.plot(100 * d.tier1_share, 100 * d.cascade_accuracy, color=col, marker=mk, ms=4, lw=0.9,
                label=f"Cascade, {'nested CV (n = 657)' if sset == 'nested CV' else 'temporal (n = 97)'}")
        for r in d.itertuples():
            ax.annotate(f"{int(round(100 * float(r.target)))}%", (100 * r.tier1_share, 100 * r.cascade_accuracy),
                        textcoords="offset points", xytext=(0, dy), ha="center", fontsize=6.0, color=col)
        ax.axhline(100 * d.cbc_bio_accuracy.iloc[0], color=col, lw=0.7, ls="--")
        cbc = sel(m, block="cascade", set=sset, metric="cbc_only_accuracy")["value"].iloc[0]
        ax.axhline(100 * cbc, color=col, lw=0.7, ls=":")
    ax.plot([], [], color="black", ls="--", lw=0.7, label="Biochemistry for all")
    ax.plot([], [], color="black", ls=":", lw=0.7, label="CBC only")
    ax.set_xlabel("Patients finalized at Tier 1 (biochemistry spared), %")
    ax.set_ylabel("End-to-end accuracy, %")
    ax.set_xlim(-2, 55); ax.set_ylim(57, 73)
    ax.legend(loc="center right", frameon=False, fontsize=6.0)
    ax.set_title("B  Cascade at the locked operating points", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    save(fig, "Fig3_selective_cascade")


# ═══════════════════════════════════════════════════════════ Fig. 4 IDA vs HGB HTZ
def fig4():
    roc = sel(pd.read_parquet(R / "figures_data/roc.parquet"), stage="MX", scenario="CBC")
    mx = sel(pd.read_parquet(R / "figures_data/mx_vs_indices.parquet"))
    fig, axs = plt.subplots(1, 2, figsize=(6.85, 3.2))
    idx_col = {"Mentzer": OK["orange"], "Green & King": OK["green"], "England & Fraser": OK["purple"],
               "RDW index": OK["sky"], "Shine & Lal": OK["grey"], "Srivastava": "#8C564B", "Ehsani": OK["yellow"]}
    for ax, (sset, sub, title) in zip(axs, (("nested CV", "all", "A  Nested CV, all patients"),
                                            ("nested CV", "MCV<80", "B  Nested CV, MCV <80 fL"))):
        d = roc[(roc.set == sset) & (roc.subgroup == sub)]
        t = mx[(mx.set == sset) & (mx.subgroup == sub)].set_index("index")
        n = int(t["n"].iloc[0])
        for name, col in idx_col.items():
            c = d[d.curve == name]
            ax.plot(c.fpr, c.tpr, color=col, lw=0.8, label=f"{name} ({t.loc[name, 'auc_index']:.3f})")
        c = d[d.curve == "model"]
        ax.plot(c.fpr, c.tpr, color=OK["blue"], lw=1.6, label=f"Model ({t['auc_model'].iloc[0]:.3f})")
        ax.plot([0, 1], [0, 1], color="black", lw=0.4, ls=":")
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity for HGB HTZ")
        ax.set_title(f"{title} (n = {n})", loc="left", weight="bold")
        ax.legend(loc="lower right", frameon=False, fontsize=5.8, title="AUC", title_fontsize=6)
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.5)
    save(fig, "Fig4_ida_vs_hgbhtz")


# ═══════════════════════════════════════════════════════════ supplementary
def figS1():
    roc = sel(pd.read_parquet(R / "figures_data/roc.parquet"), set="nested CV")
    fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.5))
    ax = axs[0]
    for scen, col, lab in (("CBC", OK["blue"], "CBC"), ("CBC_BIO", OK["verm"], "CBC + biochemistry")):
        c = roc[(roc.stage == "S1") & (roc.scenario == scen)]
        auc = sel(m, block="stage1", set="nested CV", scenario=scen, metric="auc")["value"].iloc[0]
        ax.plot(c.fpr, c.tpr, color=col, lw=1.1, label=f"{lab} ({auc:.3f})")
    ax.set_title("A  Stage 1, AAC vs OAC", loc="left", weight="bold")
    for ax, scen, title in ((axs[1], "CBC", "B  Stage 2, CBC"), (axs[2], "CBC_BIO", "C  Stage 2, CBC + biochemistry")):
        for c_, col in CLS_COL.items():
            c = roc[(roc.stage == "S2") & (roc.scenario == scen) & (roc.curve == c_)]
            auc = sel(m, block="stage2", set="nested CV", scenario=scen, metric=f"auc_{c_}")["value"].iloc[0]
            ax.plot(c.fpr, c.tpr, color=col, lw=1.0, label=f"{CLS_LAB[c_]} ({auc:.3f})")
        ax.set_title(title, loc="left", weight="bold")
    for ax in axs:
        ax.plot([0, 1], [0, 1], color="black", lw=0.4, ls=":")
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", frameon=False, fontsize=5.8, title="AUC", title_fontsize=6)
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "FigS1_roc")


def figS2():
    fig, axs = plt.subplots(1, 4, figsize=(6.85, 2.1))
    specs = [("S1", "CBC", "A  Stage 1, CBC"), ("S2", "CBC", "B  Stage 2, CBC"),
             ("S2", "CBC_BIO", "C  Stage 2, CBC + bio"), ("MX", "CBC", "D  IDA vs HGB HTZ")]
    for ax, (st, scen, title) in zip(axs, specs):
        rel = sel(pd.read_parquet(R / f"figures_data/reliability_{st}.parquet"), scenario=scen)
        for sset, col, mk in (("nested CV", OK["blue"], "o"), ("temporal", OK["verm"], "s")):
            if st == "S1" and sset == "temporal":
                continue
            d = rel[(rel.set == sset) & (rel.version == "calibrated") & (rel.n > 0)]
            ax.plot(d.mean_predicted, d.observed, marker=mk, ms=3, lw=0.8, color=col,
                    label="nested CV" if sset == "nested CV" else "temporal")
        ax.plot([0, 1], [0, 1], color="black", lw=0.4, ls=":")
        ax.set_title(title, loc="left", weight="bold", fontsize=7.5)
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axs[3].legend(loc="upper left", frameon=False, fontsize=6)
    axs[1].set_xlabel("Top-class probability"); axs[2].set_xlabel("Top-class probability")
    axs[1].set_ylabel("Observed accuracy"); axs[2].set_ylabel("Observed accuracy")
    fig.tight_layout(w_pad=0.8)
    save(fig, "FigS2_reliability")


def figS3(shap_dir=None, out_name="FigS3_shap"):
    """Shapley values of the CBC models (outer folds pooled): the ten features with the largest mean |phi| per model
    and explained class; each dot is one patient, coloured by the feature value's percentile among the explained
    patients (s12 beeswarm data)."""
    import os
    sd = Path(shap_dir or os.environ.get("SHAP_DIR", R / "shap"))
    specs = [("A1_FULL_CBC_S1", "1", "A  Stage 1: AAC"), ("A1_FULL_CBC_MX", "HGB_HTZ", "B  IDA vs HGB HTZ: HGB HTZ"),
             ("A1_FULL_CBC_S2", "IDA", "C  Stage 2: IDA"), ("A1_FULL_CBC_S2", "HA", "D  Stage 2: HA"),
             ("A1_FULL_CBC_S2", "HGB_HTZ", "E  Stage 2: HGB HTZ"), ("A1_FULL_CBC_S2", "NORMAL", "F  Stage 2: Normal")]
    if not all((sd / f"{c}_beeswarm.parquet").exists() for c, _, _ in specs):
        print("figS3: Shapley outputs not available"); return
    fig, axs = plt.subplots(2, 3, figsize=(6.85, 6.2))
    cmap = plt.get_cmap("coolwarm")
    rng = np.random.default_rng(42)
    for ax, (cfg, cls, title) in zip(axs.ravel(), specs):
        G = pd.read_csv(sd / f"{cfg}_global.csv", dtype={"explained_class": str})
        top = G[G.explained_class == cls].sort_values("rank").head(10)
        bw = pd.read_parquet(sd / f"{cfg}_beeswarm.parquet")
        bw = bw[bw.explained_class.astype(str) == cls]
        for i, f in enumerate(top.feature):
            d = bw[bw.feature == f]
            y = (len(top) - 1 - i) + rng.uniform(-0.28, 0.28, len(d))
            ax.scatter(d.phi, y, c=d.value_pct, cmap=cmap, vmin=0, vmax=1, s=1.6, lw=0, rasterized=True)
        ax.axvline(0, color="black", lw=0.4)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([shap_label(f) for f in top.feature[::-1]], fontsize=5.8)
        ax.set_title(title, loc="left", weight="bold", fontsize=7.2)
        ax.set_xlabel("Shapley value (probability)", fontsize=6.5)
        ax.tick_params(axis="x", labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=0.6, h_pad=1.0)
    cax = fig.add_axes([0.35, -0.02, 0.3, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
    cb.set_label("Feature value (percentile among the explained patients)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    save(fig, out_name)


SHAP_NAMES = {"age": "Age", "hgb_g_d_l": "HGB", "rbc_10_6_u_l": "RBC", "ret_number_10_6_l": "RET#", "mcv_f_l": "MCV",
              "mchc_g_dl": "MCHC", "rdw_sd_fl": "RDW-SD", "ret_he_pg": "RET-He", "irf_pct": "IRF",
              "micro_macro_ratio": "MicroR/MacroR", "nrbc_pct": "NRBC%", "delta_he_pg": "Delta-He", "frc_perc": "FRC",
              "hct_pct": "HCT", "mch_pg": "MCH", "rdw_cv_pct": "RDW-CV", "nrbc_number_10_3_u_l": "NRBC#",
              "wbc_10_3_u_l": "WBC", "neut_number_10_3_u_l": "NEUT#", "lymph_number_10_3_u_l": "LYMPH#",
              "mono_number_10_3_u_l": "MONO#", "eo_number_10_3_u_l": "EO#", "baso_number_10_3_u_l": "BASO#",
              "ig_number_10_3_u_l": "IG#", "plt_10_3_u_l": "PLT", "mpv_fl": "MPV", "pdw_fl": "PDW", "p_lcr_pct": "P-LCR",
              "ret_pct": "RET%", "lfr_pct": "LFR", "mfr_pct": "MFR", "hfr_pct": "HFR", "rbc_he_pg": "RBC-He",
              "micro_r_pct": "MicroR", "macro_r_pct": "MacroR", "ret_y_ch": "RET-Y", "ret_rbc_y_ch": "RET-RBC-Y",
              "irf_y_ch": "IRF-Y", "ferritin": "Ferritin", "iron": "Iron", "uibc": "UIBC", "ldh": "LDH"}


def shap_label(f):
    if "_div_" not in f:
        return SHAP_NAMES.get(f, f)
    parts = [SHAP_NAMES[x] for x in f.split("_div_")]
    return "/".join(f"({x})" if "/" in x else x for x in parts)


def figS4():
    lc = pd.read_parquet(R / "figures_data/learning_curve.parquet")
    lc = lc[(lc.engine == "tabpfn") & (lc.analysis == "A1")]
    fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.3))
    for ax, (st, title) in zip(axs, (("S1", "A  Stage 1 (AUC)"), ("S2", "B  Stage 2 (macro AUC)"),
                                     ("MX", "C  IDA vs HGB HTZ (AUC)"))):
        for part, col, mk in (("outer fold", OK["blue"], "o"), ("inner OOF", OK["grey"], "s")):
            d = lc[(lc.stage == st) & (lc.part == part)].groupby("fraction")["value"].agg(["mean", "min", "max"])
            ax.plot(100 * d.index, d["mean"], marker=mk, ms=3, color=col, lw=1.0, label=part)
            ax.fill_between(100 * d.index, d["min"], d["max"], color=col, alpha=0.15, lw=0)
        ax.set_title(title, loc="left", weight="bold", fontsize=7.5)
        ax.set_xlabel("Training patients used, %"); ax.set_xticks([25, 50, 75, 100])
        ax.set_ylim(0.65, 0.95)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("AUC"); axs[0].legend(loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout(w_pad=1.2)
    save(fig, "FigS4_learning_curve")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); figS1(); figS2(); figS3(); figS4()
