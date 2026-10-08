"""
s14_rwo.py — the analyzer's rule-based RBC Defect Workflow Optimisation (RWO) on the v4 cohort (R26)
=====================================================================================================

The Sysmex RWO decision tree (thesis section "RBC Defekti İş Akışı Optimizasyonu Algoritması";
Nivaggioni et al., Int J Lab Hematol 2020) is applied to the first sample of each of the 863 primary
patients. Nothing is fitted: the published cut-offs are used as they are.

Step 1 (CBC)
    MCHC > 36.5 g/dL                       -> RET analysis (high MCHC; MCHC-O is not exported, so the
                                              high value is taken as confirmed)
    MicroR < 12.9 %:  NRBC < 0.9 %         -> Other
                      MCHC < 33.8 g/dL     -> Other
                      otherwise            -> RET analysis
    MicroR >= 12.9 %: RDW-SD < 38.5 fL     -> HGB HTZ
                      MCHC < 33.0 g/dL     -> IDA
                      otherwise            -> RET analysis
Step 2 (RET analysis)
    RBC score = -7.6219 + 1.5892 x FRC% + 40.293 x RET# (10^6/uL);  p = 1 / (1 + exp(-score))
    p > 0.15:  IRF > 20 % -> SCD, otherwise -> HS
    p <= 0.15: Hypo-He / MicroR > 1.5 -> SAO, otherwise -> Other.  Hypo-He is not in the export, so this
               branch always ends in Other (as in the thesis, where no sample was flagged SAO).
FRC% = 100 x FRC#/RBC (the model feature frc_perc is the fraction).

Two evaluations, first sample per patient:
  (a) local characterisation in the algorithm's own scheme (thesis harmonisation): truth IDA -> IDA,
      sickle cell trait -> SCD, other heterozygous haemoglobinopathies -> HGB HTZ, HA and OAC -> Other;
      Normal patients are left out because the algorithm has no "normal" output (as in the thesis);
      hereditary spherocytosis is outside the primary cohort. Per-class precision, sensitivity, F1 and
      overall accuracy with patient bootstrap CIs (B = 2000, seed 42).
  (b) head-to-head on the same 863 patients in a common three-class scheme: IDA / HGB HTZ (sickle trait
      included) / Other (HA, OAC, Normal). RWO: SCD -> HGB HTZ, HS and SAO -> Other. Model: the CBC-only
      TabPFN-3.5 cascade (A1 FULL CBC, nested-CV outer folds, locked Stage 1 threshold; Stage 2 argmax,
      HA and Normal -> Other). Accuracy, per-class sensitivity, McNemar on correctness, bootstrap CI of the
      accuracy difference.

Outputs: reports/rwo/rwo_own_scheme.csv, rwo_vs_model.csv, rwo_confusion.csv, rwo.xlsx
Usage: python s14_rwo.py --out /path/CDS_v4 [--boot 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import s07_lock as L
import s09_evaluate as V

ENGINE = "tabpfn"


def rwo_class(r: pd.Series) -> str:
    mchc, micro, nrbc, rdw = r["mchc_g_dl"], r["micro_r_pct"], r["nrbc_pct"], r["rdw_sd_fl"]
    ret = False
    if mchc > 36.5:
        ret = True
    elif micro < 12.9:
        if nrbc < 0.9 or mchc < 33.8:
            return "Other"
        ret = True
    else:
        if rdw < 38.5:
            return "HGB HTZ"
        if mchc < 33.0:
            return "IDA"
        ret = True
    assert ret
    score = -7.6219 + 1.5892 * (100 * r["frc_perc"]) + 40.293 * r["ret_number_10_6_l"]
    p = 1 / (1 + np.exp(-score))
    if p > 0.15:
        return "SCD" if r["irf_pct"] > 20 else "HS"
    return "Other"                                    # Hypo-He not exported -> SAO branch cannot fire


def truth_own(r: pd.Series) -> str | None:
    if r["cls"] == "IDA":
        return "IDA"
    if r["cls"] == "HGB_HTZ":
        return "SCD" if r["adj_label"] == "ORAK HÜCRELİ ANEMİ" else "HGB HTZ"
    if r["cls"] in ("HA", "OAC"):
        return "Other"
    return None                                       # Normal: no RWO output for normal


def per_class(y: np.ndarray, yh: np.ndarray, classes: list[str]) -> dict:
    out = {"accuracy": float(np.mean(y == yh))}
    for c in classes:
        tp = np.sum((yh == c) & (y == c))
        out[f"precision_{c}"] = tp / np.sum(yh == c) if np.sum(yh == c) else np.nan
        out[f"sensitivity_{c}"] = tp / np.sum(y == c) if np.sum(y == c) else np.nan
        p, s = out[f"precision_{c}"], out[f"sensitivity_{c}"]
        out[f"f1_{c}"] = 2 * p * s / (p + s) if np.isfinite(p) and np.isfinite(s) and p + s > 0 else np.nan
    return out


def boot(fn, y, yh, b: int) -> dict:
    point = fn(y, yh)
    reps = {k: [] for k in point}
    for idx in V.Boot(np.arange(len(y)), b):
        r = fn(y[idx], yh[idx])
        for k in point:
            reps[k].append(r[k])
    return {k: (point[k], *np.nanpercentile(np.asarray(reps[k], float), [2.5, 97.5])) for k in point}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=V.B_DEFAULT)
    a = ap.parse_args()
    od = a.out / "reports" / "rwo"
    od.mkdir(parents=True, exist_ok=True)

    c = pd.read_parquet(a.out / "data" / "cohort" / "cohort_primary.parquet")
    assert c["patient_id"].is_unique and len(c) == 863
    c["rwo"] = c.apply(rwo_class, axis=1)
    c["truth_own"] = c.apply(truth_own, axis=1)

    # (a) own scheme
    own = c[c["truth_own"].notna()]
    own_classes = ["IDA", "HGB HTZ", "SCD", "HS", "Other"]
    res = boot(lambda y, yh: per_class(y, yh, own_classes), own["truth_own"].to_numpy(), own["rwo"].to_numpy(), a.boot)
    rows = [{"metric": k, "value": v[0], "ci_low": v[1], "ci_high": v[2]} for k, v in res.items()]
    A = pd.DataFrame(rows).assign(n=len(own))
    for cl in own_classes:
        A.loc[A["metric"].str.endswith("_" + cl), "n_true"] = int((own["truth_own"] == cl).sum())
        A.loc[A["metric"].str.endswith("_" + cl), "n_flagged"] = int((own["rwo"] == cl).sum())
    A.to_csv(od / "rwo_own_scheme.csv", index=False)
    cm = pd.crosstab(pd.Categorical(c["cls"], ["IDA", "HA", "HGB_HTZ", "NORMAL", "OAC"]),
                     pd.Categorical(c["rwo"], own_classes), dropna=False)
    cm.index.name, cm.columns.name = "true class (v4)", "RWO output"
    cm.to_csv(od / "rwo_confusion.csv")
    print(f"(a) own scheme, n = {len(own)} (Normal left out): accuracy {res['accuracy'][0]:.3f} "
          f"({res['accuracy'][1]:.3f}-{res['accuracy'][2]:.3f})")
    for cl in own_classes:
        print(f"    {cl:8s} true {int((own['truth_own'] == cl).sum()):3d} flagged {int((own['rwo'] == cl).sum()):3d}  "
              f"prec {res[f'precision_{cl}'][0]:.2f} sens {res[f'sensitivity_{cl}'][0]:.2f} F1 {res[f'f1_{cl}'][0]:.2f}")

    # (b) head-to-head, common scheme, all 863
    lock = json.loads((a.out / "lock" / "lock.json").read_text())
    D = V.Data(a.out, lock, joblib.load(a.out / "lock" / "calibrators.joblib"))
    first = lambda d: d[d["is_index_sample"].astype(bool)].set_index("record_id")
    s1, s2 = first(D.pooled(ENGINE, "A1_FULL_CBC_S1")), first(D.pooled(ENGINE, "A1_FULL_CBC_S2"))
    ids = c["record_id"]
    assert set(ids) == set(s1.index) == set(s2.index), "model predictions and cohort differ"
    s1, s2 = s1.loc[ids], s2.loc[ids]
    s2cls = np.array(V.S2)[V.P2(s2).argmax(1)]
    e2e = np.where(s1["s"].to_numpy() >= s1["thr"].to_numpy(), s2cls, "OAC")
    common = {"IDA": "IDA", "HGB_HTZ": "HGB HTZ", "HA": "Other", "NORMAL": "Other", "OAC": "Other"}
    y = c["cls"].map(common).to_numpy()
    yr = c["rwo"].replace({"SCD": "HGB HTZ", "HS": "Other", "SAO": "Other"}).to_numpy()
    ym = pd.Series(e2e).map(common).to_numpy()
    k3 = ["IDA", "HGB HTZ", "Other"]
    rr, rm = boot(lambda y, yh: per_class(y, yh, k3), y, yr, a.boot), boot(lambda y, yh: per_class(y, yh, k3), y, ym, a.boot)
    diff = []
    for idx in V.Boot(np.arange(len(y)), a.boot):
        diff.append(np.mean(ym[idx] == y[idx]) - np.mean(yr[idx] == y[idx]))
    dlo, dhi = np.percentile(diff, [2.5, 97.5])
    b_only_model, b_only_rwo, p_mc = V.mcnemar(ym == y, yr == y)
    B = pd.DataFrame([{"metric": k, "rwo": rr[k][0], "rwo_low": rr[k][1], "rwo_high": rr[k][2],
                       "model": rm[k][0], "model_low": rm[k][1], "model_high": rm[k][2]} for k in rr])
    B = pd.concat([B, pd.DataFrame([{"metric": "accuracy difference (model - RWO)",
                                     "model": float(np.mean(ym == y) - np.mean(yr == y)), "model_low": dlo,
                                     "model_high": dhi},
                                    {"metric": "McNemar: only model correct / only RWO correct / p",
                                     "model": b_only_model, "rwo": b_only_rwo, "model_high": p_mc}])],
                  ignore_index=True).assign(n=len(y))
    B.to_csv(od / "rwo_vs_model.csv", index=False)
    print(f"(b) common scheme, n = {len(y)}: accuracy RWO {rr['accuracy'][0]:.3f} vs model {rm['accuracy'][0]:.3f}"
          f" (difference {np.mean(ym == y) - np.mean(yr == y):+.3f}, {dlo:+.3f} to {dhi:+.3f}); McNemar "
          f"{b_only_model}/{b_only_rwo}, p = {p_mc:.2g}")
    for cl in k3:
        print(f"    sens {cl:8s} RWO {rr[f'sensitivity_{cl}'][0]:.2f}  model {rm[f'sensitivity_{cl}'][0]:.2f}   "
              f"prec RWO {rr[f'precision_{cl}'][0]:.2f} model {rm[f'precision_{cl}'][0]:.2f}")
    with pd.ExcelWriter(od / "rwo.xlsx") as xw:
        A.to_excel(xw, sheet_name="own_scheme", index=False)
        B.to_excel(xw, sheet_name="vs_model", index=False)
        cm.to_excel(xw, sheet_name="confusion")


if __name__ == "__main__":
    main()
