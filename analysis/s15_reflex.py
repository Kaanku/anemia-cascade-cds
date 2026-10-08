"""
s15_reflex.py — reflex test recommendations of the cascade (CDS v4.2, R27)
===========================================================================

The rule set of the earlier version of the system (14 rules: tier x predicted class x confidence zone), adapted
to the v4 cascade. The cascade is the one evaluated in s09 (block "cascade"): Tier 1 = CBC-only
Stage 1 (locked threshold) and Stage 2; a patient is finalised at Tier 1 when Stage 1 says AAC and the
Stage 2 top-class probability reaches the locked HIGH cut-off (90 % accuracy target on inner OOF).
Everyone else is escalated: the biochemistry panel is added and the CBC + biochemistry models decide
(Tier 2). Zones: HIGH = top-class probability >= the configuration's locked HIGH cut-off; LOW < 0.35;
MEDIUM in between (L4). Nothing is fitted here.

Population: A1, nested-CV outer folds, first sample, patients with measured biochemistry (657; the
cascade needs the Tier 2 panel), and the temporal cohort (97 with an analyzer record).
Reported per final rule: number and share of patients and the share whose true class matches the
predicted class of the rule (for "OAC" rules: true OAC).

Outputs: reports/reflex/reflex_rules.csv (the rule table), reflex_distribution.csv
Usage: python s15_reflex.py --out /path/CDS_v4
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
CLS = {"IDA": "IDA", "HA": "HA", "HGB_HTZ": "HGB HTZ", "NORMAL": "Normal", "OAC": "OAC"}

RULES = [  # id, tier, predicted class, zone, recommended test, urgency, rationale
    ("T1-1", 1, "IDA", "HIGH", "Ferritin + serum iron + TIBC/UIBC", "routine",
     "Confirm iron deficiency before treatment"),
    ("T1-2", 1, "HA", "HIGH", "LD + indirect bilirubin + haptoglobin + peripheral smear + direct antiglobulin test",
     "urgent", "Confirm haemolysis and screen for an immune cause (most HA in this cohort was autoimmune)"),
    ("T1-3", 1, "HGB HTZ", "HIGH", "Haemoglobin HPLC / electrophoresis", "routine",
     "Confirm the haemoglobinopathy carrier state"),
    ("T1-4", 1, "Normal", "HIGH", "No additional testing", "none", "High-confidence normal red cell profile"),
    ("T1-5", 1, "any", "MEDIUM", "Biochemistry panel (ferritin, iron, UIBC, LD) -> Tier 2", "routine",
     "Not resolved by the CBC: escalate"),
    ("T1-6", 1, "any", "LOW", "Biochemistry panel (ferritin, iron, UIBC, LD) -> Tier 2", "priority",
     "Low confidence: escalate and flag for review"),
    ("T1-7", 1, "OAC", "-", "Biochemistry panel (ferritin, iron, UIBC, LD) -> Tier 2", "routine",
     "Stage 1 predicts another cause: verify with biochemistry"),
    ("T2-1", 2, "IDA", "HIGH", "Clinical evaluation for iron therapy; repeat CBC after treatment", "routine",
     "Iron deficiency supported by CBC and iron studies"),
    ("T2-2", 2, "HA", "HIGH", "Haemolysis work-up (indirect bilirubin, haptoglobin, smear, DAT) + haematology consultation",
     "urgent", "Haemolysis supported by CBC and LD: investigate the cause"),
    ("T2-3", 2, "HGB HTZ", "HIGH", "Haemoglobin HPLC / electrophoresis; if HbA2 is normal and iron is replete, alpha-globin "
     "molecular testing; genetic counselling", "routine", "Carrier state likely; alpha-thalassaemia trait can have a normal HPLC"),
    ("T2-4", 2, "Normal", "HIGH", "No additional testing", "none", "Normal profile with full biochemistry"),
    ("T2-5", 2, "OAC", "-", "Investigate other causes (CRP, creatinine/eGFR, vitamin B12, folate, reticulocyte response)",
     "routine", "Not one of the four classes after biochemistry"),
    ("T2-6", 2, "any", "MEDIUM", "Specialist review of the CBC, smear and biochemistry", "routine",
     "Still uncertain after biochemistry: human review"),
    ("T2-7", 2, "any", "LOW", "Specialist review + extended work-up", "priority",
     "Low confidence after biochemistry"),
]


def zone(p: np.ndarray, high: np.ndarray) -> np.ndarray:
    top = p.max(1)
    return np.where(top >= high, "HIGH", np.where(top < L.LOW_CUTOFF, "LOW", "MEDIUM"))


def assign(c1, c2, b1, b2) -> pd.DataFrame:
    k = c1.index
    Pc, Pb = V.P2(c2.loc[k]), V.P2(b2.loc[k])
    s1c_aac = (c1["s"] >= c1["thr"]).to_numpy()
    s1b_aac = (b1.loc[k, "s"] >= b1.loc[k, "thr"]).to_numpy()
    zc, zb = zone(Pc, c2.loc[k, "high"].to_numpy()), zone(Pb, b2.loc[k, "high"].to_numpy())
    pc = np.array([CLS[x] for x in np.array(V.S2)[Pc.argmax(1)]])
    pb = np.array([CLS[x] for x in np.array(V.S2)[Pb.argmax(1)]])
    tier1 = s1c_aac & (zc == "HIGH")
    rule, pred = [], []
    for i in range(len(k)):
        if tier1[i]:
            r = {"IDA": "T1-1", "HA": "T1-2", "HGB HTZ": "T1-3", "Normal": "T1-4"}[pc[i]]
            rule.append(r), pred.append(pc[i])
        elif not s1b_aac[i]:
            rule.append("T2-5"), pred.append("OAC")
        elif zb[i] == "HIGH":
            rule.append({"IDA": "T2-1", "HA": "T2-2", "HGB HTZ": "T2-3", "Normal": "T2-4"}[pb[i]]), pred.append(pb[i])
        else:
            rule.append("T2-6" if zb[i] == "MEDIUM" else "T2-7"), pred.append(pb[i])
    t1why = np.where(~s1c_aac, "T1-7", np.where(zc == "LOW", "T1-6", np.where(zc == "MEDIUM", "T1-5", "")))
    return pd.DataFrame({"record_id": k, "true": [CLS[x] for x in c1["cls"]], "rule": rule, "pred": pred,
                         "tier1_rule_if_escalated": t1why})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    od = a.out / "reports" / "reflex"
    od.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(RULES, columns=["rule", "tier", "predicted_class", "zone", "recommended_test", "urgency", "rationale"])
    R.to_csv(od / "reflex_rules.csv", index=False)
    lock = json.loads((a.out / "lock" / "lock.json").read_text())
    D = V.Data(a.out, lock, joblib.load(a.out / "lock" / "calibrators.joblib"))
    first = lambda d: d[d["is_index_sample"].astype(bool)].set_index("record_id")
    out = []
    for set_name in ("nested CV", "temporal"):
        get = (lambda c: first(D.pooled(ENGINE, c))) if set_name == "nested CV" else (lambda c: first(D.temporal(ENGINE, c)))
        c1, c2, b1, b2 = (get(f"A1_FULL_{x}") for x in ("CBC_S1", "CBC_S2", "CBC_BIO_S1", "CBC_BIO_S2"))
        ids = c1.index.intersection(b1.index)
        if set_name == "temporal":
            ids = ids[c1.loc[ids, "has_analyzer_record"].astype(bool).to_numpy()]
        A = assign(c1.loc[ids], c2, b1, b2)
        A["correct"] = A["true"] == A["pred"]
        g = A.groupby("rule").agg(n=("rule", "size"), class_match=("correct", "mean")).reset_index()
        g["share"] = g["n"] / len(A)
        g["set"], g["n_total"] = set_name, len(A)
        esc = A.loc[A["tier1_rule_if_escalated"] != "", "tier1_rule_if_escalated"].value_counts()
        for r_, n_ in esc.items():
            g = pd.concat([g, pd.DataFrame([{"rule": r_ + " (escalation step)", "n": int(n_), "share": n_ / len(A),
                                             "set": set_name, "n_total": len(A)}])])
        out.append(g)
        print(f"{set_name}: {len(A)} patients")
        print(g.merge(R[["rule", "recommended_test"]], on="rule", how="left")[["rule", "n", "share", "class_match"]]
              .round(3).to_string(index=False))
    pd.concat(out, ignore_index=True).to_csv(od / "reflex_distribution.csv", index=False)


if __name__ == "__main__":
    main()
