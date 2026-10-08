"""
time_app.py — CPU timing of the demo engine: model fit, batched predictions and SHAP explanations
=================================================================================================

Runs the engine as the app does (TabPFN-3.5-fast, the ensemble size of data/lock.json, CPU) and measures
  * predict_proba on 1 to 1026 rows for the Stage 2 and Stage 1 models (the cost of one SHAP call),
  * the cascade and explain() for the synthetic examples (budget CDS_SHAP_BUDGET, default 512),
  * for two examples, the agreement of the 512-coalition values with a 4096-coalition reference
    (top-10 overlap, largest absolute difference), when 1026 rows take less than 40 s.
Run on a CPU host with the TabPFN-3.5 weights (TABPFN_TOKEN set); on a GPU machine hide the GPU first.

Usage: CUDA_VISIBLE_DEVICES="" python app/time_app.py [--out timing.json]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("CDS_THREADS", "2"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import BIO, CONFIGS, DATA, SHAP_BUDGET, Cascade, add_ratios  # noqa: E402


def rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "timing_result.json"))
    args = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    info = {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "cuda": torch.cuda.is_available()}
    t0 = time.time()
    eng = Cascade()
    info.update(model=eng.model_label, fit_seconds=eng.load_seconds, rss_after_fit_gb=rss_gb())
    print(json.dumps(info), flush=True)

    syn = pd.read_parquet(DATA / "synthetic_cohort.parquet")
    mats = {"CBC": add_ratios(syn, "CBC"),
            "CBC_BIO": add_ratios(eng.imputer.transform(syn[syn[BIO].notna().any(axis=1)]), "CBC_BIO")}
    pred = []
    for key in ("cbc_s2", "bio_s2", "cbc_s1"):
        X = mats["CBC_BIO" if key.startswith("bio") else "CBC"][eng.features[CONFIGS[key]]].to_numpy(float)
        eng.models[key].predict_proba(X[:1])                                   # warm-up
        for n in (1, 64, 256, 514, 1026):
            t = time.time()
            eng.models[key].predict_proba(X[np.arange(n) % len(X)])
            pred.append({"model": key, "rows": n, "seconds": round(time.time() - t, 2), "max_rss_gb": rss_gb()})
            print(f"predict {key} {n:5d} rows: {pred[-1]['seconds']:6.2f} s (max RSS {pred[-1]['max_rss_gb']} GB)",
                  flush=True)
    slow = max(r["seconds"] for r in pred if r["rows"] == 1026)

    ex = json.loads((DATA / "examples.json").read_text())
    cases = []
    for name, e in ex.items():
        x = {k: v for k, v in e.items() if k != "record_id"}
        t = time.time()
        r = eng.run(x)
        run_s = round(time.time() - t, 1)
        fin = r["final"]
        key = ("bio_s1" if fin["rule"] == "T2-5" else "bio_s2") if fin["tier"] == 2 else \
              ("cbc_s1" if fin["rule"] == "T1-7" else "cbc_s2")
        o = eng.explain(r, key)
        phi = np.asarray(o["phi"])
        gap = float(np.abs(phi.sum(0) - (np.asarray(o["f_x"]) - np.asarray(o["f_ref"]))).max())
        c = {"example": name, "rule": fin["rule"], "cascade_seconds": run_s, "model": key,
             "players": len(o["players"]), "coalitions": o["n_coalitions"], "shap_seconds": o["seconds"],
             "efficiency_gap": gap, "check_vs_cascade": o["check_vs_cascade"], "max_rss_gb": rss_gb()}
        if slow < 40 and name in ("HA", "HGB_HTZ"):
            ref = eng.explain(r, key, budget=4096)
            k = 0 if key.endswith("s1") else S2.index(r["tier2" if key.startswith("bio") else "tier1"]["stage2"]["top"])
            a, b = phi[:, k], np.asarray(ref["phi"])[:, k]
            top_a, top_b = set(np.argsort(-np.abs(a))[:10]), set(np.argsort(-np.abs(b))[:10])
            c.update(ref_coalitions=ref["n_coalitions"], ref_seconds=ref["seconds"],
                     top10_overlap=len(top_a & top_b), max_abs_diff=float(np.abs(a - b).max()),
                     max_abs_ref=float(np.abs(b).max()))
        cases.append(c)
        print(json.dumps(c), flush=True)
    info.update(predict=pred, cases=cases, budget=SHAP_BUDGET, total_seconds=round(time.time() - t0, 1),
                max_rss_gb=rss_gb())
    Path(args.out).write_text(json.dumps(info, indent=1))
    print("saved", args.out)


S2 = ["HA", "HGB_HTZ", "IDA", "NORMAL"]

if __name__ == "__main__":
    main()
