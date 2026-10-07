"""
app_selftest.py — end-to-end check of the demo engine with the real TabPFN backend (no interface)
=====================================================================================================

Loads the cascade exactly as the app does (synthetic training data, locked JSON), runs every example and a
case without biochemistry, checks the outputs for consistency and prints wall times. Run on the deployment
target or any CPU machine with the TabPFN-3.5 weights (TABPFN_TOKEN set).

Usage: python app/app_selftest.py
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("CDS_THREADS", "2"))   # a small host has 2 cores
import numpy as np  # noqa: E402

from engine import BIO, DATA, S2_CLASSES, Cascade  # noqa: E402

t0 = time.time()
eng = Cascade()
print(f"backend {eng.backend}, model {eng.model_label}, lock engine {eng.lock_engine}, models fitted in {eng.load_seconds} s "
      f"({eng.n_train})", flush=True)
examples = json.loads((DATA / "examples.json").read_text())
out = {}
for name, ex in examples.items():
    for variant in ("full", "no biochemistry"):
        x = {k: v for k, v in ex.items() if k != "record_id"}
        if variant != "full":
            x.update({c: None for c in BIO})
        r = eng.run(x)
        assert r["ok"], (name, r["check"])
        for tier in ("tier1", "tier2"):
            if tier in r:
                p = np.array(list(r[tier]["stage2"]["probs"].values()))
                assert abs(p.sum() - 1) < 1e-4 and list(r[tier]["stage2"]["probs"]) == S2_CLASSES
                assert 0 <= r[tier]["stage1"]["score"] <= 1
        if name in ("IDA", "HA", "HGB_HTZ", "NORMAL"):              # class examples near their class median
            if not r["tier1"]["stage1"]["aac"]:
                print(f"  check: Tier 1 Stage 1 calls the {name} example 'another cause' "
                      f"(score {r['tier1']['stage1']['score']:.3f})", flush=True)
            if r["tier1"]["stage2"]["top"] != name:
                print(f"  check: Tier 1 Stage 2 top class {r['tier1']['stage2']['top']} for the {name} example", flush=True)
        fin = r["final"]
        out[f"{name} / {variant}"] = {"rule": fin["rule"], "class": fin.get("class"), "tier": fin["tier"],
                                      "seconds": r["seconds"],
                                      "tier1_top": r["tier1"]["stage2"]["top"], "tier1_zone": r["tier1"]["stage2"]["zone"],
                                      "p_aac_tier1": round(r["tier1"]["stage1"]["p_display"], 3),
                                      "score_aac_tier1": round(r["tier1"]["stage1"]["score"], 3),
                                      "top_score_tier1": round(r["tier1"]["stage2"]["top_score"], 3)}
        print(f"{name:8s} {variant:16s} -> tier {fin['tier']} {fin['rule']:5s} {str(fin.get('class')):8s} "
              f"(Tier 1 top {r['tier1']['stage2']['top']} {r['tier1']['stage2']['zone']}, {r['seconds']} s)", flush=True)
summary = {"backend": eng.backend, "model": eng.model_label, "locked_on": eng.locked_on,
           "fit_seconds": eng.load_seconds, "threads": os.environ["OMP_NUM_THREADS"],
           "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
           "model_cache": os.environ.get("TABPFN_MODEL_CACHE_SIZE"),
           "median_run_seconds": float(np.median([v["seconds"] for v in out.values()])),
           "total_seconds": round(time.time() - t0, 1), "fit_mode": os.environ.get("CDS_FIT_MODE", "default"),
           "cases": out}
print(json.dumps({k: v for k, v in summary.items() if k != "cases"}))
(Path(__file__).parent / "selftest_result.json").write_text(json.dumps(summary, indent=1))
