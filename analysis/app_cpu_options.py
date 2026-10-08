"""
app_cpu_options.py — CPU cost of TabPFN-3.5 settings for the demo app (Colab, 2 threads; R31)
=============================================================================================

Measures, on the synthetic A1 FULL CBC Stage 1 matrix (863 rows, 50 features) and the CBC + biochemistry
Stage 2 matrix (449 rows, 103 features), the wall time of fit and of a one-patient prediction, and the resident
memory, for: the study setting (n_estimators "auto") in the default fit mode, n_estimators 1 and 2, the separate
"v3.5-fast" model, and fit_with_cache. Then four default-mode models together with the built-model cache
(TABPFN_MODEL_CACHE_SIZE=1), as the app would hold them. Output: <OUT>/app/cpu_options.json
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TABPFN_MODEL_CACHE_SIZE"] = "1"
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/CDS_v43")
proc = psutil.Process()
gb = lambda: round(proc.memory_info().rss / 1e9, 2)


def data(sc, st):
    d = pd.read_parquet(OUT / "data" / "synthetic" / f"A1_FULL_{sc}" / "train.parquet")
    if st != "S1":
        d = d[d["cls"].isin(["IDA", "HA", "HGB_HTZ", "NORMAL"])]
    f = json.loads((OUT / "data" / "features" / f"A1_FULL_{sc}" / "final" / f"features_{st}.json").read_text())
    return d[f].to_numpy(float), (d["y_s1"] if st == "S1" else d["cls"]).to_numpy()


res = {"threads": 2, "start_rss_gb": gb()}
for sc, st in (("CBC", "S1"), ("CBC_BIO", "S2")):
    X, y = data(sc, st)
    for label, ver, kw in (("auto, default", "v3.5", {}), ("n_estimators=1", "v3.5", {"n_estimators": 1}),
                           ("n_estimators=2", "v3.5", {"n_estimators": 2}), ("v3.5-fast, auto", "v3.5-fast", {}),
                           ("auto, fit_with_cache", "v3.5", {"fit_mode": "fit_with_cache"})):
        gc.collect()
        r0 = gb()
        try:
            t0 = time.time()
            clf = TabPFNClassifier.create_default_for_version(ModelVersion(ver), device="cpu", random_state=42, **kw)
            clf.fit(X, y)
            t1 = time.time()
            clf.predict_proba(X[:1])
            t2 = time.time()
            res[f"{sc}_{st} | {label}"] = {"fit_s": round(t1 - t0, 1), "predict_s": round(t2 - t1, 1),
                                           "rss_gb": gb(), "rss_added_gb": round(gb() - r0, 2)}
        except Exception as e:
            res[f"{sc}_{st} | {label}"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(f"{sc}_{st} | {label}: {res[f'{sc}_{st} | {label}']}", flush=True)
        clf = None
        gc.collect()

# four default-mode models at once, sharing the built model
gc.collect()
r0 = gb()
models, t0 = [], time.time()
for sc, st in (("CBC", "S1"), ("CBC", "S2"), ("CBC_BIO", "S1"), ("CBC_BIO", "S2")):
    X, y = data(sc, st)
    models.append(TabPFNClassifier.create_default_for_version(ModelVersion("v3.5"), device="cpu", random_state=42,
                                                              n_estimators=1).fit(X, y))
res["four models, n_estimators=1, shared"] = {"fit_s": round(time.time() - t0, 1), "rss_gb": gb(),
                                              "rss_added_gb": round(gb() - r0, 2),
                                              "same_model_object": len({id(m.model_) for m in models}) == 1}
print(res["four models, n_estimators=1, shared"], flush=True)
(OUT / "app" / "cpu_options.json").write_text(json.dumps(res, indent=1))
print("written", OUT / "app" / "cpu_options.json")
