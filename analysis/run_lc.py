"""
run_lc.py — learning-curve jobs of s05 in a second Colab session (v4)
=====================================================================

Runs the 120 learning-curve jobs of s05_train.py (phase "lc") with the same
run_job() function, in parallel with the main training session. The job names
(LC_…) differ from the main run's, so the two sessions never write the same
prediction or run file; this session logs to reports/progress_lc.log instead of
progress.log. A job is finished when its runs/<job>/model_info.json exists
(written last); finished jobs are skipped, so the script can be restarted.

Usage (Colab, CPU runtime):
    %cd /content/drive/MyDrive/CDS_v4/src
    !python run_lc.py
"""

import argparse
import time
from pathlib import Path

import s05_train as S

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/CDS_v4"))
ap.add_argument("--work", type=Path, default=Path("/content/work_lc"))
ap.add_argument("--presets", default="best_quality")
ap.add_argument("--only", nargs="*", default=None, help="job names (testing)")
ap.add_argument("--time-limit", type=int, default=None, help="testing only")
a = ap.parse_args()

jobs = S.job_list("lc")
if a.only:
    jobs = [j for j in jobs if j["name"] in set(a.only)]
if a.time_limit:
    for j in jobs:
        j["tl"] = a.time_limit
(a.out / "reports").mkdir(exist_ok=True)
log = a.out / "reports" / "progress_lc.log"


def note(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | LC | {msg}"
    print(line, flush=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")


todo = [j for j in jobs if not (a.out / "runs" / j["name"] / "model_info.json").exists()]
note(f"START: {len(jobs)} learning-curve jobs, {len(jobs) - len(todo)} already finished, {len(todo)} to run")
t0 = time.time()
for i, j in enumerate(todo, 1):
    try:
        info = S.run_job(a.out, a.work, j, a.presets)
    except Exception as e:
        note(f"ERROR {j['name']}: {type(e).__name__}: {e}")
        raise
    if info is None:
        note(f"[{i}/{len(todo)}] {j['name']}: finished elsewhere, skipped")
        continue
    note(f"[{i}/{len(todo)}] {j['name']}: {info['fit_seconds']} s, OOF macro F1 {info['oof_f1_macro_best']:.3f}"
         f" | elapsed {(time.time() - t0) / 3600:.1f} h")
note(f"DONE: {sum((a.out / 'runs' / j['name'] / 'model_info.json').exists() for j in jobs)}/{len(jobs)} finished")
