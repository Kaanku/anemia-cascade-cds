"""
run_jobs.py — finish the remaining s05 jobs in one or more parallel Colab sessions (v4)
=======================================================================================

Resumes the AutoGluon training of s05_train.py after a disconnected session.
The jobs of the given phases are taken in s05's fixed order and split by their
position in that full list (job i goes to part (i mod n) + 1), so n sessions
started at any time never run the same job twice. Finished jobs (their
runs/<job>/model_info.json exists; it is written last) are skipped. Every job
is trained with s05_train.run_job unchanged. Each part logs to
reports/progress_part<k>.log.

Usage (Colab, CPU runtime; one session per part):
    %cd /content/drive/MyDrive/CDS_v4/src
    !python run_jobs.py --part 1/2        # in the first session
    !python run_jobs.py --part 2/2        # in the second session
"""

import argparse
import time
from pathlib import Path

import s05_train as S

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/CDS_v4"))
ap.add_argument("--work", type=Path, default=Path("/content/work"))
ap.add_argument("--phases", nargs="+", default=["main", "t13"], choices=["main", "t13", "lc"])
ap.add_argument("--part", default="1/1", help="k/n: this session runs every n-th job, starting with the k-th")
ap.add_argument("--presets", default="best_quality")
ap.add_argument("--dry-run", action="store_true", help="only list this part's remaining jobs")
ap.add_argument("--max-jobs", type=int, default=None, help="testing only")
ap.add_argument("--time-limit", type=int, default=None, help="testing only")
a = ap.parse_args()

k, n = (int(x) for x in a.part.split("/"))
assert 1 <= k <= n, "--part must be k/n with 1 <= k <= n"
jobs = [j for ph in a.phases for j in S.job_list(ph)]


def done(j: dict) -> bool:
    return (a.out / "runs" / j["name"] / "model_info.json").exists()


mine = [j for i, j in enumerate(jobs) if i % n == k - 1 and not done(j)]
if a.max_jobs:
    mine = mine[:a.max_jobs]
if a.time_limit:
    for j in mine:
        j["tl"] = a.time_limit
if a.dry_run:
    print(f"part {k}/{n}: {len(mine)} jobs left", *[j["name"] for j in mine], sep="\n  ")
    raise SystemExit(0)

(a.out / "reports").mkdir(exist_ok=True)
log = a.out / "reports" / f"progress_part{k}.log"


def note(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | part {k}/{n} | {msg}"
    print(line, flush=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")


left_all = sum(not done(j) for j in jobs)
note(f"START {'+'.join(a.phases)}: {len(jobs)} jobs, {len(jobs) - left_all} finished, {left_all} left, "
     f"{len(mine)} in this part (≈ {sum(j['tl'] for j in mine) / 3600:.1f} h)")
t0 = time.time()
for i, j in enumerate(mine, 1):
    try:
        info = S.run_job(a.out, a.work, j, a.presets)
    except Exception as e:
        note(f"ERROR {j['name']}: {type(e).__name__}: {e}")
        raise
    if info is None:
        note(f"[{i}/{len(mine)}] {j['name']}: finished elsewhere, skipped")
        continue
    note(f"[{i}/{len(mine)}] {j['name']}: {info['fit_seconds']} s, OOF macro F1 {info['oof_f1_macro_best']:.3f}"
         f" | elapsed {(time.time() - t0) / 3600:.1f} h")
note(f"DONE: {sum(done(j) for j in jobs)}/{len(jobs)} jobs of {'+'.join(a.phases)} finished")
