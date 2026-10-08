"""
s16b_examples.py — example inputs for the demo app (R30)
=========================================================

The app's models hold the synthetic training cohort as context, so an example taken from that cohort would be
an in-sample prediction. The examples are drawn instead from a separate synthetic sample (the same generator,
fitted on the 863 development patients, seed 4242, 300 records per class) and are therefore new to the app's
models. Per class the record with all inputs present that is closest to the real class's median profile (age, HGB,
RBC, MCV, MCHC, RDW-SD, RET-He, IRF, ferritin, iron, LDH; log ferritin / LDH; scaled by the real class IQR) is
taken (only the class medians and IQRs of the real data are used).

Output: <APP>/data/examples.json
Usage:  python s16b_examples.py --out /path/CDS_v43 --app /path/CDS_v43/app
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import s16_synthetic as S

SEED_EXAMPLES, N_PER_CLASS = 4242, 300
KEY = ["age", "hgb_g_d_l", "rbc_10_6_u_l", "mcv_f_l", "mchc_g_dl", "rdw_sd_fl", "ret_he_pg", "irf_pct",
       "ferritin", "iron", "ldh"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--app", required=True, type=Path)
    a = ap.parse_args()
    real = pd.read_parquet(a.out / "data" / "model" / "A1.parquet")
    gen = S.Generator(real)
    pool = gen.sample(np.random.default_rng(SEED_EXAMPLES), {k: N_PER_CLASS for k in S.CLASSES})
    pool = pool[pool[S.BASE + S.BIO].notna().all(axis=1)]
    train = pd.read_parquet(a.out / "data" / "synthetic" / "synthetic_cohort.parquet")
    assert S.exact_matches(pool, train) == 0 and S.exact_matches(pool, real) == 0
    ex = {}
    logged = lambda d: d.assign(ferritin=np.log(d["ferritin"]), ldh=np.log(d["ldh"]))
    for k in S.CLASSES:
        g = pool[pool["cls"] == k]
        ref = logged(real.loc[real["cls"] == k, KEY].dropna())             # the real class profile
        z = (logged(g[KEY]) - ref.median()) / (ref.quantile(.75) - ref.quantile(.25)).replace(0, 1)
        r = g.loc[(z ** 2).sum(axis=1).idxmin()]
        ex[k] = {c: float(r[c]) for c in S.BASE + S.BIO}
        ex[k]["record_id"] = f"EX-{k}"
        print(k, len(g), {c: round(ex[k][c], 2) for c in KEY})
    (a.app / "data" / "examples.json").write_text(json.dumps(ex, indent=1))


if __name__ == "__main__":
    main()
