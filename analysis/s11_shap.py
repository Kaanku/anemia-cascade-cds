"""
s11_shap.py — Shapley explanations of the TabPFN-3.5 models (CDS v4.1, R21)
============================================================================

Explains the main model (TabPFN-3.5, R12–R15) with Shapley values computed by
shapiq (Muschalik et al., NeurIPS 2024) on imputation-based feature removal:
a feature left out of a coalition is replaced by values drawn from background
patients (marginal / interventional imputation), the training context of the
model stays fixed, so the transformer key-value cache can be reused across
coalitions (fit_mode="fit_with_cache"). First-order Shapley values only
(index "SV"), estimated with OddSHAP when available (shapiq >= 1.6, the
estimator tabpfn-extensions routes Shapley values to), else KernelSHAP.

Scope (decided with the user, 2026-10-03):
    configurations  A1 FULL CBC (S1, S2, MX) and A1 FULL CBC_BIO (S1, S2)
    outer folds     each outer-fold model explains its own outer-fold patients
                    (patients it has not seen): global importance, 863 patients
    final model     explains the temporal cohort (FULL: patients with an analyzer
                    record, as in the evaluation, N8): case examples
    explained rows  the rows s09 evaluates: S1 all patients, S2 true AAC classes,
                    MX true IDA / HGB HTZ
    explained output S1 P(AAC) · MX P(HGB HTZ) · S2 each of the four classes
Every model is refitted exactly as in s05c (same rows, features, seed); the
refit's predictions are checked against the stored s05c predictions before
anything is explained (max |difference| is recorded).

The four Stage 2 classes share one set of model calls: for a given patient the
four explainers are reset to the same random state, so they draw the same
coalitions and background rows, and the class probabilities of every batch
are computed once (CachedProba).

Outputs (finished jobs are skipped; a job resumes from its checkpoint):
    <OUT>/shap_tabpfn35/<cfg>/<fold-set>.parquet   one row per patient × explained class:
          metadata, class, base value, model prediction, phi_<feature> (Shapley values),
          x_<feature> (the patient's feature values)
    <OUT>/runs_shap35/<cfg>__<fold-set>.json        settings, versions, checks (written last)
    <OUT>/reports/progress_shap35.log

Usage (Colab, GPU; TabPFN-3.5 licence accepted, TABPFN_TOKEN set):
    pip install "tabpfn==9.0.0" "shapiq>=1.6" lightgbm
    python s11_shap.py --out /content/drive/MyDrive/CDS_v4 --data /content/cds_data --pilot
    python s11_shap.py --out /content/drive/MyDrive/CDS_v4 --data /content/cds_data \
        --settings /content/drive/MyDrive/CDS_v4/src/shap_settings.json      # chosen after the pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
VERSION = "v3.5"
TAG = "shap35"
ANALYSIS = "A1"
CONFIGS = [("FULL", "CBC", "S1"), ("FULL", "CBC", "S2"), ("FULL", "CBC", "MX"),
           ("FULL", "CBC_BIO", "S1"), ("FULL", "CBC_BIO", "S2")]
FOLDSETS = [f"outer{k}" for k in range(5)] + ["final"]
AAC = ["IDA", "HA", "HGB_HTZ", "NORMAL"]
POP = {"S1": None, "S2": AAC, "MX": ["IDA", "HGB_HTZ"]}            # training rows, as s05/s05c
EXPLAIN = {"S1": ["1"], "MX": ["HGB_HTZ"], "S2": ["HA", "HGB_HTZ", "IDA", "NORMAL"]}
META = ["record_id", "patient_id", "is_index_sample", "cls", "y_s1", "n_bio_measured"]
CHECKPOINT_EVERY = 10


# ─────────────────────────────────────────────────────────────── model
def make_model(fake: bool = False):
    """TabPFN-3.5 exactly as s05c, plus the KV cache at full precision (no int8 quantisation,
    so the explained function equals the evaluated one)."""
    if fake:                                         # local test of the pipeline only
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=60, random_state=SEED), "cpu"
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion(VERSION), device=device, random_state=SEED,
        fit_mode="fit_with_cache", kv_cache_precision="auto")
    return clf, device


class CachedProba:
    """predict_proba with a cache keyed on the input batch, shared by the explainers of one
    patient (the four Stage 2 classes draw identical batches)."""

    def __init__(self, clf):
        self.clf, self.cache, self.calls, self.hits, self.rows = clf, {}, 0, 0, 0

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=float)
        key = hashlib.sha1(X.tobytes()).hexdigest() + str(X.shape)
        self.calls += 1
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        P = self.clf.predict_proba(X)
        self.rows += len(X)
        self.cache[key] = P
        return P

    def output(self, k: int):
        return lambda X: self.proba(X)[:, k]

    def clear(self):
        self.cache = {}


def make_explainer(f, background: np.ndarray, imputer: str, sample_size: int):
    import shapiq
    n = background.shape[1]
    odd = getattr(shapiq, "OddSHAP", None)
    approximator = odd(n=n, random_state=SEED) if odd is not None else "auto"
    kw = {"sample_size": sample_size} if imputer == "marginal" else {}
    return shapiq.TabularExplainer(model=f, data=background, imputer=imputer, approximator=approximator,
                                   index="SV", max_order=1, random_state=SEED, **kw), \
        ("OddSHAP" if odd is not None else "auto (KernelSHAP)")


# ─────────────────────────────────────────────────────────────── data
def load_job(data: Path, cfg_parts: tuple, fset: str) -> dict:
    fs, sc, st = cfg_parts
    fdir = data / "features" / f"{ANALYSIS}_{fs}_{sc}" / fset
    feats = json.loads((fdir / f"features_{st}.json").read_text())
    tr = pd.read_parquet(fdir / "train.parquet")
    if POP[st] is not None:
        tr = tr[tr["cls"].isin(POP[st])]
    tr = tr.reset_index(drop=True)
    y = (tr["y_s1"] if st == "S1" else tr["cls"]).to_numpy()
    if fset == "final":
        tg = pd.read_parquet(fdir / "temporal.parquet")
        if fs == "FULL":
            tg = tg[tg["has_analyzer_record"].astype(bool)]                       # N8
    else:
        tg = pd.read_parquet(fdir / "eval.parquet")
    tg = tg[tg["is_index_sample"].astype(bool)]                                  # A1: first samples
    if POP[st] is not None:
        tg = tg[tg["cls"].isin(POP[st])]                                         # rows s09 evaluates
    assert not set(tr["patient_id"]) & set(tg["patient_id"]), "patient in training and explained rows"
    return {"feats": feats, "tr": tr, "y": y, "tg": tg.reset_index(drop=True), "st": st}


def stored_predictions(out: Path, cfg: str, fset: str) -> pd.DataFrame | None:
    p = out / "predictions_tabpfn35" / cfg / (f"{fset}_eval.parquet" if fset != "final" else "final_temporal.parquet")
    return pd.read_parquet(p) if p.exists() else None


# ─────────────────────────────────────────────────────────────── run
def explain_rows(J: dict, cp: CachedProba, classes: list, settings: dict, rows: list[int],
                 note=None) -> list[dict]:
    feats, st = J["feats"], J["st"]
    X = J["tg"][feats].to_numpy(dtype=float)
    background = J["tr"][feats].to_numpy(dtype=float)
    budget = max(settings["min_budget"], settings["budget_factor"] * len(feats))
    explainers, approx_name = {}, None
    for c in EXPLAIN[st]:
        k = classes.index(c)
        explainers[c], approx_name = make_explainer(cp.output(k), background, settings["imputer"],
                                                    settings["sample_size"])
    out = []
    for n_done, i in enumerate(rows, 1):
        cp.clear()
        x = X[i:i + 1]
        p_direct = cp.clf.predict_proba(x)[0]
        for c, ex in explainers.items():
            t0 = time.time()
            iv = ex.explain(x, budget=budget, random_state=SEED + i)
            phi = np.asarray(iv.get_n_order_values(1), dtype=float)
            base = float(iv.baseline_value)
            rec = {**{m: J["tg"].iloc[i][m] for m in META}, "explained_class": c,
                   "base_value": base, "prediction": float(p_direct[classes.index(c)]),
                   "sum_phi": float(phi.sum()), "seconds": time.time() - t0}
            rec.update({f"phi_{f}": v for f, v in zip(feats, phi)})
            rec.update({f"x_{f}": v for f, v in zip(feats, X[i])})
            out.append(rec)
        if note and n_done % CHECKPOINT_EVERY == 0:
            note(f"    {n_done}/{len(rows)} patients")
    settings["_budget"], settings["_approximator"] = budget, approx_name
    return out


def run_job(out: Path, data: Path, cfg_parts: tuple, fset: str, settings: dict, note, fake=False) -> dict | None:
    fs, sc, st = cfg_parts
    cfg = f"{ANALYSIS}_{fs}_{sc}_{st}"
    marker = out / f"runs_{TAG}" / f"{cfg}__{fset}.json"
    if marker.exists():
        return None
    t0 = time.time()
    J = load_job(data, cfg_parts, fset)
    clf, device = make_model(fake)
    clf.fit(J["tr"][J["feats"]].to_numpy(dtype=float), J["y"])
    classes = [str(c) for c in clf.classes_]
    cp = CachedProba(clf)

    # the refit must reproduce the evaluated model (s05c, fit_preprocessors, same seed)
    P = clf.predict_proba(J["tg"][J["feats"]].to_numpy(dtype=float))
    check = {"n_rows": len(J["tg"])}
    sp = stored_predictions(out, cfg, fset)
    if sp is not None and not fake:
        m = J["tg"][["record_id"]].merge(sp, on="record_id", how="left", validate="one_to_one")
        S = m[[f"prob_{c}" for c in classes]].to_numpy(float)
        check["max_abs_diff_vs_s05c"] = float(np.nanmax(np.abs(S - P)))
        check["argmax_agreement"] = float(np.mean(S.argmax(1) == P.argmax(1)))
        assert check["max_abs_diff_vs_s05c"] < 0.02, f"refit differs from s05c: {check}"

    sdir = out / f"shap_{'tabpfn35' if not fake else 'fake'}" / cfg
    sdir.mkdir(parents=True, exist_ok=True)
    part = sdir / f"{fset}.partial.parquet"
    done = pd.read_parquet(part) if part.exists() else pd.DataFrame()
    done_ids = set(done["record_id"]) if len(done) else set()
    todo = [i for i in range(len(J["tg"])) if J["tg"].iloc[i]["record_id"] not in done_ids]
    note(f"  {cfg} {fset}: {len(J['tg'])} patients × {len(EXPLAIN[st])} output(s), {len(J['feats'])} features, "
         f"{len(todo)} to explain")
    recs = []
    for s in range(0, len(todo), CHECKPOINT_EVERY):
        recs += explain_rows(J, cp, classes, settings, todo[s:s + CHECKPOINT_EVERY])
        pd.concat([done, pd.DataFrame(recs)], ignore_index=True).to_parquet(part, index=False)
        note(f"    {len(done_ids) + min(s + CHECKPOINT_EVERY, len(todo))}/{len(J['tg'])} patients "
             f"({(time.time() - t0) / 60:.1f} min)")
    res = pd.read_parquet(part)
    res.to_parquet(sdir / f"{fset}.parquet", index=False)
    part.unlink()

    gap = (res["base_value"] + res["sum_phi"] - res["prediction"]).abs()
    info = {"config": cfg, "fold_set": fset, "explained": EXPLAIN[st], "n_patients": int(res["record_id"].nunique()),
            "n_rows": len(res), "features": J["feats"], "n_background": len(J["tr"]),
            "imputer": settings["imputer"], "sample_size": settings["sample_size"] if settings["imputer"] == "marginal" else None,
            "budget": settings.get("_budget"), "approximator": settings.get("_approximator"), "index": "SV",
            "seed": SEED, "check": check, "efficiency_gap_max": float(gap.max()), "efficiency_gap_mean": float(gap.mean()),
            "model_calls": cp.calls, "cache_hits": cp.hits, "rows_predicted": cp.rows,
            "seconds": round(time.time() - t0), "device": device, "python": platform.python_version()}
    try:
        import shapiq
        import tabpfn
        import torch
        info.update({"shapiq": shapiq.__version__, "tabpfn": getattr(tabpfn, "__version__", "?"),
                     "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except ImportError:
        pass
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(info, indent=1, default=str))                       # written last
    return info


def pilot(out: Path, data: Path, note, fake=False) -> None:
    """Time per explained patient and stability for candidate settings; nothing is saved."""
    grid = [("baseline", None, 10, 512), ("marginal", 20, 10, 512), ("marginal", 50, 10, 512),
            ("marginal", 50, 20, 1024)]
    for cfg_parts in [("FULL", "CBC", "S1"), ("FULL", "CBC_BIO", "S2")]:
        J = load_job(data, cfg_parts, "outer0")
        clf, device = make_model(fake)
        t = time.time()
        clf.fit(J["tr"][J["feats"]].to_numpy(dtype=float), J["y"])
        classes = [str(c) for c in clf.classes_]
        note(f"PILOT {cfg_parts} outer0: fit {time.time() - t:.1f} s, {len(J['feats'])} features, "
             f"{len(J['tg'])} patients, device {device}")
        sp = stored_predictions(out, f"{ANALYSIS}_{'_'.join(cfg_parts)}", "outer0")
        if sp is not None and not fake:
            P = clf.predict_proba(J["tg"][J["feats"]].to_numpy(dtype=float))
            m = J["tg"][["record_id"]].merge(sp, on="record_id", how="left")
            d = np.abs(m[[f"prob_{c}" for c in classes]].to_numpy(float) - P)
            note(f"  refit vs s05c predictions: max |diff| {np.nanmax(d):.2e}")
        phis = {}
        for imp, ss, bf, mb in grid:
            cp = CachedProba(clf)
            s = {"imputer": imp, "sample_size": ss or 50, "budget_factor": bf, "min_budget": mb}
            t = time.time()
            r = pd.DataFrame(explain_rows(J, cp, classes, s, [0, 1]))
            sec = (time.time() - t) / 2
            phis[(imp, ss, bf)] = r.filter(like="phi_").to_numpy()
            gap = (r["base_value"] + r["sum_phi"] - r["prediction"]).abs().max()
            note(f"  {imp:9s} sample={ss} budget={s['_budget']}: {sec:.1f} s/patient "
                 f"(all outputs), model rows/patient {cp.rows / 2:,.0f}, cache hits {cp.hits}/{cp.calls}, "
                 f"efficiency gap {gap:.3f}")
        a, b = phis[("marginal", 50, 10)], phis[("marginal", 50, 20)]
        corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        note(f"  stability marginal-50: budget ×10 vs ×20, r = {corr:.3f}, max |Δφ| {np.abs(a - b).max():.3f}")
        c = np.corrcoef(phis[("baseline", None, 10)].ravel(), a.ravel())[0, 1]
        note(f"  baseline vs marginal-50 (×10): r = {c:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--imputer", default="marginal", choices=["marginal", "baseline"])
    ap.add_argument("--sample-size", type=int, default=50, help="background rows per coalition (marginal)")
    ap.add_argument("--budget-factor", type=int, default=10, help="budget = max(min_budget, factor × n_features)")
    ap.add_argument("--min-budget", type=int, default=512)
    ap.add_argument("--settings", type=Path, default=None,
                    help="JSON with imputer / sample_size / budget_factor / min_budget (overrides the flags)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to configs, e.g. A1_FULL_CBC_S1")
    ap.add_argument("--fake-model", action="store_true", help="local pipeline test without TabPFN")
    a = ap.parse_args()
    data = a.data or a.out / "data"
    log = a.out / "reports" / f"progress_{TAG}.log"
    log.parent.mkdir(exist_ok=True)

    def note(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | SHAP TabPFN-3.5 | {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    if a.pilot:
        pilot(a.out, data, note, a.fake_model)
        return
    settings = {"imputer": a.imputer, "sample_size": a.sample_size, "budget_factor": a.budget_factor,
                "min_budget": a.min_budget}
    if a.settings is not None:
        settings.update({k: v for k, v in json.loads(a.settings.read_text()).items() if k in settings})
    jobs = [(c, f) for c in CONFIGS for f in FOLDSETS
            if a.only is None or f"{ANALYSIS}_{'_'.join(c)}" in a.only]
    note(f"START {len(jobs)} jobs, settings {settings}")
    t = time.time()
    for k, (c, f) in enumerate(jobs, 1):
        info = run_job(a.out, data, c, f, dict(settings), note, a.fake_model)
        if info:
            note(f"[{k}/{len(jobs)}] {info['config']} {f}: {info['n_patients']} patients, {info['seconds']} s, "
                 f"check {info['check']}, efficiency gap ≤ {info['efficiency_gap_max']:.3f} "
                 f"| elapsed {(time.time() - t) / 60:.0f} min")
    done = sum((a.out / f"runs_{TAG}" / f"{ANALYSIS}_{'_'.join(c)}__{f}.json").exists() for c, f in jobs)
    note(f"DONE: {done}/{len(jobs)} jobs finished")


if __name__ == "__main__":
    main()
