"""
test_kernel_shap.py — the app's KernelSHAP against exact Shapley values (no model or weights needed)
=====================================================================================================

1. With a budget of at least 2^p − 2 coalitions, kernel_design() + shapley_regression() must reproduce the
   Shapley values of brute-force enumeration (random non-additive games, p = 2 … 8).
2. With fewer coalitions (sampled design) the values must add up to f(all) − f(none) exactly, and the error
   against the exact values must shrink as the budget grows (p = 12).

Usage: python app/test_kernel_shap.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import kernel_design, shapley_regression  # noqa: E402


def game(p: int, rng: np.random.Generator):
    a, B, C = rng.normal(size=p), rng.normal(size=(p, p)) * 0.5, rng.normal(size=(p, p, p)) * 0.2

    def v(z) -> float:
        z = np.asarray(z, float)
        return float(np.tanh(a @ z + z @ B @ z / p + np.einsum("i,j,k,ijk->", z, z, z, C) / p ** 2))
    return v


def exact(v, p: int) -> np.ndarray:
    phi = np.zeros(p)
    for i in range(p):
        rest = [j for j in range(p) if j != i]
        for s in range(p):
            w = math.factorial(s) * math.factorial(p - s - 1) / math.factorial(p)
            for S in itertools.combinations(rest, s):
                z = np.zeros(p)
                z[list(S)] = 1
                z1 = z.copy()
                z1[i] = 1
                phi[i] += w * (v(z1) - v(z))
    return phi


def kernel(v, p: int, budget: int, seed: int) -> np.ndarray:
    Z, w = kernel_design(p, budget, seed)
    Y = np.array([[v(z)] for z in Z])
    return shapley_regression(Z, w, Y, np.array([v(np.zeros(p))]), np.array([v(np.ones(p))]))[:, 0]


def main() -> None:
    rng = np.random.default_rng(2026)
    for p in range(2, 9):
        v = game(p, rng)
        err = np.abs(kernel(v, p, 2 ** p, 0) - exact(v, p)).max()
        assert err < 1e-10, (p, err)
        print(f"p = {p}: all {2 ** p - 2} coalitions, max |error| = {err:.1e}")
    p = 12
    v = game(p, rng)
    ex = exact(v, p)
    total = v(np.ones(p)) - v(np.zeros(p))
    mean_err = []
    for budget in (256, 1024, 4000):
        errs = []
        for seed in range(10):
            phi = kernel(v, p, budget, seed)
            assert abs(phi.sum() - total) < 1e-10                         # efficiency holds exactly
            errs.append(np.abs(phi - ex).max() / np.abs(ex).max())
        mean_err.append(float(np.mean(errs)))
        print(f"p = {p}, budget {budget}: mean relative max error {mean_err[-1]:.4f} (10 seeds)")
    assert mean_err[0] > mean_err[1] > mean_err[2], mean_err
    print("ok")


if __name__ == "__main__":
    main()
