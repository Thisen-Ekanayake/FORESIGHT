"""Small-sample statistics for paired navigation comparisons (same episodes under two conditions).

Bootstrap percentile intervals for means/rates; exact McNemar (two-sided binomial on the discordant pairs) for paired
binary outcomes such as success or crash; paired bootstrap for differences of means such as SPL.
"""
from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(x, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0):
    """-> (mean, lo, hi) of a 1-D sample (NaNs dropped)."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (n_boot, x.size))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0):
    """Mean of (b - a) over pairs with both values, with its percentile interval and a two-sided bootstrap p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)), float(min(p, 1.0))


def mcnemar_exact(a, b):
    """Paired binary outcomes a, b (True/False per episode). -> (n_a_only, n_b_only, two-sided p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return n10, n01, float(min(p, 1.0))
