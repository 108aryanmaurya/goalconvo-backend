"""Statistical summaries and confidence intervals for evaluation metrics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _z_for_confidence(confidence: float) -> float:
    # Common normal approximations (two-sided)
    m = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    return float(m.get(confidence, 1.96))


def mean_std_ci(
    values: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Mean, sample standard deviation, and normal-approximate CI for the mean
    (valid for large n; for small n consider bootstrap — here we use classic t-like z).
    """
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    mean = float(np.mean(arr))
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(np.std(arr, ddof=1))
    z = _z_for_confidence(confidence)
    half = z * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "confidence": confidence,
    }


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (stable at extremes)."""
    if n <= 0:
        return (0.0, 1.0)
    z = _z_for_confidence(confidence)
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def proportion_summary(successes: int, n: int, confidence: float = 0.95) -> Dict[str, Any]:
    if n <= 0:
        return {"n": 0, "rate": None, "wilson_ci_low": None, "wilson_ci_high": None}
    rate = successes / n
    lo, hi = wilson_interval(successes, n, confidence)
    return {
        "n": n,
        "successes": successes,
        "rate": rate,
        "wilson_ci_low": lo,
        "wilson_ci_high": hi,
        "confidence": confidence,
    }


def stack_metric_rows(
    per_dialogue: List[Dict[str, Any]],
    metric_keys: List[str],
    confidence: float = 0.95,
) -> Dict[str, Dict[str, Any]]:
    """Build aggregate summaries for a set of numeric keys (skips None / missing)."""
    out: Dict[str, Dict[str, Any]] = {}
    for key in metric_keys:
        vals: List[float] = []
        for r in per_dialogue:
            if key not in r:
                continue
            v = r[key]
            if v is None:
                continue
            vals.append(float(v))
        out[key] = mean_std_ci(vals, confidence) if vals else {
            "n": 0,
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
            "confidence": confidence,
        }
    return out
