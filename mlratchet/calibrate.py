"""Honest error bars: widen-only calibration on the dev set, and coverage on the exam.

The rule used across the collection: fit ONE factor per output on the dev set only, so that the
network's 68% bars cover 68% of dev residuals — but never shrink them (factor ≥ 1). Then report
the exam coverage you actually get, whatever it is.
"""
from __future__ import annotations

import numpy as np

ONE_SIGMA = 0.6826894921370859  # P(|z| < 1) for a Gaussian


def coverage(residual, sigma, factor: float = 1.0) -> float:
    """Fraction of ``|residual| <= factor * sigma``."""
    r = np.abs(np.asarray(residual, float))
    s = np.asarray(sigma, float) * factor
    ok = np.isfinite(r) & np.isfinite(s)
    return float(np.mean(r[ok] <= s[ok])) if ok.any() else float("nan")


def widen_factor(residual, sigma, target: float = ONE_SIGMA) -> float:
    """Smallest factor ``f >= 1`` with ``coverage(residual, sigma, f) >= target`` (widen-only).

    Fit this on dev residuals only. It is the ``target`` quantile of ``|residual| / sigma``,
    floored at 1 so bars are never narrowed.
    """
    z = np.abs(np.asarray(residual, float)) / np.asarray(sigma, float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 1.0
    return float(max(1.0, np.quantile(z, target)))


def wrap_degrees(a, b):
    """Signed angular difference ``a - b`` in degrees, wrapped to (-180, 180]."""
    d = (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0
    return np.where(d == -180.0, 180.0, d)


def von_mises_halfwidth(kappa) -> np.ndarray:
    """Approximate 68% half-width in degrees for a von Mises concentration ``kappa`` (57.3/√κ)."""
    return 57.29577951308232 / np.sqrt(np.maximum(np.asarray(kappa, float), 1e-6))
