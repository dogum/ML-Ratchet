"""Honest error bars: widen-only calibration, finite-sample split conformal and Mondrian strata.

The baseline rule used across the collection: fit ONE factor per output on the dev set only, so
that the network's 68% bars cover 68% of dev residuals — never shrinking them (factor >= 1).
``MondrianConformal`` extends that to per-stratum (Mondrian) split conformal calibration with
optional empirical-Bayes shrinkage toward the global pooled quantile for small dev strata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

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


def conformal_quantile(
    scores,
    alpha: float = 1.0 - ONE_SIGMA,
    floor: float = 0.0,
) -> float:
    """Exact finite-sample split-conformal quantile of nonconformity ``scores``.

    Uses order statistic ``k = ceil((n + 1) * (1 - alpha))`` (Vovk et al. 2005 / Romano et al.
    2019). When ``k > n`` (too few scores for the requested coverage, including ``n = 0``) the
    k-th score is the extra ``+inf`` of the conformal construction, so the result is ``inf``:
    no finite interval carries the guarantee. Otherwise marginal coverage is ``>= 1 - alpha`` on
    exchangeable draws from the same distribution.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    k = max(1, k)
    q = float(np.partition(s, k - 1)[k - 1])
    return float(max(floor, q))


@dataclass
class MondrianConformal:
    """Per-stratum (Mondrian) split conformal calibrator with small-stratum shrinkage.

    Operates either on studentized residuals ``|r| / sigma`` (multiplicative widening, default,
    preserving heteroscedastic ``sigma`` from the learner) or additive residuals ``|r|``
    (``additive=True``).

    When ``shrink_k > 0``, each stratum's finite-sample quantile ``q_g`` is shrunk toward the
    pooled global quantile ``q_global`` via ``w_g = n_g / (n_g + shrink_k)``. Setting
    ``shrink_k = 0.0`` recovers raw unshrunk Mondrian conformal (falling back to ``q_global`` only
    when ``n_g < min_n``).
    """

    target: float = ONE_SIGMA
    widen_only: bool = True
    shrink_k: float = 10.0
    min_n: int = 5
    additive: bool = False
    global_factor: float = 1.0
    stratum_factors: dict[str, float] = field(default_factory=dict)
    stratum_raw_factors: dict[str, float] = field(default_factory=dict)
    stratum_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        residual,
        sigma,
        strata,
        *,
        target: float = ONE_SIGMA,
        widen_only: bool = True,
        shrink_k: float = 10.0,
        min_n: int = 5,
        additive: bool = False,
    ) -> "MondrianConformal":
        """Fit pooled and per-stratum conformal quantiles strictly on dev residuals."""
        r = np.abs(np.asarray(residual, dtype=np.float64))
        st = np.asarray(strata, dtype=str)
        if additive:
            scores = r
            floor = 0.0
            ok = np.isfinite(scores)
        else:
            s = np.asarray(sigma, dtype=np.float64)
            ok = np.isfinite(r) & np.isfinite(s) & (s > 0)
            scores = np.where(ok, r / np.where(ok, s, 1.0), np.nan)
            floor = 1.0 if widen_only else 0.0

        alpha = 1.0 - float(target)
        q_global = conformal_quantile(scores[ok], alpha=alpha, floor=floor)

        s_factors: dict[str, float] = {}
        s_raw: dict[str, float] = {}
        s_counts: dict[str, int] = {}

        for g in sorted(set(st[ok].tolist())):
            m = ok & (st == g)
            ng = int(np.sum(m))
            s_counts[g] = ng
            if ng < min_n:
                s_raw[g] = q_global
                s_factors[g] = q_global
            else:
                q_raw = conformal_quantile(scores[m], alpha=alpha, floor=floor)
                s_raw[g] = q_raw
                if shrink_k > 0:
                    w = ng / (ng + shrink_k)
                    q_shrunk = w * q_raw + (1.0 - w) * q_global     # inf if either is inf (an unbounded interval)
                else:
                    q_shrunk = q_raw                               # avoid inf * 0 = NaN
                s_factors[g] = float(max(floor, q_shrunk))

        return cls(
            target=float(target),
            widen_only=bool(widen_only),
            shrink_k=float(shrink_k),
            min_n=int(min_n),
            additive=bool(additive),
            global_factor=float(q_global),
            stratum_factors=s_factors,
            stratum_raw_factors=s_raw,
            stratum_counts=s_counts,
        )

    def factor_for(self, strata) -> np.ndarray:
        """Return per-row factor (or additive half-width) for each entry of ``strata``."""
        st = np.asarray(strata, dtype=str)
        return np.array(
            [self.stratum_factors.get(g, self.global_factor) for g in st.tolist()],
            dtype=np.float64,
        )

    def halfwidth(self, sigma, strata) -> np.ndarray:
        """Calibrated prediction half-width per row."""
        f = self.factor_for(strata)
        if self.additive:
            return f
        return np.asarray(sigma, dtype=np.float64) * f

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "widen_only": self.widen_only,
            "shrink_k": self.shrink_k,
            "min_n": self.min_n,
            "additive": self.additive,
            "global_factor": self.global_factor,
            "stratum_factors": dict(self.stratum_factors),
            "stratum_raw_factors": dict(self.stratum_raw_factors),
            "stratum_counts": dict(self.stratum_counts),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MondrianConformal":
        return cls(
            target=float(d["target"]),
            widen_only=bool(d["widen_only"]),
            shrink_k=float(d["shrink_k"]),
            min_n=int(d["min_n"]),
            additive=bool(d.get("additive", False)),
            global_factor=float(d["global_factor"]),
            stratum_factors={str(k): float(v) for k, v in d["stratum_factors"].items()},
            stratum_raw_factors={str(k): float(v) for k, v in d.get("stratum_raw_factors", {}).items()},
            stratum_counts={str(k): int(v) for k, v in d.get("stratum_counts", {}).items()},
        )


def stratum_coverage(
    residual,
    halfwidth,
    strata,
    target: float = ONE_SIGMA,
) -> dict:
    """Empirical coverage, worst-stratum gap and mean half-width across strata."""
    r = np.abs(np.asarray(residual, dtype=np.float64))
    hw = np.asarray(halfwidth, dtype=np.float64)
    st = np.asarray(strata, dtype=str)
    # +inf half-widths are valid (an undersized calibration set gives an unbounded interval): they cover every
    # finite residual and make the mean width infinite. Only NaN or negative widths are dropped.
    ok = np.isfinite(r) & ~np.isnan(hw) & (hw >= 0)

    pooled_cov = float(np.mean(r[ok] <= hw[ok])) if ok.any() else float("nan")
    mean_hw = float(np.mean(hw[ok])) if ok.any() else float("nan")

    by_stratum: dict[str, dict] = {}
    worst_name = None
    worst_under = -1.0
    max_abs_gap = 0.0

    for g in sorted(set(st[ok].tolist())):
        m = ok & (st == g)
        cov_g = float(np.mean(r[m] <= hw[m]))
        gap_g = cov_g - target
        by_stratum[g] = {
            "n": int(np.sum(m)),
            "coverage": round(cov_g, 4),
            "gap": round(gap_g, 4),
            "mean_halfwidth": round(float(np.mean(hw[m])), 6),
        }
        max_abs_gap = max(max_abs_gap, abs(gap_g))
        under = target - cov_g
        if under > worst_under:
            worst_under = under
            worst_name = g

    return {
        "target": float(target),
        "n": int(np.sum(ok)),
        "pooled_coverage": round(pooled_cov, 4),
        "pooled_gap": round(pooled_cov - target, 4),
        "mean_halfwidth": round(mean_hw, 6),
        "worst_under_stratum": worst_name,
        "worst_under_gap": round(float(max(0.0, worst_under)), 4) if worst_name is not None else float("nan"),
        "max_abs_stratum_gap": round(float(max_abs_gap), 4),
        "strata": by_stratum,
    }


def wrap_degrees(a, b):
    """Signed angular difference ``a - b`` in degrees, wrapped to (-180, 180]."""
    d = (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0
    return np.where(d == -180.0, 180.0, d)


def von_mises_halfwidth(kappa) -> np.ndarray:
    """Approximate 68% half-width in degrees for a von Mises concentration ``kappa`` (57.3/√κ)."""
    return 57.29577951308232 / np.sqrt(np.maximum(np.asarray(kappa, float), 1e-6))
