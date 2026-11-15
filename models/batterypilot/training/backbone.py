"""Day-ahead value-curve backbone (numpy; v1.2).

The rolling day-ahead LP (strategies.rolling_da_lp) plans with DA prices as the forecast of RT and follows the plan as
a fixed schedule.  The same plan can be written as a value curve: the marginal value of one stored MWh at every state
of charge, from a backward dynamic programme over the known DA horizon.  Executed by the offer rule against the RT
price, that curve does what the LP does when RT equals DA, and trades the difference when it does not.

At decision hour t the curve is lambda_{t+1}(s): the derivative of the value-to-go from the end of hour t to the end
of the known DA horizon (end of today; end of tomorrow once tomorrow's DA prices clear at 14:00 local), with DA
prices as the forecast and no value left at the horizon end (as the LP).  Re-planned at the same hours as the LP.

The value function is concave and piecewise linear in s; it is carried on a grid of N+1 SOC points.  One backward
step: the optimal end-of-hour SOC for each start SOC is the offer rule's target against lambda_{t+1} at the forecast
price, the value is reward + W_{t+1}(target), and lambda_t is its slope.
"""
from __future__ import annotations

import numpy as np

PUBLISH_HOUR = 14


def _cross(lam, y, grid):
    """SOC up to which the decreasing curve lam (on grid) is >= y (piecewise linear)."""
    a, b = lam[:-1], lam[1:]
    frac = np.clip((a - y) / np.maximum(a - b, 1e-12), 0.0, 1.0)
    frac = np.where(a - b <= 1e-12, (a >= y).astype(float), frac)
    return float((frac * np.diff(grid)).sum())


def backward_pass(price, c_lin, E, P, eta_c, eta_d, N=200, terminal=0.0):
    """Value curves for a horizon of forecast prices.  Returns lam (n+1, N+1): lam[k] = slope of the value-to-go
    from step k (in $ per stored MWh) on the SOC grid; lam[n] = terminal."""
    n = len(price)
    grid = np.linspace(0.0, 1.0, N + 1)
    lam = np.empty((n + 1, N + 1))
    lam[n] = terminal
    W = terminal * grid * E
    up = P * eta_c / E; dn = P / (eta_d * E)
    for k in range(n - 1, -1, -1):
        p = price[k]; c = c_lin[k]
        y_c = p / eta_c; y_d = eta_d * (p - c)
        nxt = lam[k + 1]
        s_lo = _cross(nxt, y_c, grid); s_hi = _cross(nxt, y_d, grid)
        tgt = np.clip(grid, s_lo, max(s_hi, s_lo))
        tgt = np.clip(tgt, grid - dn, grid + up)
        tgt = np.clip(tgt, 0.0, 1.0)
        d = tgt - grid
        r = np.where(d > 0, -p * d * E / eta_c, (p - c) * (-d) * E * eta_d)
        Wn = r + np.interp(tgt, grid, W)
        g = np.gradient(Wn, grid) / E
        # keep it a decreasing curve (concavity holds up to grid error)
        lam[k] = np.minimum.accumulate(g)
        W = Wn
    return lam, grid


def value_curves(da, local_hour, day_id, c_lin, duration_h, P=1.0, eta_rt=0.85, knots=None, N=200):
    """Backbone value curves at every hour of a path: (n, K) $ per stored MWh at the SOC knots, the curve the offer
    rule uses for hour t (slope of the value-to-go from the end of hour t).  Causal: at hour t it uses only the DA
    prices known at t (same publication rule as the rolling LP)."""
    if knots is None:
        knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    da = np.asarray(da, float); n = len(da)
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,)).astype(float)
    E = P * duration_h; eta = np.sqrt(eta_rt)
    days = np.asarray(day_id)
    uniq, first = np.unique(days, return_index=True)
    order = np.argsort(first); uniq = uniq[order]; first = first[order]
    ends = np.r_[first[1:], n]
    day_pos = {d: j for j, d in enumerate(uniq)}
    replan = [0] + [t for t in range(1, n) if local_hour[t] == PUBLISH_HOUR]
    out = np.zeros((n, len(knots)))
    for j, t0 in enumerate(replan):
        dj = day_pos[days[t0]]
        known_end = ends[min(dj + 1, len(ends) - 1)] if (t0 > 0 or local_hour[t0] >= PUBLISH_HOUR) else ends[dj]
        t1 = replan[j + 1] if j + 1 < len(replan) else n
        p = da[t0:known_end]
        if np.isnan(p).any():
            p = np.where(np.isnan(p), np.nanmedian(p) if np.isfinite(p).any() else 0.0, p)
        lam, grid = backward_pass(p, cl[t0:known_end], E, P, eta, eta, N=N)
        m = t1 - t0
        # hour t0+i uses lam[i+1] (value-to-go after the hour)
        for i in range(min(m, len(p))):
            out[t0 + i] = np.interp(knots, grid, lam[i + 1])
    return out
