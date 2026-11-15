"""Real-time scarcity that the day-ahead market cannot see (GridSim physics 1.2, opt-in).

GridSim's dispatch prices one market. Real ERCOT has two: the day-ahead market (DA) clears the day before, and the
real-time market (RT) settles what actually happened. Most real price spikes are not in the DA price: on the training
years 2018, 2020 and 2022 (HB_HUBAVG), the DA price covers only 24-30 % of the RT value of the hours above $300/MWh.
What the DA cannot see are forced outages after it clears, net-load forecast misses and short ramp shortfalls.

This module draws that part as a marked point process on the hourly RT-minus-DA spread:

* an event starts in hour t with probability  p_t = r0 * H[local hour] * M[month] * g(z_t)  (not while one runs),
  with g(z) = exp(beta z) / mean(exp(beta z)) and z = (actual - forecast demand) / forecast / err_sd, clipped to
  +-4: misses of the day-ahead demand forecast make RT scarcity more likely;
* it lasts D hours, P(D = d) = q (1 - q)^(d - 1);
* each of its hours adds u + X $/MWh to the RT price, X ~ generalised Pareto(xi, sigma) (the excess over the
  threshold u), adjacent hours of one event tied by a Gaussian copula with correlation rho.

`expected(...)` gives the mean adder for each hour as a risk-neutral DA would price it: the same hour and month
structure without z (the DA does not know the miss), so DA = E[RT] still holds on average.

Opt-in: with physics 1.0 or 1.1 (the default, $GRIDSIM_PHYSICS) `rt_scarcity` returns zeros and `expected` returns
zeros, so a caller that adds them reproduces 1.1 bit for bit. Nothing else in GridSim calls this module, and physics
1.2 changes no other value (outage defaults, dispatch and frequency calibration are 1.1's).

Fitted on ERCOT only (shared/grid/checks/fit_rt_scarcity.py: real HB_HUBAVG RT and DA prices and EIA-930 demand
forecast errors of the training-input years 2018, 2020, 2022; never dev or exam prices). There is no CAISO fit:
CAISO prices exist only from Aug 2023, so region 'CISO' (or any region other than 'ERCO') returns zeros unless the
caller passes its own parameters. Checks: shared/grid/checks/check_rt_scarcity.py -> rt_scarcity.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from .fleet import physics_version

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, "data", "rt_scarcity_v1_2.json")
VERSIONS = ("1.2",)          # physics versions in which the process is on
TZ = {"ERCO": "America/Chicago", "CISO": "America/Los_Angeles"}
KEYS = ("u", "r0", "H", "M", "beta", "err_sd", "q", "xi", "sigma", "rho", "x_max")


def enabled(version=None):
    """True if the RT scarcity process is on for this physics version ($GRIDSIM_PHYSICS by default)."""
    return (version or physics_version()) in VERSIONS


def params(region="ERCO"):
    """The fitted parameters of a region (ERCO only), or None."""
    if not os.path.exists(PARAMS_FILE):
        return None
    p = json.load(open(PARAMS_FILE)).get("regions", {}).get(region)
    return None if p is None else {k: p[k] for k in KEYS}


def _local(times, region):
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        t = t.tz_localize("UTC")
    loc = t.tz_convert(TZ.get(region, "UTC"))
    return np.asarray(loc.hour), np.asarray(loc.month)


def _setup(times, region, p, version, scale):
    if not enabled(version):
        return None, None
    p = p if p is not None else params(region)
    if p is None:
        return None, None
    hour, month = _local(times, region)
    base = float(p["r0"]) * scale * np.asarray(p["H"], float)[hour] * np.asarray(p["M"], float)[month - 1]
    return p, np.clip(base, 0.0, 0.5)


def _mean_excess(p):
    """E[min(X, x_max)] for X ~ GPD(xi, sigma) (numerical; finite for any xi)."""
    x = np.linspace(0.0, float(p["x_max"]), 20001)
    sf = stats.genpareto.sf(x, float(p["xi"]), scale=float(p["sigma"]))
    return float(np.sum((sf[1:] + sf[:-1]) * 0.5 * np.diff(x)))


def rt_scarcity(times, region="ERCO", seed=0, demand_err=None, params_=None, version=None, scale=1.0,
                return_events=False):
    """RT-minus-DA scarcity adder [$/MWh] for each hour of `times` (UTC or tz-aware hourly index).

    demand_err: (actual - day-ahead forecast) / forecast demand per hour (fraction), or None (z = 0).
    params_: a parameter dict (keys KEYS) instead of the fitted region parameters.
    scale: multiplies the start rate (scenario randomisation).
    version: physics version (default $GRIDSIM_PHYSICS); the adder is zero unless it is 1.2.
    The process has its own random stream (seed), so it never changes a caller's other draws.
    Returns the adder (T,), and with return_events=True also a list of (start index, duration).
    """
    T = len(times)
    out = np.zeros(T)
    p, base = _setup(times, region, params_, version, scale)
    if p is None:
        return (out, []) if return_events else out
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 120])
    if demand_err is not None and float(p["beta"]) != 0.0:
        z = np.clip(np.nan_to_num(np.asarray(demand_err, float) / float(p["err_sd"])), -4, 4)
        g = np.exp(float(p["beta"]) * z)
        g = g / g.mean()
    else:
        g = np.ones(T)
    pr = np.clip(base * g, 0.0, 0.9)
    u_start = rng.random(T)
    q, xi, sg, rho, xmax, u = (float(p[k]) for k in ("q", "xi", "sigma", "rho", "x_max", "u"))
    events = []
    t = 0
    while t < T:
        if u_start[t] < pr[t]:
            d = int(rng.geometric(q))
            n = min(d, T - t)
            e = rng.standard_normal(n)
            zc = np.empty(n)
            zc[0] = e[0]
            for k in range(1, n):
                zc[k] = rho * zc[k - 1] + np.sqrt(1.0 - rho * rho) * e[k]
            x = stats.genpareto.ppf(stats.norm.cdf(zc), xi, scale=sg)
            out[t:t + n] = u + np.minimum(x, xmax)
            events.append((t, n))
            t += n
        else:
            t += 1
    return (out, events) if return_events else out


def expected(times, region="ERCO", params_=None, version=None, scale=1.0):
    """Mean adder [$/MWh] per hour without knowledge of the forecast miss (what a risk-neutral DA prices in):
    P(an event is running in hour t) x E[u + min(X, x_max)].  Zero unless physics 1.2."""
    T = len(times)
    p, base = _setup(times, region, params_, version, scale)
    if p is None:
        return np.zeros(T)
    q = float(p["q"])
    K = int(np.ceil(np.log(1e-4) / np.log(1.0 - q))) if q < 1 else 1
    surv = (1.0 - q) ** np.arange(K)                     # P(D > k)
    occ = np.zeros(T)
    for k in range(K):                                   # starts k hours ago still running
        occ[k:] += base[:T - k] * surv[k]
    occ = occ / (1.0 + occ)                              # no new start while an event runs (renewal correction)
    return occ * (float(p["u"]) + _mean_excess(p))
