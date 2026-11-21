"""Observation and forcing error models, so networks never train on perfect data.

streamflow_obs: gauge error as a multiplicative AR(1) lognormal error (rating-curve scatter that persists for days;
    McMillan et al. 2012 report 10-20 % typical daily uncertainty, larger at low and very high flows) plus a
    rating-shift bias that changes at random 'rating revisions', larger relative error at low flow, rounding to the
    reporting resolution, and optional gaps.
forcing_error: precipitation multiplicative error (day-to-day lognormal + a persistent basin bias, gauge undercatch
    of snow), temperature bias + AR(1) noise. Apply it to the forcing given to the NETWORK, not to the simulator,
    to mimic forcing that differs from what the basin really got.
The defaults are moderate; prefer measuring the real error structure on dev data (e.g. simulated twin vs gauge).
"""
import numpy as np
import pandas as pd


def _ar1(n, rho, rng):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    s = np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + s * e[t]
    return x


def streamflow_obs(q, rng=None, cv=0.08, rho=0.9, revision_years=3.0, bias_sd=0.05, low_flow_cv=0.25,
                   resolution=0.001, gap_frac=0.0):
    """Noisy 'observed' streamflow from a simulated series q (mm/day or m3/s; same units back)."""
    rng = np.random.default_rng(rng)
    q = np.asarray(q, float)
    n = len(q)
    med = np.median(q[q > 0]) if (q > 0).any() else 1.0
    rel = cv + low_flow_cv * np.exp(-q / (0.2 * med))          # larger relative error at low flow
    eps = _ar1(n, rho, rng) * rel
    bias = np.empty(n)
    b = rng.normal(0, bias_sd)
    for t in range(n):
        if rng.random() < 1.0 / (revision_years * 365.25):
            b = rng.normal(0, bias_sd)
        bias[t] = b
    obs = q * np.exp(eps + bias - 0.5 * rel ** 2)
    if resolution:
        obs = np.round(obs / resolution) * resolution
    obs = np.maximum(obs, 0.0)
    if gap_frac > 0:
        m = np.zeros(n, bool)
        while m.mean() < gap_frac:
            s = int(rng.integers(n))
            m[s: s + int(rng.integers(5, 60))] = True
        obs[m] = np.nan
    return obs


def forcing_error(df, rng=None, p_bias_sd=0.08, p_day_cv=0.25, snow_undercatch=0.1, t_bias_sd=0.5, t_sd=0.7):
    """Perturbed copy of a forcing frame (what an imperfect forcing product would have shown)."""
    rng = np.random.default_rng(rng)
    out = df.copy()
    n = len(df)
    pb = np.exp(rng.normal(0, p_bias_sd))
    pd_ = np.exp(rng.normal(0, p_day_cv, n) - 0.5 * p_day_cv ** 2)
    snowy = (df['T_mean'].values < 0).astype(float)
    out['P'] = df['P'].values * pb * pd_ * (1 - snow_undercatch * snowy)
    dT = rng.normal(0, t_bias_sd) + _ar1(n, 0.6, rng) * t_sd
    for c in ('T_mean', 'T_min', 'T_max'):
        if c in out:
            out[c] = out[c].values + dT
    out.attrs = dict(getattr(df, 'attrs', {}))
    out.attrs['forcing_error'] = {'p_bias': float(pb), 'p_day_cv': p_day_cv, 'snow_undercatch': snow_undercatch,
                                  't_bias_sd': t_bias_sd, 't_sd': t_sd}
    return out
