"""Small classical models used for the WaterData exam baselines (independent of agent A's watersim).

gr4j_snow   GR4J (Perrin, Michel & Andreassian 2003) with a single-layer degree-day snow store in front of it.
            Params: x1 (mm), x2 (mm/d), x3 (mm), x4 (d), tt (degC), ddf (mm/degC/d). Inputs daily P, T_mean, PET.
degree_day  Point degree-day snow model for SNOTEL: SWE from P and T with rain/snow partition, melt, refreeze-free.
            Params: tt (degC), ddf (mm/degC/d), pmult (snowfall catch multiplier), tmelt (degC).
calibrate   scipy differential_evolution, single thread, seeded.
GR4J verified against agent A's airGR-checked reference outputs (see DATA.md 'checks').
"""
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution

GR4J_BOUNDS = [(10.0, 2500.0), (-10.0, 6.0), (5.0, 600.0), (0.5, 6.0), (-2.0, 2.5), (0.5, 8.0)]
DD_BOUNDS = [(-2.0, 3.0), (0.5, 10.0), (0.6, 1.8), (-2.0, 3.0)]


@njit(cache=True)
def _uh_ordinates(x4):
    n1 = int(np.ceil(x4)); n2 = int(np.ceil(2 * x4))
    sh1 = np.zeros(n1 + 1); sh2 = np.zeros(n2 + 1)
    for t in range(n1 + 1):
        sh1[t] = 1.0 if t >= x4 else (t / x4) ** 2.5
    for t in range(n2 + 1):
        if t <= x4:
            sh2[t] = 0.5 * (t / x4) ** 2.5
        elif t < 2 * x4:
            sh2[t] = 1 - 0.5 * (2 - t / x4) ** 2.5
        else:
            sh2[t] = 1.0
    return np.diff(sh1), np.diff(sh2)


@njit(cache=True)
def gr4j_core(P, E, x1, x2, x3, x4, s0=0.3, r0=0.5):
    n = P.size
    Q = np.zeros(n)
    uh1, uh2 = _uh_ordinates(x4)
    b1 = np.zeros(uh1.size); b2 = np.zeros(uh2.size)
    S = s0 * x1; R = r0 * x3
    for t in range(n):
        p = P[t]; e = E[t]
        if p >= e:
            pn = p - e; en = 0.0
            tw = np.tanh(min(pn / x1, 13.0))
            ps = x1 * (1 - (S / x1) ** 2) * tw / (1 + S / x1 * tw)
            es = 0.0
        else:
            pn = 0.0; en = e - p
            tw = np.tanh(min(en / x1, 13.0))
            es = S * (2 - S / x1) * tw / (1 + (1 - S / x1) * tw)
            ps = 0.0
        S = S - es + ps
        perc = S * (1 - (1 + (4.0 / 9.0 * S / x1) ** 4) ** -0.25)
        S -= perc
        pr = perc + (pn - ps)
        for k in range(uh1.size - 1):
            b1[k] = b1[k + 1] + uh1[k] * pr
        b1[uh1.size - 1] = uh1[uh1.size - 1] * pr
        for k in range(uh2.size - 1):
            b2[k] = b2[k + 1] + uh2[k] * pr
        b2[uh2.size - 1] = uh2[uh2.size - 1] * pr
        q9 = 0.9 * b1[0]; q1 = 0.1 * b2[0]
        F = x2 * (R / x3) ** 3.5
        R = max(0.0, R + q9 + F)
        qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
        R -= qr
        qd = max(0.0, q1 + F)
        Q[t] = qr + qd
    return Q


@njit(cache=True)
def snow_dd(P, T, tt, ddf, pmult=1.0, tmelt=0.0, swe0=0.0):
    """Returns (liquid water out mm/d, SWE end of day mm). Snow fraction linear over tt +- 1 degC."""
    n = P.size
    out = np.zeros(n); swe = np.zeros(n)
    s = swe0
    for t in range(n):
        fs = min(1.0, max(0.0, (tt + 1.0 - T[t]) / 2.0))
        snow = P[t] * fs * pmult
        rain = P[t] * (1 - fs)
        s += snow
        m = min(s, max(0.0, ddf * (T[t] - tmelt)))
        s -= m
        out[t] = rain + m
        swe[t] = s
    return out, swe


def gr4j_snow(params, P, T, E):
    x1, x2, x3, x4, tt, ddf = params
    liq, swe = snow_dd(np.asarray(P, float), np.asarray(T, float), tt, ddf, 1.0, tt)
    return gr4j_core(liq, np.asarray(E, float), x1, x2, x3, x4)


def _nse_loss(sim, obs, m):
    o = obs[m]; s = sim[m]
    return np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def calibrate_gr4j(P, T, E, Qobs, spin, seed=0, maxiter=60, popsize=12):
    """Minimise 1-NSE on days >= spin with finite obs. Returns params, NSE."""
    P = np.nan_to_num(np.asarray(P, float)); T = np.nan_to_num(np.asarray(T, float)); E = np.nan_to_num(np.asarray(E, float))
    obs = np.asarray(Qobs, float)
    m = np.isfinite(obs); m[:spin] = False
    f = lambda th: _nse_loss(gr4j_snow(th, P, T, E), obs, m)
    r = differential_evolution(f, GR4J_BOUNDS, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6, polish=False,
                               init='sobol', updating='immediate')
    return r.x, 1 - r.fun


def degree_day(params, P, T):
    tt, ddf, pmult, tmelt = params
    _, swe = snow_dd(np.asarray(P, float), np.asarray(T, float), tt, ddf, pmult, tmelt)
    return swe


def calibrate_dd(P, T, SWE, seed=0, maxiter=50, popsize=12):
    P = np.nan_to_num(np.asarray(P, float)); T = np.nan_to_num(np.asarray(T, float))
    obs = np.asarray(SWE, float); m = np.isfinite(obs)
    f = lambda th: np.mean((degree_day(th, P, T)[m] - obs[m]) ** 2)
    r = differential_evolution(f, DD_BOUNDS, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6, polish=False,
                               init='sobol', updating='immediate')
    return r.x, float(np.sqrt(r.fun))
