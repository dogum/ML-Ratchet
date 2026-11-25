"""Ensemble Kalman filter (serial square-root EnSRF, Whitaker & Hamill 2002) around WaterSim's HBV + degree-day snow.

Model step: the vendored WaterSim kernels themselves (watersim.snow.dd_step per elevation band, watersim.models.hbv.step
with its triangular hillslope unit hydrograph), then WaterSim's 'uh' channel routing (gamma, shape 3, mean travel
L / CH_VEL, Hack length). Same band rules as watersim.engine.run_core (lapse rate, normalised precipitation gradient,
PET reduced by snow-covered area).

Filter state per member (one row of X):
    snow ice [B] | snow liquid [B] | SM, SUZ, SLZ | hillslope UH buffer [3 x L] | channel buffer [Lc] | log params [K]
Observation: y_t = log(Qobs_t + c), predicted h_t = log(Q_t + c) from each member's outflow of day t (state
augmentation: the update uses the ensemble covariance between the states at the end of day t and h_t).
Spread: daily multiplicative precipitation noise, additive temperature noise, multiplicative noise on the three HBV
stores, relaxation of the posterior spread towards the prior spread (RTPS) at each analysis, random walk on augmented log parameters.

Causality: the analysis at day t uses qobs[t] only; the forecast for day t + lead starts from the analysis at day t
(using qobs[: t + 1]) and runs `lead` model days with the (known) weather of days t+1 .. t+lead.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_up = HERE
while _up != os.path.dirname(_up) and not os.path.isdir(os.path.join(_up, 'harness', 'vendor')):
    _up = os.path.dirname(_up)
_VENDOR = os.path.join(_up, 'harness', 'vendor')   # the bake-off harness's vendored WaterSim (hashed with it)
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from numba import njit  # noqa: E402
from watersim.snow import dd_step  # noqa: E402
from watersim.models.hbv import step as hbv_step  # noqa: E402

# augmented-parameter codes (index into the member parameter vector mp, or -1 = precipitation multiplier)
# mp order = HBV PARAMS: FC LP BETA PERC UZL K0 K1 K2 MAXBAS
AUG_CODES = {'pmult': -1, 'FC': 0, 'LP': 1, 'BETA': 2, 'PERC': 3, 'UZL': 4, 'K0': 5, 'K1': 6, 'K2': 7}


@njit(cache=False)
def _member_day(t, m, P, Tm, pet, pmult, eP, eT, bfrac, bdz, pm_b, sp, mp, s, snow, uhord, uhbuf, chord, chbuf,
                f, petred):
    """Advance member m by one day; returns its outflow Q (mm/day)."""
    B = bfrac.shape[0]
    Pd = P[t] * pmult * eP
    Td = Tm[t] + eT
    tot_out = 0.0
    tot_sca = 0.0
    for b in range(B):
        fb = bfrac[b]
        Pb = Pd * pm_b[m, b]
        Tb = Td + sp[m, 6] * bdz[b]
        ice, liq, rain, sn, melt, refr, out = dd_step(snow[m, b], snow[m, B + b], Pb, Tb, sp[m, 0], sp[m, 1],
                                                       sp[m, 2], sp[m, 3], sp[m, 4], sp[m, 5], 1.0)
        snow[m, b] = ice
        snow[m, B + b] = liq
        sca = (ice + liq) / 10.0
        if sca > 1.0:
            sca = 1.0
        tot_out += fb * out
        tot_sca += fb * sca
    pe = pet[t]
    if petred:
        pe = pe * (1.0 - tot_sca)
    hbv_step(s[m], mp[m], tot_out, pe, 1.0, uhord[m], uhbuf[m], f)
    qm = f[1]
    Lc = chord.shape[0]
    for k in range(Lc):
        chbuf[m, k] += qm * chord[k]
    q = chbuf[m, 0]
    for k in range(Lc - 1):
        chbuf[m, k] = chbuf[m, k + 1]
    chbuf[m, Lc - 1] = 0.0
    return q


@njit(cache=False)
def _pack(m, X, snow, s, uhbuf, chbuf, lp, eps_s, eps_b):
    """States enter the update as log(x + eps) (Gaussian anamorphosis: multiplicative, positive updates)."""
    j = 0
    for k in range(snow.shape[1]):
        X[m, j] = math.log(max(snow[m, k], 0.0) + eps_s); j += 1
    for k in range(3):
        X[m, j] = math.log(max(s[m, k], 0.0) + eps_s); j += 1
    for c in range(3):
        for k in range(uhbuf.shape[2]):
            X[m, j] = math.log(max(uhbuf[m, c, k], 0.0) + eps_b); j += 1
    for k in range(chbuf.shape[1]):
        X[m, j] = math.log(max(chbuf[m, k], 0.0) + eps_b); j += 1
    for k in range(lp.shape[1]):
        X[m, j] = lp[m, k]; j += 1


@njit(cache=False)
def _unpack(m, X, snow, s, uhbuf, chbuf, lp, mp, sp, upd_snow, upd_buf, eps_s, eps_b):
    j = 0
    for k in range(snow.shape[1]):
        if upd_snow:
            v = math.exp(X[m, j]) - eps_s
            snow[m, k] = v if v > 0.0 else 0.0
        j += 1
    for k in range(3):
        v = math.exp(X[m, j]) - eps_s
        if v < 0.0:
            v = 0.0
        if k == 0 and v > mp[m, 0]:
            v = mp[m, 0]
        s[m, k] = v
        j += 1
    for c in range(3):
        for k in range(uhbuf.shape[2]):
            if upd_buf:
                v = math.exp(X[m, j]) - eps_b
                uhbuf[m, c, k] = v if v > 0.0 else 0.0
            j += 1
    for k in range(chbuf.shape[1]):
        if upd_buf:
            v = math.exp(X[m, j]) - eps_b
            chbuf[m, k] = v if v > 0.0 else 0.0
        j += 1
    for k in range(lp.shape[1]):
        lp[m, k] = X[m, j]
        j += 1


@njit(cache=False)
def run_filter(P, Tm, pet, qobs, t_assim0, lead, out_from, bfrac, bdz, pm_b, sp, mp0, s0, snow0, uhord, chord,
               eP, eT, eS, eW, aug_idx, lp0, lp_lo, lp_hi, rw_sd, pmult0,
               c_obs, r_obs, infl, upd_snow, upd_buf, petred, eps_s, inn_clip):
    """Run the filter over all T days and return (pred (T, N) forecast outflow for day t made at day t - lead (NaN
    where t - lead < out_from - lead), analysis info (T, 3): innovation, var_h, n_assim flag).
    P, Tm, pet, qobs: (T,). eP, eT: (T, N) multiplicative / additive forcing noise; eS: (T, N) store noise factor.
    aug_idx (K,): codes of augmented log parameters (-1 = precipitation multiplier, else index into mp).
    eW: (T, N, K) standard normals for the parameter random walk (sd rw_sd per parameter)."""
    T = P.shape[0]
    N = mp0.shape[0]
    B = bfrac.shape[0]
    L = uhord.shape[2]
    Lc = chord.shape[0]
    K = aug_idx.shape[0]
    mp = mp0.copy()
    s = s0.copy()
    snow = snow0.copy()
    uhbuf = np.zeros((N, 3, L))
    chbuf = np.zeros((N, Lc))
    lp = lp0.copy()
    pmult = pmult0.copy()
    D = 2 * B + 3 + 3 * L + Lc + K
    X = np.zeros((N, D))
    h = np.zeros(N)
    f = np.zeros(11)
    pred = np.full((T, N), np.nan)
    info = np.full((T, 3), np.nan)
    # forecast scratch
    fs = np.zeros((N, 3))
    fsnow = np.zeros((N, 2 * B))
    fuh = np.zeros((N, 3, L))
    fch = np.zeros((N, Lc))
    for t in range(T):
        # apply augmented parameters to the members
        for m in range(N):
            for k in range(K):
                v = math.exp(lp[m, k])
                if aug_idx[k] < 0:
                    pmult[m] = v
                else:
                    mp[m, aug_idx[k]] = v
        # propagate
        for m in range(N):
            q = _member_day(t, m, P, Tm, pet, pmult[m], eP[t, m], eT[t, m], bfrac, bdz, pm_b, sp, mp, s, snow,
                            uhord, uhbuf, chord, chbuf, f, petred)
            for k in range(3):
                s[m, k] *= eS[t, m]
            h[m] = math.log(max(q, 0.0) + c_obs)
        # analysis
        y = qobs[t]
        if t >= t_assim0 and not math.isnan(y):
            yo = math.log(max(y, 0.0) + c_obs)
            for m in range(N):
                _pack(m, X, snow, s, uhbuf, chbuf, lp, eps_s, c_obs)
            hb = 0.0
            for m in range(N):
                hb += h[m]
            hb /= N
            xb = np.zeros(D)
            for m in range(N):
                for j in range(D):
                    xb[j] += X[m, j]
            for j in range(D):
                xb[j] /= N
            var_h = 0.0
            for m in range(N):
                a = h[m] - hb
                var_h += a * a
            var_h /= (N - 1)
            if var_h > 1e-12:
                cov = np.zeros(D)
                sdb = np.zeros(D)
                for m in range(N):
                    a = h[m] - hb
                    for j in range(D):
                        dx = X[m, j] - xb[j]
                        cov[j] += dx * a
                        sdb[j] += dx * dx
                for j in range(D):
                    cov[j] /= (N - 1)
                den = var_h + r_obs
                alpha = 1.0 / (1.0 + math.sqrt(r_obs / den))
                inn = yo - hb
                lim = inn_clip * math.sqrt(den)
                if inn > lim:
                    inn = lim
                elif inn < -lim:
                    inn = -lim
                for m in range(N):
                    a = h[m] - hb
                    for j in range(D):
                        X[m, j] = X[m, j] + cov[j] / den * (inn - alpha * a)
                # relaxation to prior spread (RTPS, Whitaker & Hamill 2012): sd_a <- sd_a + rtps (sd_b - sd_a)
                if infl > 0.0:
                    xa = np.zeros(D)
                    for m in range(N):
                        for j in range(D):
                            xa[j] += X[m, j]
                    for j in range(D):
                        xa[j] /= N
                    for j in range(D):
                        sda = 0.0
                        for m in range(N):
                            sda += (X[m, j] - xa[j]) ** 2
                        sda = math.sqrt(sda)
                        sb = math.sqrt(sdb[j])
                        if sda > 1e-12:
                            g = 1.0 + infl * (sb - sda) / sda
                            for m in range(N):
                                X[m, j] = xa[j] + g * (X[m, j] - xa[j])
                info[t, 0] = inn
                info[t, 1] = var_h
                info[t, 2] = 1.0
            for m in range(N):
                _unpack(m, X, snow, s, uhbuf, chbuf, lp, mp, sp, upd_snow, upd_buf, eps_s, c_obs)
            # parameter random walk + bounds
            for m in range(N):
                for k in range(K):
                    v = lp[m, k] + rw_sd[k] * eW[t, m, k]
                    if v < lp_lo[k]:
                        v = lp_lo[k]
                    if v > lp_hi[k]:
                        v = lp_hi[k]
                    lp[m, k] = v
        # forecast `lead` days from the analysis at day t
        tt = t + lead
        if tt < T and tt >= out_from:
            for m in range(N):
                for k in range(3):
                    fs[m, k] = s[m, k]
                for k in range(2 * B):
                    fsnow[m, k] = snow[m, k]
                for c in range(3):
                    for k in range(L):
                        fuh[m, c, k] = uhbuf[m, c, k]
                for k in range(Lc):
                    fch[m, k] = chbuf[m, k]
            for d in range(1, lead + 1):
                for m in range(N):
                    for k in range(K):
                        v = math.exp(lp[m, k])
                        if aug_idx[k] < 0:
                            pmult[m] = v
                        else:
                            mp[m, aug_idx[k]] = v
                    q = _member_day(t + d, m, P, Tm, pet, pmult[m], eP[t + d, m], eT[t + d, m], bfrac, bdz, pm_b,
                                    sp, mp, fs, fsnow, uhord, fuh, chord, fch, f, petred)
                    if d == lead:
                        pred[tt, m] = q
    return pred, info
