"""Physics rung v2 (dev only): how wrong is the forcing precipitation?

ERA5-Land undercatches mountain / snowy precipitation (B's gotcha 2: 50 basins with runoff ratio > 1). In the bank,
the TRUE precipitation that drives the simulator is P_seen x m. v1 used log m ~ N(0, 0.08).
Estimator (applied identically to simulated and real basins, calibration period only): the multiplier m_B that puts
(m P, PET, Q) on the Fu curve with the global w = 2.6, i.e. Q / (m P) = 1 - Fu(PET / (m P)).
m_B mixes the forcing bias with each basin's own Budyko scatter, so it is compared between the v1 frozen simulated
set and the 52 real dev basins, per snow group (frac_snow < 0.15 / >= 0.15): the shift of the median of log m_B and the
extra spread needed are moved into the v2 prior of log m:
    log m ~ N(mu0 + mu_snow * frac_snow, sd)
Writes models/pbias_v2.json and logs/calib_pbias.json.
"""
import common  # noqa: F401
import json

import numpy as np
from scipy.optimize import brentq

from common import D
from net import load_bank
import evaluate as E


def fu(phi, w=2.6):
    return 1 + phi - (1 + phi ** w) ** (1 / w)


def m_budyko(P, PET, Q):
    g = lambda lm: (1 - fu(PET / (np.exp(lm) * P))) - Q / (np.exp(lm) * P)
    lo, hi = np.log(0.2), np.log(6.0)
    try:
        if g(lo) * g(hi) > 0:
            return np.nan
        return brentq(g, lo, hi)
    except Exception:
        return np.nan


def est(X, names, qmean):
    iP, iE, iS = names.index('c_logP'), names.index('c_logPET'), names.index('c_fsnow')
    P, PET, fs = np.exp(X[:, iP]), np.exp(X[:, iE]), X[:, iS]
    lm = np.array([m_budyko(p, e, q) for p, e, q in zip(P, PET, qmean)])
    return lm, fs


def main():
    Xs, labs, names = load_bank('v1', range(900, 904))
    lm_s, fs_s = est(Xs, names, labs['cal_q_mean'].values)
    Xr, meta, names_r = E.real_set('dev')
    lm_r, fs_r = est(Xr, names_r, meta['cal_q_mean'].values)
    out = {'what': 'log Budyko-required precipitation multiplier, calibration period', 'groups': {}}
    G = [('rain', lambda f: f < 0.15), ('snow', lambda f: f >= 0.15)]
    for g, sel in G:
        s = lm_s[sel(fs_s) & np.isfinite(lm_s)]
        r = lm_r[sel(fs_r) & np.isfinite(lm_r)]
        iq = lambda x: float(np.subtract(*np.percentile(x, [75, 25])) / 1.349)
        out['groups'][g] = {'n_sim': int(s.size), 'n_real': int(r.size), 'median_sim': float(np.median(s)),
                            'median_real': float(np.median(r)), 'sd_sim': iq(s), 'sd_real': iq(r),
                            'mean_fsnow_real': float(np.mean(fs_r[sel(fs_r)]))}
    gr, gs = out['groups']['rain'], out['groups']['snow']
    shift_r = gr['median_real'] - gr['median_sim']
    shift_s = gs['median_real'] - gs['median_sim']
    fr, fsn = gr['mean_fsnow_real'], gs['mean_fsnow_real']
    mu_snow = (shift_s - shift_r) / max(fsn - fr, 1e-3)
    mu0 = shift_r - mu_snow * fr
    # extra spread: pooled over groups, only ever widened
    ex = []
    for g in (gr, gs):
        ex.append(max(g['sd_real'] ** 2 - g['sd_sim'] ** 2, 0.0))
    sd = float(np.sqrt(0.08 ** 2 + np.mean(ex)))
    p = {'mu0': float(mu0), 'mu_snow': float(mu_snow), 'sd': sd, 'snow_extra': 0.1, 'tsd': 0.7}
    out['v2_prior'] = p
    out['sim_estimator_vs_true_rho'] = float(np.corrcoef(lm_s[np.isfinite(lm_s)],
                                                         labs['log_pmult'].values[np.isfinite(lm_s)])[0, 1])
    json.dump(p, open(f'{D}/models/pbias_v2.json', 'w'), indent=1)
    json.dump(out, open(f'{D}/logs/calib_pbias.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
