"""BasinDetective v1.1: what a basin's own gauge record says about the decade it has not seen.

    import basindetective as bd
    det = bd.load()                                   # weights/basindetective_v1_1.npz (v1.1, 27 Nov 2026)
    det10 = bd.load(bd.V1_0)                          # the v1.0 release, kept for its exam record
    res = bd.predict(det, cal, q_cal, tgt)

cal   DataFrame, daily, columns P (mm/d), T_mean (C), PET (mm/d): the calibration (gauged) period, ideally 9-10 years
q_cal array/Series on cal's index: observed streamflow in mm/day over the basin (NaN = missing)
tgt   DataFrame like cal for the period you want the basin's response for (forcing only, no flow)
"""
import os

import numpy as np

from .features import featurize, EPSQ
from .model import Detective, UNITS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, '..', 'weights', 'basindetective_v1_1.npz')   # v1.1: symmetric precipitation-change prior
V1_0 = os.path.join(HERE, '..', 'weights', 'basindetective_v1.npz')

LEVEL = {'d_logq': 'q_mean', 'd_rr': 'runoff_ratio', 'd_bfi': 'bfi_lh', 'd_logq5': 'q5', 'd_logq95': 'q95',
         'd_hfd': 'hfd_mean'}
LOGT = {'d_logq', 'd_logq5', 'd_logq95'}


def load(path=DEFAULT):
    return Detective(path)


def predict(det, cal, q_cal, tgt):
    q = np.asarray(q_cal, float)
    ok = np.isfinite(q)
    # Input checks (audit, 25 Nov 2026): all-NaN or short flow failed with KeyError('bfi_lh'), and a record of zeros
    # returned a finite estimate.
    if ok.sum() < 365:
        raise ValueError(f'q_cal needs at least 365 finite daily values, got {int(ok.sum())}')
    if not (q[ok] > 0).any():
        raise ValueError('q_cal has no positive flow; the calibration signatures are undefined')
    x, s_cal, names = featurize(cal, q, tgt)
    assert names == det.names, 'feature list mismatch'
    r = det.predict(x[None, :])[0]
    out = {'changes': {}, 'target_period': {}, 'hidden': {}, 'regulation': {}}
    for t, lv in LEVEL.items():
        mu, sd = r[t]
        c = s_cal.get(lv, np.nan)
        if t in LOGT:
            lev = lambda d: np.exp(np.log(c + EPSQ) + d) - EPSQ
        else:
            lev = lambda d: c + d
        out['changes'][t] = {'mean': mu, 'sd': sd, 'unit': UNITS[t]}
        out['target_period'][lv] = {'calibration': c, 'predicted': float(lev(mu)), 'lo68': float(lev(mu - sd)),
                                    'hi68': float(lev(mu + sd))}
    for t in ('log_kslow', 'gw_share', 'log_soilcap', 'log_pmult'):
        out['hidden'][t] = {'mean': r[t][0], 'sd': r[t][1], 'unit': UNITS[t]}
    out['hidden']['slow_store_days'] = float(np.exp(r['log_kslow'][0]))
    out['hidden']['soil_capacity_mm'] = float(np.exp(r['log_soilcap'][0]))
    out['hidden']['precip_multiplier'] = float(np.exp(r['log_pmult'][0]))
    out['regulation'] = {'p_regulated': r['p_regulated'], 'log10_storage_over_annual_flow': r['log_dor'],
                         'note': 'regulated = reservoir storage above 10 % of mean annual flow'}
    out['calibration_signatures'] = s_cal
    return out
