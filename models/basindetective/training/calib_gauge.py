"""Physics rung v2 (dev only): realistic gauge records.

The v1 twin test showed the biggest real-vs-twin feature gaps in day-to-day texture: real dev records have many more
'flat' days (|dQ| < 1 % of Q; +1.4 IQR) and fewer rising days (-1.2 IQR) than their simulated twins. watersim's
default gauge noise (AR(1) rho 0.9, cv 0.08) jitters every day. Real USGS daily values are smooth between rating
shifts and are published to 3 significant figures (cfs), with 0.01 cfs resolution at low flow.

This script re-simulates the clean calibrated twins of the 52 dev basins (calibration window only), applies a grid of
gauge-noise variants and picks the one whose day-to-day feature medians are closest to the real dev records
(sum over the dynamics features of |median difference| / real IQR). Writes models/gauge_v2.json, logs/calib_gauge.json.
"""
import common  # noqa: F401
from _paths import DEV_PARAMS
import itertools
import json

import numpy as np
import pandas as pd

import watersim as ws
import waterdata as wd
from common import CAL, D
from features import featurize
import evaluate as E

CFS = 0.028316846592
DYN = ['q_flat', 'q_rise', 'q_flash', 'q_ac1', 'q_ac3', 'q_pq0', 'q_pq1', 'q_pq2', 'q_bfi_lh', 'q_bfi_eck',
       'q_logreck', 'q_recb', 'q_fdc95', 'q_fdc99', 'q_lowfreq']


def round_sig(q_mm, area_km2, sig=3):
    """Round mm/day flows as USGS publishes cfs: `sig` significant figures, at least 0.01 cfs resolution."""
    if sig is None:
        return q_mm
    k = area_km2 / 86.4 / CFS          # mm/day -> cfs
    c = np.asarray(q_mm, float) * k
    out = np.zeros_like(c)
    pos = c > 0
    e = np.floor(np.log10(c[pos]))
    step = np.maximum(10.0 ** (e - sig + 1), 0.01)
    out[pos] = np.round(c[pos] / step) * step
    return out / k


def gauge(q, rng, area, rho=0.9, cv=0.08, sig=None, gap_frac=0.0):
    o = ws.noise.streamflow_obs(q, rng, cv=cv, rho=rho, resolution=0, gap_frac=gap_frac)
    nan = np.isnan(o)
    o = round_sig(np.nan_to_num(o), area, sig)
    o[nan] = np.nan
    return o


def main():
    P = json.load(open(DEV_PARAMS))['params']
    Xr, meta, names = E.real_set('dev')
    idx = [names.index(f) for f in DYN]
    real_med = np.nanmedian(Xr[:, idx], 0)
    real_iqr = np.subtract(*np.nanpercentile(Xr[:, idx], [75, 25], 0))
    clean = []
    for b in wd.basins('camels', 'dev'):
        s = wd.spec(b)
        fc = wd.forcing(b, *CAL)
        qr = wd.streamflow(b, 'caravan').Q_mm.reindex(fc.index)
        for pair in ('hbv+degree_day', 'gr4j+degree_day', 'sacsma+degree_day', 'vic_lite+degree_day'):
            pr = P[b][pair]
            pr = eval(pr) if isinstance(pr, str) else pr
            m, sn = pair.split('+')
            o = ws.simulate(s, (fc, wd.forcing_meta(b)), *CAL, model=m, snow=sn, params=pr, labels=False)
            clean.append((b, fc[['P', 'T_mean', 'PET']], o['ts']['Q'].values, qr.notna().values, s['area_km2']))
    res = []
    for rho, cv, sig in itertools.product([0.9, 0.99, 0.999], [0.0, 0.02, 0.04, 0.08], [None, 3]):
        rng = np.random.default_rng(1)
        F = []
        for b, f, q, ok, area in clean:
            o = gauge(q, rng, area, rho, cv, sig)
            o[~ok] = np.nan
            x, _, _ = featurize(f, o, f)
            F.append(x[idx])
        med = np.nanmedian(np.array(F), 0)
        gap = (med - real_med) / real_iqr
        res.append({'rho': rho, 'cv': cv, 'sig': sig, 'score': float(np.abs(gap).sum()),
                    'gap': {d: float(g) for d, g in zip(DYN, gap)}})
        print(rho, cv, sig, round(res[-1]['score'], 3), {d: round(float(g), 2) for d, g in zip(DYN[:3], gap[:3])},
              flush=True)
    best = min(res, key=lambda r: r['score'])
    json.dump({'rho': best['rho'], 'cv': best['cv'], 'sig': best['sig']}, open(f'{D}/models/gauge_v2.json', 'w'))
    json.dump({'grid': res, 'best': best}, open(f'{D}/logs/calib_gauge.json', 'w'), indent=1)
    print('best', best)


if __name__ == '__main__':
    main()
