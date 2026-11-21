"""Physics rung v4 (dev only): year-to-year error of the forcing precipitation.

After v2, the largest remaining real-vs-twin feature gap that is about the forcing is q_annpcorr (correlation of
annual streamflow with annual ERA5-Land precipitation): real dev records correlate less with the forcing than their
twins do (-0.44 IQR). The bank's daily precipitation noise averages out over a year, so it cannot produce this.
Here every dev twin is driven by TRUE P = ERA5 P x exp(e_y), one e_y ~ N(0, s) per water year, while its features
read the ERA5 P; s is picked from a grid so the twin median of q_annpcorr matches the real dev median.
Writes models/annual_v4.json and logs/calib_annual.json.
"""
import common  # noqa: F401
from _paths import DEV_PARAMS
import json

import numpy as np

import watersim as ws
import waterdata as wd
from common import CAL, D
from features import featurize, water_year
from calib_gauge import gauge
import evaluate as E

PAIRS = ['hbv+degree_day', 'gr4j+degree_day', 'sacsma+degree_day', 'vic_lite+degree_day']


def main():
    P = json.load(open(DEV_PARAMS))['params']
    gp = json.load(open(f'{D}/models/gauge_v2.json'))
    Xr, meta, names = E.real_set('dev')
    j = names.index('q_annpcorr')
    real_med = float(np.nanmedian(Xr[:, j]))
    res = []
    for s in (0.0, 0.05, 0.08, 0.12, 0.16):
        rng = np.random.default_rng(5)
        vals = []
        for b in wd.basins('camels', 'dev'):
            sp = wd.spec(b)
            fc = wd.forcing(b, *CAL)
            qr = wd.streamflow(b, 'caravan').Q_mm.reindex(fc.index)
            wy = water_year(fc.index)
            e = {y: rng.normal(0, s) for y in np.unique(wy)}
            ft = fc.copy()
            ft['P'] = fc.P.values * np.exp(np.array([e[y] for y in wy]))
            for pair in PAIRS[:2]:
                pr = P[b][pair]
                pr = eval(pr) if isinstance(pr, str) else pr
                m, sn = pair.split('+')
                o = ws.simulate(sp, (ft, wd.forcing_meta(b)), *CAL, model=m, snow=sn, params=pr, labels=False)
                q = gauge(o['ts']['Q'].values, rng, sp['area_km2'], gp['rho'], gp['cv'], gp['sig'])
                q[qr.isna().values] = np.nan
                x, _, _ = featurize(fc[['P', 'T_mean', 'PET']], q, fc[['P', 'T_mean', 'PET']])
                vals.append(x[j])
        med = float(np.nanmedian(vals))
        res.append({'s': s, 'twin_median': med})
        print(s, round(med, 3), 'real', round(real_med, 3), flush=True)
    best = min(res, key=lambda r: abs(r['twin_median'] - real_med))
    json.dump({'sd': best['s']}, open(f'{D}/models/annual_v4.json', 'w'))
    json.dump({'real_median': real_med, 'grid': res, 'best': best}, open(f'{D}/logs/calib_annual.json', 'w'), indent=1)
    print('best', best)


if __name__ == '__main__':
    main()
