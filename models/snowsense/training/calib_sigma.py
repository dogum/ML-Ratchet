"""Widen-only error-bar calibration on the DEV stations (WY2011-2025): one factor per output.
  daily  : distance of each band from the median x k so that 68% of snow days (obs or estimate > 1 mm) fall
           inside the q16-q84 band
  peak   : comonotone peak-SWE interval x k to 68% coverage of station-years
  melt   : comonotone melt-out interval x k to 68% coverage
k = max(1, smallest grid value reaching 68%). python3 calib_sigma.py <run> -> results/k_<run>.json
"""
import json
import sys
import numpy as np
from common import *
from evalset import station_set
from evaluate import load, predict, annual, band, melt_out_idx

run = sys.argv[1]
net = load(run)
R, years = [], []
for sid, lat, elev, ok in station_set('dev'):
    df = load_station(sid)
    X, D, Y, wys = station_windows(sid, list(range(2011, 2026)), lat, elev, df)
    Qa = predict(net, X)
    for j, y in enumerate(wys):
        if y not in ok:
            continue
        wm = wy_of(D[j]) == y
        o = df['SWE_eod'].reindex(D[j][wm]).values
        Q = Qa[j][wm]
        med = band(Q, 2)
        f = np.isfinite(o) & ((o > 1) | (med > 1))
        z = np.sqrt(np.maximum(o[f], 0)) / 10
        q = Q[f]
        up = z > q[:, 2]
        half = np.where(up, q[:, 3] - q[:, 2], q[:, 2] - q[:, 1])
        R.append(np.abs(z - q[:, 2]) / np.maximum(half, 1e-6))
        if np.isfinite(o).sum() >= 300 and np.nanmax(o) >= 25:
            years.append((Q, float(np.nanmax(o)), melt_out_idx(np.nan_to_num(o))))
R = np.concatenate(R)
k_daily = max(1.0, float(np.quantile(R, 0.68)))


def cov(kind, k):
    hit = []
    for Q, pk, mo in years:
        a = annual(Q, k, k)
        if kind == 'peak':
            hit.append(a['peak_lo68'] <= pk <= a['peak_hi68'])
        elif mo is not None and a['melt'] is not None:
            lo = a['melt_lo68'] if a['melt_lo68'] is not None else -1e9
            hi = a['melt_hi68'] if a['melt_hi68'] is not None else 1e9
            hit.append(lo <= mo <= hi)
    return float(np.mean(hit))


out = {'daily': round(k_daily, 3), 'raw_cov68_daily': float((R <= 1).mean())}
for kind in ('peak', 'melt'):
    out[f'raw_cov68_{kind}'] = cov(kind, 1.0)
    k = 1.0
    while cov(kind, k) < 0.68 and k < 6:
        k += 0.05
    out[kind] = round(k, 2)
json.dump(out, open(f'{ROOT}/results/k_{run}.json', 'w'), indent=1)
print(out)
