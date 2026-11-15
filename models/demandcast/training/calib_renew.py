"""Dev calibration of the wind/solar physics (2019, 2021 only):
  k_BA   = sum(EIA-930 output) / sum(physics output) over dev years (reporting scope, reanalysis bias, curtailment)
  error statistics of real vs k x physics: hourly nMAE (MAE / mean output), lag-1 h and lag-24 h autocorrelation of the
  capacity-factor error, std of the daily ratio -> used to set the simulator's weather-error ranges.
-> data/calib/renew_dev.json"""
import json, numpy as np
from common import *


def stats(real, phys, cap, m):
    ok = m & np.isfinite(real) & (cap > 0)
    k = real[ok].sum() / max(phys[ok].sum(), 1e-9)
    e = np.where(ok, (real - k * phys) / np.maximum(cap, 1), np.nan)
    nmae = np.nanmean(np.abs(real[ok] - k * phys[ok])) / np.nanmean(real[ok])
    ee = np.nan_to_num(e)
    ac1 = np.corrcoef(ee[1:], ee[:-1])[0, 1]; ac24 = np.corrcoef(ee[24:], ee[:-24])[0, 1]
    # daily ratio
    n = len(real) // 24 * 24
    rd = np.nansum(np.where(ok, real, np.nan)[:n].reshape(-1, 24), 1); pdd = np.nansum(np.where(ok, k * phys, np.nan)[:n].reshape(-1, 24), 1)
    good = pdd > 0.05 * np.nanmean(pdd)
    dr = np.std(np.log(np.clip(rd[good] / pdd[good], 0.05, 20)))
    return dict(k=float(k), nmae=float(nmae), ac1=float(ac1), ac24=float(ac24), daily_logratio_sd=float(dr),
                cf_err_sd=float(np.nanstd(e)))


import sys
which = sys.argv[1] if len(sys.argv) > 1 else 'dev'
YEARS = {'dev': DEV_YEARS, 'dev2019': (2019,), 'dev2021': (2021,)}[which]
out = {}
for ba in BAS:
    P = np.load(f'{DATA}/prior/{ba}_renew.npz')
    d = gd.eia930(ba, columns=['ng_WND', 'ng_SUN']).reindex(HOURS)
    m = np.isin(HOURS.year, YEARS)
    out[ba] = dict(wind=stats(d.ng_WND.values.astype(float), P['wind'].astype(float), P['wind_cap'], m),
                   solar=stats(d.ng_SUN.values.astype(float), P['solar'].astype(float), P['solar_cap'], m)
                   if np.nansum(d.ng_SUN.values[m]) > 0 else None)
    if out[ba]['solar'] is not None:      # EIA-930 solar timestamp offset vs the physics (best hourly shift on dev)
        y = d.ng_SUN.values.astype(float); ok = m & np.isfinite(y)
        cs = {s: np.corrcoef(y[ok], np.roll(P['solar'], s)[ok])[0, 1] for s in (-1, 0, 1)}
        out[ba]['solar']['lag'] = int(max(cs, key=cs.get)); out[ba]['solar']['lag_corr'] = cs
    print(ba, json.dumps(out[ba]))
jdump(out, f'{DATA}/calib/renew_{which}.json')
