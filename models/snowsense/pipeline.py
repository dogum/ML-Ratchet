"""SnowSense pipeline: CSV in, daily SWE and annual snow dates out.

    python3 pipeline.py examples/349_MT_SNTL.csv --lat 46.414 --elev 2185 [--out result.csv]

CSV columns: date, P (mm/day, the local-day gauge total), T_mean, T_min, T_max (deg C); optional SWE_obs (mm, start of
day as SNOTEL reports it) is only used for scoring. Gaps are allowed: missing P counts as 0 and is flagged, missing T
is interpolated and flagged, exactly as in training.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snowsense import SnowSense


def load_csv(path):
    df = pd.read_csv(path, parse_dates=['date']).set_index('date').asfreq('D')
    return df


def nse(obs, sim):
    m = obs.notna() & sim.notna()
    o, s = obs[m], sim[m]
    return float(1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()) if len(o) > 30 else float('nan')


def run(path, lat, elev, model=None):
    model = model or SnowSense()
    df = load_csv(path)
    out = model.predict(df, lat, elev)
    daily = out['daily']
    if 'SWE_obs' in df:
        daily['obs_eod'] = df['SWE_obs'].shift(-1).reindex(daily.index)   # SNOTEL is start-of-day
    return daily, out['annual']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--elev', type=float, required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    daily, annual = run(a.csv, a.lat, a.elev)
    if 'obs_eod' in daily:
        print('NSE vs observed SWE:', round(nse(daily['obs_eod'], daily['swe']), 3))
    print(annual[['peak_swe_mm', 'peak_lo68', 'peak_hi68', 'peak_date', 'melt_out_date', 'melt_lo68', 'melt_hi68']].round(0).to_string())
    if a.out:
        daily.round(1).to_csv(a.out)
