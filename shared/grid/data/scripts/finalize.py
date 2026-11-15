"""Consolidate fetched pieces into the proc/ files the griddata loader reads.
  proc/frequency/GB/minute_summary.parquet   per-minute GB frequency summary (UTC), all scanned months
  proc/frequency/GB/inertia.parquet          NESO outturn / market inertia per settlement period (time_utc)
  proc/frequency/GB/genmix.parquet           NESO historic generation mix (time_utc), 2018+
  proc/prices/caiso_dam.parquet, caiso_rtpd.parquet
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, glob, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RAW = f'{GRID_ROOT}/data/raw'; PROC = f'{GRID_ROOT}/data/proc'
os.makedirs(f'{PROC}/frequency/GB', exist_ok=True)


def minute_utc_vec(m, tz):
    t = pd.to_datetime(m.str.replace('T', ' '), format='mixed')
    tz = tz.fillna('').str.strip()
    off = tz.apply(lambda z: 0 if not z else (1 if z[0] == '+' else -1) * (int(z[1:3]) * 60 + int(z[3:5])))
    return (t - pd.to_timedelta(off, unit='m')).dt.tz_localize('UTC')


d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{RAW}/neso/scan/*.parquet'))], ignore_index=True)
d['time_utc'] = minute_utc_vec(d.m, d.tz)
d = d.drop(columns=['m', 'tz']).sort_values('time_utc').drop_duplicates('time_utc')
d.to_parquet(f'{PROC}/frequency/GB/minute_summary.parquet', index=False)
print('minutes', len(d), d.time_utc.min(), d.time_utc.max())

i = pd.read_parquet(f'{RAW}/neso/inertia.parquet')
dt = pd.to_datetime(i['Settlement Date'].str[:10], format='%Y-%m-%d')
sp = pd.to_numeric(i['Settlement Period'])
loc = dt.dt.tz_localize('Europe/London', ambiguous='NaT', nonexistent='shift_forward')
# settlement period k starts (k-1)*30 min after local midnight (elapsed time, so DST days have 46/50 periods)
t = loc.dt.tz_convert('UTC') + pd.to_timedelta((sp - 1) * 30, unit='m')
o = pd.DataFrame({'time_utc': t, 'settlement_date': dt.dt.date.astype(str), 'settlement_period': sp,
                  'inertia_outturn_GVAs': pd.to_numeric(i['Outturn Inertia'], errors='coerce'),
                  'inertia_market_GVAs': pd.to_numeric(i['Market Provided Inertia'], errors='coerce')})
o = o.dropna(subset=['time_utc']).sort_values('time_utc').drop_duplicates('time_utc')
o.to_parquet(f'{PROC}/frequency/GB/inertia.parquet', index=False)
print('inertia', len(o), o.time_utc.min(), o.time_utc.max())

g = pd.read_parquet(f'{RAW}/neso/genmix.parquet')
g['time_utc'] = pd.to_datetime(g.DATETIME, utc=True)
cols = [c for c in g.columns if c not in ('DATETIME', 'time_utc', '_id')]
for c in cols: g[c] = pd.to_numeric(g[c], errors='coerce').astype('float32')
g = g[['time_utc'] + cols].sort_values('time_utc').drop_duplicates('time_utc')
g.to_parquet(f'{PROC}/frequency/GB/genmix.parquet', index=False)
print('genmix', len(g), g.time_utc.min(), g.time_utc.max())

for k in ('dam', 'rtpd'):
    fs = sorted(glob.glob(f'{RAW}/caiso/{k}_*.parquet'))
    if not fs: continue
    c = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(['time_utc', 'node']).sort_values(['node', 'time_utc'])
    for col in ('LMP', 'MCC', 'MCE', 'MCL'):
        if col in c: c[col] = c[col].astype('float32')
    c['node'] = c.node.astype('category')
    c.to_parquet(f'{PROC}/prices/caiso_{k}.parquet', index=False)
    print('caiso', k, len(c), c.time_utc.min(), c.time_utc.max())
