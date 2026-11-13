"""Twin test on DEV only: simulate each dev VED trip with the dev car's partial spec (completed generically), compare
simulated vs logged energy. Writes data/twin_<tag>.parquet. Usage: python twin.py <tag> [wrapper]"""
import sys, time, json
from common import *
import numpy as np, pandas as pd
import realdata as rd
from roadsim import simulate
import wrappers

tag = sys.argv[1] if len(sys.argv) > 1 else 'raw'
wrap = sys.argv[2] if len(sys.argv) > 2 else 'none'
use_ts = len(sys.argv) > 3 and sys.argv[3] == 'ts'
TS = json.load(open(f'{EN}/data/twin_specs_dev.json')) if use_ts else {}
rows = []
t0 = time.time()
for meta, arr in rd.load_trips('dev', label_ok=True):
    vid = meta['vehicle_id']; arch = meta['architecture']; vt = meta['ved_type']
    spec = TS[vid]['spec'] if (use_ts and vid in TS) else complete_spec(rd.load_vehicle(vid))
    trip = {k: arr[k] for k in ('t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface')}
    soc0 = None
    if arch != 'conv':
        s = arr['soc'][np.isfinite(arr['soc'])]
        if len(s): soc0 = float(s[0])
    T0 = float(np.nanmean(arr['T_amb']))
    try:
        out = wrappers.run(spec, trip, wrap, init_soc=soc0, init_temps=warm_temps(T0), meta=meta)
    except Exception as e:
        print('fail', meta['trip_id'], e); continue
    ok = arr['valid'] == 1
    if 'fuel_rate_maf' in out and meta.get('fuel_source', '').startswith(('maf', 'absload')):
        out = dict(out, fuel_rate=out['fuel_rate_maf'])
    ys = sim_target(out, arch); yr = real_target(arr, vt)
    L = out['ledger']
    v = arr['v'].astype(float)
    rows.append(dict(trip_id=meta['trip_id'], vehicle_id=vid, arch=arch, vt=vt, T=T0, v_mean=v.mean(), idle=float((v < 0.5).mean()),
                     dist=float(v[ok].sum()), dur=int(ok.sum()), climb=float(np.sum(np.clip(np.diff(arr['elev']), 0, None))),
                     y_real=float(yr.sum()), y_sim=float(ys[ok].sum()), fuel_sim=float(np.nansum(out['fuel_rate'][ok]) * LHV),
                     aux_sim=L['aux'] / 1e3, eng_loss=L['eng_loss'] / 1e3, wet=meta.get('wet_frac', 0), snow=meta.get('snow_frac', 0),
                     soc0=soc0 if soc0 is not None else np.nan,
                     soc_drop=float(arr['soc'][np.isfinite(arr['soc'])][0] - arr['soc'][np.isfinite(arr['soc'])][-1]) if np.isfinite(arr['soc']).any() else np.nan,
                     aux_real_kJ=float(np.nansum(arr['aux_power'][ok]) / 1e3), hour=meta.get('hour_local'),
                     fuel_real_zero_moving=float(((np.nan_to_num(arr['fuel_rate']) < 0.05) & (v > 3) & ok).mean()),
                     fuel_sim_zero_moving=float(((out['fuel_rate'] < 0.05) & (out['v'] > 3)).mean())))
    if len(rows) % 200 == 0: print(len(rows), round(time.time() - t0), flush=True)
d = pd.DataFrame(rows); d['lr'] = np.log(d.y_sim / d.y_real)
d.to_parquet(f'{EN}/data/twin_{tag}.parquet')
print('done', len(d), round(time.time() - t0))
for a, g in d[d.y_real > 0.02].groupby('arch'):
    print(a, len(g), 'median log(sim/real) %.3f  MdAPE %.3f' % (g.lr.median(), np.median(np.abs(np.exp(g.lr) - 1))))
