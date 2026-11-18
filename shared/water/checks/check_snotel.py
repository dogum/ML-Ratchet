"""Snow vs SNOTEL SWE at DEV stations (B's splits['snotel']['dev']), station P and T as forcing (radiation from the
Hargreaves-Samani DTR formula, vp from T_min, wind 2 m/s for the energy-balance model).
Degree-day: (a) attribute-free default parameters, (b) per-station calibration of TT, CFMAX, SFCF (NSE on SWE)
on water years 2001-2010, validated on WY 2011-2025 (B's SNOTEL protocol; dev stations may be used fully).
Energy balance: defaults and calibration of TT, SFCF, ALB_MIN, ALB_DECAY, TURB, Z0. SNOTEL SWE is the start-of-day value, so
simulated end-of-day SWE[t] is compared with observed SWE[t+1]. Stations near dev basins (waterdata.snotel_near)
are flagged as 'co-located'."""
import time

import numpy as np
import pandas as pd

from _common import write
import waterdata as wd
import watersim as ws
from watersim.calibrate import calibrate
from watersim import metrics as M

CALW = ('2000-10-01', '2010-09-30')
VALW = ('2010-10-01', '2025-09-30')
SPIN = ('1999-10-01', '2000-09-30')
t0 = time.time()
sp = wd.splits()
dev = sp['snotel']['dev']
meta = wd.snotel_meta().set_index('station_id')
near = set()
for b in wd.basins('camels', split='dev'):
    try:
        near |= set(wd.snotel_near(b, max_km=30).station_id)
    except Exception:
        pass


def scores(sim, obs):
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 300:
        return None
    o, s = obs[m], sim[m]
    df = pd.DataFrame({'o': obs, 's': sim}, index=sim_index)
    wy = df.index.year + (df.index.month >= 10)
    pk, mo = [], []
    for y, g in df.groupby(wy):
        if g['o'].notna().sum() < 300 or g['o'].max() < 50:
            continue
        pk.append((g['s'].max() - g['o'].max()) / g['o'].max())
        io, is_ = int(np.nanargmax(g['o'].values)), int(np.nanargmax(g['s'].values))
        ao = np.where(g['o'].values[io:] < 5)[0]
        as_ = np.where(g['s'].values[is_:] < 5)[0]
        if len(ao) and len(as_):
            mo.append((is_ + as_[0]) - (io + ao[0]))
    return {'NSE': M.nse(o, s), 'KGE': M.kge(o, s), 'peak_rel_err_median': float(np.median(pk)) if pk else None,
            'melt_out_err_days_median': float(np.median(mo)) if mo else None,
            'melt_out_abs_err_days_median': float(np.median(np.abs(mo))) if mo else None, 'n_years': len(pk)}


rows = []
skipped = {}
for sid in dev:
    d = wd.snotel(sid, SPIN[0], VALW[1])
    ncal = d['SWE'].loc[CALW[0]:CALW[1]].notna().sum()
    if ncal < 5 * 365 or d['SWE'].loc[VALW[0]:VALW[1]].notna().sum() < 3 * 365:
        skipped[sid] = f'too little SWE in the windows ({ncal} calibration days)'
        continue
    d = d.asfreq('D')
    f = pd.DataFrame({'P': d['P'].fillna(0).clip(lower=0), 'T_mean': d['T_mean'], 'T_min': d['T_min'],
                      'T_max': d['T_max']}, index=d.index)
    f[['T_mean', 'T_min', 'T_max']] = f[['T_mean', 'T_min', 'T_max']].interpolate(limit_direction='both')
    st = meta.loc[sid]
    spec = {'id': sid, 'area_km2': 1.0, 'lat': float(st.lat), 'lon': float(st.lon), 'elev_mean_m': float(st.elev_m)}
    obs = d['SWE'].shift(-1)          # start-of-next-day = end of this day
    rec = {'station': sid, 'elev_m': float(st.elev_m), 'lat': float(st.lat), 'co_located_with_dev_basin': sid in near}
    for snow, fixed in (('degree_day', ['TLAPSE', 'PGRAD', 'CFR', 'CWH', 'TTI']),
                        ('energy_balance', ['TLAPSE', 'PGRAD', 'CWH', 'TTI', 'ALB_MAX', 'GHEAT', 'CANOPY'])):
        res = {}
        for tag, par in (('default', None),):
            for per, (a, b) in (('cal', CALW), ('val', VALW)):
                o = ws.simulate(spec, f, a, b, model='none', snow=snow, routing='none', pet='hargreaves',
                                params=par, labels=False)
                sim_index = o['ts'].index
                res[f'{tag}_{per}'] = scores(o['ts']['SWE'].values, obs.reindex(sim_index).values)
        try:
            r = calibrate(spec, f, d['SWE'], *CALW, model='none', snow=snow, objective='nse', pop=24, gens=40,
                          target='SWE', obs_shift=1, fixed=fixed, pet='hargreaves')
            res['params'] = r['params']
            for per, (a, b) in (('cal', CALW), ('val', VALW)):
                o = ws.simulate(spec, f, a, b, model='none', snow=snow, routing='none', pet='hargreaves',
                                params=r['params'], labels=False)
                sim_index = o['ts'].index
                res[f'calibrated_{per}'] = scores(o['ts']['SWE'].values, obs.reindex(sim_index).values)
        except Exception as e:
            res['error'] = repr(e)
        rec[snow] = res
    rows.append(rec)
    print(sid, {s: {k: (v and round(v['NSE'], 3)) for k, v in rec[s].items() if k.endswith(('cal', 'val'))}
                for s in ('degree_day', 'energy_balance')}, flush=True)


def med(snow, key, metric):
    v = [r[snow][key][metric] for r in rows if r[snow].get(key) and r[snow][key].get(metric) is not None]
    return float(np.median(v)) if v else None


summary = {snow: {key: {m: med(snow, key, m) for m in ('NSE', 'KGE', 'peak_rel_err_median', 'melt_out_abs_err_days_median')}
                  for key in ('default_cal', 'default_val', 'calibrated_cal', 'calibrated_val')}
           for snow in ('degree_day', 'energy_balance')}
write('snotel.json', {'what': __doc__.strip().split('\n')[0], 'windows': {'cal': CALW, 'val': VALW, 'spinup': SPIN},
                      'n_stations': len(rows), 'skipped': skipped, 'n_co_located': sum(r['co_located_with_dev_basin'] for r in rows),
                      'summary_median': summary, 'stations': rows, 'wall_s': time.time() - t0})
print(summary)
