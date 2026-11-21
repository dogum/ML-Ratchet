"""Per fire-day table (PM -> next PM) for Phase 2 (FireDetective / FireCast features and targets).
-> proc/baselines/fire_days_dev.parquet and fire_days_exam.parquet (EXAM: final scoring only).
Columns: fire_id, k, t (FEDS PM label, local ~13:30), area_km2 (90 m raster at issue), d_prev_m / d_next_m (equivalent-
radius growth over the previous / next 24 h), ros_eq_m_min = d_next_m / 1440, new_obs_km2, new_outside_window,
has_fireline, flinelen_km / sumFRP / n_newpixels at issue (FEDS), and gridMET daily at the fire centroid for the issue
day (_d0) and the verification day (_d1): erc, bi, fm100, fm1000, vs (m/s 10 m), th (deg from), rmin, rmax, tmmx (degC),
pr (mm), vpd (kPa)."""
import json, glob
import numpy as np, pandas as pd
import os
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
P = f'{_DATA}/data/proc'
WX = ['erc', 'bi', 'fm100', 'fm1000', 'vs', 'th', 'rmin', 'rmax', 'tmmx', 'pr', 'vpd']
f = pd.read_parquet(f'{P}/fires.parquet').set_index('Event_ID')
for split in ['dev', 'exam']:
    rows = []
    for l in open(f'{P}/baselines/spread_{split}.jsonl'):
        for r in json.loads(l)['rows']:
            if r['method'] == 'persist':
                rows.append({k: r[k] for k in ['fire_id', 'k', 't', 'area_km2', 'd_prev_m', 'd_next_m', 'size_bin',
                                                'has_fireline', 'new_outside_window', 'new_obs_km2', 'n_cand']})
    d = pd.DataFrame(rows); d['t'] = pd.to_datetime(d.t)
    d['ros_eq_m_min'] = d.d_next_m / 1440.0
    d['year'] = d.fire_id.map(f.year)
    feds = []
    for y in sorted(d.year.unique()):
        p = pd.read_parquet(f'{P}/feds/perims/{y}.parquet', columns=['Event_ID', 't', 'flinelen', 'sumFRP', 'n_newpixels'])
        feds.append(p[p.Event_ID.isin(d.fire_id.unique())])
    feds = pd.concat(feds).rename(columns={'Event_ID': 'fire_id', 'flinelen': 'flinelen_km'})
    d = d.merge(feds, on=['fire_id', 't'], how='left')
    gm = []
    for y in sorted(d.year.unique()):
        fn = f'{P}/weather/gridmet/fires_{y}.parquet'
        try: g = pd.read_parquet(fn, filters=[('Event_ID', 'in', list(d.fire_id.unique()))])
        except FileNotFoundError: continue
        gm.append(g)
    gm = pd.concat(gm).rename(columns={'Event_ID': 'fire_id'})
    gm['date'] = pd.to_datetime(gm.date).dt.normalize()
    d['date0'] = d.t.dt.normalize(); d['date1'] = d.date0 + pd.Timedelta(days=1)
    for s, c in [('_d0', 'date0'), ('_d1', 'date1')]:
        g = gm.rename(columns={v: v + s for v in WX}).rename(columns={'date': c})
        d = d.merge(g[['fire_id', c] + [v + s for v in WX]], on=['fire_id', c], how='left')
    d = d.drop(columns=['date0', 'date1'])
    d.to_parquet(f'{P}/baselines/fire_days_{split}.parquet')
    print(split, d.shape, 'gridMET missing', float(d.erc_d1.isna().mean()))
