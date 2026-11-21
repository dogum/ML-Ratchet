"""Reproduction re-score of the released SnowSense on the rebuilt SNOTEL exam, plus the dev-tuned physics baselines.
Added on import (18 Nov 2026). SEEN EXAM: a reproduction, not a rung; nothing is chosen or changed on its result.

    ML_RATCHET_WATER=<data root> python training/rescore_exam.py      (.venv-train: numba makes WaterSim fast)

1. The 621 exam stations exactly as the shipped exam chose them (evalset.station_set('exam'): the exam stations of
   WaterData's snow_scores.parquet with >= 5 complete water years in 2011-2025), checked against splits.json.
2. SnowSense: the shipped numpy release (snowsense.py + snowsense_weights.npz, widen factors k from the npz) through
   the shipped exam path (evaluate.real_score), with the torch checkpoint loader swapped for the numpy model (no
   checkpoints were shipped; numpy matches torch to about 1e-6 in z).
3. Physics on the same inputs SnowSense reads (common.load_station), continuous WY2011-2025 with a one-year spin-up
   (evalset.physics): WaterSim's energy balance and degree-day with the default + attribute-transfer parameters (the
   shipped baselines) and with the frozen dev-tuned parameters of devtuned_params.json.
4. WaterData's own climatology and per-station calibrated degree-day (shared/water/data/scripts/run_baselines_snow.py),
   re-run on the rebuilt series into WORK, compared station by station with the shipped snow_scores.parquet.
Writes WORK/rescore/*.parquet (intermediates) and training/rescore_exam_stations.csv (per-station scores, no series).
Then run training/rescore_summary.py for medians, paired bootstrap intervals and verdicts.
"""
import importlib.util
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*dayofyear.*')
import numpy as np
import pandas as pd
from common import *
import evaluate as E
from evalset import station_set, physics
from _paths import WATER

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = f'{WORK}/rescore'
for d in (OUT, f'{WORK}/results/rescore'):
    os.makedirs(d, exist_ok=True)
METRICS = ['nse', 'rmse_mm', 'peak_swe_abs_err_pct', 'peak_swe_bias_pct', 'meltout_mae_days', 'meltout_bias_days',
           'n_years']
REG = {'WA': 'Pacific', 'OR': 'Pacific', 'CA': 'Pacific', 'ID': 'Northern Rockies', 'MT': 'Northern Rockies',
       'WY': 'Central Rockies', 'CO': 'Central Rockies', 'UT': 'Central Rockies', 'NV': 'Great Basin',
       'AZ': 'Southwest', 'NM': 'Southwest', 'AK': 'Alaska', 'SD': 'Central Rockies'}

# ---- 1. stations
sp = wd().splits()['snotel']
rows = station_set('exam')
ids = [r[0] for r in rows]
assert set(ids) <= set(sp['exam']) and not set(ids) & set(sp['dev']), 'exam stations outside splits.json exam'
check = {'n_scored': len(ids), 'n_splits_exam': len(sp['exam']), 'n_splits_dev': len(sp['dev']),
         'all_in_splits_exam': True}
print('exam stations', len(ids), 'of', len(sp['exam']), flush=True)

# ---- 2. SnowSense, shipped weights, shipped exam path
sys.path.insert(0, MODEL)
import snowsense  # noqa: E402
NET = snowsense.SnowSense(os.path.join(MODEL, 'snowsense_weights.npz'))
E.load = lambda run: NET
E.predict = lambda net, X, bs=256: np.concatenate(
    [net.forward(np.asarray(X[i:i + bs], np.float64)) for i in range(0, len(X), bs)])
k = dict(NET.k)
net_summary = E.real_score('r6', 'exam', k, tag='rescore/net_exam')
print('SnowSense', {c: round(net_summary[c], 3) for c in ('nse', 'rmse_mm', 'peak_swe_abs_err_pct')}, flush=True)
net = pd.read_parquet(f'{WORK}/results/rescore/net_exam.parquet')

# ---- 3. physics, default and dev-tuned
DT = json.load(open(os.path.join(SRC_DIR, 'devtuned_params.json')))
variants = [('eb_default', 'energy_balance', None), ('dd_default', 'degree_day', None),
            ('eb_devtuned', 'energy_balance', DT['energy_balance']), ('dd_devtuned', 'degree_day', DT['degree_day'])]
phys = []
for j, (sid, lat, elev, ok) in enumerate(rows):
    df = load_station(sid)
    for tag, snow, prm in variants:
        s = physics(sid, lat, elev, snow, params=dict(prm) if prm else None, df=df)
        r = score(df['SWE_eod'], s, ok)
        r.update(station_id=sid, method=tag)
        phys.append(r)
    if j % 100 == 0:
        print('physics', j, sid, flush=True)
phys = pd.DataFrame(phys)
phys.to_parquet(f'{OUT}/physics_exam.parquet')
import watersim as ws  # noqa: E402  (record the resolved parameter sets once; they are the same at every station)
sid, lat, elev, ok = rows[0]
f = load_station(sid).loc['2009-10-01':'2025-09-30', ['P', 'T_mean', 'T_min', 'T_max']]
spec = {'id': sid, 'area_km2': 1.0, 'lat': lat, 'lon': -110.0, 'elev_mean_m': elev}
resolved = {tag: ws.simulate(spec, f, '2010-10-01', '2025-09-30', model='none', snow=snow, routing='none',
                             pet='hargreaves', params=dict(prm) if prm else None, labels=False)['params']['snow']
            for tag, snow, prm in variants}

# ---- 4. WaterData's climatology and calibrated degree-day, re-run on the rebuilt series (into WORK)
spec_ = importlib.util.spec_from_file_location('rbs', os.path.join(WATER, 'data', 'scripts', 'run_baselines_snow.py'))
rbs = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(rbs)
rbs.ROOT = f'{OUT}/waterdata'
os.makedirs(f'{rbs.ROOT}/proc/baselines', exist_ok=True)
rbs.main()
wnew = pd.read_parquet(f'{rbs.ROOT}/proc/baselines/snow_scores.parquet')
wold = pd.read_parquet(SNOW_SCORES)
cmp = {}
for m in ('clim', 'dd'):
    a = wold[(wold.split == 'exam') & (wold.method == m)].set_index('station_id')
    b = wnew[(wnew.split == 'exam') & (wnew.method == m)].set_index('station_id')
    common_ids = a.index.intersection(b.index)
    d = (b.loc[common_ids, 'nse'] - a.loc[common_ids, 'nse']).abs()
    cmp[m] = {'n_shipped': int(len(a)), 'n_rescored': int(len(b)), 'n_common': int(len(common_ids)),
              'only_shipped': sorted(set(a.index) - set(b.index)), 'only_rescored': sorted(set(b.index) - set(a.index)),
              'max_abs_nse_diff': float(d.max()), 'n_nse_diff_gt_1e-6': int((d > 1e-6).sum()),
              'n_nse_diff_gt_0.01': int((d > 0.01).sum()),
              'median_nse_shipped_621': float(a.loc[ids, 'nse'].median()),
              'median_nse_rescored_621': float(b.reindex(ids)['nse'].median())}
print('WaterData baselines vs shipped', json.dumps({m: {kk: v for kk, v in c.items() if not isinstance(v, list)}
                                                   for m, c in cmp.items()}), flush=True)

# ---- per-station table (wide; no time series)
meta = wd().snotel_meta().set_index('station_id')
tab = pd.DataFrame({'station_id': ids})
tab['state'] = [meta.loc[s, 'state'] for s in ids]
tab['region'] = tab.state.map(REG)
tab['elev_m'] = [round(float(meta.loc[s, 'elev_m']), 1) for s in ids]
blocks = {'net': net, **{t: phys[phys.method == t] for t, _, _ in variants},
          'clim_shipped': wold[(wold.split == 'exam') & (wold.method == 'clim')],
          'ddcal_shipped': wold[(wold.split == 'exam') & (wold.method == 'dd')],
          'clim_rescored': wnew[(wnew.split == 'exam') & (wnew.method == 'clim')],
          'ddcal_rescored': wnew[(wnew.split == 'exam') & (wnew.method == 'dd')]}
for name, g in blocks.items():
    g = g.set_index('station_id').reindex(ids)
    for c in METRICS:
        tab[f'{name}_{c}'] = g[c].values
tab.round(5).to_csv(os.path.join(SRC_DIR, 'rescore_exam_stations.csv'), index=False)
json.dump({'stations': check, 'net_summary': net_summary, 'k': k, 'resolved_params': resolved,
           'waterdata_vs_shipped': cmp}, open(f'{OUT}/rescore_run.json', 'w'), indent=1, default=float)
print('wrote', os.path.join(SRC_DIR, 'rescore_exam_stations.csv'))
