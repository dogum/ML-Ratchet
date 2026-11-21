"""The final exam (run once, after the ratchet chose the release rung on sim + dev only).

python3 final_exam.py <release_run>
Scores every rung on the 621 exam stations (for the ladder table; nothing is chosen here), the release rung with its
dev-fitted widen factors, and gathers the baselines: B's climatology and per-station calibrated degree-day
(snow_scores.parquet), and the uncalibrated WaterSim degree-day and energy-balance models (physics_exam.parquet).
-> results/final_summary.json, results/exam_by_rung.json
"""
import json
import sys
import numpy as np
import pandas as pd
from common import *
import evaluate as E

rel = sys.argv[1]
R = json.load(open(f'{ROOT}/ratchet.json'))
by = {}
for r in R['rungs']:
    tag = f"{r['run']}_exam"
    if r['run'] == rel:
        continue
    if not os.path.exists(f'{ROOT}/results/{tag}.parquet'):
        E.real_score(r['run'], 'exam')
    sc = pd.read_parquet(f'{ROOT}/results/{tag}.parquet')
    yr = pd.read_parquet(f'{ROOT}/results/{tag}_years.parquet')
    by[r['rung']] = E.summarize(sc, yr)
    print(r['rung'], round(by[r['rung']]['nse'], 3), flush=True)
k = json.load(open(f'{ROOT}/results/k_{rel}.json'))
net = E.real_score(rel, 'exam', k)
rel_rung = [r['rung'] for r in R['rungs'] if r['run'] == rel][0]
by[rel_rung] = net
json.dump(by, open(f'{ROOT}/results/exam_by_rung.json', 'w'), indent=1)

cols = ['nse', 'rmse_mm', 'peak_swe_abs_err_pct', 'peak_swe_bias_pct', 'meltout_mae_days', 'meltout_bias_days']
b = pd.read_parquet(SNOW_SCORES)
b = b[b.split == 'exam']
phy = pd.read_parquet(f'{ROOT}/results/physics_exam.parquet')
sc = pd.read_parquet(f'{ROOT}/results/{rel}_exam.parquet')
out = {'net': net, 'release_run': rel, 'release_rung': rel_rung, 'k': k}
for name, g in (('clim', b[b.method == 'clim']), ('dd_cal', b[b.method == 'dd']),
                ('dd_uncal', phy[phy.method == 'dd_uncal']), ('eb_uncal', phy[phy.method == 'eb_uncal'])):
    out[name] = {c: float(g[c].median()) for c in cols if c in g}
    out[name]['n'] = int(len(g))
# head to head
m = sc.set_index('station_id')
for name, g in (('eb_uncal', phy[phy.method == 'eb_uncal']), ('dd_cal', b[b.method == 'dd']),
                ('clim', b[b.method == 'clim'])):
    g = g.set_index('station_id').reindex(m.index)
    out[f'win_frac_vs_{name}'] = float((m.nse > g.nse).mean())
meta = wd().snotel_meta().set_index('station_id')
REG = {'WA': 'Pacific', 'OR': 'Pacific', 'CA': 'Pacific', 'ID': 'Northern Rockies', 'MT': 'Northern Rockies',
       'WY': 'Central Rockies', 'CO': 'Central Rockies', 'UT': 'Central Rockies', 'NV': 'Great Basin',
       'AZ': 'Southwest', 'NM': 'Southwest', 'AK': 'Alaska', 'SD': 'Central Rockies'}
m['region'] = [REG[meta.loc[s, 'state']] for s in m.index]
eb = phy[phy.method == 'eb_uncal'].set_index('station_id').reindex(m.index)
dd = b[b.method == 'dd'].set_index('station_id').reindex(m.index)
m['eb'], m['dd'] = eb.nse, dd.nse
out['by_region'] = {reg: {'n': int(len(g)), 'net': float(g.nse.median()), 'eb_uncal': float(g.eb.median()),
                          'dd_cal': float(g.dd.median())} for reg, g in m.groupby('region')}
m['elev_band'] = pd.cut([meta.loc[s, 'elev_m'] for s in m.index], [0, 1500, 2200, 2800, 5000],
                        labels=['<1500 m', '1500-2200 m', '2200-2800 m', '>2800 m'])
out['by_elev'] = {str(e): {'n': int(len(g)), 'net': float(g.nse.median()), 'eb_uncal': float(g.eb.median()),
                           'dd_cal': float(g.dd.median())} for e, g in m.groupby('elev_band', observed=True)}
out['frac_nse_below_0.5'] = {'net': float((m.nse < 0.5).mean()), 'eb_uncal': float((m.eb < 0.5).mean()),
                             'dd_cal': float((m.dd < 0.5).mean())}
json.dump(out, open(f'{ROOT}/results/final_summary.json', 'w'), indent=1)
print(json.dumps({k2: v for k2, v in out.items() if k2 not in ('by_region', 'by_elev')}, indent=1))
print(pd.DataFrame(out['by_region']).T.round(3))
print(pd.DataFrame(out['by_elev']).T.round(3))
