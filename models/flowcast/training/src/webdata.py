"""Data for the FlowCast page and the release examples (after the exam). usage: python3 webdata.py <ckpt> <exam_tag>
Examples: a fixed seeded draw of exam basins within strata (snow / arid / humid / other), never hand-picked."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
import evalreal as ER

ck, tag = sys.argv[1], sys.argv[2]
BL = f'{WATER_DATA}/data/proc/baselines'
a = wd.attributes('camels')
ex_ids = wd.basins('camels', split='exam')
fc = pd.read_parquet(f'{ROOT}/results/{tag}.parquet')
bl = pd.read_parquet(f'{BL}/flow_scores.parquet')
nn = pd.read_parquet(f'{BL}/sims/nn_dev.parquet')
nwm = pd.read_parquet(f'{WATER_DATA}/data/proc/nwm/daily_local.parquet')
gr = pd.read_parquet(f'{BL}/sims/gr4j_gauged.parquet')
fv = fc[fc.period == 'val'].set_index('basin')
# ---- map: every exam basin
mp = []
for b in ex_ids:
    if b not in fv.index:
        continue
    r = {'id': b, 'lat': round(float(a.loc[b, 'gauge_lat'] if 'gauge_lat' in a else wd.spec(b)['lat']), 3),
         'lon': round(float(a.loc[b, 'gauge_lon'] if 'gauge_lon' in a else wd.spec(b)['lon']), 3),
         'fc': round(float(fv.loc[b, 'nse']), 3), 'cov': round(float(fv.loc[b, 'cov90']), 3)}
    for m in ('nn_dev', 'nwm', 'gr4j'):
        g = bl[(bl.basin == b) & (bl.period == 'val') & (bl.method == m)]
        r[m] = round(float(g.nse.iloc[0]), 3) if len(g) else None
    mp.append(r)
# ---- examples: seeded draw within strata
rng = np.random.default_rng(20261118)
ar = a['aridity_FAO_PM']; fs = a['frac_snow']
strata = {'snow': [b for b in ex_ids if fs[b] > 0.3],
          'arid': [b for b in ex_ids if ar[b] > 1 and fs[b] <= 0.3],
          'humid': [b for b in ex_ids if ar[b] < 0.6 and fs[b] <= 0.3],
          'between': [b for b in ex_ids if 0.6 <= ar[b] <= 1 and fs[b] <= 0.3]}
S0, S1, T0, T1 = '1993-10-01', '1996-10-01', '1994-10-01', '1996-09-30'
exs = []
for sname, pool in strata.items():
    pool = [b for b in pool if b in fv.index]
    for b in rng.choice(pool, 4, replace=False):
        sp = wd.spec(b)
        f = wd.forcing(b, S0, S1)
        obs = wd.streamflow(b, 'caravan').Q_mm.reindex(pd.date_range(T0, T1))
        att = spec_attrs(sp); att['elev_min_m'] = sp.get('elev_min_m') or att['elev_mean_m'] * 0.7
        att['elev_max_m'] = sp.get('elev_max_m') or att['elev_mean_m'] * 1.3
        g = b.split('_')[1]
        e = {'id': b, 'name': sp.get('name'), 'stratum': sname, 'lat': sp['lat'], 'lon': sp['lon'],
             'area_km2': round(att['area_km2'], 1), 'attrs': {k: round(float(v), 4) for k, v in att.items()},
             'F': np.round(f[DYN].astype(float).interpolate(limit_direction='both').values, 2).tolist(),
             'doy': f.index.dayofyear.tolist(), 'start': T0,
             'obs': [None if not np.isfinite(x) else round(float(x), 3) for x in obs.values],
             'nn_dev': np.round(nn[b].reindex(obs.index).values.astype(float), 3).tolist(),
             'gr4j': np.round(gr[b].reindex(obs.index).values.astype(float), 3).tolist(),
             'nwm': (np.round((nwm[g] * 86.4 / sp['area_km2']).reindex(obs.index).values.astype(float), 3).tolist()
                     if g in nwm.columns else None),
             'nse_val_decade': {'flowcast': round(float(fv.loc[b, 'nse']), 3)}}
        for m in ('nn_dev', 'nwm', 'gr4j'):
            gg = bl[(bl.basin == b) & (bl.period == 'val') & (bl.method == m)]
            e['nse_val_decade'][m] = round(float(gg.nse.iloc[0]), 3) if len(gg) else None
        exs.append(e)
json.dump({'map': mp, 'examples': exs}, open(f'{ROOT}/web/data.json', 'w'), separators=(',', ':'))
print('examples', [(e['id'], e['stratum']) for e in exs], 'map', len(mp))
# ---- release examples: first example of 3 strata
os.makedirs(f'{ROOT}/release/examples', exist_ok=True)
rel = []
for e in [x for x in exs if x['stratum'] == 'snow'][:1] + [x for x in exs if x['stratum'] == 'arid'][:1] + [x for x in exs if x['stratum'] == 'humid'][:1]:
    f = wd.forcing(e['id'], S0, S1)[DYN]
    f.to_csv(f'{ROOT}/release/examples/{e["id"]}_forcing.csv', float_format='%.3f')
    pd.DataFrame({'Q_mm': e['obs']}, index=pd.date_range(T0, T1)).to_csv(f'{ROOT}/release/examples/{e["id"]}_obs.csv')
    rel.append({'id': e['id'], 'name': e['name'], 'attrs': e['attrs'], 'start': T0, 'end': T1,
                'forcing_csv': f'{e["id"]}_forcing.csv', 'obs_csv': f'{e["id"]}_obs.csv'})
json.dump(rel, open(f'{ROOT}/release/examples/examples.json', 'w'), indent=1)
