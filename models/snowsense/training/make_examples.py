"""Release examples: a fixed random draw (seed 20261118) of one exam station from each of three regions.
Writes release/examples/<id>.csv (date, P, T_mean, T_min, T_max, SWE_obs start-of-day) for WY2016-2025 plus the
preceding Aug-Sep, and examples/stations.json (lat, elev, name)."""
import json
import numpy as np
from common import *
from evalset import station_set

meta = wd().snotel_meta().set_index('station_id')
groups = {'Pacific': ('WA', 'OR', 'CA'), 'Central Rockies': ('CO', 'UT', 'WY'), 'Northern Rockies': ('ID', 'MT')}
rng = np.random.default_rng(SEED + 7)
out = {}
ex = station_set('exam')
for reg, states in groups.items():
    c = [s for s in ex if meta.loc[s[0], 'state'] in states]
    sid, lat, elev, ok = c[rng.integers(len(c))]
    d = wd().snotel(sid, '2015-08-01', '2025-10-01')[['P', 'T_mean', 'T_min', 'T_max', 'SWE']]
    d.rename(columns={'SWE': 'SWE_obs'}).round(2).to_csv(f'{ROOT}/release/examples/{sid}.csv', index_label='date')
    out[sid] = {'name': meta.loc[sid, 'name'], 'state': meta.loc[sid, 'state'], 'region': reg, 'lat': lat, 'elev_m': elev}
json.dump(out, open(f'{ROOT}/release/examples/stations.json', 'w'), indent=1)
print(out)
