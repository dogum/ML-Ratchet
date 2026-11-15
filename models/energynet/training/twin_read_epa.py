"""EPA twin read (DEV only, 15 Nov 2026): the network's dev EPA errors next to RoadSim's own twins of the same dev test
cars (shared/vehicle/checks/epa_pool__roadsim-physics-v2.json: 235 dev conventional cars, RoadSim's EPA procedure
emulation). Fuel per mile, + = more fuel than the test.
  pred / real   what the dev gate sees
  twin / real   the simulator vs the test: a gap here is the physics
  pred / twin   the network vs the simulator on the same car: a gap here is the generator or the network
python twin_read_epa.py <rung> [--epa-hyb]   -> work/twin_read_epa_<rung>.json"""
import _paths as _P
import sys, os, json
sys.path.insert(0, _P.SRC)
import numpy as np, pandas as pd
import evaluate as EV
from ratchet2 import RUNGS, load

name = sys.argv[1]
m = load({x['name']: x for x in RUNGS}[name])
r, o = EV.epa_metrics(EV.epa_frame(m, 'dev'))
pool = json.load(open(os.path.join(_P.VEHICLE, 'checks', 'epa_dev_pool.json')))['vehicles']
tw = json.load(open(os.path.join(_P.VEHICLE, 'checks', 'epa_pool__roadsim-physics-v2.json')))['vehicles']
rows = []
TK = {'FTP': 'FTP', 'HWY': 'HWY', 'US06': 'US06', 'SC03': 'SC03', 'COLD': 'FTPcold'}
for p, t in zip(pool, tw):
    assert p['name'] == t['name']
    for k, e in (t.get('err_pct') or {}).items():
        if k in TK and e is not None:
            rows.append(dict(vehicle_id=p['row'], test=TK[k], twin_fuel=100.0 / (100.0 + e)))   # twin fuel / real fuel
T = pd.DataFrame(rows)
d = o[o.arch == 'conv'].merge(T, on=['vehicle_id', 'test'])
d['pred_real'] = d.pred / d.y; d['pred_twin'] = d.pred_real / d.twin_fuel
out = {}
for t, g in d.groupby('test'):
    out[t] = dict(n=int(len(g)), pred_real=float(np.median(g.pred_real) - 1), twin_real=float(np.median(g.twin_fuel) - 1),
                  pred_twin=float(np.median(g.pred_twin) - 1),
                  all_dev_pred_real=float(r[f'conv_{t}']['bias']), all_dev_n=int(r[f'conv_{t}']['n']))
json.dump(dict(rung=name, by_test=out), open(f'{_P.WORK}/twin_read_epa_{name}.json', 'w'), indent=1)
print(pd.DataFrame(out).T.round(3).to_string())
