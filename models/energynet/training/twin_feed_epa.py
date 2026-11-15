"""Does the network read RoadSim's own twins of the dev EPA cars right? (DEV only, 15 Nov 2026)
For dev-pool cars (shared/vehicle/checks/epa_dev_pool.json, RoadSim's dev conventional twins), simulate the bank's way
(physics.sim on a dyno spec: cold / hot UDDS at 24 C, HWFET) and compare with the network's prediction from the real
EPA record's inputs (the same inputs the dev gate uses). pred / sim is the network vs the simulator on the same car
and cycle: a gap here is the generator's fleet or the network, not the physics.
python twin_feed_epa.py <rung> [n]  -> work/twin_feed_epa_<rung>.json"""
import _paths as _P
import sys, os, json
sys.path.insert(0, _P.SRC)
from common import *
import numpy as np, pandas as pd
import evaluate as EV, physics as PHY, bank as BK
from ratchet2 import RUNGS, load

name = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
m = load({x['name']: x for x in RUNGS}[name])
B, T, X, E, M = EV.real_set('epa_dev')
p, s, c = m.predict(X, E)
M = M.copy(); M['pred'] = p * B['vr']
pool = json.load(open(os.path.join(_P.VEHICLE, 'checks', 'epa_dev_pool.json')))['vehicles']
rng = np.random.default_rng(3); pick = rng.choice(len(pool), min(n, len(pool)), replace=False)
cy = BK.cycles(); rows = []
for i in pick:
    v = pool[i]; spec = v['spec']
    for test, cyc, cold in (('FTP', 'UDDS', True), ('FTP', 'UDDS', False), ('HWY', 'HWFET', False)):
        mm = M[(M.vehicle_id == v['row']) & (M.test == test) & (M.cycle == cyc) & (M.cold == cold)]
        if not len(mm): continue
        tr = BK._fill(cy[cyc]); N = len(tr['v']); tr = dict(tr)
        tr.update(T_amb=np.full(N, 24.0, np.float32), wind_speed=np.zeros(N, np.float32), grade=np.zeros(N, np.float32))
        o = PHY.sim(spec, tr, dyno=True, init_temps=None if cold else warm_temps(24.0), record=True)
        e = PHY.targets(o, spec, tr['T_amb'], True)[0]
        rows.append(dict(vehicle=v['row'], map=v['map'] if 'map' in v else spec['engine']['bsfc_map'], test=test, cold=cold,
                         sim_MJ=e, pred_MJ=float(mm.pred.iloc[0]), real_MJ_equiv=np.nan))
d = pd.DataFrame(rows); d['pred_sim'] = d.pred_MJ / d.sim_MJ - 1
out = {f'{t}_{"cold" if c else "hot"}': dict(n=int(len(g)), pred_sim=float(g.pred_sim.median())) for (t, c), g in d.groupby(['test', 'cold'])}
out['by_map'] = {f'{mp}_{t}_{"cold" if c else "hot"}': dict(n=int(len(g)), pred_sim=float(g.pred_sim.median())) for (mp, t, c), g in d.groupby(['map', 'test', 'cold'])}
json.dump(dict(rung=name, summary=out), open(f'{_P.WORK}/twin_feed_epa_{name}.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
