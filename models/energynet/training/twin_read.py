"""Twin test (DEV only, method.md step 5): does the network read the dev cars' RoadSim twins as it reads the real cars?
For every dev VED trip: the network's prediction (real features, as in the dev gate), the logged energy, and the energy of
the car's label-matched RoadSim v2 twin on the same trip through the OBD label model (shared/vehicle/checks/
ved_twin__roadsim-physics-v2.json, written by the RoadSim calibration passes on dev cars only).
  pred / real   what the dev gate sees
  pred / twin   the network vs the simulator on the same car: a gap here is the prior / generator, not the physics
  twin / real   the simulator vs reality: a gap here is the physics
python twin_read.py <rung>   -> work/twin_read_<rung>.json"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json
sys.path.insert(0, _P.SRC)
import numpy as np, pandas as pd
import evaluate as EV
from ratchet2 import RUNGS, load

name = sys.argv[1]
r = {x['name']: x for x in RUNGS}[name]
m = load(r)
M = EV.predict_real(m, 'ved_dev')
tw = pd.DataFrame(json.load(open(os.path.join(_P.VEHICLE, 'checks', 'ved_twin__roadsim-physics-v2.json')))['trips']).set_index('trip_id')
for k in ('energy_sim_MJ', 'energy_real_MJ'):
    if k not in tw: tw[k] = np.nan
ice = tw.arch.isin(['conv', 'hev_powersplit', 'hev_parallel_p2', 'hev_series']); ev = tw.arch == 'bev'
tw.loc[ice, 'energy_sim_MJ'] = tw.fuel_sim_g * 0.042; tw.loc[ice, 'energy_real_MJ'] = tw.fuel_real_g * 0.042
tw.loc[ev, 'energy_sim_MJ'] = tw.batt_sim_kWh * 3.6; tw.loc[ev, 'energy_real_MJ'] = tw.batt_real_kWh * 3.6
d = M.join(tw[['energy_sim_MJ', 'energy_real_MJ']], on='trip_id')
d = d[(d.y > d.group.map(EV.FLOOR)) & d.energy_sim_MJ.notna() & (d.energy_sim_MJ > 0)]
d['p_r'] = np.log(d.pred / d.y); d['p_t'] = np.log(d.pred / d.energy_sim_MJ); d['t_r'] = np.log(d.energy_sim_MJ / d.energy_real_MJ)
out = {}
for (g, v), x in d.groupby(['group', 'vehicle_id']):
    out[v] = dict(group=g, arch=x.arch.iloc[0], n=int(len(x)), pred_real=float(np.expm1(x.p_r.median())),
                  pred_twin=float(np.expm1(x.p_t.median())), twin_real=float(np.expm1(x.t_r.median())))
grp = {g: dict(n=int(len(x)), pred_real=float(np.expm1(x.p_r.median())), pred_twin=float(np.expm1(x.p_t.median())),
               twin_real=float(np.expm1(x.t_r.median()))) for g, x in d.groupby('group')}
json.dump(dict(rung=name, by_car=out, by_group=grp), open(f'{_P.WORK}/twin_read_{name}.json', 'w'), indent=1)
print(pd.DataFrame(out).T.to_string()); print(pd.DataFrame(grp).T.to_string())
