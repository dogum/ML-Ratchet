"""P0 dev check (ICCT, dev truck only): is the trucks' dev gap in RoadSim or in the bank?

The bank never runs a heavy-duty vehicle on a dynamometer (training/bank.py draw_route: heavy duty is 80 % synthetic
road drives and 20 % cycles driven on the road), while the ICCT records are chassis-dynamometer tests (dyno flag 1).
Twin of the dev truck BIC-US (the only ICCT dev truck): RoadSim's Class 8 sleeper reference (ref_class8_sleeper_diesel)
at the test mass, on its three dev cycles with traces, at 22 C, run (a) as the bank runs heavy duty (on the road:
physics.sim(dyno=False), plus physics.targets' on-road version) and (b) as a dynamometer test (physics.sim(dyno=True):
dyno spec, no tyre-temperature effects, no on-road load, HVAC off). Fuel g/km vs the published value.
Dev only: the exam trucks (BIC-EU, AVG-EU) are never read.
-> round2/physics/icct_twin.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import physics as PHY  # noqa: E402
import realdata as rd  # noqa: E402
import routes  # noqa: E402
from bank import TR_KEYS, _fill  # noqa: E402
from roadsim import load_vehicle  # noqa: E402

WARM = {'engine': 88.0, 'tire_warm': 1.0, 'driveline': 80.0, 'cabin': 21.0}


def main(preset=None):
    if preset:
        PHY.PHYSICS = preset
    t = rd.hd_data()['icct_trucks']
    t = t[t.cycle_available & (t.split == 'dev')]
    cyc = rd.cycles()
    rows = []
    for r in t.itertuples():
        tr = _fill({k: cyc[r.cycle_id][1][k] for k in TR_KEYS if k in cyc[r.cycle_id][1]}); N = len(tr['v'])
        tr['T_amb'] = np.full(N, 22.0, np.float32); tr['wind_speed'] = np.zeros(N, np.float32)
        tr['rho_air'] = routes.rho_air(tr['T_amb'], np.zeros(N)).astype(np.float32)
        spec = load_vehicle('ref_class8_sleeper_diesel')
        m0 = float(spec['mass_kg']) + float(spec.get('payload_kg') or 0)
        spec['mass_kg'] = float(r.test_mass_kg); spec['payload_kg'] = 0.0
        dist = float(np.sum(tr['v'])) / 1000.0
        res = {}
        for mode, dyno in (('road', False), ('dyno', True)):
            o = PHY.sim(spec, tr, dyno=dyno, init_temps=dict(WARM), record=True, rng=0)
            g = float(np.nansum(o['fuel_rate']))
            res[mode] = g / dist
        rows.append(dict(truck=r.truck, cycle=r.cycle, test_mass_kg=r.test_mass_kg, ref_mass_kg=m0, measured_g_km=r.g_per_km,
                         twin_road_g_km=res['road'], twin_dyno_g_km=res['dyno'],
                         err_road=res['road'] / r.g_per_km - 1, err_dyno=res['dyno'] / r.g_per_km - 1,
                         road_over_dyno=res['road'] / res['dyno'] - 1))
    d = pd.DataFrame(rows)
    out = {'preset': PHY.PHYSICS, 'rows': d.to_dict('records'),
           'median_err_road': float(d.err_road.median()), 'median_err_dyno': float(d.err_dyno.median()),
           'median_road_over_dyno': float(d.road_over_dyno.median())}
    print(d.round(3).to_string())
    print({k: round(v, 4) for k, v in out.items() if k.startswith('median')})
    json.dump(out, open(os.path.join(HERE, f'icct_twin{"__" + preset if preset else ""}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
