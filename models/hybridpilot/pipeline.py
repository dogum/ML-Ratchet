"""Run HybridPilot (and RoadSim's built-in strategies) on a contract trip with a vehicle spec.
Needs the RoadSim package (Phase 1) on the path for the powertrain model; the network and features are numpy only.

    from pipeline import compare
    rows = compare(spec_dict, trip_arrays, init_soc=0.6, warm=True, strategies=('hybridpilot', 'aecms', 'rule'))
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# RoadSim lives in the repo's shared vehicle platform (or set ROADSIM_PATH / ML_RATCHET_VEHICLE)
_VEH = os.environ.get('ML_RATCHET_VEHICLE') or os.path.join(HERE, '..', '..', 'shared', 'vehicle')
for p in (os.environ.get('ROADSIM_PATH', ''), os.path.join(_VEH, 'roadsim')):
    if p and os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))
import hybridpilot as HP

RULE = {'hev_powersplit': 'powersplit_rule', 'hev_parallel_p2': 'parallel_rule', 'hev_series': 'series_thermostat',
        'phev_powersplit': 'phev_cdcs', 'phev_series': 'phev_cdcs'}
WARM = {'engine': 85.0, 'driveline': 70.0, 'tire_warm': 1.0, 'cabin': 21.0}


def starts(on):
    on = np.asarray(on) > 0.5
    return int(np.sum(on[1:] & ~on[:-1]))


def compare(spec, trip, init_soc=None, warm=True, strategies=('hybridpilot', 'aecms', 'rule'), **hp_params):
    from roadsim import simulate
    from roadsim.sim import get_vehicle
    arch = get_vehicle(spec).arch
    rows = []
    for s in strategies:
        st = HP.make_roadsim_strategy(**hp_params) if s == 'hybridpilot' else (RULE[arch] if s == 'rule' else s)
        o = simulate(spec, trip, strategy=st, init_soc=init_soc, init_temps=dict(WARM) if warm else None, record=True, options='roadsim-physics-v1')
        S = o['summary']
        rows.append(dict(strategy=s if s != 'rule' else RULE[arch], fuel_g=S['fuel_g'], L_per_100km=S.get('L_per_100km'),
                         batt_kWh_out=S['batt_kWh_out'], soc_start=S['soc_start'], soc_end=S['soc_end'],
                         engine_starts=starts(o['eng_on']), distance_km=S['distance_m'] / 1e3, wall_s=S['wall_s']))
    return rows


def load_example(trip_id):
    z = np.load(os.path.join(HERE, 'examples', trip_id + '.npz'))
    trip = {k: z[k].astype(float) for k in z.files}
    trip['surface'] = np.nan_to_num(trip['surface']).astype(int)
    for k in ('grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air', 'elev'):
        trip[k] = np.nan_to_num(trip[k])
    spec = json.load(open(os.path.join(HERE, 'examples', trip_id + '_twin.json')))
    return spec, trip
