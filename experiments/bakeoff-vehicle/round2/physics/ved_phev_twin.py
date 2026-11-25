"""P0 dev check (plug-ins, VED dev cars only): the bank's plug-in start state.

EnergyNet's bank (training/bank.py init_state) gives every trip that starts moving, and 55-60 % of the others, a WARM
engine (85 C coolant). That is a conventional-car rule. A plug-in with charge in its pack drives on the battery, so its
engine is usually cold; the warm coolant then heats the cabin for free and the simulated plug-in uses too little
battery in the cold (RoadSim CALIBRATION.md H25 / H28 found this on the VED dev twins and fixed it in the twin harness
with the LOGGED engine speed, which a bank cannot have).
Candidate rule (bank recipe, physics unchanged): a plug-in trip that starts with usable charge (start charge fraction
disp > 0.05 of the SOC window) starts with its engine and oil at ambient; a charge-sustaining start keeps the bank's
draw. This script measures the rule on the VED dev plug-in twins (RoadSim's own v2 dev twins, checks/ved_twin__
roadsim-physics-v2.json; VED_11 Ford Energi, VED_537 Volt-type), on their own logged dev trips, with the twin
harness's run_trip (imported unchanged): 'warm' (the bank's current rule for moving starts) vs 'soc' (the candidate).
Dev only. -> round2/physics/ved_phev_twin.json
"""
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.dirname(HERE)
sys.path.insert(0, R2)
import paths as P  # noqa: E402
import numpy as np  # noqa: E402
CHK = os.path.join(P.REPO if hasattr(P, 'REPO') else os.path.dirname(os.path.dirname(P.BAKEOFF)), 'shared', 'vehicle', 'checks')
sys.path.insert(0, CHK)
import check_ved_twin as CV  # noqa: E402
import realdata as rd  # noqa: E402

CARS = ['VED_11', 'VED_537']


def disp_of(spec, arr):
    s = arr['soc'][np.isfinite(arr['soc'])]
    if not len(s) or not (0.02 < float(s[0]) <= 1.0):
        return None
    lo, hi = (spec.get('battery') or {}).get('soc_window') or [0.2, 0.9]
    return (float(s[0]) - lo) / (hi - lo)


def job_for(meta, arr, spec, opt, rule):
    a = dict(arr)
    if rule == 'soc':
        d = disp_of(spec, arr)
        cold = d is not None and d > 0.05
        a['engine_rpm'] = np.zeros(len(arr['v'])) if cold else np.full(len(arr['v']), 1500.0)
        return dict(meta=meta, arr=a, spec=spec, opt=opt, phev_engine_start='logged')
    return dict(meta=meta, arr=a, spec=spec, opt=opt, phev_engine_start='warm')


def main(workers=4):
    T = json.load(open(os.path.join(CHK, 'ved_twin__roadsim-physics-v2.json')))
    opt = T['options']
    twins = {v: T['twins'][v]['spec'] for v in CARS}
    jobs = {'warm': [], 'soc': []}
    for meta, arr in rd.load_trips('dev', label_ok=True, vehicles=CARS):
        for r in jobs:
            jobs[r].append(job_for(meta, arr, twins[meta['vehicle_id']], opt, r))
    out = {}
    with Pool(workers) as pool:
        for r, js in jobs.items():
            rows = [x for x in pool.map(CV.run_trip, js, chunksize=4) if 'error' not in x]
            S = CV.summarize(rows)
            out[r] = {'by_car': {v: {'n': c['n_trips'], 'err_primary': c.get('err_primary')} for v, c in S['by_car'].items()},
                      'cold_trips_below_10C': {}}
            for v in CARS:
                e = np.array([x['err_primary'] for x in rows if x['vehicle_id'] == v and x['T'] < 10 and x.get('err_primary') is not None])
                out[r]['cold_trips_below_10C'][v] = {'n': int(len(e)), 'median': float(np.median(e)) if len(e) else None}
            print(r, json.dumps(out[r]), flush=True)
    json.dump(out, open(os.path.join(HERE, 'ved_phev_twin.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
