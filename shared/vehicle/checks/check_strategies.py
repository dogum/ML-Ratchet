"""Check: hybrid energy-management strategies compared on UDDS and HWFET (hot start, HVAC off, dyno road load).

Main vehicle: 2024 Toyota Camry Hybrid LE (power-split; EPA validation spec). Also the 2024 Hyundai Elantra Hybrid
(parallel P2) and the 2024 Honda Accord Hybrid (modelled as series). Each strategy starts at the same SOC (window
middle); fuel is corrected for the net battery energy with the same 1/(0.33 x 0.90) equivalence as the EPA emulation,
and the SOC change is reported so the correction size is visible. DP (offline, knows the whole cycle) ends at its start
SOC by construction (terminal penalty).
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
from roadsim import simulate, make_trip, list_strategies
from roadsim.sim import get_vehicle
from roadsim.constants import MILE, GAL, FUELS
from epa_procedures import test_spec, ETA_BATT_EQ, RHO_REF

V = {v['name']: v for v in json.load(open(os.path.join(HERE, 'epa_vehicles.json')))}
PICK = ['2024 TOYOTA CAMRY HYBRID LE', '2024 HYUNDAI Elantra Hybrid', '2024 HONDA ACCORD']


def run(spec, cyc, strat):
    veh = get_vehicle(spec)
    soc0 = 0.5 * (veh.battery.soc_lo + veh.battery.soc_hi)
    t = time.perf_counter()
    o = simulate(spec, make_trip(cyc, T_amb=23.9, rho_air=RHO_REF), strategy=strat, init_soc=soc0,
                 init_temps=dict(engine=88.0, tire_warm=1.0, driveline=80.0))
    wall = time.perf_counter() - t
    S = o['summary']; L = o['ledger']
    Eb = L['batt_chem_out'] * 1e3
    g = S['fuel_g'] + Eb / ETA_BATT_EQ / FUELS['gasoline']['lhv'] * 1e3
    mi = S['distance_m'] / MILE
    on = o['eng_on'][1:] > 0.5
    starts = int(np.sum(np.diff(o['eng_on']) > 0.5))
    P = o['eng_power'][1:][on].astype(float); F = o['fuel_rate'][1:][on].astype(float)
    eff = float(P.sum() / (F.sum() / 1e3 * FUELS['gasoline']['lhv'])) if F.sum() > 0 else 0.0
    return dict(mpg_soc_corrected=round(mi / (g / FUELS['gasoline']['density'] / GAL), 2),
                mpg_raw=round(mi / (S['fuel_L'] / GAL), 2) if S['fuel_L'] > 0 else None,
                dsoc=round(S['soc_end'] - S['soc_start'], 4), engine_on_frac=round(float(on.mean()), 3), engine_starts=starts,
                mean_engine_eff=round(eff, 4), trace_rms_mps=round(S['trace_rms_mps'], 3), closure=L['closure_rel'], wall_s=round(wall, 3))


def main():
    out = dict(check='hybrid strategy comparison on UDDS/HWFET', method=__doc__.strip(), vehicles={})
    for name in PICK:
        v = V[name]; spec = test_spec(v['spec']); arch = spec['architecture']
        res = {}
        for cyc in ('udds', 'hwfet'):
            res[cyc] = {}
            for s in list_strategies(arch):
                res[cyc][s] = run(spec, cyc, s)
                print(name, cyc, s, res[cyc][s], flush=True)
            best = res[cyc]['dp']['mpg_soc_corrected']
            for s in res[cyc]:
                res[cyc][s]['pct_of_dp'] = round(100 * res[cyc][s]['mpg_soc_corrected'] / best, 1)
        out['vehicles'][name] = dict(architecture=arch, epa={'FTP': v['epa'].get('FTP'), 'HWY': v['epa'].get('HWY')}, results=res)
    json.dump(out, open(os.path.join(HERE, 'strategies_hev.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
