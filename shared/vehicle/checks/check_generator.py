"""Check: is the generated vehicle library realistic?

400 generated light-duty vehicles (car/suv/pickup/van, all architectures) are driven over hot UDDS and HWFET on the
dyno settings (HVAC off) and a wide-open-throttle 0-60 mph run. Their distributions are compared with the real
MY2022-24 EPA test-car list: conventional gasoline vehicles (FTP < 45 mpg, so hybrids are mostly excluded),
hybrids (gasoline FTP >= 45 mpg) and BEVs (charge-depleting UDDS/HWY in MPGe; records reported in kWh/100 mi are
converted). Physical consistency: power-to-mass and 0-60 mph times per class.
Note: simulated UDDS is hot-start while EPA FTP includes a cold start (bag 1), worth about -3 to -5 %.
"""
import os, sys, json
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
from roadsim import simulate, make_trip
from roadsim.generator import generate_vehicles
from roadsim.constants import MPH, MILE, GAL, FUELS, KWH, EPA_KWH_PER_GAL
from roadsim.sim import get_vehicle
from epa_procedures import test_spec, RHO_REF, ETA_CHARGE, ETA_BATT_EQ


def q(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    return dict(n=int(len(x)), p10=round(float(np.percentile(x, 10)), 1), median=round(float(np.median(x)), 1),
                p90=round(float(np.percentile(x, 90)), 1)) if len(x) else None


def fe(spec, cyc):
    veh = get_vehicle(spec)
    o = simulate(spec, make_trip(cyc, T_amb=23.9, rho_air=RHO_REF), init_temps=dict(engine=88.0, tire_warm=1.0, driveline=80.0),
                 init_soc=None if veh.arch in ('conv', 'bev') else 0.5 * (veh.battery.soc_lo + veh.battery.soc_hi), record=False)
    S = o['summary']; mi = S['distance_m'] / MILE
    if veh.arch == 'bev':
        return EPA_KWH_PER_GAL / (o['ledger']['batt_chem_out'] * 1e3 / KWH / mi / ETA_CHARGE)
    if veh.arch.startswith('phev'):
        return None
    g = S['fuel_g'] + o['ledger']['batt_chem_out'] * 1e3 / ETA_BATT_EQ / FUELS[veh.fuel]['lhv'] * 1e3
    return mi / (g / FUELS[veh.fuel]['density'] / GAL)


def zero60(spec):
    v = np.minimum(np.arange(40) * 20.0, 70 * MPH)          # demand far above capability -> WOT
    o = simulate(spec, make_trip(v, T_amb=20), init_temps=dict(engine=88.0, tire_warm=1.0, driveline=80.0),
                 driver={'a_max': 12.0}, check=False)
    va = o['v']
    i = np.argmax(va >= 60 * MPH)
    if va[i] < 60 * MPH:
        return float('nan')
    return float(i - 1 + (60 * MPH - va[i - 1]) / max(va[i] - va[i - 1], 1e-6))


def epa_reference():
    d = pd.read_csv(os.path.join(HERE, '..', 'roadsim', 'refdata', 'epa_candidates.csv'))
    d = d[d['Model Year'] >= 2022]
    d = d.drop_duplicates(subset=['Represented Test Veh Make', 'Represented Test Veh Model', 'Equivalent Test Weight (lbs.)', 'FE_FTP', 'FE_CD_UDDS'])
    gas = d[d.fuels.str.contains('Gasoline', na=False) & d.FE_FTP.notna()]
    conv = gas[gas.FE_FTP < 45]; hev = gas[gas.FE_FTP >= 45]
    bev = d[d.fuels.str.contains('Electricity', na=False) & d.FE_CD_UDDS.notna() & d.FE_CD_HWY.notna() & d.FE_FTP.isna()]
    conv_u = lambda x: np.where(x < 60, EPA_KWH_PER_GAL / (x / 100.0), x)
    return dict(conv=dict(FTP=q(conv.FE_FTP), HWY=q(conv.FE_HWY)), hev=dict(FTP=q(hev.FE_FTP), HWY=q(hev.FE_HWY)),
                bev=dict(UDDS_MPGe=q(conv_u(bev.FE_CD_UDDS.values)), HWY_MPGe=q(conv_u(bev.FE_CD_HWY.values))),
                pw_kW_per_t=q(d['Rated Horsepower'] * 0.7457 / (d['Equivalent Test Weight (lbs.)'] * 0.4536 / 1000)))


def main():
    V = generate_vehicles(400, seed=23, strategy=True)
    V = [v for v in V if v['class'] in ('car', 'suv', 'pickup', 'van')][:400]
    V += [v for v in generate_vehicles(800, seed=29) if v['class'] in ('car', 'suv', 'pickup', 'van')][:400 - len(V)]
    rows = []
    for v in V:
        s = test_spec(v)
        a = v['architecture']
        grp = 'conv' if a == 'conv' else ('bev' if a == 'bev' else ('phev' if a.startswith('phev') else 'hev'))
        veh = get_vehicle(s)
        P = (veh.engine.P_max if veh.engine else 0) + (max((m.P_max for r, m in veh.motors.items() if r in ('P2', 'MG2', 'traction')), default=0) if grp in ('bev',) else 0)
        if grp in ('hev', 'phev'):
            P = veh.engine.P_max / 0.75
        r = dict(cls=v['class'], arch=a, grp=grp, mass=veh.mass, pw=P / 1e3 / (veh.mass / 1e3),
                 udds=fe(s, 'udds'), hwy=fe(s, 'hwfet'), zero60=zero60(s))
        rows.append(r)
    ref = epa_reference()
    sim = {}
    for g in ('conv', 'hev', 'bev'):
        rr = [r for r in rows if r['grp'] == g]
        key = ('UDDS_MPGe', 'HWY_MPGe') if g == 'bev' else ('UDDS_hot', 'HWY')
        sim[g] = {key[0]: q([r['udds'] for r in rr]), key[1]: q([r['hwy'] for r in rr])}
    accel = {c: q([r['zero60'] for r in rows if r['cls'] == c]) for c in ('car', 'suv', 'pickup', 'van')}
    accel_arch = {g: q([r['zero60'] for r in rows if r['grp'] == g]) for g in ('conv', 'hev', 'phev', 'bev')}
    out = dict(check='generated light-duty library vs the real EPA fleet', method=__doc__.strip(), n=len(rows),
               counts={f"{c}|{g}": sum(1 for r in rows if r['cls'] == c and r['grp'] == g) for c in ('car', 'suv', 'pickup', 'van') for g in ('conv', 'hev', 'phev', 'bev')},
               simulated=sim, epa_testcar_MY2022_24=ref, zero_to_60_s_by_class=accel, zero_to_60_s_by_group=accel_arch,
               power_to_mass_kW_per_t_generated=q([r['pw'] for r in rows]),
               reference_zero_to_60_s='typical MY2024: cars 6-9 s, SUVs 6.5-9.5 s, pickups 5.5-8.5 s, minivans 7-8 s, BEVs 3.5-7 s (manufacturer/press figures)',
               rows=rows)
    json.dump(out, open(os.path.join(HERE, 'generator.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'method')}, indent=1))


if __name__ == '__main__':
    main()
