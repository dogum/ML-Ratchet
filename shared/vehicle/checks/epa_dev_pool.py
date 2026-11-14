"""A larger EPA *dev* pool of conventional gasoline vehicles for the v2 calibration (engine efficiency tilt and spread,
cold FTP, SC03). Writes epa_dev_pool.json.

Source: RoadData's EPA test-car table (data/proc/epa/testcars_vehicles.parquet), rows whose model family is 'dev' in
the frozen split (data/proc/splits.json). Excluded: every family of a held-out vehicle of checks/epa_vehicles.json
(FORD|TRANSIT, TESLA|MODEL Y, VOLKSWAGEN|ID.4 are RoadData-dev families with a held-out check vehicle), and the
families of the check's own dev vehicles (they are in check_cycles already). Kept: MY2018 or later, gasoline,
FTP and HWY results, target ABC, rated power and displacement. At most 6 test vehicles per family, drawn with a fixed
seed (never by results). Specs are built exactly as checks/epa_vehicles.py builds a conventional vehicle (same map
rule, gear sets, N/V and road-load handling); turbocharging is only known from the rule P/displacement > 85 kW/L.
"""
import os, sys, json, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VEH, 'roadsim')); sys.path.insert(0, VEH)
from roadsim.vehicle import GEAR_SETS, spec_from_abc
from roadsim.constants import LB, MPH

SEED = 20261114
PER_FAMILY = 6
PICKUP = ('F150', 'F-150', 'SILVERADO', 'SIERRA', 'RAM', 'TACOMA', 'TUNDRA', 'COLORADO', 'CANYON', 'FRONTIER', 'RANGER',
          'GLADIATOR', 'TITAN', 'RIDGELINE', 'MAVERICK', 'SANTA CRUZ', 'K10', 'C10', 'K15', 'C15')
VAN = ('SIENNA', 'ODYSSEY', 'PACIFICA', 'CARNIVAL', 'TRANSIT', 'EXPRESS', 'SAVANA', 'PROMASTER', 'SPRINTER', 'VOYAGER')


def map_rule(disp, P):
    turbo = P / disp > 85
    if turbo:
        return 'gas_turbo_honda_15' if disp < 2.2 else 'gas_turbo_ford_27'
    if disp >= 4.5:
        return 'gas_v8_ford_73'
    if disp >= 3.0:
        return 'gas_v6_gm_43'
    return 'gas_na_toyota_a25a'


def trans_of(t, n):
    t = str(t)
    if 'Continuously' in t:
        return 'CVT', n
    if 'Automated Manual' in t or t == 'Manual':
        return 'DCT', n
    return 'AT', n


def spec_of(r):
    model = str(r.model).upper()
    cls = 'car' if r.veh_type == 'Car' else ('pickup' if any(k in model for k in PICKUP) else 'van' if any(k in model for k in VAN) else 'suv')
    mass = float(r.etw_lb) * LB
    disp = float(r.displacement_L); P = float(r.rated_hp) * 0.7457
    nv = float(r.nv_ratio); axle = float(r.axle_ratio) if pd.notna(r.axle_ratio) else 0.0
    spec = {'id': 'epapool_%s' % r.name, 'name': f'{int(r.model_year)} {r.make} {r.model}', 'class': cls, 'architecture': 'conv',
            'mass_kg': mass, 'rot_mass_frac': 0.015, 'tire': {'K_t': 0.0, 'temp_effects': False}}
    rr = 0.33 if cls == 'car' else 0.36 if cls in ('suv', 'van') else 0.39
    spec['wheel_radius_m'] = rr
    k_nv = 2 * math.pi * rr / (60 * MPH)
    m = map_rule(disp, P)
    spec['engine'] = dict(fuel='gasoline', displacement_L=disp, max_power_kW=P, bsfc_map=m)
    n = int(r.gears) if pd.notna(r.gears) else 6
    t, n = trans_of(r.trans_type, n)
    if t == 'CVT':
        top = 0.40
        fd = axle if axle > 2 else nv * k_nv / top
        spec['transmission'] = dict(type='CVT', ratio_range=[2.6, nv * k_nv / fd])
        spec['final_drive'] = fd
    else:
        gears = GEAR_SETS.get((t, n)) or GEAR_SETS.get(('AT', n)) or list(np.round(np.geomspace(4.6, 0.65, n), 3))
        spec['transmission'] = dict(type=t, gears=gears)
        spec['final_drive'] = nv * k_nv / gears[-1]
    return spec_from_abc(spec, float(r.A_lbf), float(r.B_lbf_mph), float(r.C_lbf_mph2), mass_kg=mass)


def build():
    import realdata as rd
    e = rd.epa_testcars()
    V = json.load(open(os.path.join(HERE, 'epa_vehicles.json')))
    cand = pd.read_csv(os.path.join(VEH, 'roadsim', 'refdata', 'epa_candidates.csv'))
    excl = set()
    for v in V:
        c = cand.loc[v['row']]
        excl |= set(e[(e.model_year == c['Model Year']) & (e.test_vehicle_id == c['Test Vehicle ID'])].family)
    d = e[(e.split == 'dev') & (e.architecture == 'conv') & (e.fuel == 'gasoline') & (e.model_year >= 2018)
          & e.FTP_mpg.notna() & e.HWY_mpg.notna() & e.A_lbf.notna() & e.C_lbf_mph2.notna() & e.rated_hp.notna()
          & e.displacement_L.notna() & e.nv_ratio.notna() & e.etw_lb.notna() & ~e.family.isin(excl)]
    d = d.drop_duplicates(subset=['model_year', 'make', 'model', 'displacement_L', 'trans_type', 'gears', 'etw_lb', 'FTP_mpg'])
    rng = np.random.default_rng(SEED)
    keep = []
    for fam, g in d.groupby('family', sort=True):
        idx = list(g.index)
        if len(idx) > PER_FAMILY:
            idx = list(rng.choice(idx, PER_FAMILY, replace=False))
        keep += idx
    out = []
    for i in sorted(keep):
        r = d.loc[i]
        try:
            s = spec_of(r)
        except Exception as ex:
            print('skip', i, ex); continue
        epa = {}
        for k, c in (('FTP', 'FTP_mpg'), ('HWY', 'HWY_mpg'), ('US06', 'US06_mpg'), ('SC03', 'SC03_mpg'), ('COLD', 'FTP_cold_mpg')):
            if pd.notna(r[c]):
                epa[k] = float(r[c])
        out.append(dict(row=str(i), name=s['name'], family=r.family, spec=s, epa=epa, split='dev', source='RoadData EPA dev family'))
    json.dump(dict(about=__doc__.strip(), excluded_families=sorted(excl), vehicles=out),
              open(os.path.join(HERE, 'epa_dev_pool.json'), 'w'), indent=1)
    maps = pd.Series([o['spec']['engine']['bsfc_map'] for o in out]).value_counts().to_dict()
    print(len(out), 'vehicles,', len({o['family'] for o in out}), 'families; maps', maps,
          '; cold', sum('COLD' in o['epa'] for o in out), 'SC03', sum('SC03' in o['epa'] for o in out), 'US06', sum('US06' in o['epa'] for o in out))
    return out


if __name__ == '__main__':
    build()
