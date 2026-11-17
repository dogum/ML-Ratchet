"""Page lab: a grid of BuildingSim buildings where insulation, thermal mass and heating system are moved one at a time
from two base buildings (a mid-size office at the Eagle site, 5A; a detached house at the Rat site, 4A), real 2017
site weather, 1 h, W1 intensity (as the bank). Meters stored companded to uint8 for the page; exact labels kept.
usage: python3 labgrid.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import common as C
from simone import simulate_one
from wrappers import load_cal

warnings.filterwarnings('ignore')
INS = [0.4, 0.7, 1.0, 1.5, 2.2]          # multiplier on wall / roof / window U and air leakage
MASS = ['light', 'medium', 'heavy']
SYS = dict(office=['gas', 'heat_pump', 'resistance', 'district'], home=['gas', 'heat_pump', 'resistance'])
OUT = os.path.join(C.ROOT, 'web', 'lab_grid.json')


def base(kind):
    import buildingsim as bs
    if kind == 'office':
        s = bs.generate_buildings(1, seed=4242, primary_use='office', climate_zone='5A')[0]
        s['floor_area_m2'], s['n_floors'], s['year_built'] = 6000.0, 3, 1985
    else:
        s = bs.generate_buildings(1, seed=4243, primary_use='residential_single', climate_zone='4A')[0]
        s['floor_area_m2'], s['n_floors'], s['year_built'] = 190.0, 2, 1978
    return s


def variant(kind, ins, mass, sysname):
    import buildingsim as bs
    f = bs.fill_spec(base(kind))
    for k in ('wall_U', 'roof_U', 'window_U', 'infiltration_ach', 'ach50'):
        if f.get(k) is not None:
            f[k] = float(f[k]) * ins
    f['thermal_mass'] = mass
    h = f['hvac']
    campus = False
    if kind == 'office':
        f['dhw'] = dict(f['dhw'], fuel='district' if sysname == 'district' else ('gas' if sysname == 'gas' else 'electric'),
                        efficiency=1.0 if sysname == 'district' else (0.8 if sysname == 'gas' else 0.95))
        if sysname == 'gas':
            f['system'] = 'packaged_rtu'; h['heat_source'] = None; h['fuel'] = 'gas'
        elif sysname == 'heat_pump':
            f['system'] = 'packaged_rtu'; h['heat_source'] = 'hp'; h['fuel'] = 'electricity'
        elif sysname == 'resistance':
            f['system'] = 'packaged_rtu'; h['heat_source'] = 'elec'; h['fuel'] = 'electricity'
        else:
            f['system'] = 'packaged_rtu'; campus = 'both'
    else:
        if sysname == 'gas':
            f['system'] = 'split_ac_furnace'; h['heat_source'] = None; h['fuel'] = 'gas'
        elif sysname == 'heat_pump':
            f['system'] = 'heat_pump_air'; h['heat_source'] = None; h['fuel'] = 'electricity'
        else:
            f['system'] = 'baseboard_resistance'; h['heat_source'] = 'elec'; h['fuel'] = 'electricity'
    return f, campus


def comp(x):
    x = np.maximum(np.asarray(x, float), 0)
    mx = float(x.max()) if len(x) else 0.0
    q = np.round(np.sqrt(x / mx) * 254).astype(np.uint8) if mx > 0 else np.zeros(len(x), np.uint8)
    return q, mx


def decomp(q, mx):
    return (np.asarray(q, float) / 254) ** 2 * mx


def main():
    import base64
    cal = load_cal()
    cal = dict(cal, campus=dict(cal['campus'], eff_sd=0.0))           # lab: the median distribution factor
    items = []
    for kind, site in (('office', 'Eagle'), ('home', 'Rat')):
        for si, sysname in enumerate(SYS[kind]):
            for ii, ins in enumerate(INS):
                for mi, mass in enumerate(MASS):
                    f, campus = variant(kind, ins, mass, sysname)
                    rng = np.random.default_rng(7)
                    r = simulate_one(f, rng, cal, campus=campus, wx_key=('site', site, 2017))
                    met = {}
                    for ch, x in r['meters'].items():
                        q, mx = comp(x)
                        met[ch] = dict(b64=base64.b64encode(q.tobytes()).decode(), max=mx)
                    items.append(dict(kind=kind, sys=sysname, ins=ins, mass=mass, site=site, meters=met,
                                      y={k: (None if not np.isfinite(v) else float(v)) for k, v in r['y'].items()},
                                      cls=r['cls'], info={k: r['info'][k] for k in ('UA', 'tau_h', 'A', 'year', 'EUI', 'system')}))
                    print(kind, sysname, ins, mass, r['cls'], 'UA %.0f tau %.1f' % (r['info']['UA'], r['info']['tau_h']), flush=True)
    json.dump(dict(INS=INS, MASS=MASS, SYS=SYS, items=items), open(OUT, 'w'))


if __name__ == '__main__':
    main()
