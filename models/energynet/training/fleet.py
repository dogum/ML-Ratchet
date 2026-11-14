"""Training/validation fleets: roadsim.generator vehicles, broadened to cover older and less efficient vehicles
(VED is MY2005-2017; dev twins needed engine efficiency scales 0.72-1.07 to match their own lab mpg), plus
articulated buses. Each light-duty vehicle gets simulated label values (EPA-style):
  city  = FTP-75 approx: 0.43 x cold UDDS + 0.57 x hot UDDS (fuel per mile); hwy = hot HWFET
  adjusted (window sticker) by the EPA derived 5-cycle formulas: 1/(0.003259 + 1.1805/city), 1/(0.001376 + 1.3466/hwy)
  BEV: wall energy = DC / 0.88, adjusted MPGe = 0.7 x unadjusted
  PHEV: charge-sustaining fuel (start at the bottom of the SOC window) and charge-depleting electricity (start full)
Labels are stored as energy per km in target units (MJ/km): fuel g x 0.042, wall kWh x 3.6.

python fleet.py <name> <n> <seed>     (resumable chunks of 50 in data/fleet_<name>/)
"""
import sys, os, json, copy, time
from common import *
import numpy as np
from roadsim import simulate, make_trip
from roadsim.generator import generate_vehicle
from roadsim.vehicle import fill_spec

MI = 1609.344; GAL_L = 3.785411784
_TR = {}


def _trip(c):
    if c not in _TR: _TR[c] = make_trip(c, T_amb=24.0)
    return _TR[c]


def broaden(s, rng):
    """widen the generator's distribution (documented in the model card)."""
    s = copy.deepcopy(s)
    cls, arch = s['class'], s['architecture']
    hd = cls in HD
    if s.get('engine'):
        if not hd:
            s['engine']['eff_scale'] = round(float(rng.uniform(0.68, 1.08) if arch == 'conv' else rng.uniform(0.72, 1.10)), 3)
        else:
            s['engine']['eff_scale'] = round(float(rng.uniform(0.88, 1.04)), 3)
    if not hd:
        s['CdA_m2'] = round(s['CdA_m2'] * float(np.exp(rng.normal(0, 0.10))), 4)
        s['Crr'] = round(s['Crr'] * float(np.exp(rng.normal(0, 0.12))), 5)
        if arch == 'conv' and s.get('transmission', {}).get('type') in ('AT', 'DCT', 'CVT') and rng.random() < 0.35:
            n = int(rng.choice([4, 5, 6]))
            s['transmission'] = {'type': 'AT', 'n_gears': n}
    if arch == 'bev' and s.get('motors'):
        for m in s['motors']:
            m['loss_scale'] = round(0.47 * float(rng.uniform(0.7, 1.5)), 3)
    if cls == 'bus' and rng.random() < 0.35:              # articulated (18 m) bus
        s['mass_kg'] = round(s['mass_kg'] * float(rng.uniform(1.35, 1.55)), 1)
        s['payload_kg'] = round(s['payload_kg'] * 1.5, 1)
        s['CdA_m2'] = round(s['CdA_m2'] * 1.05, 3)
        s.setdefault('aux', {})['base_W'] = round(s['aux'].get('base_W', 4000) * 1.4)
        s['aux'].setdefault('HVAC', {})['UA_W_K'] = 1350.0
        for m in s.get('motors') or []: m['max_power_kW'] = round(m['max_power_kW'] * 1.3, 1)
        if s.get('engine'): s['engine']['max_power_kW'] = round(s['engine']['max_power_kW'] * 1.3, 1)
        if s.get('battery') and arch == 'bev': s['battery']['capacity_kWh'] = round(s['battery']['capacity_kWh'] * 1.3)
        s['articulated'] = True
    return s


def per_mile(out, fuel):
    s = out['summary']
    d = s['distance_m'] / MI
    return s['fuel_g'] / d, s['batt_kWh_out'] / d          # g/mi, DC kWh/mi


def lab_labels(spec):
    """simulated label values (MJ/km) for light-duty vehicles; None for heavy duty."""
    cls, arch = spec['class'], spec['architecture']
    if cls in HD: return None
    fuel = (spec.get('engine') or {}).get('fuel', 'gasoline')
    full = fill_spec(spec)
    b = full.get('battery') or {}
    lo, hi = (b.get('soc_window') or [0.15, 0.95])
    W = warm_temps(24)
    r = {}
    def run(c, cold=False, soc=None):
        return simulate(spec, _trip(c), init_temps=None if cold else W, init_soc=soc, record=False, options='roadsim-physics-v1')
    if arch == 'bev':
        u = per_mile(run('udds', soc=0.9), fuel)[1]; h = per_mile(run('hwfet', soc=0.9), fuel)[1]
        for k, dc in (('city', u), ('hwy', h)):
            wall = dc / 0.88; mpge = 33.705 / wall; adj = 0.7 * mpge
            r[f'label_{k}_elec'] = 33.705 / adj * 3.6 / (MI / 1000)        # wall MJ/km
            r[f'unadj_{k}_elec'] = wall * 3.6 / (MI / 1000)
        return r
    soc_cs = (lo + 0.03) if arch.startswith('phev') else None
    c = 0.43 * per_mile(run('udds', cold=True, soc=soc_cs), fuel)[0] + 0.57 * per_mile(run('udds', soc=soc_cs), fuel)[0]
    h = per_mile(run('hwfet', soc=soc_cs), fuel)[0]
    dens = 832.0 if fuel == 'diesel' else 745.0
    for k, gpm, A, B in (('city', c, 0.003259, 1.1805), ('hwy', h, 0.001376, 1.3466)):
        mpg = dens * GAL_L / max(gpm, 1e-3)
        adj = 1 / (A + B / mpg)
        r[f'label_{k}_fuel'] = dens * GAL_L / adj * LHV / (MI / 1000)
        r[f'unadj_{k}_fuel'] = gpm * LHV / (MI / 1000)
    if arch.startswith('phev'):
        u = per_mile(run('udds', soc=hi - 0.01), fuel)[1]; hh = per_mile(run('hwfet', soc=hi - 0.01), fuel)[1]
        for k, dc in (('city', u), ('hwy', hh)):
            wall = max(dc, 0.05) / 0.88
            r[f'label_{k}_elec'] = wall / 0.7 * 3.6 / (MI / 1000)
            r[f'unadj_{k}_elec'] = wall * 3.6 / (MI / 1000)
    return r


def make_vehicle(rng):
    s = broaden(generate_vehicle(rng), rng)
    fill_spec(s)
    return s


if __name__ == '__main__':
    name, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = f'{EN}/data/fleet_{name}'; os.makedirs(out, exist_ok=True)
    CH = 50
    for ci in range((n + CH - 1) // CH):
        p = f'{out}/chunk_{ci:03d}.json'
        if os.path.exists(p): continue
        rng = np.random.default_rng([seed, ci])
        res = []; t0 = time.time()
        for j in range(min(CH, n - ci * CH)):
            for attempt in range(5):
                try:
                    s = make_vehicle(rng); lab = lab_labels(s)
                    break
                except Exception as e:
                    print('retry', e, flush=True)
            s['id'] = f'{name}_{ci:03d}_{j:02d}'
            res.append({'spec': s, 'labels': lab})
        json.dump(res, open(p + '.tmp', 'w'), default=float); os.replace(p + '.tmp', p)
        print('chunk', ci, round(time.time() - t0, 1), flush=True)
