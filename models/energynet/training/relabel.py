"""Phase D rung r11 (hybrid / EV bias): window-sticker labels of the simulated fleets by RoadSim's own EPA procedure
emulation (shared/vehicle/checks/epa_procedures.py, read-only import, physics v2 pinned) instead of fleet.py's shortcut.

fleet.py's labels: city = 0.43 cold UDDS + 0.57 hot UDDS per mile, hybrids from the middle of the SOC window with the
battery's net energy ignored, no hot soak. The procedure emulation: FTP-75 with three bags and a 10-minute hot soak,
hybrids preconditioned to their test SOC and corrected for net battery energy (SAE J1711-style), plug-ins in charge
sustaining for the fuel label, tyres and HVAC as on the test. Same 5-cycle adjustment and units as fleet.py.

python relabel.py <fleet> [workers]   -> data/fleet_<fleet>/labels_epa.json  {vehicle id: labels}
python relabel.py --bank <bank> <fleet>   -> adds 'lab_epa' to every chunk of data/bank_<bank>/ (by vehicle index)"""
import sys, os, json, glob
from common import *
import numpy as np
import physics as PHY
import _paths as _P
sys.path.insert(0, os.path.join(_P.VEHICLE, 'checks'))

MI = 1609.344; GAL_L = 3.785411784


def labels(spec):
    import _opts, epa_procedures as EP
    _opts.set(PHY.PHYSICS)
    cls, arch = spec['class'], spec['architecture']
    if cls in HD: return None
    from roadsim.vehicle import fill_spec
    s = PHY.dyno_spec(spec, hvac=False)
    full = fill_spec(s)
    fuel = (s.get('engine') or {}).get('fuel', 'gasoline')
    r = {}
    if arch == 'bev':
        for k, cyc in (('city', 'udds'), ('hwy', 'hwfet')):
            o = EP.hot_cycle(s, cyc)
            wall = o['kWh_100mi_ac'] / 100.0; adj = 0.7 * 33.705 / wall      # AC (wall) kWh/mi = DC / 0.88
            r[f'label_{k}_elec'] = 33.705 / adj * 3.6 / (MI / 1000); r[f'unadj_{k}_elec'] = wall * 3.6 / (MI / 1000)
        return r
    b = full.get('battery') or {}
    lo, hi = (b.get('soc_window') or [0.15, 0.95])
    soc_cs = (lo + 0.03) if arch.startswith('phev') else None
    city = EP.ftp(s, soc0=soc_cs)['FE']; hwy = EP.hot_cycle(s, 'hwfet', soc0=soc_cs)['FE']
    dens = 832.0 if fuel == 'diesel' else 745.0
    for k, mpg, A, B in (('city', city, 0.003259, 1.1805), ('hwy', hwy, 0.001376, 1.3466)):
        adj = 1 / (A + B / mpg)
        r[f'label_{k}_fuel'] = dens * GAL_L / adj * LHV / (MI / 1000)
        r[f'unadj_{k}_fuel'] = dens * GAL_L / mpg * LHV / (MI / 1000)
    if arch.startswith('phev'):
        for k, cyc in (('city', 'udds'), ('hwy', 'hwfet')):
            o = EP.hot_cycle(s, cyc, soc0=hi - 0.01)
            kwh_mi = o['batt_J'] / 3.6e6 / (o['o']['summary']['distance_m'] / MI)
            wall = max(kwh_mi, 0.05) / 0.88
            r[f'label_{k}_elec'] = wall / 0.7 * 3.6 / (MI / 1000); r[f'unadj_{k}_elec'] = wall * 3.6 / (MI / 1000)
    return r


def one(path):
    out = {}
    for V in json.load(open(path)):
        try:
            out[V['spec']['id']] = labels(V['spec'])
        except Exception as e:
            print('fail', V['spec']['id'], e, flush=True); out[V['spec']['id']] = None
    print('done', os.path.basename(path), flush=True)
    return out


def main_fleet(fleet, workers):
    fs = sorted(glob.glob(f'{EN}/data/fleet_{fleet}/chunk_*.json'))
    from multiprocessing import Pool
    res = {}
    with Pool(workers) as p:
        for r in p.imap_unordered(one, fs): res.update(r)
    json.dump(res, open(f'{EN}/data/fleet_{fleet}/labels_epa.json', 'w'))


def main_bank(bank, fleet):
    L = json.load(open(f'{EN}/data/fleet_{fleet}/labels_epa.json'))
    for cp in sorted(glob.glob(f'{EN}/data/bank_{bank}/chunk_*.npz')):
        ci = int(cp[-7:-4]); D = dict(np.load(cp))
        lab = np.full_like(D['lab'], np.nan)
        for i, vid in enumerate(D['vid']):
            sid = f'{fleet}_{ci:03d}_{int(vid) - ci * 100:02d}'
            l = L.get(sid) or {}
            lab[i] = [l.get('label_city_fuel', np.nan), l.get('label_hwy_fuel', np.nan), l.get('label_city_elec', np.nan), l.get('label_hwy_elec', np.nan)]
        D['lab_epa'] = lab.astype(np.float32)
        np.savez_compressed(cp + '.tmp.npz', **D); os.replace(cp + '.tmp.npz', cp)
    print('bank', bank, 'lab_epa added')


if __name__ == '__main__':
    if sys.argv[1] == '--bank':
        main_bank(sys.argv[2], sys.argv[3])
    else:
        main_fleet(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
