"""Trips: build trip inputs from drive cycles, and read/write the shared contract format (.npz + .json meta)."""
import os, json
import numpy as np
from .constants import MPH

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_CYC = {}

CONTRACT_INPUTS = ['t', 'v', 'grade', 'elev', 'lat', 'lon', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface']
CONTRACT_MEASURED = ['fuel_rate', 'engine_rpm', 'batt_current', 'batt_voltage', 'soc', 'aux_power']


def cycle(name):
    """speed trace (m/s, 1 Hz) of a built-in cycle: udds hwfet us06 sc03 nycc la92 hd_transient gem_55 gem_65 cbd14."""
    if not _CYC:
        z = np.load(os.path.join(DATA, 'cycles.npz'))
        _CYC.update({k: z[k].astype(float) for k in z.files})
    return _CYC[name].copy()


def list_cycles():
    cycle('udds')
    return sorted(k for k in _CYC if not k.endswith('_grade'))


def make_trip(v, T_amb=20.0, grade=0.0, wind_speed=0.0, wind_dir=0.0, heading=0.0, elev=0.0, rho_air=None,
              surface=0, solar=0.0, p_Pa=None):
    """trip-input dict from a speed trace (array or cycle name); scalars are broadcast."""
    if isinstance(v, str):
        name = v
        v = cycle(name)
        if name in ('gem_55', 'gem_65') and np.isscalar(grade) and grade == 0.0:
            grade = cycle(name + '_grade')
    v = np.asarray(v, float)
    N = len(v)
    b = lambda x: np.full(N, float(x)) if np.isscalar(x) else np.asarray(x, float)
    trip = dict(t=np.arange(N, dtype=float), v=v, grade=b(grade), elev=b(elev), T_amb=b(T_amb), wind_speed=b(wind_speed),
                wind_dir=b(wind_dir), heading=b(heading), surface=b(surface).astype(int), solar=b(solar))
    if rho_air is not None:
        trip['rho_air'] = b(rho_air)
    elif p_Pa is not None:
        trip['rho_air'] = b(p_Pa) / (287.05 * (trip['T_amb'] + 273.15))
    if np.all(trip['elev'] == trip['elev'][0]) and np.any(trip['grade'] != 0):
        ds = np.concatenate([[0.0], 0.5 * (v[1:] + v[:-1])])
        trip['elev'] = trip['elev'][0] + np.cumsum(ds * trip['grade'])
    return trip


def load_trip(path):
    """read a contract trip (.npz [+ .json meta]) -> (trip dict, meta dict)."""
    base = path[:-4] if path.endswith('.npz') else path
    z = np.load(base + '.npz')
    trip = {k: z[k].astype(float) for k in z.files}
    meta = json.load(open(base + '.json')) if os.path.exists(base + '.json') else {}
    return trip, meta


def save_trip(path, trip, meta=None):
    base = path[:-4] if path.endswith('.npz') else path
    np.savez_compressed(base + '.npz', **{k: np.asarray(v, np.float32) for k, v in trip.items() if k in CONTRACT_INPUTS + CONTRACT_MEASURED + ['solar']})
    if meta is not None:
        json.dump(meta, open(base + '.json', 'w'), indent=1)


def sim_to_contract(out, fuel='gasoline'):
    """map simulator outputs onto the contract's measured-output names (for twin tests against real logs)."""
    return dict(fuel_rate=out['fuel_rate'], engine_rpm=out['eng_rpm'], batt_current=out['batt_current'],
                batt_voltage=out['batt_voltage'], soc=out['soc'], aux_power=out['aux_power'])
