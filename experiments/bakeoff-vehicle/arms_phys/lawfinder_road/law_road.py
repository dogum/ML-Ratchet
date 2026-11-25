"""Law 5 config: RoadSim's trip-energy law for conventional vehicles (fuel energy per km of a real trip).

Built in the Vehicle Bake-off (agent C, 23 Nov 2026) on the bake-off's frozen RoadSim bank (physics roadsim-physics-v2,
EnergyNet v2.1's bank recipe) and dev gate; drop into LawFinder's src/ (see MERGE_NOTES.md). Reads the bake-off
harness (BAKEOFF_HARNESS) for the bank and the dev records; nothing here reads exam data.

The law is one of three the bake-off formula arm uses (ice / hev / bev, plug-ins mix hev and bev); LawFinder's page shows
the conventional one (`road`); the hybrid and battery-electric companions are in road_law_entry.json ("companions").
"""
import os
import sys

import numpy as np
import pandas as pd

BAKEOFF_HARNESS = os.environ.get('BAKEOFF_HARNESS', '/home/claude/bakeoff/vehicle/harness')
HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'road'
TITLE = 'Trip fuel energy (conventional vehicles)'
VERSIONS = ['v2']                                     # roadsim-physics-v2 (EnergyNet v2.1's pinned physics)
VARS = ['p', 'n', 't', 'm', 'T', 'c', 'd', 'dyn']      # the released law uses a subset (see its 'vars')
VAR_INFO = {
    'p': {'label': 'positive wheel work per km (road load: aero + rolling + climbing + accelerating)', 'symbol': 'p',
          'unit': 'MJ/km', 'range': [0.05, 3.0]},
    'n': {'label': 'negative wheel work per km (braking / descending)', 'symbol': 'n', 'unit': 'MJ/km', 'range': [0.0, 1.5]},
    't': {'label': 'time per km (60 / mean speed in km/h)', 'symbol': 't', 'unit': 'min/km', 'range': [0.5, 6.0]},
    'm': {'label': 'vehicle mass', 'symbol': 'm', 'unit': 't', 'range': [1.0, 40.0]},
    'T': {'label': 'mean ambient temperature', 'symbol': 'T', 'unit': '°C', 'range': [-25, 40]},
    'c': {'label': 'start state: -1 warm, 0 unknown, +1 cold soak', 'symbol': 'c', 'unit': '-', 'range': [-1, 1]},
    'd': {'label': 'trip distance', 'symbol': 'd', 'unit': 'km', 'range': [0.5, 100.0]},
    'dyn': {'label': 'dynamometer test (1) or road trip (0)', 'symbol': 'D', 'unit': '-', 'range': [0, 1]},
}
TARGET = {'name': 'e', 'label': 'trip fuel energy per km', 'symbol': 'e', 'unit': 'MJ/km'}
SEARCH_SECONDS = 1200
SEARCH_KW = {'log_target': True}                     # needs the log_target option (lawsearch_logtarget.patch)


def _harness():
    if BAKEOFF_HARNESS not in sys.path:
        sys.path.insert(0, BAKEOFF_HARNESS)
    sys.path.insert(0, HERE)
    import bake
    import lawvars as LV
    return bake, LV


def bank_ready(ver):
    return os.path.exists(os.path.join(BAKEOFF_HARNESS, 'bank', 'train', 'MANIFEST.json'))


def _frame(V, y):
    d = dict(V); d['d'] = V['dist_km']
    df = pd.DataFrame({k: np.asarray(d[k], float) for k in VARS})
    df['e'] = y / V['dist_km']; df['arch'] = V['arch']; df['cls'] = V['cls']; df['E_phys'] = V['E_phys']
    return df


def load_bank(ver, split):
    """conventional trips of the bake-off bank: train = two tier draws (rng 11, 12), val = the frozen tiers (777)."""
    bake, LV = _harness()
    B = bake.load_bank(split)
    out = []
    for T in ([bake.draw_tiers(B, np.random.default_rng(s)) for s in (11, 12)] if split == 'train' else [bake.val_tiers()]):
        D = bake.sim_table(B, T)
        V = LV.from_table(D['X'])
        ok = (D['y'] > 0.02) & (V['dist_km'] > 0.3) & (D['arch'] == 0)
        out.append(_frame({k: v[ok] for k, v in V.items()}, D['y'][ok]))
    return pd.concat(out, ignore_index=True)


def X_of(df):
    return df[VARS].values.astype(float)


def target(df):
    return df['e'].values.astype(float)


def load_dev():
    """the bake-off dev gate's VED ICE trips (16 dev cars; 645 scored), whole-trip fuel energy y / vr in MJ.
    EPA (mpg per test vehicle, bag-weighted), Mersin (LHV-scaled grams) and ICCT records carry their labels in other
    units; the law's dev score on them is the bake-off's (mean MdAPE over the conventional D16 sets, road_ratchet.json)."""
    bake, LV = _harness()
    Fr = bake.dev_frame()
    X = np.zeros((len(Fr), 151))
    off = {name: int(Fr.index[Fr.recset == name][0]) for name in bake.SETS}
    for name, s, b in bake._real_batches('dev', 'table'):
        X[off[name] + s.start:off[name] + s.start + b['n']] = b['X']
    V = LV.from_table(X)
    y = Fr.y.values * np.where(Fr.recset == 'ved', 1.0 / Fr.vr.values, 1.0)   # whole-trip energy (MJ)
    df = _frame(V, y)
    df['set'] = Fr['set'].values; df['recset'] = Fr.recset.values
    keep = (V['arch'] == 0) & np.isfinite(df.e) & (df.e > 0) & (Fr.recset == 'ved').values & \
        (Fr.y.values > 1 * 0.042)                                    # EnergyNet's VED ICE floor (1 g)
    return df[keep].reset_index(drop=True)


def dev_error(y, yh):
    return float(np.median(np.abs(yh / y - 1)))          # MdAPE as a fraction


def road_load(df, coef):
    """Textbook rival: the road-load estimate (EnergyNet's E_phys, conventional form) per km:
    e = p / eta + k (m / 1.5 t)^0.7 t; coef = (1 / eta, k)."""
    a, k = coef
    return a * df['p'].values + k * (df['m'].values / 1.5) ** 0.7 * df['t'].values


ROAD_LOAD_ENET = (1 / 0.2845, 4642.0 * 60 / 1e6)          # EnergyNet's PHYS_K['ice'] in these units


def fit_road_load(df):
    """(1 / eta, k) by least squares in log space on df (LawFinder fits the textbook constants on dev)."""
    from scipy.optimize import least_squares
    y = np.log(df['e'].values)
    r = least_squares(lambda c: np.log(np.maximum(road_load(df, c), 1e-9)) - y, np.array(ROAD_LOAD_ENET))
    return [float(v) for v in r.x]
