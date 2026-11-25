"""Run the harness's vendored WaterSim (HBV + degree-day snow + gamma channel UH) on one harness batch with a given
parameter set. Shared by the physics-in-the-loop and formula arms (copied into each arm folder).

The spec is built from the batch's attributes only: area, elevation mean and range (-> 5 equal-area elevation
bands, as WaterSim builds them from elev_min/max), slope, forest, porosity, ksat, carbonate, urban, irrigation share
(WaterSim's irrigation module runs on it), lat/lon. No reservoirs (their specs are not inputs). PET = the forcing's PET
column. Spin-up = the batch's n_warm warm-up rows. PMULT multiplies the forcing precipitation before WaterSim.
"""
import os
import sys

import numpy as np
import pandas as pd

HARNESS_VENDOR = '/home/claude/bakeoff/river/harness/vendor'
if HARNESS_VENDOR not in sys.path:
    sys.path.insert(0, HARNESS_VENDOR)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import watersim as ws  # noqa: E402

DYN = ['P', 'PET', 'T_mean', 'T_min', 'T_max', 'SRAD', 'vp', 'wind']
FIXED = {'CFR': 0.05, 'CWH': 0.1, 'CH_N': 0.035}
MODEL_KEYS = ['FC', 'LP', 'BETA', 'PERC', 'UZL', 'K0', 'K1', 'K2', 'MAXBAS']
SNOW_KEYS = ['TT', 'TTI', 'CFMAX', 'SFCF', 'TLAPSE', 'PGRAD']


def spec_of(b):
    a = np.asarray(b['attrs'], float)
    em, rg = float(a[1]), float(a[2])
    return {'area_km2': float(a[0]), 'elev_mean_m': em, 'elev_min_m': em - rg / 2, 'elev_max_m': em + rg / 2,
            'slope_mean': float(a[3]), 'forest_frac': float(a[4]), 'soil_porosity': float(a[5]),
            'soil_conductivity': float(a[6]), 'carbonate_frac': float(a[7]), 'irrigation_frac': float(a[9]),
            'urban_frac': float(a[10]), 'lat': float(b['lat']), 'lon': float(b['lon']), 'reservoirs': None}


def run_batch(b, prm):
    """prm: dict with MODEL_KEYS + SNOW_KEYS + CH_VEL + PMULT. Returns Q (n_target,) mm/day (WaterSim ts['Q'])."""
    nw, nt = int(b['n_warm']), int(b['n_target'])
    F = np.asarray(b['forcing'], np.float64)[:nw + nt]
    idx = pd.DatetimeIndex(np.asarray(b['dates'])[:nw + nt])
    df = pd.DataFrame(F, index=idx, columns=DYN)
    df['P'] = np.maximum(df['P'].values, 0.0) * float(prm['PMULT'])
    params = {k: float(prm[k]) for k in MODEL_KEYS + SNOW_KEYS}
    params.update({'CFR': FIXED['CFR'], 'CWH': FIXED['CWH'], 'CH_VEL': float(prm['CH_VEL']), 'CH_N': FIXED['CH_N']})
    out = ws.simulate(spec_of(b), (df, {'srad_is_net': True}), idx[nw], idx[nw + nt - 1], model='hbv',
                      params=params, snow='degree_day', routing='uh', reservoirs=[], spinup=nw, labels=False,
                      pet='forcing')
    q = out['ts']['Q'].values
    assert len(q) == nt
    return q
