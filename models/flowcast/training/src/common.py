"""FlowCast shared helpers: paths, Phase-1 imports, feature extraction (identical for sim and real)."""
import os, sys, math, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # training/ (for _paths)
from _paths import WORK, TRAIN, MODEL, WATER, WATER_DATA, SRC, latest   # noqa: E402  (also puts shared/water on sys.path)
os.environ.setdefault('NUMBA_CACHE_DIR', f'{WORK}/.numba')
import numpy as np
import pandas as pd

ROOT = WORK                      # every output (bank/, ckpt/, results/, preds/, web/, release/, ratchet.json)
PHYS = f'{TRAIN}/phys'           # shipped physics versions p0-p4 (read-only)
CONFIGS = f'{TRAIN}/configs'     # shipped rung configs (read-only)

FREE = [('1951-01-01', '1988-09-30'), ('1999-10-01', '2015-09-30')]
VAL = ('1989-10-01', '1999-09-30')
REC = ('2015-10-01', '2023-09-30')
CAL = ('1999-10-01', '2008-09-30')
DEVWIN = ('2000-10-01', '2014-09-30')     # dev gate window; its 365-d warm-up starts 1999-10-01 (all non-exam)

DYN = ['P', 'PET', 'T_mean', 'T_min', 'T_max', 'SRAD', 'vp', 'wind']   # stored raw forcing channels
N_DYN_FEAT = 11
STATIC_NAMES = ['log_area', 'elev_mean_km', 'elev_range_km', 'log_slope_mkm', 'forest_frac', 'soil_porosity',
                'log_ksat_cmh', 'carbonate_frac', 'reservoir_frac', 'irrigation_frac', 'urban_frac',
                'log_p_mean', 'pet_mean', 'log_aridity', 'frac_snow', 'hp_freq', 'lp_freq', 't_mean_c']
# fallbacks for missing real attributes (sim-prior medians)
FALLBACK = {'forest_frac': 0.5, 'soil_porosity': 0.44, 'soil_conductivity_cmh': 1.6, 'carbonate_frac': 0.0,
            'reservoir_storage_frac': 0.0, 'irrigation_frac': 0.0, 'urban_frac': 0.0, 'slope_mkm': 60.0}


def _f(x, d):
    try:
        x = float(x)
        return x if np.isfinite(x) else d
    except Exception:
        return d


def climate_stats(P, PET, T):
    """Climate indices from (noisy) forcing arrays: exactly the same for sim and real basins."""
    P = np.asarray(P, float); PET = np.asarray(PET, float); T = np.asarray(T, float)
    pm = max(np.nanmean(P), 0.05)
    em = max(np.nanmean(PET), 0.05)
    fs = float(np.nansum(P[T < 0]) / max(np.nansum(P), 1e-6))
    return {'p_mean': pm, 'pet_mean': em, 'aridity': em / pm, 'frac_snow': fs,
            'hp_freq': float(np.nanmean(P > 5 * pm)), 'lp_freq': float(np.nanmean(P < 1.0)),
            't_mean': float(np.nanmean(T))}


def spec_attrs(spec):
    """Physical attributes in sim units (slope m/km, ksat cm/h) from a sim or real spec."""
    sl = _f(spec.get('slope_mean'), None)
    if sl is None:
        sl = FALLBACK['slope_mkm']
    elif str(spec.get('slope_unit', '')).startswith('deg'):
        sl = math.tan(math.radians(sl)) * 1000.0
    k = _f(spec.get('soil_conductivity'), None)
    if k is None:
        k = FALLBACK['soil_conductivity_cmh']
    elif spec.get('soil_conductivity_unit') == 'mm/h':
        k = k / 10.0
    em = _f(spec.get('elev_mean_m'), 500.0)
    lo = _f(spec.get('elev_min_m'), em * 0.7)
    hi = _f(spec.get('elev_max_m'), em * 1.3)
    return {'area_km2': max(_f(spec.get('area_km2'), 300.0), 1.0), 'elev_mean_m': em, 'elev_range_m': max(hi - lo, 0.0),
            'slope_mkm': max(sl, 0.5), 'forest_frac': _f(spec.get('forest_frac'), FALLBACK['forest_frac']),
            'soil_porosity': _f(spec.get('soil_porosity'), FALLBACK['soil_porosity']), 'ksat_cmh': max(k, 0.01),
            'carbonate_frac': _f(spec.get('carbonate_frac'), 0.0),
            'reservoir_storage_frac': _f(spec.get('reservoir_storage_frac'), 0.0),
            'irrigation_frac': _f(spec.get('irrigation_frac'), 0.0), 'urban_frac': _f(spec.get('urban_frac'), 0.0)}


def static_vector(attrs, clim):
    a, c = attrs, clim
    return np.array([
        math.log10(a['area_km2']), a['elev_mean_m'] / 1000, a['elev_range_m'] / 1000, math.log10(a['slope_mkm']),
        a['forest_frac'], a['soil_porosity'], math.log10(a['ksat_cmh']), a['carbonate_frac'],
        min(a['reservoir_storage_frac'], 2.0), a['irrigation_frac'], a['urban_frac'],
        math.log(c['p_mean']), c['pet_mean'], math.log(c['aridity']), c['frac_snow'], c['hp_freq'], c['lp_freq'],
        c['t_mean'] / 10], np.float32)


def dyn_features(F, doy):
    """F: (T, 8) raw forcing in DYN order; doy: (T,) day of year. Returns (T, 11) float32 (fixed scaling)."""
    F = np.asarray(F, np.float32)
    P = np.maximum(F[:, 0], 0)
    out = np.empty((len(F), N_DYN_FEAT), np.float32)
    out[:, 0] = P / 10.0
    out[:, 1] = np.log1p(P)
    out[:, 2] = F[:, 1] / 5.0
    out[:, 3] = F[:, 2] / 10.0
    out[:, 4] = F[:, 3] / 10.0
    out[:, 5] = F[:, 4] / 10.0
    out[:, 6] = F[:, 5] / 150.0
    out[:, 7] = F[:, 6] / 1000.0
    out[:, 8] = F[:, 7] / 3.0
    w = 2 * np.pi * np.asarray(doy, np.float32) / 365.25
    out[:, 9] = np.sin(w)
    out[:, 10] = np.cos(w)
    return out


def forcing_array(df):
    """(T, 8) float32 in DYN order from a contract forcing frame (fills small gaps)."""
    X = df[DYN].astype('float64').interpolate(limit_direction='both').values
    return X.astype(np.float32)


def real_inputs(bid, start, end, warm_days=365, wd=None):
    """Forcing (T,8), doy, statics for a REAL basin over [start - warm, end]; never touches Q."""
    if wd is None:
        import waterdata as wd
    s0 = (pd.Timestamp(start) - pd.Timedelta(days=warm_days))
    df = wd.forcing(bid, str(s0.date()), end)
    F = forcing_array(df)
    clim = climate_stats(F[:, 0], F[:, 1], F[:, 2])
    st = static_vector(spec_attrs(wd.spec(bid)), clim)
    return F, df.index.dayofyear.values, st, df.index


def jdump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
