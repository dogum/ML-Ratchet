"""Input helpers shared by the bank builder and the data API. Copied unchanged from FlowCast
(models/flowcast/training/src/common.py and flowcast.py) so that sim and real inputs are built exactly as FlowCast
built them. numpy + pandas only."""
import math
import numpy as np

DYN = ['P', 'PET', 'T_mean', 'T_min', 'T_max', 'SRAD', 'vp', 'wind']
FALLBACK = {'forest_frac': 0.5, 'soil_porosity': 0.44, 'soil_conductivity_cmh': 1.6, 'carbonate_frac': 0.0,
            'reservoir_storage_frac': 0.0, 'irrigation_frac': 0.0, 'urban_frac': 0.0, 'slope_mkm': 60.0}
ATTR_NAMES = ['area_km2', 'elev_mean_m', 'elev_range_m', 'slope_mkm', 'forest_frac', 'soil_porosity', 'ksat_cmh',
              'carbonate_frac', 'reservoir_storage_frac', 'irrigation_frac', 'urban_frac']


def _f(x, d):
    try:
        x = float(x)
        return x if np.isfinite(x) else d
    except Exception:
        return d


def climate_stats(P, PET, T):
    """Climate indices from forcing arrays (identical for sim and real basins)."""
    P = np.asarray(P, float); PET = np.asarray(PET, float); T = np.asarray(T, float)
    pm = max(np.nanmean(P), 0.05)
    em = max(np.nanmean(PET), 0.05)
    fs = float(np.nansum(P[T < 0]) / max(np.nansum(P), 1e-6))
    return {'p_mean': pm, 'pet_mean': em, 'aridity': em / pm, 'frac_snow': fs,
            'hp_freq': float(np.nanmean(P > 5 * pm)), 'lp_freq': float(np.nanmean(P < 1.0)),
            't_mean': float(np.nanmean(T))}


def spec_attrs(spec):
    """Physical attributes in sim units (slope m/km, ksat cm/h) from a sim or real spec (FlowCast fallbacks)."""
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


def attr_vector(a):
    return np.array([a[k] for k in ATTR_NAMES], np.float32)


def attr_dict(v):
    return {k: float(x) for k, x in zip(ATTR_NAMES, v)}


def static_vector(a, c):
    """FlowCast's 18 static features: 11 attributes (transformed) + 7 climate indices."""
    return np.array([
        math.log10(a['area_km2']), a['elev_mean_m'] / 1000, a['elev_range_m'] / 1000, math.log10(a['slope_mkm']),
        a['forest_frac'], a['soil_porosity'], math.log10(a['ksat_cmh']), a['carbonate_frac'],
        min(a['reservoir_storage_frac'], 2.0), a['irrigation_frac'], a['urban_frac'],
        math.log(c['p_mean']), c['pet_mean'], math.log(c['aridity']), c['frac_snow'], c['hp_freq'], c['lp_freq'],
        c['t_mean'] / 10], np.float32)


def forcing_array(df):
    """(T, 8) float32 in DYN order from a contract forcing frame (fills small gaps by interpolation)."""
    X = df[DYN].astype('float64').interpolate(limit_direction='both').values
    return X.astype(np.float32)


def dyn_features(F, doy):
    """FlowCast's 11 scaled daily features (optional helper; arms may use their own)."""
    F = np.asarray(F, np.float32)
    P = np.maximum(F[:, 0], 0)
    out = np.empty((len(F), 11), np.float32)
    out[:, 0] = P / 10.0; out[:, 1] = np.log1p(P); out[:, 2] = F[:, 1] / 5.0
    out[:, 3] = F[:, 2] / 10.0; out[:, 4] = F[:, 3] / 10.0; out[:, 5] = F[:, 4] / 10.0
    out[:, 6] = F[:, 5] / 150.0; out[:, 7] = F[:, 6] / 1000.0; out[:, 8] = F[:, 7] / 3.0
    w = 2 * np.pi * np.asarray(doy, np.float32) / 365.25
    out[:, 9] = np.sin(w); out[:, 10] = np.cos(w)
    return out


def classes_from(frac_snow, aridity, reservoir_storage_frac, area_km2):
    """Basin classes as in FlowCast's breakdowns (a basin can be in several; 'other' if none)."""
    c = []
    if frac_snow > 0.3:
        c.append('snow')
    if aridity > 1.0:
        c.append('arid')
    if aridity < 0.6:
        c.append('humid')
    if reservoir_storage_frac >= 0.1:
        c.append('regulated')
    if area_km2 > 2000:
        c.append('large')
    return c or ['other']


def fu_runoff_ratio(aridity, w=2.6):
    phi = np.asarray(aridity, float)
    return np.clip((1 + phi ** w) ** (1 / w) - phi, 0.0, 1.0)
