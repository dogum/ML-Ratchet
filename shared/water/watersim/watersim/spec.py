"""Basin spec (shared contract) defaults, elevation bands and parameter resolution."""
import copy
import math

import numpy as np

from .models import MODULES
from . import snow as snowmod

# field: (default, unit, source of the default)
SPEC_DEFAULTS = {
    'id': ('basin', '-', 'placeholder'),
    'area_km2': (500.0, 'km2', 'placeholder; set it (Q in m3/s depends on it)'),
    'lat': (40.0, 'deg', 'placeholder; PET and snow radiation depend on it'),
    'lon': (-100.0, 'deg', 'placeholder'),
    'elev_mean_m': (500.0, 'm', 'placeholder'),
    'elev_bands': (None, '-', 'built from elev_min_m/elev_max_m or elev_std_m when given, else one band'),
    'slope_mean': (90.0, 'm/km', 'HydroATLAS-based (Caravan slp_dg_sav, degrees -> m/km) CAMELS-US dev median ~118'),
    'aridity': (None, '-', 'computed from forcing (PET/P) when None'),
    'p_mean': (None, 'mm/day', 'computed from forcing when None'),
    'frac_snow': (None, '-', 'computed from simulation when None'),
    'forest_frac': (0.5, '-', 'CAMELS-US median ~0.6'),
    'lai': (2.0, 'm2/m2', 'CAMELS-US lai_max median ~3.4, mean ~2'),
    'soil_depth_m': (1.3, 'm', 'CAMELS-US soil_depth_pelletier/statsgo typical'),
    'soil_porosity': (0.44, '-', 'CAMELS-US median'),
    'soil_conductivity': (1.6, 'cm/h', 'CAMELS-US median'),
    'max_water_content_mm': (500.0, 'mm', 'CAMELS-US max_water_content median ~0.5 m'),
    'geology_permeability': (-14.0, 'log10 m2', 'GLHYMPS typical'),
    'carbonate_frac': (0.0, '-', 'assume none'),
    'dam_count': (0, '-', 'assume unregulated'),
    'reservoir_storage_frac': (0.0, '-', 'assume unregulated'),
    'irrigation_frac': (0.0, '-', 'assume none'),
    'urban_frac': (0.0, '-', 'assume none'),
    'forcing_elev_m': (None, 'm', 'reference elevation of the forcing = elev_mean_m'),
    'wind_height': (10.0, 'm', 'ERA5-Land 10 m wind'),
    'channel': (None, '-', 'Hack length, hydraulic-geometry width, slope from slope_mean (routing.channel_geometry)'),
    'params': (None, '-', 'model parameters (simulated basins only)'),
}


def fill_spec(spec):
    """Return a completed copy of `spec`; spec['src_fill'] lists each defaulted field and where its value came from."""
    s = copy.deepcopy(dict(spec or {}))
    src = {}
    for k, (v, unit, why) in SPEC_DEFAULTS.items():
        if s.get(k) is None:
            s[k] = copy.deepcopy(v)
            if v is not None:
                src[k] = why
    if str(s.get('slope_unit', '')).startswith('deg') and s.get('slope_mean') is not None:
        s['slope_mean'] = math.tan(math.radians(float(s['slope_mean']))) * 1000.0   # degrees -> m/km
        s['slope_unit'] = 'm/km (converted from degrees)'
    if s.get('soil_conductivity_unit') == 'mm/h' and s.get('soil_conductivity') is not None:
        s['soil_conductivity'] = float(s['soil_conductivity']) / 10.0
        s['soil_conductivity_unit'] = 'cm/h (converted from mm/h)'
    if s.get('max_water_content_mm') is None or 'max_water_content_mm' in src:
        if s.get('soil_depth_m') is not None and s.get('soil_porosity') is not None and \
                spec and spec.get('max_water_content_mm') is None:
            s['max_water_content_mm'] = float(s['soil_depth_m']) * float(s['soil_porosity']) * 1000 * 0.5
            src['max_water_content_mm'] = 'soil_depth_m x soil_porosity x 1000 x 0.5 (plant-available share)'
    if 'lai' in src:
        s['lai'] = 1.0 + 3.0 * float(s['forest_frac'])
        src['lai'] = '1 + 3 forest_frac (seasonal-max LAI proxy)'
    if s.get('forcing_elev_m') is None:
        s['forcing_elev_m'] = s['elev_mean_m']
    s['src_fill'] = {**(spec or {}).get('src_fill', {}), **src} if isinstance((spec or {}).get('src_fill'), dict) else src
    return s


def build_bands(spec, n=5):
    """Elevation bands as (frac, elev_m) arrays. Accepted forms of spec['elev_bands']:
    {'frac': [...], 'elev': [...]}, a list of {'frac', 'elev_m'} dicts, or a list of (frac, elev) pairs.
    Without bands: 5 equal-area bands from elev_min_m/elev_max_m (uniform hypsometry) or elev_std_m (normal)."""
    eb = spec.get('elev_bands')
    if eb:
        if isinstance(eb, dict):
            frac = np.asarray(eb['frac'], float)
            elev = np.asarray(eb.get('elev', eb.get('elev_m')), float)
        else:
            rows = [(r['frac'], r.get('elev_m', r.get('elev'))) if isinstance(r, dict) else tuple(r) for r in eb]
            frac = np.array([r[0] for r in rows], float)
            elev = np.array([r[1] for r in rows], float)
        frac = frac / frac.sum()
        return frac, elev
    zm = spec.get('elev_mean_m')
    zm = 500.0 if zm is None else float(zm)       # 0 m (sea level) is a valid elevation; only a missing one is filled
    if spec.get('elev_std_m'):
        from scipy.stats import norm
        q = (np.arange(n) + 0.5) / n
        return np.full(n, 1.0 / n), zm + float(spec['elev_std_m']) * norm.ppf(q)
    if spec.get('elev_min_m') is not None and spec.get('elev_max_m') is not None:
        lo, hi = float(spec['elev_min_m']), float(spec['elev_max_m'])
        if hi - lo > 100:
            e = lo + (hi - lo) * (np.arange(n) + 0.5) / n
            return np.full(n, 1.0 / n), e - (e.mean() - zm)
    return np.array([1.0]), np.array([zm])


def param_table(model, snow='degree_day'):
    """List of (name, default, lo, hi, unit, description, group) for the model, the snow routine and routing."""
    rows = [r + ('model',) for r in MODULES[model].PARAMS] if model in MODULES else []
    if snow == 'degree_day':
        rows += [r + ('snow',) for r in snowmod.DD_PARAMS]
    elif snow == 'energy_balance':
        rows += [r + ('snow',) for r in snowmod.EB_PARAMS]
    rows += [r + ('routing',) for r in ROUTING_PARAMS]
    return rows


ROUTING_PARAMS = [
    ('CH_VEL', 1.0, 0.3, 3.0, 'm/s', "mean channel-network velocity for routing='uh' (gamma, shape 3)"),
    ('CH_N', 0.035, 0.02, 0.08, 's/m^(1/3)', "Manning n for routing='muskingum'/'kinematic'"),
]


def resolve_params(model, snow, spec, params=None):
    """Merge parameters: defaults < attribute transfer functions (if spec has no params) < spec['params'] < params.
    `params` may be flat ({'FC': 300, 'CFMAX': 4}) or nested ({'hbv': {...}, 'snow': {...}}).
    Returns (model dict, snow dict, routing dict, sources dict)."""
    table = param_table(model, snow)
    out = {'model': {}, 'snow': {}, 'routing': {}}
    src = {}
    for name, d, lo, hi, unit, desc, grp in table:
        out[grp][name] = float(d)
        src[name] = 'default'
    sp = spec.get('params') or {}
    layers = []
    if not sp:
        from .generator import params_from_attributes
        att = params_from_attributes(spec, model=model, snow=snow)
        layers.append(('attributes', att))
    else:
        layers.append(('spec', sp))
    if params:
        layers.append(('arg', params))
    snow_key = {'degree_day': 'snow_dd', 'energy_balance': 'snow_eb'}.get(snow, 'snow')
    for tag, lay in layers:
        flat = {}
        for k, v in lay.items():
            if isinstance(v, dict):
                if k in (model, 'snow', snow_key, 'routing'):
                    flat.update(v)
            elif v is not None:
                flat[k] = v
        for k, v in flat.items():
            for grp in ('model', 'snow', 'routing'):
                if k in out[grp]:
                    out[grp][k] = float(v)
                    src[k] = tag
                    break
            else:
                if tag == 'arg':
                    raise KeyError(f'unknown parameter {k!r} for model={model!r}, snow={snow!r}')
    return out['model'], out['snow'], out['routing'], src


def clip_params(model, snow, pm, ps):
    """Clip parameter dicts into their documented bounds (returns copies)."""
    pm = dict(pm)
    ps = dict(ps)
    for name, d, lo, hi, *_ in MODULES[model].PARAMS:
        pm[name] = float(min(max(pm[name], lo), hi))
    tab = snowmod.DD_PARAMS if snow == 'degree_day' else snowmod.EB_PARAMS if snow == 'energy_balance' else []
    for name, d, lo, hi, *_ in tab:
        if name in ps:
            ps[name] = float(min(max(ps[name], lo), hi))
    return pm, ps


def channel_for(spec, area_km2, rp):
    from .routing import channel_geometry
    ch = dict(spec.get('channel') or {})
    slope = ch.get('slope')
    if slope is None:
        sm = spec.get('slope_mean') or 90.0
        slope = float(np.clip(sm / 1000.0 * 0.05, 2e-4, 0.03))   # channel ~5 % of the hillslope gradient
    g = channel_geometry(area_km2, slope=slope, length_km=ch.get('length_km'), width_m=ch.get('width_m'),
                         manning_n=ch.get('manning_n', rp.get('CH_N', 0.035)))
    return g


def log_uniform(rng, lo, hi, size=None):
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size))
