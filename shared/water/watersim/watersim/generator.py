"""Random basins for training: attributes from documented priors (parametric, or resampled from real non-exam
CAMELS/Caravan attributes), and parameter sets for every model drawn through attribute -> parameter transfer
functions plus scatter. Four shared latent factors (storage, permeability, quickness, snow) drive the parameters of
all models, so one basin is the 'same' basin under HBV, GR4J, SAC-SMA and VIC-lite.

Transfer functions (medians; scatter is lognormal, sigma ~0.25-0.45 in log space, so attributes explain only part
of the parameter variance, as in regionalisation studies, e.g. Merz & Bloeschl 2004, Beck et al. 2016):
  storage S = 0.75 max_water_content (0.75 + 0.5 forest)    -> FC, X1, UZTWM+LZTWM, W1M
  permeability g = clip((log10 k + 15.5)/4 + 0.4 carbonate)  -> PERC, K2 (slower with g), LZPK, X3, DSMAX, X2 < 0 in karst
  quickness q = log(slope/90 m/km) - log(area)              -> K0, K1, UZK, MAXBAS, X4, UH scale
  forest                                                     -> CFMAX lower, canopy transmissivity lower
  aridity                                                    -> BETA, LP higher in dry basins
Ranges are clipped into each model's documented bounds.
"""
import math

import numpy as np

from .models import MODULES
from . import snow as snowmod
from .spec import SPEC_DEFAULTS

PRIOR_NOTES = {
    'area_km2': 'lognormal, median 330 km2, sigma(ln) 1.2, clip 5-25000 (CAMELS-US spans 4-25,800 km2, median ~340)',
    'lat/lon': 'uniform over the contiguous US box (25-49 N, 124-68 W) for prior="default"; "global" widens to 10-68 N/S',
    'elev_mean_m': 'lognormal, median 550 m, sigma 0.9, clip 10-3600',
    'relief': 'elev range = elev_mean (0.3-1.6) -> 5 equal-area bands',
    'slope_mean': 'lognormal, median 90 m/km, sigma ~0.95, correlated with elevation, clip 2-600 (HydroATLAS slp_dg_sav '
                  'converted to m/km, as in B\'s specs: CAMELS-US dev median 118 m/km)',
    'p_mean': 'lognormal, median 2.9 mm/day, sigma 0.45, clip 0.4-10 (CAMELS-US 0.6-8.9)',
    'climate': 'T_ann = 27.5 - 0.0085 lat^2 - 6 z_km (+- 2 C; ~ zonal-mean sea-level temperature); P seasonality amplitude U(-0.6, 0.7)',
    'forest_frac': 'beta, mean rising with p_mean',
    'soil': 'depth U(0.4, 2.0) m; porosity N(0.44, 0.04); max_water_content = depth x porosity x U(0.4, 0.9)',
    'geology_permeability': 'N(-14, 1.2) log10 m2, clip -17..-10.5 (GLHYMPS range); carbonate 20 % of basins',
    'human': 'dams: 30 % of basins; reservoir storage 0.02-1.5 years of mean flow; storage-depth exponent b ~ '
             '2.5 exp(N(0, 0.35)) (dev ResOpsUS fits: median 2.5, range 1.4-6); irrigation in 25 %; urban small',
}


def _clip(x, lo, hi):
    return float(min(max(x, lo), hi))


def _bounds(model):
    return {p[0]: (p[2], p[3]) for p in MODULES[model].PARAMS}


_ADJ = {}


def prior_adjustments(reload=False):
    """Dev-calibrated corrections of the transfer functions (data/prior_adjust.json, written by
    tools/fit_prior.py from the dev-basin calibrations only): per parameter a median factor (or shift) and the
    extra log-scatter needed to match the spread of the calibrated values."""
    import json
    import os
    if reload or 'table' not in _ADJ:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'prior_adjust.json')
        _ADJ['table'] = json.load(open(path)) if os.path.exists(path) else {}
    return _ADJ['table']


def params_from_attributes(spec, model='hbv', snow='degree_day', latent=None, rng=None, scatter=1.0, adjust=True):
    """Parameters for `model` (+ snow) from basin attributes. latent: dict of standard-normal factors
    {'storage','perm','quick','snow'} (0 = median); rng adds independent per-parameter scatter.
    adjust: apply the dev-calibrated corrections (prior_adjustments()).
    Returns a flat dict of model and snow parameters (within bounds)."""
    s = {k: (spec.get(k) if spec.get(k) is not None else SPEC_DEFAULTS[k][0]) for k in SPEC_DEFAULTS
         if k not in ('elev_bands', 'channel', 'params', 'id')}
    z = {'storage': 0.0, 'perm': 0.0, 'quick': 0.0, 'snow': 0.0, **(latent or {})}
    rng = rng if rng is not None else None

    def e(sig):  # independent scatter
        return float(rng.normal(0, sig * scatter)) if rng is not None else 0.0

    forest = float(s['forest_frac'])
    mwc = float(s['max_water_content_mm'])
    if mwc < 10:          # given in metres (CAMELS units)
        mwc *= 1000.0
    S = 0.75 * mwc * (0.75 + 0.5 * forest) * math.exp(0.35 * z['storage'] * scatter)
    perm = float(s['geology_permeability'])
    g = _clip((perm + 15.5) / 4.0 + 0.4 * float(s['carbonate_frac']) + 0.25 * z['perm'] * scatter, 0.0, 1.0)
    slope = max(float(s['slope_mean']), 0.5)
    area = max(float(s['area_km2']), 1.0)
    qk = _clip(0.5 + 0.2 * math.log(slope / 90.0) - 0.08 * math.log(area / 330.0) + 0.2 * z['quick'] * scatter,
               0.0, 1.0)
    arid = spec.get('aridity')
    arid = 1.0 if arid is None else float(arid)
    urban = float(s['urban_frac'] or 0.0)
    out = {}
    if model == 'hbv':
        b = _bounds('hbv')
        out['FC'] = S * math.exp(e(0.3))
        out['LP'] = 0.55 + 0.25 * _clip(arid - 0.5, 0, 1) + e(0.08)
        out['BETA'] = 1.5 + 2.0 * _clip(arid - 0.5, 0, 1.5) + e(0.6)
        out['PERC'] = (0.3 + 3.5 * g) * math.exp(e(0.35))
        out['UZL'] = 10 + 40 * (1 - qk) + e(10)
        out['K0'] = 0.12 + 0.3 * qk + e(0.05)
        out['K1'] = 0.03 + 0.15 * qk + e(0.02)
        out['K2'] = math.exp(math.log(0.06) + (math.log(0.003) - math.log(0.06)) * g + e(0.4))
        out['MAXBAS'] = 1.0 + 0.5 * math.log10(area) * (1.3 - qk) + e(0.3)
    elif model == 'gr4j':
        b = _bounds('gr4j')
        out['X1'] = 1.1 * S * math.exp(e(0.35))
        out['X2'] = -0.3 - 1.5 * float(s['carbonate_frac']) + 0.4 * z['perm'] * scatter + e(0.5)
        out['X3'] = 25 + 220 * g * math.exp(e(0.35)) + 30 * (1 - qk)
        out['X4'] = (0.8 + 0.45 * math.log10(area) * (1.3 - qk)) * math.exp(e(0.25))
    elif model == 'sacsma':
        b = _bounds('sacsma')
        out['UZTWM'] = 0.25 * S * math.exp(e(0.35))
        out['UZFWM'] = (15 + 50 * g) * math.exp(e(0.35))
        out['UZK'] = 0.2 + 0.4 * qk + e(0.07)
        out['PCTIM'] = 0.005 + 0.3 * urban + abs(e(0.01))
        out['ADIMP'] = 0.02 + 0.5 * urban + abs(e(0.04))
        out['RIVA'] = abs(e(0.01))
        out['ZPERC'] = (20 + 80 * (1 - g)) * math.exp(e(0.4))
        out['REXP'] = 1.8 + e(0.4)
        out['LZTWM'] = 0.6 * S * math.exp(e(0.35))
        out['LZFSM'] = (15 + 60 * g) * math.exp(e(0.4))
        out['LZFPM'] = (40 + 300 * g) * math.exp(e(0.4))
        out['LZSK'] = 0.03 + 0.12 * (1 - g) + e(0.02)
        out['LZPK'] = math.exp(math.log(0.02) + (math.log(0.002) - math.log(0.02)) * g + e(0.4))
        out['PFREE'] = 0.05 + 0.4 * g + e(0.05)
        out['SIDE'] = 0.0
        out['RSERV'] = 0.3
        out['UH_SHAPE'] = 2.0 + e(0.4)
        out['UH_SCALE'] = (0.35 + 0.3 * math.log10(area) * (1.3 - qk)) * math.exp(e(0.25))
    elif model == 'vic_lite':
        b = _bounds('vic_lite')
        out['W1M'] = 0.5 * S * math.exp(e(0.3))
        out['B_INF'] = math.exp(math.log(0.25) + 0.8 * (qk - 0.5) + e(0.5))
        out['ROOT1'] = 0.55 + 0.25 * (1 - forest) + e(0.08)
        out['KSAT'] = (3 + 60 * g) * math.exp(e(0.5))
        out['EXPT'] = 6 + 8 * (1 - g) + e(2)
        out['W2M'] = (150 + 900 * g) * math.exp(e(0.35))
        out['DSMAX'] = (0.5 + 12 * g) * math.exp(e(0.5))
        out['DS'] = 0.05 + 0.3 * (1 - g) + e(0.05)
        out['WS'] = 0.75 + e(0.08)
        out['UH_SHAPE'] = 2.0 + e(0.4)
        out['UH_SCALE'] = (0.35 + 0.3 * math.log10(area) * (1.3 - qk)) * math.exp(e(0.25))
    else:
        b = {}
    adj = prior_adjustments() if adjust else {}

    def _apply(group, d):
        for k, a in (adj.get(group) or {}).items():
            if k not in d:
                continue
            x = e(a.get('extra_sigma', 0.0))
            if a.get('mode') == 'mult':
                d[k] = d[k] * a.get('factor', 1.0) * math.exp(x)
            else:
                d[k] = d[k] + a.get('shift', 0.0) + x
    _apply(model, out)
    for k, (lo, hi) in b.items():
        if k in out:
            out[k] = _clip(out[k], lo, hi)
    # snow
    zs = z['snow'] * scatter
    if snow == 'degree_day':
        sb = {p[0]: (p[2], p[3]) for p in snowmod.DD_PARAMS}
        sn = {'TT': 0.0 + 0.6 * zs + e(0.4), 'TTI': 2.0 + e(0.5),
              'CFMAX': (4.2 - 1.6 * forest) * math.exp(0.15 * zs + e(0.2)), 'SFCF': 1.0 + e(0.08),
              'CFR': 0.05, 'CWH': 0.1, 'TLAPSE': -6.0 + e(0.6), 'PGRAD': 0.3 + abs(e(0.15))}
    elif snow == 'energy_balance':
        sb = {p[0]: (p[2], p[3]) for p in snowmod.EB_PARAMS}
        lai = float(s['lai'])
        sn = {'TT': 0.5 + 0.5 * zs + e(0.3), 'TTI': 2.0 + e(0.5), 'SFCF': 1.0 + e(0.08), 'CWH': 0.05,
              'ALB_MAX': 0.85 + e(0.02), 'ALB_MIN': 0.45 + e(0.05), 'ALB_DECAY': 12 * math.exp(e(0.3)),
              'Z0': 0.002 * math.exp(e(0.6)), 'TURB': math.exp(0.2 * zs + e(0.25)), 'GHEAT': 0.173,
              'CANOPY': math.exp(-0.5 * lai * forest), 'TLAPSE': -6.0 + e(0.6), 'PGRAD': 0.3 + abs(e(0.15))}
    else:
        sb, sn = {}, {}
    _apply({'degree_day': 'snow_dd', 'energy_balance': 'snow_eb'}.get(snow, 'none'), sn)
    for k, (lo, hi) in sb.items():
        if k in sn:
            sn[k] = _clip(sn[k], lo, hi)
    out.update(sn)
    return out


def _sample_attributes(rng, prior='default'):
    glob = prior == 'global'
    a = {}
    a['area_km2'] = _clip(330 * math.exp(rng.normal(0, 1.2)), 5, 25000)
    if glob:
        a['lat'] = float(rng.uniform(10, 68) * rng.choice([1, 1, 1, -1]))
        a['lon'] = float(rng.uniform(-180, 180))
    else:
        a['lat'] = float(rng.uniform(25, 49))
        a['lon'] = float(rng.uniform(-124, -68))
    zr = rng.normal()
    a['elev_mean_m'] = _clip(550 * math.exp(0.9 * zr), 10, 3600)
    rel = a['elev_mean_m'] * rng.uniform(0.3, 1.6)
    a['elev_min_m'] = max(a['elev_mean_m'] - rel * 0.45, 0.0)
    a['elev_max_m'] = a['elev_min_m'] + rel
    a['slope_mean'] = _clip(90 * math.exp(0.6 * zr + 0.75 * rng.normal()), 2, 600)
    a['p_mean'] = _clip(2.9 * math.exp(rng.normal(0, 0.45)), 0.4, 10)
    wet = (math.log(a['p_mean'] / 2.9) / 0.45)
    a['forest_frac'] = float(rng.beta(1.0 + max(1.5 * wet + 1.2, 0.2), 1.5))
    a['lai'] = float(0.4 + 4.0 * a['forest_frac'] * rng.uniform(0.6, 1.1))
    a['soil_depth_m'] = float(rng.uniform(0.4, 2.0))
    a['soil_porosity'] = _clip(rng.normal(0.44, 0.04), 0.3, 0.6)
    a['soil_conductivity'] = _clip(1.6 * math.exp(rng.normal(0, 0.5)), 0.2, 15)
    a['max_water_content_mm'] = a['soil_depth_m'] * a['soil_porosity'] * 1000 * float(rng.uniform(0.4, 0.9))
    a['geology_permeability'] = _clip(rng.normal(-14, 1.2), -17, -10.5)
    a['carbonate_frac'] = float(rng.beta(1, 2)) if rng.random() < 0.2 else 0.0
    dams = rng.random() < 0.3
    a['dam_count'] = int(1 + rng.poisson(2)) if dams else 0
    a['reservoir_storage_frac'] = _clip(0.15 * math.exp(rng.normal(0, 1.1)), 0.02, 1.5) if dams else 0.0
    a['irrigation_frac'] = float(rng.beta(1, 6)) if rng.random() < 0.25 else 0.0
    a['urban_frac'] = float(rng.beta(1, 6)) if rng.random() < 0.2 else float(rng.beta(1, 60))
    # climate for the weather generator
    T_ann = 27.5 - 0.0085 * a['lat'] ** 2 - 6.0 * a['elev_mean_m'] / 1000.0 + rng.normal(0, 2.0)
    a['climate'] = {
        'T_ann': float(T_ann), 'T_amp': float(max(3.0 + 0.3 * abs(a['lat']) + rng.normal(0, 2.0), 1.0)),
        'P_mean': a['p_mean'], 'P_amp': float(rng.uniform(-0.6, 0.7)),
        'P_phase_doy': float(rng.uniform(0, 365)), 'wet_frac': _clip(0.16 + 0.07 * a['p_mean'] + rng.normal(0, 0.05), 0.08, 0.7),
        'persistence': float(rng.uniform(0.2, 0.5)), 'gamma_shape': float(rng.uniform(0.5, 0.9)),
        'dtr': float(rng.uniform(7, 16) + 3 * (1 - a['forest_frac'])),
        'dew_depression': float(max(rng.normal(3.0, 1.5) + 2.0 * (a['p_mean'] < 1.5), 0.3)),
        'wind_mean': float(rng.uniform(1.5, 5.0)), 'T_sigma': float(rng.uniform(1.8, 3.5)),
    }
    return a


CAMELS_MAP = {   # spec field <- candidate attribute column names (CAMELS-US / Caravan)
    'area_km2': ['area_km2', 'area', 'area_gages2', 'area_geospa_fabric'],
    'lat': ['lat', 'gauge_lat'], 'lon': ['lon', 'gauge_lon'],
    'elev_mean_m': ['elev_mean_m', 'elev_mean', 'ele_mt_sav'],
    'slope_mean': ['slope_mean', 'slp_dg_sav'],
    'p_mean': ['p_mean'], 'aridity': ['aridity', 'aridity_fao_pm', 'ari_ix_sav'],
    'frac_snow': ['frac_snow', 'frac_snow_daily'],
    'forest_frac': ['forest_frac', 'frac_forest', 'for_pc_sse'],
    'lai': ['lai', 'lai_max'],
    'soil_depth_m': ['soil_depth_m', 'soil_depth_pelletier', 'soil_depth_statsgo'],
    'soil_porosity': ['soil_porosity'], 'soil_conductivity': ['soil_conductivity'],
    'max_water_content_mm': ['max_water_content_mm', 'max_water_content'],
    'geology_permeability': ['geology_permeability', 'geol_permeability'],
    'carbonate_frac': ['carbonate_frac', 'carbonate_rocks_frac', 'kar_pc_sse'],
    'urban_frac': ['urban_frac', 'urb_pc_sse'], 'irrigation_frac': ['irrigation_frac', 'ire_pc_sse'],
    'dam_count': ['dam_count', 'dor_pc_pva'], 'reservoir_storage_frac': ['reservoir_storage_frac'],
}


def attributes_from_row(row):
    """Map one attribute table row (CAMELS-US/Caravan names) to spec fields (with unit fixes)."""
    a = {}
    for k, names in CAMELS_MAP.items():
        for n in names:
            if n in row and row[n] is not None and np.isfinite(_num(row[n])):
                a[k] = _num(row[n])
                if n in ('for_pc_sse', 'urb_pc_sse', 'ire_pc_sse', 'kar_pc_sse'):
                    a[k] = a[k] / 100.0
                if n == 'slp_dg_sav':   # HydroATLAS: degrees x 10 -> m/km
                    a[k] = math.tan(math.radians(a[k] / 10.0)) * 1000.0
                if n == 'max_water_content':  # CAMELS: metres
                    a[k] = a[k] * 1000.0
                if n == 'geol_permeability':
                    a[k] = float(a[k])
                break
    return a


def _num(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def generate_basins(n, rng=None, prior='default', models=('hbv', 'gr4j', 'sacsma', 'vic_lite'), regulation=True,
                    attributes=None, jitter=0.15, scatter=1.0):
    """Draw n random basin specs with parameters for every model.

    prior: 'default' (parametric, CAMELS-US-like, PRIOR_NOTES), 'global', or 'camels' with `attributes` = a pandas
           DataFrame of REAL NON-EXAM basins (index = basin id; CAMELS/Caravan column names, see CAMELS_MAP).
           With 'camels', each draw copies a donor basin's attributes, multiplies positive fields by
           exp(N(0, jitter)) and records spec['donor_id'] so its (non-exam-period) forcing can be reused.
    regulation: add reservoirs (and irrigation) where the attributes say so.
    Returns a list of spec dicts; spec['params'] = {model: {...}, 'snow_dd': {...}, 'snow_eb': {...}},
    spec['latent'] = the four shared latent factors, spec['climate'] = weather-generator parameters."""
    rng = np.random.default_rng(rng)
    specs = []
    donors = None
    if prior == 'camels':
        if attributes is None:
            from .realdata import donor_table
            attributes = donor_table()          # B's dev basins only
        donors = attributes
    for i in range(n):
        a = _sample_attributes(rng, 'global' if prior == 'global' else 'default')
        if donors is not None:
            j = int(rng.integers(len(donors)))
            row = donors.iloc[j]
            real = attributes_from_row(row.to_dict())
            for k, v in real.items():
                if k in ('lat', 'lon'):
                    a[k] = float(v)
                elif k in ('dam_count',):
                    a[k] = int(v)
                elif v > 0:
                    a[k] = float(v) * math.exp(rng.normal(0, jitter))
                else:
                    a[k] = float(v)
            a['donor_id'] = str(donors.index[j])
            if 'p_mean' in real:
                a['climate']['P_mean'] = a['p_mean']
            rel = a['elev_mean_m'] * rng.uniform(0.3, 1.6)
            a['elev_min_m'] = max(a['elev_mean_m'] - rel * 0.45, 0.0)
            a['elev_max_m'] = a['elev_min_m'] + rel
        latent = {k: float(rng.normal()) for k in ('storage', 'perm', 'quick', 'snow')}
        spec = {'id': f'sim{i:06d}', **a, 'latent': latent, 'prior': prior}
        params = {}
        for m in models:
            params[m] = {k: v for k, v in params_from_attributes(spec, m, 'none', latent, rng, scatter).items()}
        params['snow_dd'] = params_from_attributes(spec, 'none', 'degree_day', latent, rng, scatter)
        params['snow_eb'] = params_from_attributes(spec, 'none', 'energy_balance', latent, rng, scatter)
        params['routing'] = {'CH_VEL': _clip(1.0 * math.exp(rng.normal(0, 0.35)), 0.3, 3.0),
                             'CH_N': _clip(0.035 * math.exp(rng.normal(0, 0.25)), 0.02, 0.08)}
        spec['params'] = params
        if regulation and a.get('reservoir_storage_frac', 0) > 0 and a.get('dam_count', 0) > 0:
            spec['reservoirs'] = _random_reservoirs(a, rng)
        specs.append(spec)
    return specs


def _random_reservoirs(a, rng):
    """One reservoir (or two in series for multi-dam basins) holding reservoir_storage_frac x mean annual flow."""
    nres = 1 if a['dam_count'] < 3 or rng.random() < 0.6 else 2
    total = a['reservoir_storage_frac']
    up = float(rng.uniform(0.3, 0.95))
    pol = str(rng.choice(['sop', 'rule_curve', 'hanasaki', 'sop']))
    res = []
    for k in range(nres):
        frac = up if k == nres - 1 else up * float(rng.uniform(0.3, 0.8))
        r = {'name': f'res{k}', 'capacity_years': total / nres, 'upstream_frac': frac,
             'policy': pol, 'init_frac': float(rng.uniform(0.4, 0.9)), 'dead_frac': float(rng.uniform(0.02, 0.15)),
             'shape_b': _clip(2.5 * math.exp(rng.normal(0, 0.35)), 1.3, 6.0), 'policy_params': {
                 's_lo': float(rng.uniform(0.3, 0.6)), 's_hi': float(rng.uniform(0.75, 0.98)),
                 'doy_full': float(rng.uniform(100, 220)), 't_adjust_days': float(rng.uniform(10, 60)),
                 'withdraw': bool(rng.random() < 0.4)},
             'demand_m3s': None, 'efficiency': float(rng.uniform(0.8, 0.92)), 'consumptive_frac': float(rng.uniform(0.3, 0.9))}
        if k > 0:
            r['inflow_from'] = k - 1
        res.append(r)
    return res


def labels_of_spec(spec):
    """Attribute-level labels of a generated spec (no simulation needed)."""
    keep = ['area_km2', 'lat', 'lon', 'elev_mean_m', 'slope_mean', 'p_mean', 'forest_frac', 'lai', 'soil_depth_m',
            'soil_porosity', 'soil_conductivity', 'max_water_content_mm', 'geology_permeability', 'carbonate_frac',
            'dam_count', 'reservoir_storage_frac', 'irrigation_frac', 'urban_frac']
    return {k: spec.get(k) for k in keep} | {f'latent_{k}': v for k, v in (spec.get('latent') or {}).items()}
