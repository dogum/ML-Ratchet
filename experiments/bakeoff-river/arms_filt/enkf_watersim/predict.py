"""EnKF on WaterSim (gauged track): predict(batch) for the River Bake-off harness.

Per batch: attribute-prior HBV + degree-day snow parameters (watersim.generator.params_from_attributes, the same
transfer functions + dev-fitted prior adjustments WaterSim uses for a basin spec without parameters), elevation bands
from the elevation mean and range, WaterSim's gamma channel UH, a cyclic one-year spin-up on the warm-up forcing (no
flow used), then the ensemble filter over warm-up + target rows (enkf_core.run_filter), assimilating qobs up to the
forecast origin and forecasting `lead` days. Settings: hyper.json (the released rung). Deterministic (fixed seeds).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import enkf_core as EK  # noqa: E402  (puts the harness's vendored watersim on sys.path)
from watersim.spec import fill_spec, resolve_params  # noqa: E402
from watersim.generator import params_from_attributes  # noqa: E402
from watersim.models import hbv as _hbv  # noqa: E402
from watersim import snow as _snowmod  # noqa: E402
from watersim.routing import triangle, gamma_uh, channel_geometry  # noqa: E402

HYPER = json.load(open(os.path.join(HERE, 'hyper.json')))
HBV_NAMES = [p[0] for p in _hbv.PARAMS]
HBV_BOUNDS = {p[0]: (p[2], p[3]) for p in _hbv.PARAMS}
DD_NAMES = [p[0] for p in _snowmod.DD_PARAMS]
ATTR_NAMES = ['area_km2', 'elev_mean_m', 'elev_range_m', 'slope_mkm', 'forest_frac', 'soil_porosity', 'ksat_cmh',
              'carbonate_frac', 'reservoir_storage_frac', 'irrigation_frac', 'urban_frac']
PMULT_BOUNDS = (0.4, 2.5)
_MAPPING = None


def basin_spec(batch):
    """A WaterSim basin spec from the batch's 11 attributes and its forcing climate (static vector)."""
    a = {k: float(v) for k, v in zip(ATTR_NAMES, np.asarray(batch['attrs'], float))}
    st = np.asarray(batch['static'], float)
    zm, zr = a['elev_mean_m'], max(a['elev_range_m'], 0.0)
    spec = {'area_km2': max(a['area_km2'], 1.0), 'lat': float(batch['lat']), 'lon': float(batch['lon']),
            'elev_mean_m': zm, 'elev_min_m': max(zm - 0.5 * zr, 0.0), 'elev_max_m': max(zm - 0.5 * zr, 0.0) + zr,
            'slope_mean': max(a['slope_mkm'], 0.5), 'forest_frac': a['forest_frac'],
            'soil_porosity': a['soil_porosity'], 'soil_conductivity': max(a['ksat_cmh'], 0.01),
            'carbonate_frac': a['carbonate_frac'], 'reservoir_storage_frac': a['reservoir_storage_frac'],
            'irrigation_frac': a['irrigation_frac'], 'urban_frac': a['urban_frac'],
            'aridity': float(math.exp(st[13])), 'p_mean': float(math.exp(st[11]))}
    return fill_spec(spec)


def bands(spec, nb=5):
    zm = float(spec['elev_mean_m'])
    lo, hi = float(spec['elev_min_m']), float(spec['elev_max_m'])
    if hi - lo > 100:
        e = lo + (hi - lo) * (np.arange(nb) + 0.5) / nb
        e = e - (e.mean() - zm)
        return np.full(nb, 1.0 / nb), (e - zm) / 1000.0
    return np.array([1.0]), np.array([0.0])


def pmult_prior(batch):
    """WaterSim p1 water-yield multiplier median from attributes + forcing climate (FlowCast's p1 physics):
    log m = -0.088 + 0.058 elev_km + 0.153 frac_snow + 0.397 log(P / 3 mm)."""
    st = np.asarray(batch['static'], float)
    lm = -0.088 + 0.058 * float(st[1]) + 0.153 * float(st[14]) + 0.397 * math.log(math.exp(st[11]) / 3.0)
    return float(np.clip(math.exp(lm), *PMULT_BOUNDS))


def _mapped_params(batch, H):
    """Optional rung: parameters from arm B's physics-in-the-loop mapping (if hyper.json names one)."""
    global _MAPPING
    if not H.get('param_source', 'prior').startswith('mapping'):
        return None
    if _MAPPING is None:
        sys.path.insert(0, os.path.join(HERE, H['mapping_dir']))
        import importlib
        _MAPPING = importlib.import_module(H['mapping_module'])
    return _MAPPING.hbv_params(batch)


def setup(batch, H=None):
    H = H or HYPER
    N = int(H['n_members'])
    rng = np.random.default_rng(int(H.get('seed', 20261123)))
    spec = basin_spec(batch)
    pm_med, ps_med, prp, _ = resolve_params('hbv', 'degree_day', spec)
    mapped = _mapped_params(batch, H)
    if mapped is not None:
        pm_med.update({k: v for k, v in mapped.items() if k in pm_med})
        ps_med.update({k: v for k, v in mapped.items() if k in ps_med})
    sc = float(H.get('param_scatter', 0.0))
    mp = np.zeros((N, len(HBV_NAMES)))
    sp = np.zeros((N, len(DD_NAMES)))
    for m in range(N):
        if sc > 0 and m > 0:
            d = params_from_attributes(spec, model='hbv', snow='degree_day', rng=rng, scatter=sc)
            # scatter around the (possibly mapped) median: apply the draw's log ratio to the median
            base = params_from_attributes(spec, model='hbv', snow='degree_day')
            for j, k in enumerate(HBV_NAMES):
                lo, hi = HBV_BOUNDS[k]
                if k in ('UZL', 'PERC'):
                    v = pm_med[k] + (d[k] - base[k])
                else:
                    v = pm_med[k] * d[k] / max(base[k], 1e-9)
                mp[m, j] = min(max(v, lo), hi)
            for j, k in enumerate(DD_NAMES):
                sp[m, j] = ps_med[k] + (d[k] - base[k])
        else:
            mp[m] = [pm_med[k] for k in HBV_NAMES]
            sp[m] = [ps_med[k] for k in DD_NAMES]
    frac, dz = bands(spec)
    B = len(frac)
    pm_b = np.zeros((N, B))
    for m in range(N):
        g = np.maximum(1.0 + sp[m, 7] * dz, 0.0)
        nrm = float(np.sum(frac * g))
        pm_b[m] = g / nrm if nrm > 0 else 1.0
    # hillslope UH (triangle, MAXBAS per member) padded to a common length
    tris = [triangle(mp[m, 8]) for m in range(N)]
    L = max(len(o) for o in tris)
    uhord = np.zeros((N, 3, L))
    for m, o in enumerate(tris):
        uhord[m, :, :len(o)] = o
    # channel UH: WaterSim routing='uh' (gamma shape 3, mean travel = Hack length / CH_VEL)
    ch = channel_geometry(spec['area_km2'], slope=float(np.clip(spec['slope_mean'] / 1000 * 0.05, 2e-4, 0.03)))
    travel_d = ch['length_km'] * 1000.0 / prp['CH_VEL'] / 86400.0
    chord = np.asarray(gamma_uh(3.0, max(travel_d / 3.0, 1e-3)), float)
    # initial stores (HBV defaults), snow empty
    s0 = np.zeros((N, 3))
    for m in range(N):
        s0[m] = [0.6 * mp[m, 0], 5.0, 30.0]
    # augmented log parameters
    aug = list(H.get('augment', []))
    K = len(aug)
    aug_idx = np.array([EK.AUG_CODES[k] for k in aug], np.int64)
    pm0 = pmult_prior(batch) if H.get('pmult_prior', True) else 1.0
    lp0 = np.zeros((N, K))
    lo = np.zeros(K)
    hi = np.zeros(K)
    rw = np.zeros(K)
    asd = H.get('augment_init_sd', {})
    ars = H.get('augment_rw_sd', {})
    z = rng.standard_normal((N, K))
    for k, name in enumerate(aug):
        if name == 'pmult':
            centre, (a, b) = math.log(pm0), PMULT_BOUNDS
        else:
            j = HBV_NAMES.index(name)
            centre, (a, b) = None, HBV_BOUNDS[name]
        a = max(a, 1e-3)                    # PERC / K bounds may start at 0: log space needs a floor
        lo[k], hi[k] = math.log(a), math.log(b)
        rw[k] = float(ars.get(name, 0.0))
        for m in range(N):
            c = centre if centre is not None else math.log(max(mp[m, j], a))
            lp0[m, k] = min(max(c + float(asd.get(name, 0.0)) * z[m, k], lo[k]), hi[k])
    pmult0 = np.full(N, pm0)
    return dict(N=N, mp=mp, sp=sp, frac=frac, dz=dz, pm_b=pm_b, uhord=uhord, chord=chord, s0=s0,
                aug_idx=aug_idx, lp0=lp0, lo=lo, hi=hi, rw=rw, pmult0=pmult0, spec=spec)


def run(batch, H=None, lead=None, return_members=False):
    H = H or HYPER
    S = setup(batch, H)
    N = S['N']
    F = np.asarray(batch['forcing'], float)
    nw, nt = int(batch['n_warm']), int(batch['n_target'])
    lead = int(lead if lead is not None else (batch.get('lead') or 1))
    nspin = min(365, nw) if H.get('spinup', True) else 0
    P = np.concatenate([F[:nspin, 0], F[:, 0]])
    PET = np.maximum(np.concatenate([F[:nspin, 1], F[:, 1]]), 0.0)
    Tm = np.concatenate([F[:nspin, 2], F[:, 2]])
    q = np.asarray(batch['qobs'], float) if batch.get('qobs') is not None else np.full(len(F), np.nan)
    qo = np.concatenate([np.full(nspin, np.nan), q])
    T = len(P)
    rng = np.random.default_rng(int(H.get('seed', 20261123)) + 1)
    sP, sT, sS = float(H['sigma_P']), float(H['sigma_T']), float(H['sigma_S'])
    eP = np.exp(sP * rng.standard_normal((T, N)) - 0.5 * sP * sP)
    eT = sT * rng.standard_normal((T, N))
    eS = np.exp(sS * rng.standard_normal((T, N)) - 0.5 * sS * sS)
    K = len(S['aug_idx'])
    eW = rng.standard_normal((T, N, max(K, 1)))[:, :, :K].copy()
    p_mean = float(math.exp(np.asarray(batch['static'], float)[11]))
    c_obs = float(H['c_frac']) * p_mean
    snow0 = np.zeros((N, 2 * len(S['frac'])))
    pred, info = EK.run_filter(P, Tm, PET, qo, nspin, lead, nspin + nw, S['frac'], S['dz'], S['pm_b'], S['sp'],
                               S['mp'], S['s0'], snow0, S['uhord'], S['chord'], eP, eT, eS, eW, S['aug_idx'],
                               S['lp0'], S['lo'], S['hi'], S['rw'], S['pmult0'], c_obs, float(H['sigma_obs']) ** 2,
                               float(H['inflation']), bool(H.get('update_snow', False)),
                               bool(H.get('update_buffers', True)), True,
                               float(H.get('eps_store', 1.0)), float(H.get('innov_clip', 3.0)))
    M = pred[nspin + nw: nspin + nw + nt]
    if np.isnan(M).any():   # first target rows before an origin exists cannot happen (lead <= 7 < n_warm)
        M = np.where(np.isnan(M), np.nanmean(M, 1, keepdims=True), M)
    M = np.maximum(M, 0.0)
    phi = float(H.get('ar_phi', 0.0))
    if phi > 0:
        # output error correction (AR(1) on the log innovation at the forecast origin; causal: origin = row - lead)
        inn = info[nspin:, 0]
        last = np.full(len(inn), np.nan)
        age = np.zeros(len(inn))
        cur, ca = np.nan, 0.0
        for r in range(len(inn)):
            if np.isfinite(inn[r]):
                cur, ca = inn[r], 0.0
            else:
                ca += 1.0
            last[r], age[r] = cur, ca
        rows = np.arange(nw, nw + nt) - lead
        e = np.where(np.isfinite(last[rows]), last[rows] * phi ** (lead + age[rows]), 0.0)
        M = np.maximum((M + c_obs) * np.exp(e)[:, None] - c_obs, 0.0)
    if return_members:
        return M, info[nspin:], S
    return M, c_obs


def _summarise(M, c_obs, H):
    mean = M.mean(1)
    sf = float(H.get('sigma_fcst_obs', 0.0))
    if sf > 0:
        z = np.random.default_rng(7).standard_normal(int(H.get('n_noise_draws', 8)))
        S = ((M[:, :, None] + c_obs) * np.exp(sf * z[None, None, :]) - c_obs).reshape(len(M), -1)
        S = np.maximum(S, 0.0)
    else:
        S = M
    qs = np.quantile(S, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
    out = {'mean': mean}
    for k, v in zip(('q05', 'q25', 'q50', 'q75', 'q95'), qs):
        out[k] = v
    return out


def predict(batch, H=None):
    H = H or HYPER
    if batch.get('protocol') != 'gauged':
        raise ValueError('enkf_watersim is a gauged-track arm (needs qobs)')
    M, c_obs = run(batch, H)
    return _summarise(M, c_obs, H)
