"""simulate(): one basin through one model, snow routine, channel routing, irrigation and reservoirs, with a closing
water-balance ledger and labels. simulate_batch(): many parameter sets / basins at once (numba)."""
import time

import numpy as np
import pandas as pd

from . import engine, routing as rt
from .constants import mm_day_to_m3s
from .forcing import prepare, engine_matrix
from .models import MODULES, MODEL_IDS
from .pet import compute_pet, penman_open_water
from .reservoir import from_spec as reservoir_from_spec, DT_HM3
from .spec import fill_spec, build_bands, resolve_params, channel_for

SNOW_KINDS = {'degree_day': engine.SNOW_DD, 'energy_balance': engine.SNOW_EB, 'none': engine.SNOW_NONE, None: engine.SNOW_NONE}
ROUTINGS = ('none', 'uh', 'muskingum', 'kinematic')


def list_models():
    """Available runoff models, snow routines, routing methods and PET methods with their parameter tables."""
    from . import snow as sn
    from .spec import ROUTING_PARAMS
    return {
        'models': {k: {'params': m.PARAMS, 'states': m.STATES, 'doc': (m.__doc__ or '').strip().split('\n')[0]}
                   for k, m in MODULES.items()},
        'snow': {'degree_day': sn.DD_PARAMS, 'energy_balance': sn.EB_PARAMS, 'none': []},
        'routing': {'none': 'no channel routing', 'uh': 'gamma channel UH (shape 3, mean travel L/CH_VEL)',
                    'muskingum': 'variable-parameter Muskingum-Cunge over the main channel (centroid length L/2)',
                    'kinematic': 'kinematic wave with uniform lateral inflow along the main channel',
                    'params': ROUTING_PARAMS},
        'pet': ['forcing', 'penman_monteith', 'priestley_taylor', 'hargreaves', 'oudin'],
        'reservoir_policies': ['sop', 'rule_curve', 'hanasaki', 'linear', 'run_of_river', 'callable(obs)->m3/s'],
    }


def _spin_days(spinup):
    if spinup in (None, False, 0, '0', ''):
        return 0
    if isinstance(spinup, (int, np.integer)):
        return int(spinup)
    s = str(spinup).strip().lower()
    num = float(s[:-1]) if s[-1] in 'ymd' else float(s)
    unit = s[-1] if s[-1] in 'ymd' else 'd'
    return int(round(num * {'y': 365.25, 'm': 30.44, 'd': 1}[unit]))


def _merge_meta(spec, meta):
    s = dict(spec or {})
    alias = {'area_km2': ['area_km2', 'area'], 'lat': ['lat', 'centroid_lat', 'gauge_lat'],
             'lon': ['lon', 'centroid_lon', 'gauge_lon'], 'elev_mean_m': ['elev_mean_m', 'elevation_mean', 'elev_mean'],
             'elev_bands': ['elev_bands'], 'id': ['basin_id', 'id']}
    for k, names in alias.items():
        if s.get(k) is None:
            for n in names:
                if meta.get(n) is not None:
                    s[k] = meta[n]
                    break
    return s


def state_storage(model, s, pm):
    """(soil, groundwater) storage in mm from a model state vector."""
    if model == 'hbv':
        return s[0], s[1] + s[2]
    if model in ('gr4j', 'vic_lite'):
        return s[0], s[1]
    if model == 'sacsma':
        pa = 1 - pm['PCTIM'] - pm['ADIMP']
        return (s[0] + s[2]) * pa + s[5] * pm['ADIMP'], (s[1] + s[3] + s[4]) * pa
    return 0.0, 0.0


def _snow_vec(snow, ps):
    from . import snow as sn
    if snow == 'degree_day':
        return np.array([ps[p[0]] for p in sn.DD_PARAMS], float)
    if snow == 'energy_balance':
        return np.array([ps[p[0]] for p in sn.EB_PARAMS], float)
    return np.zeros(13)


def _model_vec(model, pm):
    if model not in MODULES:
        return np.zeros(1)
    return np.array([pm[p[0]] for p in MODULES[model].PARAMS], float)


class _Prepared:
    pass


def prepare_run(spec, forcing, model='hbv', params=None, snow='degree_day', pet=None):
    """Resolve spec, forcing, parameters, PET and bands once (shared by simulate and the calibrator)."""
    if model not in MODULES and model != 'none':
        raise ValueError(f'unknown model {model!r}; available: {sorted(MODULES)}')
    if snow not in SNOW_KINDS:
        raise ValueError(f'unknown snow routine {snow!r}')
    df, fmeta, notes = prepare(forcing)
    sp = fill_spec(_merge_meta(spec, fmeta))
    dt = float(np.median(np.diff(df.index.values).astype('timedelta64[s]').astype(float))) / 86400.0 \
        if len(df) > 1 else 1.0
    if abs(dt - 1.0) > 1e-6:
        if 1.0 / dt != round(1.0 / dt):
            raise ValueError(f'time step {dt} days does not divide a day')
        if model == 'gr4j':
            raise NotImplementedError('GR4J is a daily model (use GR4H constants for hourly); use hbv, sacsma or '
                                      'vic_lite for sub-daily steps')
        if snow == 'energy_balance':
            notes.append('energy-balance snow at a sub-daily step uses daily-type surface physics (see README)')
    pm, ps, prp, src = resolve_params(model, snow or 'none', sp, params) if model != 'none' else \
        ({}, *resolve_params('hbv', snow or 'none', sp, params)[1:])
    if model == 'none':
        pm = {}
    lat = float(sp['lat'])
    zref = float(sp['forcing_elev_m'])
    method = pet or sp.get('pet_method') or ('forcing' if 'PET' in df and df['PET'].notna().mean() > 0.9
                                             else 'penman_monteith')
    net = bool(sp.get('srad_is_net') or fmeta.get('srad_is_net'))
    if abs(dt - 1.0) > 1e-6 and method != 'forcing':
        # sub-daily: daily PET from daily aggregates, spread over the day by solar elevation
        from .pet import disaggregate_daily
        dd = pd.DataFrame({'P': df['P'].resample('D').sum(), 'T_mean': df['T_mean'].resample('D').mean(),
                           'T_min': df['T_mean'].resample('D').min(), 'T_max': df['T_mean'].resample('D').max(),
                           'SRAD': df['SRAD'].resample('D').mean(), 'vp': df['vp'].resample('D').mean(),
                           'wind': df['wind'].resample('D').mean()})
        pd_, pet_used = compute_pet(dd, lat, zref, method, float(sp.get('wind_height') or 10.0), srad_is_net=net)
        pet_arr = disaggregate_daily(pd.Series(pd_, index=dd.index), df.index, lat, float(sp.get('lon') or 0.0))
        pet_used += ' (daily, disaggregated by solar elevation)'
    else:
        pet_arr, pet_used = compute_pet(df, lat, zref, method, float(sp.get('wind_height') or 10.0),
                                        srad_is_net=net)
        if abs(dt - 1.0) > 1e-6:
            pet_used += ' (per-step amounts as given)'
    frac, elev = build_bands(sp)
    F, rso = engine_matrix(df, lat, zref, srad_is_net=net)
    o = _Prepared()
    o.df, o.meta, o.notes, o.spec = df, fmeta, notes, sp
    o.pm, o.ps, o.prp, o.src = pm, ps, prp, src
    o.pet, o.pet_method = pet_arr, pet_used
    o.frac, o.elev, o.dz = frac, elev, (elev - zref) / 1000.0
    o.F, o.rso = F, rso
    o.model, o.snow = model, snow or 'none'
    o.dt = dt
    return o


def _window(df, start, end):
    idx = df.index
    s = pd.Timestamp(start) if start is not None else idx[0]
    e = pd.Timestamp(end) if end is not None else idx[-1]
    i0 = int(idx.searchsorted(s))
    i1 = int(idx.searchsorted(e, side='right'))
    if i1 <= i0:
        raise ValueError(f'empty simulation window {start}..{end}')
    return i0, i1


def _run(o, a, b, s, snowst, uhbuf, uhord, pet_red, infil_cap=0.0, uhx=None):
    T = b - a
    rec = np.zeros((T, engine.NREC))
    swe_b = np.zeros((T, len(o.frac)))
    if uhx is None:
        uhx = np.zeros(uhord.shape[1])
    engine.run_core(MODEL_IDS.get(o.model, -1), SNOW_KINDS[o.snow], o.F[a:b], o.pet[a:b], o.rso[a:b], o.dt,
                    o.frac, o.dz, o.elev, _snow_vec(o.snow, o.ps), _model_vec(o.model, o.pm), s, snowst, uhord,
                    uhbuf, pet_red, rec, swe_b, float(infil_cap), uhx)
    return rec, swe_b


def infiltration_capacity(spec, dt, infiltration_excess):
    """Infiltration capacity in mm per step: None/'auto' -> on for sub-daily steps only, from the saturated
    conductivity (spec soil_conductivity, cm/h) reduced by the urban share; a number = capacity in mm/h."""
    if infiltration_excess is False or (infiltration_excess in (None, 'auto') and dt >= 1.0 - 1e-9):
        return 0.0
    if isinstance(infiltration_excess, (int, float)) and not isinstance(infiltration_excess, bool):
        mmh = float(infiltration_excess)
    else:
        mmh = 10.0 * float(spec.get('soil_conductivity') or 1.6) * (1.0 - float(spec.get('urban_frac') or 0.0))
    return max(mmh, 0.01) * 24.0 * dt


def simulate(spec, forcing, start=None, end=None, model='hbv', params=None, snow='degree_day', routing='uh',
             reservoirs=None, init=None, spinup='1y', rng=None, pet=None, irrigation=None, pet_snow_reduction=True,
             return_bands=False, labels=True, infiltration_excess=None, obs_noise=None):
    """Simulate one basin (daily step). See the README for every argument and output.

    spec      basin spec dict (contract); missing fields get documented defaults (out['spec']['src_fill'])
    forcing   DataFrame (UTC daily index, contract columns) or (DataFrame, meta) or a parquet path
    start/end inclusive dates of the reported window (default: whole forcing)
    model     'hbv' | 'gr4j' | 'sacsma' | 'vic_lite' | 'none' (snow + routing only)
    params    flat or nested dict overriding spec['params'] (model, snow and routing parameters)
    snow      'degree_day' | 'energy_balance' | 'none'
    routing   'uh' | 'muskingum' | 'kinematic' | 'none' (channel routing after the model's own hillslope UH)
    reservoirs list of reservoir dicts / Reservoir objects (default: spec.get('reservoirs'))
    init      a previous out['final_state'] (then no spin-up) or {'model': {...}, 'snow_swe_mm': x}. A final_state
              carries reservoir storage too, so a chained run continues each reservoir where it stopped. The channel
              routing buffer is not carried (chained runs with routing other than 'none' differ by ~1e-6 mm/day), and
              reservoir settings derived from the window (capacity_years, the default demand of 0.6 x mean inflow,
              policies that read the window's mean inflow or history) are recomputed per window
    spinup    warm-up length before `start` ('1y', '2y', '180d', 0); taken from the forcing before `start` when it
              exists, otherwise the first part of the window is run once as a warm-up
    irrigation None (use spec['irrigation_frac']), False, or a dict of irrigation settings
    infiltration_excess None/'auto' (on for sub-daily steps), True, False, or a capacity in mm/h
    obs_noise None, True or a dict of noise.streamflow_obs settings: adds ts['Q_obs'] (gauge-like, uses rng)
    The step is taken from the forcing index (daily, or e.g. hourly for hbv/sacsma/vic_lite; amounts per step).
    """
    t0 = time.time()
    rng = np.random.default_rng(rng)
    o = prepare_run(spec, forcing, model, params, snow, pet)
    sp = o.spec
    area = float(sp['area_km2'])
    i0, i1 = _window(o.df, start, end)
    mod = MODULES.get(model)
    # ------------------------------------------------------------------ initial state + spin-up
    B = len(o.frac)
    dt = o.dt
    uhord = rt.model_uh(model, o.pm, dt)
    L = uhord.shape[1]
    icap = infiltration_capacity(sp, dt, infiltration_excess)
    uhx = np.zeros(L)
    if init is not None and 'model_state' in init:
        s = np.array(init['model_state'], float)
        snowst = np.array(init['snow'], float)
        uhbuf = np.array(init['uhbuf'], float)
        if init.get('uhx') is not None:
            uhx = np.array(init['uhx'], float)
        if snowst.shape != (B, 4) or uhbuf.shape != (3, L):
            raise ValueError('init final_state does not match this spec/model (bands or UH length)')
        nspin = 0
        spin_src = 'init'
    else:
        s = np.array(mod.init_state(o.pm, (init or {}).get('model')) if mod else [0.0], float)
        snowst = np.zeros((B, 4))
        if init and init.get('snow_swe_mm'):
            snowst[:, 0] = float(init['snow_swe_mm'])
        uhbuf = np.zeros((3, L))
        nspin = 0 if init is not None else int(round(_spin_days(spinup) / dt))
        spin_src = 'none'
    if nspin > 0:
        if i0 >= int(0.9 * nspin):
            a = max(i0 - nspin, 0)
            _run(o, a, i0, s, snowst, uhbuf, uhord, pet_snow_reduction, icap, uhx)
            spin_src = f'forcing {o.df.index[a].date()}..{o.df.index[i0 - 1].date()}'
        else:
            b = min(i0 + nspin, i1)
            _run(o, i0, b, s, snowst, uhbuf, uhord, pet_snow_reduction, icap, uhx)
            spin_src = f'cyclic: first {b - i0} steps of the window run once as warm-up'
    soil0, gw0 = state_storage(model, s, o.pm)
    snow0 = float(np.sum(o.frac * (snowst[:, 0] + snowst[:, 1])))
    mroute0 = float(uhbuf.sum() + uhx.sum())
    s_init = s.copy()
    # ------------------------------------------------------------------ main run
    rec, swe_b = _run(o, i0, i1, s, snowst, uhbuf, uhord, pet_snow_reduction, icap, uhx)
    idx = o.df.index[i0:i1]
    ts = pd.DataFrame(rec, index=idx, columns=engine.REC)
    ts.insert(0, 'P_forcing', o.df['P'].values[i0:i1])
    ts['T_mean'] = o.df['T_mean'].values[i0:i1]
    qm = ts['Q_model'].values
    # ------------------------------------------------------------------ channel routing
    ch = channel_for(sp, area, o.prp)
    comps = ts[['Q_quick', 'Q_inter', 'Q_base']].values
    if routing in (None, 'none'):
        q_nat, ch_store, ch0 = qm.copy(), np.zeros(len(qm)), 0.0
        comp_r = comps.copy()
    elif routing == 'uh':
        travel_d = ch['length_km'] * 1000.0 / o.prp['CH_VEL'] / 86400.0
        ordc = rt.gamma_uh(3.0, max(travel_d / 3.0, 1e-3), dt)
        q_nat, ch_store, ch0 = _uh_route(qm, ordc)
        comp_r = np.column_stack([_uh_route(comps[:, j], ordc)[0] for j in range(3)])
    elif routing in ('muskingum', 'kinematic'):
        qin = mm_day_to_m3s(qm / dt, area)
        if routing == 'muskingum':
            qo, S = rt.muskingum_cunge(qin, 0.5 * ch['length_km'], ch['slope'], ch['width_m'], ch['manning_n'],
                                       dt_days=dt)
            ch0 = 0.0
        else:
            qo, S = rt.kinematic(qin, ch['length_km'], ch['slope'], ch['width_m'], ch['manning_n'], dt_days=dt)
            ch0 = rt.kinematic_initial_storage(qin[0], ch['length_km'], ch['slope'], ch['width_m'], ch['manning_n']) \
                * 86.4 / area
        q_nat = qo * 86.4 / area * dt
        ch_store = S * 86.4 / area
        # component shares from a linear (gamma) channel response of the same mean travel time
        travel_d = max(ch['length_km'] * 500.0 / 86400.0, 1e-3)
        ordc = rt.gamma_uh(3.0, travel_d / 3.0, dt)
        cr = np.column_stack([_uh_route(comps[:, j], ordc)[0] for j in range(3)])
        tot = cr.sum(1)
        share = np.where(tot[:, None] > 0, cr / np.where(tot > 0, tot, 1)[:, None], 1.0 / 3)
        comp_r = share * q_nat[:, None]
    else:
        raise ValueError(f'unknown routing {routing!r}; choose from {ROUTINGS}')
    ts['Q_natural'] = q_nat
    ts['channel_storage'] = ch_store
    ts['Q_quick_routed'], ts['Q_inter_routed'], ts['Q_base_routed'] = comp_r[:, 0], comp_r[:, 1], comp_r[:, 2]
    # ------------------------------------------------------------------ irrigation
    q = q_nat.copy()
    irr = _irrigation(sp, irrigation, ts, o.pet[i0:i1], q)
    q = irr['q']
    ts['withdrawal'] = irr['withdrawal']
    ts['return_flow'] = irr['return_flow']
    ts['irrigation_store'] = irr['store']
    # ------------------------------------------------------------------ reservoirs
    res_list = reservoirs if reservoirs is not None else sp.get('reservoirs')
    res_out = []
    res_evap_mm = np.zeros(len(q))
    res_withdraw_mm = np.zeros(len(q))
    res_store_mm = np.zeros(len(q))
    res_store0_mm = 0.0
    res_final = None
    if res_list and dt < 1.0 - 1e-9:
        raise NotImplementedError('reservoirs run at a daily step; aggregate the forcing to daily')
    if res_list:
        dfm = o.df.iloc[i0:i1]
        e_ow = penman_open_water(dfm['T_mean'].values, dfm['T_max'].values, dfm['T_min'].values, o.F[i0:i1, 4],
                                 o.F[i0:i1, 5], o.F[i0:i1, 6], float(sp['lat']), idx.dayofyear.values,
                                 float(sp['forcing_elev_m']), float(sp.get('wind_height') or 10))
        q_m3s = mm_day_to_m3s(q, area)
        imean = float(q_m3s.mean())
        objs = [reservoir_from_spec(r, imean) for r in res_list]
        if init is not None and init.get('reservoir_storage_hm3') is not None:
            S_prev = [float(v) for v in init['reservoir_storage_hm3']]
            if len(S_prev) != len(objs):
                raise ValueError('init final_state has a different number of reservoirs than this run')
            for r, v in zip(objs, S_prev):
                r.reset(storage_hm3=v)
        outflow = {}
        captured = np.zeros(len(objs))
        for k, (r, rd) in enumerate(zip(objs, res_list)):
            frm = rd.get('inflow_from') if isinstance(rd, dict) else getattr(r, 'inflow_from', None)
            a_r = r.upstream_frac
            if frm is not None:
                inflow = q_m3s * max(a_r - objs[frm].upstream_frac, 0.0) + outflow[frm]
                captured[frm] = 1   # upstream reservoir's outflow is not terminal
            else:
                inflow = q_m3s * a_r
            res0 = r.S
            ro = r.run(inflow, idx, e_ow)
            ro['inflow_m3s'] = inflow
            outflow[k] = ro['release_m3s'] - ro['withdrawal_m3s'] + ro['spill_m3s']
            res_evap_mm += ro['evap_hm3'] * 1000.0 / area
            res_withdraw_mm += ro['withdrawal_m3s'] * DT_HM3 * 1000.0 / area
            res_store_mm += ro['storage_hm3'] * 1000.0 / area
            res_store0_mm += res0 * 1000.0 / area
            res_out.append({'name': r.name, 'reservoir': r, 'series': pd.DataFrame(
                {k2: v for k2, v in ro.items() if isinstance(v, np.ndarray)}, index=idx)})
        # natural flow that reaches the outlet without passing a reservoir: outside the terminal reservoirs' areas
        bypass = 1.0 - sum(objs[k].upstream_frac for k in range(len(objs)) if not captured[k])
        q_out_m3s = q_m3s * max(bypass, 0.0)
        for k in range(len(objs)):
            if not captured[k]:
                q_out_m3s = q_out_m3s + outflow[k]
        q = q_out_m3s * 86.4 / area
        ts['res_storage_hm3'] = res_store_mm * area / 1000.0
        res_final = [float(r.S) for r in objs]
        ts['res_release_m3s'] = sum(ro['series']['release_m3s'] for ro in res_out)
        ts['res_spill_m3s'] = sum(ro['series']['spill_m3s'] for ro in res_out)
        ts['res_evap_mm'] = res_evap_mm
        ts['res_power_MW'] = sum(ro['series']['power_MW'] for ro in res_out)
    ts['res_withdrawal_mm'] = res_withdraw_mm
    ts['Q'] = q
    ts['Q_m3s'] = mm_day_to_m3s(q / dt, area)
    if obs_noise:
        from .noise import streamflow_obs
        ts['Q_obs'] = streamflow_obs(q, rng, **(obs_noise if isinstance(obs_noise, dict) else {}))
    ts['ET'] = ts['ET_soil'] + ts['sublimation'] + res_evap_mm
    # ------------------------------------------------------------------ ledger
    ledger = _ledger(ts, snow0, soil0, gw0, mroute0, ch0, irr['store0'], res_store0_mm, res_store_mm, res_evap_mm,
                     res_withdraw_mm)
    out = {'ts': ts, 'ledger': ledger, 'spec': sp, 'model': model, 'snow': o.snow, 'routing': routing,
           'params': {'model': o.pm, 'snow': o.ps, 'routing': o.prp}, 'param_source': o.src,
           'pet_method': o.pet_method, 'forcing_notes': o.notes, 'spinup': spin_src, 'channel': ch,
           'bands': {'frac': o.frac.tolist(), 'elev_m': o.elev.tolist()}, 'reservoirs': res_out,
           'dt_days': dt, 'infiltration_capacity_mm_per_step': icap,
           'final_state': {'model_state': s.tolist(), 'snow': snowst.tolist(), 'uhbuf': uhbuf.tolist(),
                           'uhx': uhx.tolist(), 'reservoir_storage_hm3': res_final,
                           'model': model, 'snow_kind': o.snow, 'date': str(idx[-1].date())},
           'initial_state': {'model_state': s_init.tolist()}}
    if return_bands:
        out['swe_bands'] = pd.DataFrame(swe_b, index=idx, columns=[f'band{b}_{int(z)}m' for b, z in enumerate(o.elev)])
    if labels:
        from .labels import compute_labels
        out['labels'] = compute_labels(out)
    out['wall_s'] = time.time() - t0
    out['x_realtime'] = len(idx) * dt * 86400.0 / max(out['wall_s'], 1e-9)
    return out


def _uh_route(x, ordc):
    """Convolve with a steady history equal to x[0]; return (out, pending storage after each step, initial storage)."""
    L = len(ordc)
    pre = np.full(L, x[0])
    xx = np.concatenate([pre, x])
    y = rt.convolve_uh(xx, ordc)[L:]
    rem = 1.0 - np.cumsum(ordc)
    rem[rem < 0] = 0.0
    st = rt.convolve_uh(xx, rem)[L - 1:]
    return y, st[1:], st[0]


def _irrigation(sp, irrigation, ts, pet, q):
    n = len(q)
    zero = {'q': q, 'withdrawal': np.zeros(n), 'return_flow': np.zeros(n), 'store': np.zeros(n), 'store0': 0.0,
            'consumptive': np.zeros(n)}
    if irrigation is False:
        return zero
    cfg = {'frac': sp.get('irrigation_frac') or 0.0, 'efficiency': 0.6, 'max_take': 0.6, 'return_frac': 0.3,
           'k_return': 0.05, 'season_T': 10.0}
    if isinstance(irrigation, dict):
        cfg.update(irrigation)
    if cfg['frac'] <= 0:
        return zero
    # crop water deficit on the irrigated share of the basin (PET minus rain), withdrawn from the river
    deficit = np.maximum(pet - ts['rain'].values, 0.0) * (ts['T_mean'].values > cfg['season_T'])
    demand = cfg['frac'] * deficit / cfg['efficiency']
    w = np.minimum(demand, cfg['max_take'] * q)
    ret = np.zeros(n)
    store = np.zeros(n)
    S = 0.0
    for t in range(n):
        S += cfg['return_frac'] * w[t]
        r = cfg['k_return'] * S
        S -= r
        ret[t] = r
        store[t] = S
    qq = q - w + ret
    return {'q': qq, 'withdrawal': w, 'return_flow': ret, 'store': store, 'store0': 0.0,
            'consumptive': (1 - cfg['return_frac']) * w}


def _ledger(ts, snow0, soil0, gw0, mroute0, ch0, irr0, res0, res_store, res_evap, res_withdraw):
    """Water balance over the window in mm over the basin area. Closure must be ~0 (round-off)."""
    n = len(ts)
    P = ts['P'].values
    exch = ts['exchange'].values
    et = ts['ET_soil'].values
    subl = ts['sublimation'].values
    Q = ts['Q'].values
    stores = {
        'snow': (snow0, ts['SWE'].values),
        'soil': (soil0, ts['soil'].values),
        'groundwater': (gw0, ts['groundwater'].values),
        'model_routing': (mroute0, ts['routing_store'].values),
        'channel': (ch0, ts['channel_storage'].values),
        'irrigation_return': (irr0, ts['irrigation_store'].values),
        'reservoirs': (res0, res_store),
    }
    # irrigation: consumptive loss = withdrawal - return flow - change of the return-flow store
    irr_loss = ts['withdrawal'].values - ts['return_flow'].values - np.diff(
        np.concatenate([[irr0], ts['irrigation_store'].values]))
    totS = np.zeros(n + 1)
    dS = {}
    for name, (s0, s) in stores.items():
        dS[name] = float(s[-1] - s0)
        totS += np.concatenate([[s0], s])
    resid = P + exch - et - subl - Q - res_evap - res_withdraw - irr_loss - np.diff(totS)
    Psum = float(P.sum())
    led = {
        'units': 'mm over the basin area, whole window',
        'P_forcing': float(ts['P_forcing'].sum()),
        'snowfall_correction': float(P.sum() - ts['P_forcing'].sum()),
        'P_in': Psum,
        'exchange_gain': float(exch[exch > 0].sum()),
        'exchange_loss': float(-exch[exch < 0].sum()),
        'ET_soil': float(et.sum()),
        'sublimation': float(subl.sum()),
        'reservoir_evaporation': float(res_evap.sum()),
        'Q_out': float(Q.sum()),
        'withdrawals_consumptive': float(irr_loss.sum() + res_withdraw.sum()),
        'dS': dS,
        'dS_total': float(sum(dS.values())),
    }
    lhs = led['P_in'] + led['exchange_gain']
    rhs = led['exchange_loss'] + led['ET_soil'] + led['sublimation'] + led['reservoir_evaporation'] + \
        led['Q_out'] + led['withdrawals_consumptive'] + led['dS_total']
    led['closure_mm'] = float(lhs - rhs)
    scale = max(Psum, led['Q_out'], led['ET_soil'], abs(led['dS_total']), 1.0)
    led['closure_rel'] = float(abs(lhs - rhs) / scale)
    led['max_daily_residual_mm'] = float(np.max(np.abs(resid))) if n else 0.0
    led['ok'] = bool(led['closure_rel'] < 1e-8 and led['max_daily_residual_mm'] < 1e-8 * max(1.0, P.max(), Q.max()))
    return led


# ---------------------------------------------------------------------------------------------- batch
def _spin_rows(i0, i1, nspin):
    """Rows of the forcing a batch run steps through, and how many leading steps to drop: the `nspin` steps before
    the window when at least 90 % of them exist, otherwise the first steps of the window replayed (cyclic), as
    simulate() does."""
    if nspin > 0 and i0 >= int(0.9 * nspin):
        a = max(i0 - nspin, 0)
        return np.arange(a, i1), i0 - a
    if nspin > 0:
        b = min(i0 + nspin, i1)
        return np.concatenate([np.arange(i0, b), np.arange(i0, i1)]), b - i0
    return np.arange(i0, i1), 0


def simulate_batch(spec, forcing, params_list, model='hbv', snow='degree_day', start=None, end=None, spinup='1y',
                   outputs=('Q',), pet=None, pet_snow_reduction=True, routing='none', o=None, infiltration_excess=None):
    """Run many parameter sets for one basin (one forcing) in a single numba call.
    params_list: list of flat dicts (model + snow + routing parameters; missing ones from spec/defaults) or an
    (N, n_model + n_snow) array in param_table order. Returns dict with 'Q' (N, T) mm/day [and 'SWE', 'ET', 'Q_base'],
    'index' (dates). Spin-up: the `spinup` days before `start` (or the first days of the window, cyclic) are run
    and discarded. routing: 'none' or 'uh' (gamma channel UH; the nonlinear ones are single-run only)."""
    if o is None:
        o = prepare_run(spec, forcing, model, None, snow, pet)
    i0, i1 = _window(o.df, start, end)
    nspin = int(round(_spin_days(spinup) / o.dt))
    rows, drop = _spin_rows(i0, i1, nspin)
    F, petv, rso = o.F[rows], o.pet[rows], o.rso[rows]
    pms, pss, prs = [], [], []
    mod = MODULES.get(model)
    from .spec import param_table
    names = [r[0] for r in param_table(model, snow) if r[-1] in ('model', 'snow')]
    if isinstance(params_list, np.ndarray):
        params_list = [dict(zip(names, row)) for row in params_list]
    for p in params_list:
        pm = {**o.pm, **{k: v for k, v in p.items() if k in o.pm}}
        ps = {**o.ps, **{k: v for k, v in p.items() if k in o.ps}}
        pr = {**o.prp, **{k: v for k, v in p.items() if k in o.prp}}
        pms.append(pm)
        pss.append(ps)
        prs.append(pr)
    N = len(pms)
    mp = np.array([_model_vec(model, pm) for pm in pms])
    spv = np.array([_snow_vec(o.snow, ps) for ps in pss])
    s0 = np.array([mod.init_state(pm) for pm in pms], float) if mod else np.zeros((N, 1))
    B = len(o.frac)
    uh = rt.stack_uh([rt.model_uh(model, pm, o.dt) for pm in pms])
    icap = infiltration_capacity(o.spec, o.dt, infiltration_excess)
    T = len(F)
    Q = np.zeros((N, T))
    SWE = np.zeros((N, T)) if 'SWE' in outputs else np.zeros((1, 1))
    ET = np.zeros((N, T)) if 'ET' in outputs else np.zeros((1, 1))
    QB = np.zeros((N, T)) if 'Q_base' in outputs else np.zeros((1, 1))
    engine.run_batch(MODEL_IDS[model], SNOW_KINDS[o.snow], F[None], petv[None], rso[None], o.dt,
                     np.repeat(o.frac[None], N, 0), np.repeat(o.dz[None], N, 0), np.repeat(o.elev[None], N, 0),
                     spv, mp, s0, np.zeros((N, B, 4)), uh, pet_snow_reduction, np.zeros(N, np.int64),
                     'SWE' in outputs, 'ET' in outputs, 'Q_base' in outputs, Q, SWE, ET, QB, np.full(N, icap))
    res = {'Q': Q[:, drop:], 'index': o.df.index[i0:i1]}
    if routing == 'uh':
        ch = channel_for(o.spec, float(o.spec['area_km2']), o.prp)
        ords = {}                                        # each member's own CH_VEL (as simulate() would use)
        routed = []
        for qq, pr in zip(res['Q'], prs):
            v = float(pr['CH_VEL'])
            if v not in ords:
                travel_d = ch['length_km'] * 1000.0 / v / 86400.0
                ords[v] = rt.gamma_uh(3.0, max(travel_d / 3.0, 1e-3), o.dt)
            routed.append(_uh_route(qq, ords[v])[0])
        res['Q'] = np.array(routed)
    if 'SWE' in outputs:
        res['SWE'] = SWE[:, drop:]
    if 'ET' in outputs:
        res['ET'] = ET[:, drop:]
    if 'Q_base' in outputs:
        res['Q_base'] = QB[:, drop:]
    return res


def simulate_many(specs, forcings, model='hbv', snow='degree_day', start=None, end=None, spinup='1y',
                  outputs=('Q',), pet=None):
    """Many basins (each spec with its own params and forcing frame) in one numba call. All forcings must cover
    the same window. Returns dict of (N, T) arrays plus 'index'."""
    preps = [prepare_run(s, f, model, None, snow, pet) for s, f in zip(specs, forcings)]
    for k, p in enumerate(preps[1:], 1):            # rows are shared, so the dates must be too
        if not p.df.index.equals(preps[0].df.index):
            raise ValueError(f'simulate_many: forcing {k} has a different date index from forcing 0; '
                             'give every basin the same forcing dates (reindex or slice them first)')
    i0, i1 = _window(preps[0].df, start, end)
    dt = preps[0].dt
    nspin = int(round(_spin_days(spinup) / dt))
    rows, drop = _spin_rows(i0, i1, nspin)            # same warm-up rule as simulate() and simulate_batch()
    T = len(rows)
    N = len(preps)
    Bmax = max(len(p.frac) for p in preps)
    F = np.stack([p.F[rows] for p in preps])
    petv = np.stack([p.pet[rows] for p in preps])
    rso = np.stack([p.rso[rows] for p in preps])
    bf = np.zeros((N, Bmax))
    bdz = np.zeros((N, Bmax))
    bz = np.zeros((N, Bmax))
    for i, p in enumerate(preps):
        nb = len(p.frac)
        bf[i, :nb], bdz[i, :nb], bz[i, :nb] = p.frac, p.dz, p.elev
    mod = MODULES[model]
    mp = np.array([_model_vec(model, p.pm) for p in preps])
    spv = np.array([_snow_vec(p.snow, p.ps) for p in preps])
    s0 = np.array([mod.init_state(p.pm) for p in preps], float)
    uh = rt.stack_uh([rt.model_uh(model, p.pm, dt) for p in preps])
    icap = np.array([infiltration_capacity(p.spec, dt, None) for p in preps])
    Q = np.zeros((N, T))
    SWE = np.zeros((N, T)) if 'SWE' in outputs else np.zeros((1, 1))
    ET = np.zeros((N, T)) if 'ET' in outputs else np.zeros((1, 1))
    QB = np.zeros((N, T)) if 'Q_base' in outputs else np.zeros((1, 1))
    engine.run_batch(MODEL_IDS[model], SNOW_KINDS[preps[0].snow], F, petv, rso, dt, bf, bdz, bz, spv, mp, s0,
                     np.zeros((N, Bmax, 4)), uh, True, np.arange(N, dtype=np.int64), 'SWE' in outputs,
                     'ET' in outputs, 'Q_base' in outputs, Q, SWE, ET, QB, icap)
    out = {'Q': Q[:, drop:], 'index': preps[0].df.index[i0:i1]}
    for k, arr in (('SWE', SWE), ('ET', ET), ('Q_base', QB)):
        if k in outputs:
            out[k] = arr[:, drop:]
    return out


def simulate_network(subbasins, forcings, topology, start=None, end=None, model='hbv', snow='degree_day',
                     local_routing='uh', **kw):
    """Several sub-basins joined by Muskingum-Cunge reaches.
    subbasins: {id: spec}; forcings: {id: forcing}; topology: {id: {'down': downstream id or None, 'length_km',
    'slope', 'width_m', 'manning_n'}} where the reach runs from this sub-basin's outlet to the downstream node
    (the outlet node has down=None). Each sub-basin is simulated with `local_routing` for its own channel; extra
    keyword arguments go to simulate(). Returns {'nodes': DataFrame of total flow at each node (m3/s), 'outlet',
    'subbasins': {id: out}, 'ledger': network balance in hm3}."""
    outs = {i: simulate(subbasins[i], forcings[i], start, end, model=model, snow=snow, routing=local_routing, **kw)
            for i in subbasins}
    idx = next(iter(outs.values()))['ts'].index
    dts = {round(float(o['dt_days']), 9) for o in outs.values()}
    if len(dts) != 1:
        raise ValueError(f'simulate_network: sub-basins have different time steps {sorted(dts)} (days)')
    dt = dts.pop()
    local = {i: o['ts']['Q_m3s'].reindex(idx).values for i, o in outs.items()}
    at_node, storage, outlet = rt.route_network(local, topology, dt_days=dt)
    nodes = pd.DataFrame(at_node, index=idx)
    # network ledger (hm3): sum of sub-basin balances + reach storage change; outlet flow replaces local outflows
    hm3 = 0.0864
    P = sum(o['ledger']['P_in'] * float(o['spec']['area_km2']) / 1000 for o in outs.values())
    gain = sum(o['ledger']['exchange_gain'] * float(o['spec']['area_km2']) / 1000 for o in outs.values())
    loss = sum((o['ledger']['exchange_loss'] + o['ledger']['ET_soil'] + o['ledger']['sublimation'] +
                o['ledger']['reservoir_evaporation'] + o['ledger']['withdrawals_consumptive'] + o['ledger']['dS_total'])
               * float(o['spec']['area_km2']) / 1000 for o in outs.values())
    reach_dS = sum(float(s[-1]) for s in storage.values()) * hm3
    q_out = float(nodes[outlet].sum()) * hm3 * dt      # m3/s samples x step (days); reach storage is in m3/s x days
    closure = P + gain - loss - reach_dS - q_out
    return {'nodes': nodes, 'outlet': outlet, 'subbasins': outs, 'reach_storage': storage,
            'ledger': {'units': 'hm3', 'P_in': P, 'Q_outlet': q_out, 'reach_dS': reach_dS, 'closure_hm3': closure,
                       'closure_rel': abs(closure) / max(P, 1e-9), 'ok': abs(closure) / max(P, 1e-9) < 1e-8 and
                       all(o['ledger']['ok'] for o in outs.values())}}
