"""Simulate one generated building with the BuildingDetective wrappers and return meters + labels."""
import numpy as np
import pandas as pd

import common as C
from wrappers import apply_intensity, make_campus, campus_meters

_WX = {}


def get_wx(key):
    if key not in _WX:
        kind, name, yr = key
        if kind == 'site':
            w, m = C.site_wx(name)
            start = pd.Timestamp(f'{yr}-01-01', tz=m['tz']).tz_convert('UTC')
            end = pd.Timestamp(f'{yr + 1}-01-01', tz=m['tz']).tz_convert('UTC')
        else:
            w, m = C.tmy_wx(name)
            start, end = w.index[0], w.index[-1] + pd.Timedelta('1h')
        _WX[key] = (w, m, start, end)
    return _WX[key]


def pick_wx(cz, rng, p_site=0.6):
    sites, tm = C.wx_keys_for_cz(cz)
    if sites and (rng.random() < p_site or not tm):
        return sites[rng.integers(len(sites))]
    return tm[rng.integers(len(tm))]


def wx_name(key):
    return f'{key[0]}:{key[1]}:{key[2]}'


def simulate_one(spec, rng, cal, campus=None, wx_key=None, intensity=True):
    """campus: None = random by cal['campus']['p_campus'] (commercial only), False = never, or a kind string."""
    import buildingsim as bs
    from buildingsim.generator import labels
    f = bs.fill_spec(spec)
    use = f['primary_use']
    if intensity:
        apply_intensity(f, cal['intensity'].get(use, 1.0))
    kind = None
    if not use.startswith('residential'):
        if campus is None and rng.random() < cal['campus']['p_campus']:
            f, kind = make_campus(f, rng, cal)
        elif isinstance(campus, str):
            f, kind = make_campus(f, rng, cal, kind=campus)
    if wx_key is None:
        wx_key = pick_wx(f.get('climate_zone') or '4A', rng)
    w, m, start, end = get_wx(wx_key)
    out = bs.simulate(f, (w, m), start, end, dt='1h', record_trend=False)
    ts = out['ts']
    if kind is not None:
        r = f['hvac']['district_eff'] / f['hvac']['_eff_chw']
        campus_meters(ts, f)
        out['summary']['kWh_by_end_use']['cooling_district'] *= r
        out['summary']['kWh_by_meter']['chilled_water'] *= r
    L = labels(f, out)
    y, cls = C.derived_labels(f, out, L, m['tz'])
    meters = {ch: ts[col].to_numpy(np.float64) for ch, col in C.SIM_COLS.items() if col in ts}
    if 'meter_hot_water_kW' in ts:          # district hot water and steam both read as the district-heat channel
        meters['dheat'] = meters.get('dheat', 0.0) + ts['meter_hot_water_kW'].to_numpy(np.float64)
    if kind == 'cool':          # heat from the campus but not metered at this building
        meters['dheat'] = np.zeros(len(ts))
    meters = {ch: v for ch, v in meters.items() if np.nansum(np.abs(v)) > 1e-6}
    eu = out['summary']['kWh_by_end_use']
    info = dict(id=f['id'], use=use, system=f['system'], cz=f.get('climate_zone'), wx=wx_name(wx_key),
                A=float(f['floor_area_m2']), year=float(f['year_built']), campus=kind,
                intensity=float(f.get('_w1_intensity', 1.0)), heat=cls['heat'], cool=cls['cool'],
                site_kWh=float(out['summary']['site_kWh']), EUI=float(out['summary']['EUI_kWh_m2_yr']),
                kWh_by_meter={k: float(v) for k, v in out['summary']['kWh_by_meter'].items()},
                UA=float(L['UA_network_W_K']), tau_h=float(L['tau_h']), heat_sp=float(L['heat_sp']),
                cool_sp=float(L['cool_sp']), setback=L.get('setback_heat'),
                district_eff=float(f['hvac'].get('district_eff', 1.0)), eff_chw=f['hvac'].get('_eff_chw'),
                heat_source=f['hvac'].get('heat_source'), fuel=f['hvac'].get('fuel'),
                eu={k: float(v) for k, v in eu.items()})
    T = ts['T_out'].to_numpy(np.float32)
    G = ts['GHI'].to_numpy(np.float32)
    return dict(meters=meters, y=y, cls=cls, info=info, T=T, GHI=G, index=ts.index, tz=m['tz'], lat=m['lat'],
                wx_key=wx_key)
