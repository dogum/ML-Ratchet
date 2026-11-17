"""Real-data readers for BuildingDetective: BDG2 (hourly), RECS / CBECS (annual), HUE (hourly homes).

The Detective sees only meters + weather. Labels are returned separately and are never features.
BDG2 channel map: electricity -> elec, gas -> fuel, steam + hotwater -> dheat, chilledwater -> dcool.
Input year: local calendar 2017 (as B's PRISM / floor-area baselines). A meter is used when it has >= 60 days
with >= 12 valid hours in 2017 (bad hours are already NaN in bd.bdg2).
"""
import os

import numpy as np
import pandas as pd

import common as C

_BD = {}


def _bd():
    if 'bd' not in _BD:
        import buildingdata as bd
        _BD['bd'] = bd
    return _BD['bd']


def bdg2_buildings(split):
    bd = _bd()
    return bd.bdg2_meta(split=split)


_WCACHE = {}


def bdg2_case(bid, year=2017, meta_row=None):
    """Returns dict(meters, T, GHI, loc, lat, labels, meters_present) for one BDG2 building, or None."""
    bd = _bd()
    site = bid.split('_')[0]
    tz = bd.SITE_TZ[site]
    if site not in _WCACHE:
        w = bd.site_weather(site)
        _WCACHE[site] = (w, bd.site_weather_meta(site))
    w, wm = _WCACHE[site]
    start = pd.Timestamp(f'{year}-01-01', tz=tz).tz_convert('UTC')
    end = pd.Timestamp(f'{year + 1}-01-01', tz=tz).tz_convert('UTC')
    idx = pd.date_range(start, end, freq='h', inclusive='left')
    row = meta_row if meta_row is not None else bd.bdg2_meta().set_index('id').loc[bid]
    present = [m for m in str(row['meters']).split(',') if m]
    meters = {}
    used = []
    for ch, mm in C.BDG2_METER.items():
        acc = None
        for m in (mm if isinstance(mm, list) else [mm]):
            if m not in present:
                continue
            y = bd.bdg2(m, building=bid, start=str(start - pd.Timedelta('1D')), end=str(end + pd.Timedelta('1D')))
            if y is None or y.shape[1] == 0:
                continue
            s = y.iloc[:, 0].reindex(idx).to_numpy(float)
            d = pd.Series(np.isfinite(s)).groupby(np.arange(len(s)) // 24).sum()
            if (d >= 12).sum() < 60 or np.nansum(s) <= 0:
                continue
            acc = s if acc is None else np.where(np.isfinite(acc) & np.isfinite(s), acc + s, np.nan)
            used.append(m)
        if acc is not None:
            meters[ch] = acc
    if not meters:
        return None
    T = w['T_out'].reindex(idx).interpolate(limit=6).to_numpy(float)
    G = w['GHI'].reindex(idx).fillna(0).to_numpy(float)
    loc = idx.tz_convert(tz)
    lab = dict(use=row['primary_use'], A=float(row['floor_area_m2']) if pd.notna(row['floor_area_m2']) else np.nan,
               year=float(row['year_built']) if pd.notna(row['year_built']) else np.nan,
               meters=present, used=used, site=site, cz=row['climate_zone'],
               has_chw='chilledwater' in present, has_dheat=('steam' in present) or ('hotwater' in present),
               has_gas='gas' in present)
    return dict(meters=meters, T=T, GHI=G, loc=loc, lat=float(wm['lat']), labels=lab, id=bid)


def recs_rows(split):
    bd = _bd()
    r = bd.recs(split=split)
    rows = []
    for x in r.itertuples():
        fuel = sum(v for v in (x.gas_kWh, x.propane_kWh, x.fueloil_kWh, x.wood_kWh) if v is not None and np.isfinite(v))
        kwh = dict(elec=x.elec_kWh, fuel=fuel)
        rows.append(dict(id=x.id, kwh=kwh, hdd=x.hdd18C_year, cdd=x.cdd18C_year, row=x))
    return r, rows


def cbecs_rows(split):
    bd = _bd()
    c = bd.cbecs(split=split)
    rows = []
    for x in c.itertuples():
        fuel = sum(v for v in (x.gas_kWh, x.fueloil_kWh) if v is not None and np.isfinite(v))
        kwh = dict(elec=x.elec_kWh, fuel=fuel, dheat=x.district_heat_kWh if np.isfinite(x.district_heat_kWh or np.nan) else 0)
        rows.append(dict(id=x.id, kwh=kwh, hdd=x.hdd18C_year, cdd=x.cdd18C_year, row=x))
    return c, rows


def recs_heat_label(row):
    """RECS heating equipment -> Detective heat class (None if not mappable)."""
    he, fu = row.heating_equipment, row.heating_fuel
    if he in ('heat_pump_air', 'heat_pump_ground'):
        return 'heat_pump'
    if he in ('baseboard_resistance', 'electric_furnace') or (fu == 'electricity' and he in ('portable_electric',)):
        return 'resistance'
    if fu in ('natural_gas', 'propane', 'fuel_oil'):
        return 'fuel'
    if fu == 'none' or he == 'none':
        return 'none'
    return None


def cbecs_heat_label(row):
    if row.heating_fuel == 'district':
        return 'district'
    if row.htpmph == 1 and row.heating_fuel == 'electricity':
        return 'heat_pump'
    if row.heating_fuel == 'electricity':
        return 'resistance'
    if row.heating_fuel in ('natural_gas', 'propane', 'fuel_oil'):
        return 'fuel'
    if row.heating_fuel == 'none':
        return 'none'
    return None


def cbecs_cool_label(row):
    if row.system == 'district_steam_chw':
        return 'district'
    if row.chillr == 1:
        return 'chiller'
    if row.main_cooling_code is None or not np.isfinite(row.main_cooling_code):
        return 'none'
    return 'dx'


def hue_cases(min_days=300):
    """One year (the most complete 365 days) per HUE home, with electricity, YVR/WYJ temperature, clear-sky GHI."""
    import pvlib
    from _paths import BLDG
    P = os.path.join(BLDG, 'data', 'proc', 'other') + os.sep
    m = pd.read_parquet(P + 'hue_meters.parquet')
    hs = pd.read_parquet(P + 'hue_houses.parquet').set_index('house')
    wx = pd.read_parquet(P + 'hue_weather.parquet')
    out = []
    for h, g in m.groupby('house'):
        if h not in hs.index or not isinstance(hs.loc[h, 'HVAC'], str):
            continue
        st = hs.loc[h, 'Region'] if isinstance(hs.loc[h, 'Region'], str) else 'YVR'
        s = g.set_index('time_utc').kWh.sort_index()
        s = s[~s.index.duplicated()]
        # choose the most complete 365-day window starting on a month boundary
        best = None
        for t0 in pd.date_range(s.index[0].normalize(), s.index[-1] - pd.Timedelta('365D'), freq='MS'):
            n = s[t0:t0 + pd.Timedelta('365D')].notna().sum()
            if best is None or n > best[0]:
                best = (n, t0)
        if best is None or best[0] < min_days * 24:
            continue
        t0 = best[1]
        idx = pd.date_range(t0, t0 + pd.Timedelta('365D'), freq='h', inclusive='left', tz='UTC')
        e = s.reindex(idx).to_numpy(float)
        w = wx[wx.station == st].set_index('time_utc').T_out
        w = w[~w.index.duplicated()].reindex(idx).interpolate(limit=12).bfill().ffill().to_numpy(float)
        lat, lon = (49.19, -123.18) if st == 'YVR' else (48.65, -123.43)
        loc_ = pvlib.location.Location(lat, lon)
        cs = loc_.get_clearsky(idx + pd.Timedelta('30min'))['ghi'].to_numpy() * 0.6
        codes = str(hs.loc[h, 'heating_codes'])
        cl = codes.split(',')
        elec = bool(hs.loc[h, 'electric_heat_main'])
        heat = 'heat_pump' if ('HP' in cl or 'GEOTH' in cl) else ('resistance' if elec else 'fuel')
        out.append(dict(id=f'HUE_{h}', meters=dict(elec=e), T=w, GHI=cs, loc=idx.tz_convert('America/Vancouver'),
                        lat=lat, labels=dict(heat=heat, codes=codes, electric_heat=bool(hs.loc[h, 'electric_heat_main']),
                                             use=hs.loc[h, 'primary_use'], has_ac=bool(hs.loc[h, 'has_ac']),
                                             house_type=hs.loc[h, 'HouseType'])))
    return out
