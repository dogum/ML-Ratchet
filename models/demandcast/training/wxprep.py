"""Per-BA hourly weather bundles 2018-2025 (UTC hours) + per-point solar capacity factors + real wind/solar fleets.

data/wx/<BA>.npz:
  T, RH, q, ghi, ghi_clr, ws10       population-weighted (T, RH from ASOS where present, else NASA POWER; GHI NASA POWER)
  btm_cf                             rooftop PV capacity factor (fixed 20 deg south, DC/AC 1.15, pvlib), pop-weighted
  pT, pRH, pws10, pws50, pps, pghi, pclr   per weather point [n_pts, NH]
  pcf_track, pcf_fixed               per-point utility PV capacity factor (pvlib; single-axis / fixed 25 deg)
  plat, plon, pw, pkind              point metadata (kind 0 pop, 1 wind, 2 solar)
data/fleet/<BA>_wind.parquet, <BA>_solar.parquet  real farms (lat, lon, MW, online, retired, hub, SP / tracking, dc_ac,
  nearest point index)
"""
import os, sys, json
import numpy as np, pandas as pd
from common import *
from gridsim.renewables import solar_power, specific_power_from_spec

os.makedirs(f'{DATA}/wx', exist_ok=True); os.makedirs(f'{DATA}/fleet', exist_ok=True)
KIND = {'pop': 0, 'wind': 1, 'solar': 2}


def fill(a):
    s = pd.Series(a).interpolate(limit_direction='both')
    return s.values.astype(np.float32)


def build(ba):
    w = gd.weather(ba).reindex(HOURS)
    pts = gd.weather_points(ba)['points']
    P = {k: [] for k in ('T', 'RH', 'ws10', 'ws50', 'ps', 'ghi', 'clr', 'cft', 'cff', 'btm')}
    for p in pts:
        x = gd.weather_point(p['pid']).reindex(HOURS)
        ghi = x.ALLSKY_SFC_SW_DWN.values.astype(float)
        clr = x.CLRSKY_SFC_SW_DWN.values.astype(float)
        # irradiance gaps (CERES lag): fall back to 0.75 x clear sky, then 0 at night
        ghi = np.where(np.isnan(ghi), 0.75 * np.nan_to_num(clr), ghi)
        T = fill(x.T2M.values); ws = fill(x.WS10M.values)
        P['T'].append(T); P['RH'].append(fill(x.RH2M.values)); P['ws10'].append(ws); P['ws50'].append(fill(x.WS50M.values))
        P['ps'].append(fill(x.PS.values)); P['ghi'].append(ghi.astype(np.float32)); P['clr'].append(fill(clr))
        P['cft'].append(solar_power(HOURS, p['lat'], p['lon'], ghi, temp_air=T, wind_speed=ws, capacity_MWac=1.0,
                                    dc_ac=1.3, tracking='single_axis').astype(np.float32))
        P['cff'].append(solar_power(HOURS, p['lat'], p['lon'], ghi, temp_air=T, wind_speed=ws, capacity_MWac=1.0,
                                    dc_ac=1.2, tracking='fixed', tilt=25.0).astype(np.float32))
        P['btm'].append(solar_power(HOURS, p['lat'], p['lon'], ghi, temp_air=T, wind_speed=ws, capacity_MWac=1.0,
                                    dc_ac=1.15, tracking='fixed', tilt=20.0).astype(np.float32) if p['kind'] == 'pop'
                        else np.zeros(NH, np.float32))
        print(ba, p['pid'], flush=True)
    kind = np.array([KIND[p['kind']] for p in pts]); wts = np.array([p['w'] for p in pts])
    pop = kind == 0
    wp = wts * pop; wp = wp / wp.sum()
    T = w.T_asos_C.values if 'T_asos_C' in w else np.full(NH, np.nan)
    T = np.where(np.isnan(T), w.T2m_C.values, T)
    RH = w.RH_asos_pct.values if 'RH_asos_pct' in w else np.full(NH, np.nan)
    RH = np.where(np.isnan(RH), w.RH2m_pct.values, RH)
    clr_pop = (wp[:, None] * np.array(P['clr'])).sum(0)
    ghi_pop = (wp[:, None] * np.array(P['ghi'])).sum(0)
    out = dict(T=fill(T), RH=fill(RH), ghi=ghi_pop.astype(np.float32), ghi_clr=clr_pop.astype(np.float32),
               ws10=fill(w.wind10m_pop_ms.values), ps=fill(w.ps_kPa.values * 10),
               btm_cf=(wp[:, None] * np.array(P['btm'])).sum(0).astype(np.float32),
               pT=np.array(P['T']), pRH=np.array(P['RH']), pws10=np.array(P['ws10']), pws50=np.array(P['ws50']),
               pps=np.array(P['ps']), pghi=np.array(P['ghi']), pclr=np.array(P['clr']),
               pcf_track=np.array(P['cft']), pcf_fixed=np.array(P['cff']),
               plat=np.array([p['lat'] for p in pts]), plon=np.array([p['lon'] for p in pts]), pw=wts, pkind=kind)
    from gridsim.demand import specific_humidity
    out['q'] = (specific_humidity(out['T'], out['RH'] / 100.0, out['ps']) * 1000).astype(np.float32)   # g/kg
    np.savez_compressed(f'{DATA}/wx/{ba}.npz', **out)
    fleets(ba, out['plat'], out['plon'])


def nearest(lat, lon, plat, plon):
    d = (plat[None] - lat[:, None]) ** 2 + ((plon[None] - lon[:, None]) * np.cos(np.radians(lat[:, None]))) ** 2
    return d.argmin(1)


def fleets(ba, plat, plon):
    fl = gd.fleet(ba, as_frame=True)
    for tech, name in (('wind', 'wind'), ('solar_pv', 'solar')):
        f = fl[(fl.tech == tech) & fl.lat.notna() & (fl.pmax_MW > 0)].copy()
        f = f[f.status_latest.isin(['existing', 'retired'])]
        o = pd.DataFrame({'lat': f.lat.values, 'lon': f.lon.values, 'MW': f.pmax_MW.values,
                          'online': pd.to_datetime(f.online_date).values, 'retired': pd.to_datetime(f.retired_date).values})
        o['pt'] = nearest(o.lat.values, o.lon.values, plat, plon)
        if tech == 'wind':
            o['hub'] = f.hub_height_m.fillna(80.0).clip(40, 140).values
            o['sp'] = [specific_power_from_spec(r) for r in f.to_dict('records')]
        else:
            o['track'] = (~f.tracking.astype(str).str.lower().isin(['fixed', 'none', 'nan'])).values.astype(int)
            o['dc_ac'] = f.dc_ac.fillna(1.3).clip(1.0, 1.6).values
        o.to_parquet(f'{DATA}/fleet/{ba}_{name}.parquet')
        print(ba, name, len(o), round(o.MW.sum()))


if __name__ == '__main__':
    for ba in (sys.argv[1:] or BAS):
        build(ba)
