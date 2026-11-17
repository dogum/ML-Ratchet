"""BuildingDetective shared helpers: paths, Phase 1 imports, weather pool, label definitions.

Label conventions (all exact, from BuildingSim on generated buildings):
  UA       = labels()['UA_network_W_K'] (what an air-temperature-controlled meter sees; incl. mean air exchange)
  tau_h    = labels()['tau_h']
  Tb_heat  = labels()['balance_point_heat_C'] = heat_sp - mean(gains) / UA
  Tb_cool  = cool_sp - mean(gains) / UA          (same degree-day definition, cooling side)
  S_heat   = UA * (site heating energy / heat delivered)  W/K of site energy per K (only if heated)
  S_cool   = UA * (site cooling energy / sensible cooling delivered)  W/K (only if cooled)
  base_share = 1 - (heating + cooling + fans + pumps + heat rejection) / site energy
  op_hours = mean occupied hours per weekday (from the simulated schedule)
  weekend  = weekend occupied hours / weekday occupied hours
  heat class: heat_pump | resistance | fuel | district | none   (from the simulated end uses)
  cool class: none | dx | chiller | district
"""
import os
import sys

import numpy as np

from _paths import WORK, SRC  # noqa: E402  (also puts BLDG and BLDG/buildingsim on sys.path)

ROOT = WORK                     # the sandbox ROOT was bldg/detective
sys.path.insert(0, SRC)

USES = ['office', 'education', 'lodging', 'retail', 'healthcare', 'assembly', 'warehouse', 'residential_single',
        'residential_multi', 'food', 'lab', 'other']
HEAT = ['heat_pump', 'resistance', 'fuel', 'district', 'none']
COOL = ['none', 'dx', 'chiller', 'district']
# regression targets: name, transform note
REG = ['logA', 'year', 'logUA', 'logUAm2', 'logtau', 'Tb_heat', 'Tb_cool', 'logS_heat', 'logS_cool', 'base_share',
       'op_hours', 'weekend']
CHANNELS = ['elec', 'fuel', 'dheat', 'dcool']          # electricity, on-site fuel (gas meter), district heat, district chw
SIM_COLS = {'elec': 'meter_electricity_kW', 'fuel': 'meter_gas_kW', 'dheat': 'meter_steam_kW',
            'dcool': 'meter_chilled_water_kW'}
BDG2_METER = {'elec': 'electricity', 'fuel': 'gas', 'dheat': ['steam', 'hotwater'], 'dcool': 'chilledwater'}


def heat_class(out):
    eu = out['summary']['kWh_by_end_use']
    s = out['spec']
    he, hg, hd = eu.get('heating_electricity', 0), eu.get('heating_gas', 0), eu.get('heating_district', 0)
    tot = he + hg + hd
    A = s['floor_area_m2']
    if s['system'] == 'none' or tot < 0.5 * A:
        return 'none'
    if hd > 0.05 * tot:
        return 'district'
    if hg > he:
        return 'fuel'
    q = float(out['ts']['Q_heat_kW'].clip(lower=0).sum())
    return 'heat_pump' if q / max(he, 1e-9) > 1.3 else 'resistance'


def cool_class(out):
    eu = out['summary']['kWh_by_end_use']
    s = out['spec']
    if eu.get('cooling_district', 0) > 0:
        return 'district'
    if eu.get('cooling_electricity', 0) < 0.2 * s['floor_area_m2']:
        return 'none'
    if s['system'] in ('vav_chiller_boiler', 'fan_coil', 'radiant') and s['hvac'].get('cooling_source', 'chiller') != 'dx':
        return 'chiller'
    return 'dx'


def derived_labels(spec, out, L, tz):
    """Regression + class labels (dict) from labels() L and the simulation output."""
    ts = out['ts']
    eu = out['summary']['kWh_by_end_use']
    A = float(L['floor_area_m2'])
    UA = float(L['UA_network_W_K'])
    site = float(out['summary']['site_kWh'])
    heat_site = eu.get('heating_electricity', 0) + eu.get('heating_gas', 0) + eu.get('heating_district', 0)
    cool_site = eu.get('cooling_electricity', 0) + eu.get('cooling_district', 0)
    hvac_aux = eu.get('fans_electricity', 0) + eu.get('pumps_electricity', 0) + eu.get('heat_rejection_electricity', 0)
    q_heat = float(ts['Q_heat_kW'].clip(lower=0).sum())
    q_cool = float(ts['Q_cool_sens_kW'].abs().sum())
    hc, cc = heat_class(out), cool_class(out)
    S_heat = UA * heat_site / q_heat if (hc != 'none' and q_heat > 0) else np.nan
    S_cool = UA * cool_site / q_cool if (cc != 'none' and q_cool > 0) else np.nan
    loc = ts.index.tz_convert(tz)
    occ = ts['occupied'].to_numpy() > 0.5
    wk = np.asarray(loc.dayofweek < 5)
    op_wd = occ[wk].mean() * 24
    op_we = occ[~wk].mean() * 24
    gains = float(L['mean_gains_W'])
    y = dict(logA=np.log10(A), year=float(L['year_built']), logUA=np.log10(UA), logUAm2=np.log10(UA / A),
             logtau=np.log10(max(float(L['tau_h']), 0.5)), Tb_heat=float(L['balance_point_heat_C']),
             Tb_cool=float(L['cool_sp']) - gains / UA,
             logS_heat=np.log10(S_heat) if np.isfinite(S_heat) and S_heat > 0 else np.nan,
             logS_cool=np.log10(S_cool) if np.isfinite(S_cool) and S_cool > 0 else np.nan,
             base_share=1.0 - (heat_site + cool_site + hvac_aux) / max(site, 1e-9),
             op_hours=float(op_wd), weekend=float(op_we / max(op_wd, 0.5)))
    if L['primary_use'].startswith('residential'):
        y['op_hours'] = np.nan          # residential 'occupied' = someone home; not an operating schedule
        y['weekend'] = np.nan
    return y, dict(use=L['primary_use'], heat=hc, cool=cc)


# ----------------------------------------------------------------------------- weather pool
BDG2_SITES_SIM = ['Bear', 'Bobcat', 'Bull', 'Cockatoo', 'Crow', 'Eagle', 'Fox', 'Gator', 'Hog', 'Lamb', 'Moose',
                  'Mouse', 'Panther', 'Rat', 'Robin', 'Shrew', 'Swan', 'Wolf']        # Peacock left out (weather mismatch)
SITE_CZ = dict(Bear='3C', Swan='3C', Bobcat='5B', Bull='2A', Cockatoo='6A', Eagle='5A', Peacock='5A', Fox='2B',
               Gator='2A', Panther='2A', Hog='6A', Rat='4A', Crow='6A', Moose='6A', Lamb='4A', Mouse='4A', Robin='4A',
               Shrew='4A', Wolf='4A')


def site_wx(site):
    """(frame, meta) of a BDG2 site's real 2016-17 weather in the simulator contract."""
    import buildingdata as bd
    w = bd.site_weather(site).copy()
    if 'LW_down' in w:
        w['IR_horiz'] = w['LW_down']
    m = bd.site_weather_meta(site)
    return w, dict(lat=m['lat'], lon=m['lon'], elev=m.get('elev', 0.0) or 0.0, tz=m['tz'])


def tmy_wx(city):
    import buildingsim as bs
    return bs.read_epw(bs.epw_path(city))


def wx_keys_for_cz(cz):
    """Candidate weather keys for a climate zone: ('site', name, year) or ('tmy', city, None)."""
    from buildingsim.weather import EPW_CLIMATE_ZONE
    from buildingsim.generator import weather_hint
    sites = [('site', s, y) for s in BDG2_SITES_SIM if SITE_CZ[s] == cz for y in (2016, 2017)]
    tm = [('tmy', c, None) for c, z in EPW_CLIMATE_ZONE.items() if z == cz or (len(z) == 1 and str(cz).startswith(z))]
    if not tm:
        tm = [('tmy', weather_hint(cz), None)]
    return sites, tm


def clean_labels(Y):
    """Label hygiene (applied on load, train and evaluation alike): base_share is undefined when the metered site
    energy is ~0 (NaN outside [-1, 1], else clipped to [-0.2, 1]); balance points clipped to [-25, 30] degC (a building
    whose gains exceed its losses at any outdoor temperature has a balance point 'below -25')."""
    Y = np.array(Y, np.float32, copy=True)
    j = REG.index('base_share')
    b = Y[:, j]
    Y[:, j] = np.where(np.abs(b) > 1, np.nan, np.clip(b, -0.2, 1.0))
    for k in ('Tb_heat', 'Tb_cool'):
        j = REG.index(k)
        Y[:, j] = np.clip(Y[:, j], -25, 30)
    return Y
