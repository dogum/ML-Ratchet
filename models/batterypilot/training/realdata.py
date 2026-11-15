"""Real price / weather frames for BatteryPilot (GridData loaders; dev and exam use only)."""
import sys
import numpy as np
import pandas as pd

import _paths  # noqa: E402,F401  (GridData on sys.path)
import griddata as gd  # noqa: E402

TZ = {'ERCOT': 'America/Chicago', 'CAISO': 'America/Los_Angeles'}
BA = {'ERCOT': 'ERCO', 'CAISO': 'CISO'}
ERCOT_NODES = ['HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'LZ_HOUSTON', 'LZ_NORTH', 'LZ_SOUTH', 'LZ_WEST',
               'LZ_AEN', 'LZ_CPS', 'LZ_LCRA', 'LZ_RAYBN']
CAISO_NODES = ['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND', 'DLAP_PGAE-APND', 'DLAP_SCE-APND',
               'DLAP_SDGE-APND']


def node_frame(market, node, start, end):
    """Hourly frame (UTC index, full hourly grid) with rt, da ($/MWh), demand forecast df and demand d (EIA-930, MW),
    temperature T (degC, BA population-weighted NASA POWER), local hour / day id / weekday / month."""
    s, e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    if market == 'ERCOT':
        rt = gd.prices('ERCOT', node, s, e, kind='rt').price
        da = gd.prices('ERCOT', node, s, e, kind='da').price
    else:
        rt = gd.prices('CAISO', node, s, e, kind='rt').LMP
        da = gd.prices('CAISO', node, s, e, kind='da').LMP
    idx = pd.date_range(s, e, freq='h', inclusive='left')
    f = pd.DataFrame(index=idx)
    f['rt'] = rt[~rt.index.duplicated()].reindex(idx).astype(float)
    f['da'] = da[~da.index.duplicated()].reindex(idx).astype(float)
    ei = gd.eia930(BA[market], s, e, columns=None)
    ei = ei.set_index('time_utc') if 'time_utc' in ei else ei
    f['d'] = pd.to_numeric(ei['d'], errors='coerce').reindex(idx).astype(float)
    f['df'] = pd.to_numeric(ei['df'], errors='coerce').reindex(idx).astype(float)
    w = gd.weather(BA[market], s, e)
    f['T'] = w['T2m_C'].reindex(idx).astype(float)
    add_calendar(f, market)
    return f


def add_calendar(f, market):
    loc = f.index.tz_convert(TZ[market])
    f['hour'] = loc.hour
    f['dow'] = loc.dayofweek
    f['month'] = loc.month
    f['doy'] = loc.dayofyear
    day = loc.normalize().tz_localize(None)
    f['day'] = (day - pd.Timestamp('2000-01-01')).days.values
    return f


def fill(f):
    """Fill short gaps (interpolate up to 6 h); report the share that was missing."""
    miss = f[['rt', 'da']].isna().mean().to_dict()
    g = f.copy()
    for c in ['rt', 'da', 'd', 'df', 'T']:
        g[c] = g[c].interpolate(limit=6, limit_direction='both')
    g['df'] = g['df'].fillna(g['d'])
    g['T'] = g['T'].ffill().bfill()
    return g, miss
