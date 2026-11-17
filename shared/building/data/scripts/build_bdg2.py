"""Build the BDG2 part of BuildingData (resumable stages).

  python3 -I build_bdg2.py points     # site points (coords, tz, time conventions, ASOS candidates) -> proc/bdg2/site_points.json
  python3 -I build_bdg2.py meters     # per meter: UTC wide parquet (kWh per hour, litres for water) + uint16 flag bitmask
  python3 -I build_bdg2.py summary    # per building-meter quality table -> proc/bdg2/meter_quality.parquet
  python3 -I build_bdg2.py weather    # per site weather in the contract (after weather_fetch.py asos/power)
  python3 -I build_bdg2.py spec       # building specs (contract) -> proc/bdg2/buildings.parquet / specs.json

Time conventions (DATA.md section 3): meter stamps are local clock time WITH DST at all sites except Eagle, Moose and
Shrew (local standard time; per-building DST-shift vote + absence of DST hallmarks). Fox is Arizona (no DST). BDG2
weather stamps are local standard time at every site (checked against ASOS in UTC: Panther vs KMCO MAE 0.03 degC at
a constant +5 h in winter and summer).
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
GRID_CODE = _os.path.join(_os.path.dirname(BLDG_CODE), 'grid')                     # shared/grid (GridData)
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE
import sys, os, json, glob
import numpy as np, pandas as pd

D = f'{BLDG_ROOT}/data'
RAW = f'{D}/raw/bdg2'; P = f'{D}/proc/bdg2'
os.makedirs(f'{P}/meters', exist_ok=True); os.makedirs(f'{P}/weather', exist_ok=True)
METERS = ['electricity', 'chilledwater', 'steam', 'hotwater', 'gas', 'water', 'irrigation', 'solar']
TZ = {'Bear': 'America/Los_Angeles', 'Bobcat': 'America/Denver', 'Bull': 'America/Chicago', 'Cockatoo': 'America/New_York',
      'Crow': 'America/Toronto', 'Eagle': 'America/New_York', 'Fox': 'America/Phoenix', 'Gator': 'America/New_York',
      'Hog': 'America/Chicago', 'Lamb': 'Europe/London', 'Moose': 'America/Toronto', 'Mouse': 'Europe/London',
      'Panther': 'America/New_York', 'Peacock': 'America/New_York', 'Rat': 'America/New_York', 'Robin': 'Europe/London',
      'Shrew': 'Europe/London', 'Swan': 'America/Los_Angeles', 'Wolf': 'Europe/Dublin'}
STD_OFF = {'America/Los_Angeles': -8, 'America/Denver': -7, 'America/Phoenix': -7, 'America/Chicago': -6,
           'America/New_York': -5, 'America/Toronto': -5, 'Europe/London': 0, 'Europe/Dublin': 0}
METER_LST = {'Eagle', 'Moose', 'Shrew'}
# Unit corrections (DATA.md): Eagle chilled/hot water EUI medians 195,709 / 97,305 kWh/m2/yr = ~1000x every other site;
# consistent with kBTU readings converted with BDG2's mmBTU factor (293.071 instead of 0.293071). Divided by 1000.
UNIT_FIX = {('Eagle', 'chilledwater'): 1e-3, ('Eagle', 'hotwater'): 1e-3}          # meter stamps in local standard time
ASOS_CAND = {'Bear': ['OAK', 'HWD', 'CCR', 'SFO'], 'Bobcat': ['OGD', 'SLC', 'HIF'], 'Bull': ['AUS', 'ATT', 'EDC'],
             'Cockatoo': ['ITH', 'SYR', 'ELM'], 'Crow': ['CYOW', 'CYND'], 'Eagle': ['TTN', 'PNE', 'PHL'],
             'Fox': ['PHX', 'CHD', 'SDL', 'FFZ'], 'Gator': ['MCO', 'ORL', 'SFB'], 'Hog': ['MSP', 'STP', 'FCM'],
             'Lamb': ['EGFF', 'EGGD'], 'Moose': ['CYOW', 'CYND'], 'Mouse': ['EGLL', 'EGLC', 'EGKK'], 'Panther': ['MCO', 'ORL'],
             'Peacock': ['TTN', 'EWR', 'PNE'], 'Rat': ['DCA', 'IAD', 'BWI', 'ADW'], 'Robin': ['EGLL', 'EGLC', 'EGKK'],
             'Shrew': ['EGLL', 'EGLC', 'EGKK'], 'Swan': ['NUQ', 'SJC', 'PAO'], 'Wolf': ['EIDW', 'EIME']}
# flag bits (uint16)
F = dict(MISSING=1, ZERO=2, ZERO_RUN=4, NEGATIVE=8, SPIKE=16, FLATLINE=32, INTERP=64, LEVEL_SHIFT=128,
         BDG2_OUTLIER=256, DST=512, SEASON_SIGN=1024, ZERO_RUN_INSEASON=2048)
_core = F['MISSING'] | F['NEGATIVE'] | F['SPIKE'] | F['FLATLINE'] | F['INTERP'] | F['DST']
BAD_MASK = {'electricity': _core | F['ZERO'] | F['ZERO_RUN'] | F['LEVEL_SHIFT'],
            'water': _core | F['ZERO_RUN'] | F['LEVEL_SHIFT'],
            'chilledwater': _core | F['ZERO_RUN_INSEASON'], 'steam': _core | F['ZERO_RUN_INSEASON'],
            'hotwater': _core | F['ZERO_RUN_INSEASON'], 'gas': _core | F['ZERO_RUN_INSEASON'],
            'irrigation': _core, 'solar': _core}
THERMAL = ('chilledwater', 'steam', 'hotwater', 'gas')
UTC_INDEX = pd.date_range('2016-01-01 00:00', '2018-01-01 07:00', freq='h', tz='UTC')


# ------------------------------------------------------------------ points
def points():
    m = pd.read_csv(f'{RAW}/metadata.csv')
    g = m.groupby('site_id').agg(lat=('lat', 'median'), lon=('lng', 'median'), n_buildings=('building_id', 'size'))
    out = {}
    scr = pd.read_csv(f'{D}/tmp/site_screen.csv') if os.path.exists(f'{D}/tmp/site_screen.csv') else None
    st = pd.read_parquet(f'{GRID_ROOT}/data/raw/weather/asos_stations.parquet')   # read-only (GridData's IEM list)
    for s, r in g.iterrows():
        p = dict(site_id=s, tz=TZ[s], tz_bdg2=m[m.site_id == s].timezone.iloc[0], n_buildings=int(r.n_buildings),
                 meter_time=('local_standard' if s in METER_LST else 'local_clock_dst'), weather_time='local_standard')
        if np.isfinite(r.lat):
            p.update(lat=float(r.lat), lon=float(r.lon), coord_source='BDG2 metadata (median of building coordinates)')
            if s == 'Wolf':
                p['lon'] = -abs(p['lon']); p['coord_source'] += '; longitude sign fixed (+6.26 is in the North Sea; Dublin is -6.26)'
        else:
            b = scr[scr.site == s].sort_values('mae').iloc[0]
            p.update(asos_match=b.station, asos_match_mae_C=round(float(b.mae), 3), asos_match_exact_frac=round(float(b.exact), 3))
            a_ = pd.read_csv(f'{D}/raw/asos/screen_{b.station}_2016.csv', nrows=5)     # IEM rows carry station lat/lon
            p.update(lat=float(a_.lat.iloc[0]), lon=float(a_.lon.iloc[0]), elev=float(a_.elevation.iloc[0]))
            p['coord_source'] = (f'INFERRED: BDG2 has no coordinates; site air temperature matches ASOS {b.station} '
                                 f'(MAE {b.mae:.2f} degC, {100 * b.exact:.0f}% of 2016 hours identical) - station location used')
        if s == 'Fox': p['meter_time'] = 'local_standard (Arizona: no DST)'
        p['asos_candidates'] = ASOS_CAND[s]
        out[s] = p
    json.dump(out, open(f'{P}/site_points.json', 'w'), indent=1)
    print(json.dumps(out, indent=1)[:3000])


# ------------------------------------------------------------------ weather
def _asos_hourly(f):
    a = pd.read_csv(f, parse_dates=['valid'], low_memory=False)
    a['t'] = (a.valid + pd.Timedelta(minutes=10)).dt.floor('h')
    a = a.sort_values('valid').groupby('t').last()
    a.index = a.index.tz_localize('UTC')
    okta = {'CLR': 0, 'SKC': 0, 'NSC': 0, 'NCD': 0, 'FEW': 1.5, 'SCT': 3.5, 'BKN': 6, 'OVC': 8, 'VV ': 8}
    sky = a[['skyc1', 'skyc2', 'skyc3']].apply(lambda c: c.map(okta)).max(axis=1)
    return pd.DataFrame({'T': (a.tmpf - 32) / 1.8, 'Td': (a.dwpf - 32) / 1.8, 'ws': a.sknt * 0.514444, 'wd': a.drct,
                         'p_alt_Pa': a.alti * 3386.39, 'cloud_okta': sky, 'precip_mm': a.p01i * 25.4,
                         'lat': a.lat, 'lon': a.lon, 'elev': a.elevation})


def rh_from(T, Td):
    a, b = 17.625, 243.04
    return 100 * np.exp(a * Td / (b + Td) - a * T / (b + T))


def weather():
    pts = json.load(open(f'{P}/site_points.json'))
    w = pd.read_csv(f'{RAW}/weather.csv', parse_dates=['timestamp'])
    rows = {}
    for site, p in pts.items():
        g = w[w.site_id == site].set_index('timestamp')
        g.index = (g.index - pd.Timedelta(hours=STD_OFF[TZ[site]])).tz_localize('UTC')
        g = g[~g.index.duplicated()].reindex(UTC_INDEX)
        pw = pd.read_parquet(f'{D}/raw/nasa_power/{site}.parquet').set_index('time_utc').reindex(UTC_INDEX)
        best, stats = None, {}
        for st in p['asos_candidates']:
            f = f'{D}/raw/asos/{st}_2016_2017.csv'
            if not os.path.exists(f): continue
            a = _asos_hourly(f).reindex(UTC_INDEX)
            d = (g.airTemperature - a['T']).abs()
            stats[st] = dict(mae=round(float(d.mean()), 3), exact=round(float((d < 0.06).mean()), 3), n=int(d.notna().sum()))
            if best is None or stats[st]['mae'] < stats[best[0]]['mae']: best = (st, a)
        st, a = best
        T = g.airTemperature.copy(); src = pd.Series(np.where(T.notna(), 0, -1), index=UTC_INDEX)
        T = T.fillna(a['T']); src[(src < 0) & T.notna()] = 1
        T = T.fillna(pw.T2M); src[(src < 0) & T.notna()] = 2
        Td = g.dewTemperature.fillna(a.Td).fillna(pw.T2MDEW)
        out = pd.DataFrame({'T_out': T, 'dewpoint': Td, 'RH': rh_from(T, Td).clip(0, 100),
                            'GHI': pw.ALLSKY_SFC_SW_DWN, 'DNI': pw.ALLSKY_SFC_SW_DNI, 'DHI': pw.ALLSKY_SFC_SW_DIFF,
                            'GHI_clearsky': pw.CLRSKY_SFC_SW_DWN, 'LW_down': pw.ALLSKY_SFC_LW_DWN,
                            'wind_speed': g.windSpeed.fillna(a.ws).fillna(pw.WS10M), 'wind_dir': g.windDirection.fillna(a.wd).fillna(pw.WD10M),
                            'pressure': pw.PS * 1000.0, 'pressure_sealevel': g.seaLvlPressure * 100.0,
                            'cloud': (g.cloudCoverage / 8.0).clip(0, 1).fillna(a.cloud_okta / 8.0).fillna(pw.CLOUD_AMT / 100.0),
                            'precip_mm': g.precipDepth1HR.clip(lower=0).fillna(a.precip_mm),
                            'T_out_src': src.astype('int8'), 'T_out_asos': a['T'], 'T_out_nasa': pw.T2M, 'RH_nasa': pw.RH2M})
        out.index.name = 'time_utc'
        out = out.astype({c: 'float32' for c in out.columns if c != 'T_out_src'})
        out.to_parquet(f'{P}/weather/{site}.parquet', compression='zstd')
        dn = (out.T_out_nasa - g.airTemperature)
        meta = dict(site_id=site, lat=p['lat'], lon=p['lon'], elev=float(a.elev.dropna().iloc[0]) if a.elev.notna().any() else None,
                    tz=TZ[site], coord_source=p['coord_source'], asos_station=st, asos_match=stats,
                    source={'T_out/dewpoint/wind/cloud/pressure_sealevel/precip': 'BDG2 weather (NOAA ISD-Lite), gaps filled from ASOS then NASA POWER',
                            'GHI/DNI/DHI/GHI_clearsky/LW_down/pressure': 'NASA POWER hourly (CERES SYN1deg irradiance, MERRA-2 meteorology)',
                            'RH': 'computed from T_out and dewpoint (Magnus, Alduchov-Eskridge)', 'T_out_src': '0 BDG2, 1 ASOS, 2 NASA POWER'},
                    coverage_T_bdg2=float(g.airTemperature.notna().mean()), filled_asos=float((src == 1).mean()), filled_nasa=float((src == 2).mean()),
                    nasa_minus_bdg2_T_bias_C=round(float(dn.mean()), 2), nasa_vs_bdg2_T_mae_C=round(float(dn.abs().mean()), 2),
                    GHI_coverage=float(out.GHI.notna().mean()))
        json.dump(meta, open(f'{P}/weather/{site}.json', 'w'), indent=1)
        rows[site] = meta
        print(site, st, stats[st], 'nasa bias', meta['nasa_minus_bdg2_T_bias_C'], 'mae', meta['nasa_vs_bdg2_T_mae_C'], 'ghi cov', round(meta['GHI_coverage'], 3), flush=True)
    json.dump(rows, open(f'{P}/weather/index.json', 'w'), indent=1)


# ------------------------------------------------------------------ meters
def to_utc(df, site_of):
    """df: local-stamped wide frame (index naive). Returns UTC-indexed frame on UTC_INDEX and a DST flag frame."""
    out = pd.DataFrame(np.nan, index=UTC_INDEX, columns=df.columns, dtype='float32')
    dst = pd.DataFrame(0, index=UTC_INDEX, columns=df.columns, dtype='uint16')
    loc = df.index
    for s in sorted(set(site_of[c] for c in df.columns)):
        cols = [c for c in df.columns if site_of[c] == s]
        tz = TZ[s]
        if s in METER_LST or tz == 'America/Phoenix':
            utc = (loc - pd.Timedelta(hours=STD_OFF[tz])).tz_localize('UTC'); keep = np.ones(len(loc), bool); amb = np.zeros(len(loc), bool)
        else:
            u = loc.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
            first = loc.tz_localize(tz, ambiguous=np.ones(len(loc), bool), nonexistent='NaT')   # DST reading of ambiguous hour
            amb = u.isna() & first.notna()
            u = u.where(~amb, first)
            keep = u.notna(); utc = u[keep].tz_convert('UTC')
        x = df.loc[keep, cols].to_numpy()
        pos = UTC_INDEX.get_indexer(utc)
        ok = pos >= 0
        out.iloc[pos[ok], [out.columns.get_loc(c) for c in cols]] = x[ok]
        ci = [out.columns.get_loc(c) for c in cols]
        if amb.any():   # fall-back hour: one local row stands for two UTC hours; flag the row and its missing twin
            ap = pos[amb[keep]]
            for p_ in ap:
                dst.iloc[p_, ci] |= F['DST']
                if p_ + 1 < len(UTC_INDEX): dst.iloc[p_ + 1, ci] |= F['DST']
        # spring-forward local hour (nonexistent) is dropped; its reading is not used
    return out, dst


def run_len(mask):
    """Length of the run of True each element belongs to (0 where False). mask: 1-D bool."""
    n = len(mask)
    if not mask.any(): return np.zeros(n, np.int32)
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    out = np.zeros(n, np.int32)
    for a, b in zip(st, en): out[a:b] = b - a
    return out


def flags_for(x, m, daily_T):
    """x: 1-D float array on UTC_INDEX (hourly). Returns uint16 flags and a dict of per-meter findings."""
    n = len(x); f = np.zeros(n, np.uint16); info = {}
    miss = ~np.isfinite(x); f[miss] |= F['MISSING']
    v = np.where(miss, np.nan, x)
    zero = (v == 0); f[zero] |= F['ZERO']
    zr = run_len(zero); f[zr >= 24] |= F['ZERO_RUN']
    if m in THERMAL and daily_T is not None and (zr >= 24).any():
        Th = daily_T.reindex(UTC_INDEX.floor('D')).to_numpy()
        act = (Th > 18) if m == 'chilledwater' else (Th < 10)      # meter's active season (daily mean outdoor T)
        f[(zr >= 24) & act] |= F['ZERO_RUN_INSEASON']
    f[v < 0] |= F['NEGATIVE']
    pos = v[np.isfinite(v) & (v > 0)]
    if len(pos) < 500:
        info['too_short'] = True
        return f, info
    med_all = float(np.median(pos)); info['median'] = med_all
    # flatline: identical consecutive non-zero values >= 24 h
    same = np.r_[False, (np.diff(v) == 0)] & (v != 0) & np.isfinite(v)
    rl = run_len(same); f[rl >= 23] |= F['FLATLINE']
    # interpolation: constant non-zero first difference for >= 12 h (linear fill of a gap)
    d1 = np.r_[np.nan, np.diff(v)]
    lin = np.r_[False, np.abs(np.diff(d1)) <= 1e-4 * max(med_all, 1e-3)] & (np.abs(d1) > 1e-6 * max(med_all, 1e-3))
    rl = run_len(lin & np.isfinite(d1)); f[rl >= 11] |= F['INTERP']
    s = pd.Series(v, index=UTC_INDEX)
    good = s.where((s > 0) | (m != 'electricity') & (s == 0))
    med24 = good.rolling(25, center=True, min_periods=6).median()
    r = good - med24
    scale = r.abs().rolling(24 * 7 + 1, center=True, min_periods=48).median() * 1.4826 + 0.02 * med_all
    spike = (r > 8 * scale) & (good > 2.0 * med24.clip(lower=0.05 * med_all))
    p999 = float(np.nanpercentile(pos, 99.9))
    spike |= good > 10 * p999 if p999 > 0 else False
    f[spike.fillna(False).to_numpy()] |= F['SPIKE']
    # level shift / meter swap or replacement: 14-day rolling median of daily medians vs whole-record median
    dm = good.where((f & (F['SPIKE'] | F['FLATLINE'] | F['INTERP'])) == 0).resample('D').median()
    d14 = dm.rolling(14, center=True, min_periods=7).median()
    G = float(dm.median())
    if G > 0 and m in ('electricity', 'water'):
        shift = (d14 > 3 * G) | (d14 < G / 3)
        shift &= dm.notna()
        sd = shift[shift].index
        if len(sd):
            hrs = UTC_INDEX.floor('D').isin(sd.tz_convert('UTC') if sd.tz is not None else sd)
            # only flag hours that are not already zeros (zero runs are their own flag)
            f[hrs & ~zero] |= F['LEVEL_SHIFT']
            info['level_shift_days'] = int(len(sd))
    if G > 0:
        # breakpoints: 28-day median after / before (recorded for every meter; thermal ones can be seasonal)
        a = dm.rolling(28, min_periods=14).median(); b = dm[::-1].rolling(28, min_periods=14).median()[::-1]
        ratio = (b.shift(-1) / a).replace([np.inf, -np.inf], np.nan)
        br = ratio[(ratio > 2.5) | (ratio < 0.4)]
        if len(br):
            # keep the strongest per 28-day window
            days, last = [], None
            for t_, q in br.items():
                if last is None or (t_ - last).days > 28: days.append(str(t_.date())); last = t_
            info['breakpoints'] = days[:10]
    # seasonal sign check (meter swap between heating and cooling meters)
    if daily_T is not None and m in ('chilledwater', 'steam', 'hotwater', 'gas'):
        dd = good.resample('D').sum(min_count=20)
        j = pd.concat([dd, daily_T], axis=1, join='inner').dropna()
        if len(j) > 120 and j.iloc[:, 0].std() > 0:
            c = float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]); info['corr_daily_T'] = round(c, 3)
            if (m == 'chilledwater' and c < -0.3) or (m != 'chilledwater' and c > 0.3):
                f |= np.uint16(F['SEASON_SIGN']); info['season_sign_suspect'] = True
    return f, info


def site_daily_T():
    w = pd.read_csv(f'{RAW}/weather.csv', parse_dates=['timestamp'])
    out = {}
    for s, g in w.groupby('site_id'):
        t = g.set_index('timestamp').airTemperature
        t.index = (t.index - pd.Timedelta(hours=STD_OFF[TZ[s]])).tz_localize('UTC')
        out[s] = t.resample('D').mean()
    return out


def meters():
    meta = pd.read_csv(f'{RAW}/metadata.csv').set_index('building_id')
    site_of = meta.site_id.to_dict()
    dT = site_daily_T()
    for m in METERS:
        fo = f'{P}/meters/{m}.parquet'
        if os.path.exists(fo) and os.path.exists(f'{P}/meters/{m}_flags.parquet') and os.path.exists(f'{P}/meters/{m}_info.json'):
            print('skip', m); continue
        gz = lambda f: f if os.path.exists(f) else f + '.gz'      # raw CSVs are gzipped after the build
        raw = pd.read_csv(gz(f'{RAW}/raw/{m}.csv'), index_col=0, parse_dates=True).astype('float32')
        cl = pd.read_csv(gz(f'{RAW}/cleaned/{m}_cleaned.csv'), index_col=0, parse_dates=True).astype('float32').reindex(index=raw.index, columns=raw.columns)
        bdg2_nan = (raw.notna() & cl.isna())
        for (st_, mm), fac in UNIT_FIX.items():
            if mm == m:
                cols_ = [c for c in raw.columns if site_of[c] == st_]; raw[cols_] = raw[cols_] * fac
        u, dst = to_utc(raw, site_of)
        bn, _ = to_utc(bdg2_nan.astype('float32'), site_of)
        flags = np.zeros(u.shape, np.uint16); info = {}
        for j, c in enumerate(u.columns):
            f, inf = flags_for(u[c].to_numpy(np.float64), m, dT.get(site_of[c]))
            f |= dst[c].to_numpy(np.uint16)
            f[(bn[c].to_numpy() == 1) & ((f & (F['ZERO_RUN'] | F['ZERO'])) == 0)] |= F['BDG2_OUTLIER']
            flags[:, j] = f; info[c] = inf
        # duplicates: identical series (two building ids carrying one meter)
        h = {}
        for c in u.columns:
            x = u[c].to_numpy(); ok = np.isfinite(x)
            if ok.sum() < 500 or np.nanmax(np.abs(x)) == 0: continue
            key = (int(ok.sum()), round(float(np.nansum(x)), 2), round(float(np.nansum(x[::7] ** 2)), 1))
            h.setdefault(key, []).append(c)
        for k, cs in h.items():
            if len(cs) > 1:
                for c in cs: info[c]['duplicate_of'] = [x for x in cs if x != c]
        u.index.name = 'time_utc'
        u.to_parquet(fo + '.tmp', compression='zstd'); os.replace(fo + '.tmp', fo)
        fl = pd.DataFrame(flags, index=u.index, columns=u.columns)
        fl.to_parquet(f'{P}/meters/{m}_flags.parquet', compression='zstd')
        json.dump(info, open(f'{P}/meters/{m}_info.json', 'w'))
        print(m, u.shape, 'bad frac', float(((flags & BAD_MASK[m]) > 0).mean()), 'missing', float(((flags & 1) > 0).mean()), flush=True)


# ------------------------------------------------------------------ summary
def summary():
    meta = pd.read_csv(f'{RAW}/metadata.csv').set_index('building_id')
    rows = []
    for m in METERS:
        u = pd.read_parquet(f'{P}/meters/{m}.parquet'); fl = pd.read_parquet(f'{P}/meters/{m}_flags.parquet')
        info = json.load(open(f'{P}/meters/{m}_info.json'))
        tz_local = {b: TZ[meta.site_id[b]] for b in u.columns}
        for c in u.columns:
            f = fl[c].to_numpy(); x = u[c].to_numpy(); bad = (f & BAD_MASK[m]) > 0
            loc_year = u.index.tz_convert(tz_local[c]).year
            r = dict(building_id=c, site_id=meta.site_id[c], meter=m, n_hours=int(np.isfinite(x).sum()),
                     unit_factor_applied=UNIT_FIX.get((meta.site_id[c], m), 1.0))
            for k, b in F.items(): r[f'frac_{k.lower()}'] = float(((f & b) > 0).mean())
            r['frac_bad'] = float(bad.mean())
            for y in (2016, 2017):
                sel = loc_year == y
                r[f'good_frac_{y}'] = float((~bad[sel]).mean())
                xs = np.where(bad[sel], np.nan, x[sel])
                r[f'annual_{y}'] = float(np.nanmean(xs) * sel.sum()) if np.isfinite(xs).any() else np.nan
            both = ~bad & np.isfinite(x)
            loc_doy = u.index.tz_convert(tz_local[c]).dayofyear
            a16 = (loc_year == 2016) & both; a17 = (loc_year == 2017) & both
            common = np.intersect1d(np.unique(loc_doy[a16]), np.unique(loc_doy[a17]))
            if len(common) >= 120:
                s16 = np.nanmean(x[a16 & np.isin(loc_doy, common)]); s17 = np.nanmean(x[a17 & np.isin(loc_doy, common)])
                r['yoy_ratio'] = float(s17 / s16) if s16 > 0 else np.nan
            else:
                r['yoy_ratio'] = np.nan
            # native resolution: share of local days (months) whose readings take <= 2 distinct values
            xs_ = pd.Series(np.where(np.isfinite(x) & (x != 0), x, np.nan), index=u.index.tz_convert(tz_local[c]).tz_localize(None))
            dn = xs_.resample('D').nunique()[xs_.resample('D').count() >= 12]
            mn = xs_.resample('MS').nunique()[xs_.resample('MS').count() >= 300]
            if len(dn) < 30: r['native_res'] = 'unknown'
            elif len(mn) >= 3 and (mn <= 2).mean() > 0.8: r['native_res'] = 'monthly'
            elif (dn <= 2).mean() > 0.8: r['native_res'] = 'daily'
            else: r['native_res'] = 'hourly'
            r['frac_days_le2_values'] = float((dn <= 2).mean()) if len(dn) else np.nan
            inf = info.get(c, {})
            r['median_hourly'] = inf.get('median', np.nan)
            r['level_shift_days'] = inf.get('level_shift_days', 0)
            r['breakpoints'] = ','.join(inf.get('breakpoints', []))
            r['duplicate_of'] = ','.join(inf.get('duplicate_of', []))
            r['corr_daily_T'] = inf.get('corr_daily_T', np.nan)
            r['season_sign_suspect'] = bool(inf.get('season_sign_suspect', False))
            sqm = meta.sqm.get(c)
            r['eui_2017'] = r['annual_2017'] / sqm if (m not in ('water', 'irrigation') and sqm and sqm > 0) else np.nan
            rows.append(r)
    q = pd.DataFrame(rows)
    # unit / scale suspicion: electricity EUI outside 10..2000 kWh/m2/yr (thermal meters: > 3000)
    hi = np.where(q.meter == 'electricity', 2000, 3000)
    q['eui_suspect'] = (q.eui_2017 > hi) | ((q.meter == 'electricity') & (q.eui_2017 < 10))
    base = (q.level_shift_days < 30) & ~q.season_sign_suspect & (q.duplicate_of == '') & ~q.eui_suspect.fillna(False)
    # daily usability ignores FLATLINE (a daily meter spread over 24 h is flat within the day by construction)
    for y in (2016, 2017):
        q[f'q_ok_daily_{y}'] = (q[f'good_frac_{y}'] + q.frac_flatline >= 0.85) & base & q.native_res.isin(['hourly', 'daily'])
        q[f'q_ok_{y}'] = (q[f'good_frac_{y}'] >= 0.85) & base & (q.native_res == 'hourly')
    # both years usable AND no year-to-year jump beyond x2.5 (a level change between the years breaks history models)
    q['q_ok'] = q.q_ok_2016 & q.q_ok_2017 & q.yoy_ratio.between(0.4, 2.5)
    q['q_ok_daily'] = q.q_ok_daily_2016 & q.q_ok_daily_2017 & q.yoy_ratio.between(0.4, 2.5)
    q.to_parquet(f'{P}/meter_quality.parquet', index=False)
    print(q.groupby('meter').agg(n=('building_id', 'size'), ok16=('q_ok_2016', 'sum'), ok17=('q_ok_2017', 'sum'), ok=('q_ok', 'sum'),
                                 bad=('frac_bad', 'median'), dup=('duplicate_of', lambda s: (s != '').sum()), okd=('q_ok_daily', 'sum'), daily=('native_res', lambda s: (s == 'daily').sum()), monthly=('native_res', lambda s: (s == 'monthly').sum()),
                                 sign=('season_sign_suspect', 'sum'), lvl=('level_shift_days', lambda s: (s > 0).sum()),
                                 eui=('eui_suspect', 'sum')).to_string())


# ------------------------------------------------------------------ spec
USE_MAP = {'Education': 'education', 'Office': 'office', 'Entertainment/public assembly': 'assembly', 'Lodging/residential': 'lodging',
           'Public services': 'other', 'Healthcare': 'healthcare', 'Parking': 'other', 'Warehouse/storage': 'warehouse',
           'Manufacturing/industrial': 'other', 'Retail': 'retail', 'Services': 'other', 'Technology/science': 'lab',
           'Food sales and service': 'food', 'Utility': 'other', 'Religious worship': 'assembly', 'Other': 'other'}
DEFAULT_SRC = 'unknown in BDG2 metadata: None. Phase 2 default = ComStock 2025R3 prior by (primary_use, climate_zone, vintage) via buildingdata.priors()'


# US site -> county (NHGIS GISJOIN) from the site's city (coordinates in BDG2 metadata / matched ASOS station)
SITE_COUNTY = {'Bear': ('G0600010', 'Alameda CA (Berkeley)'), 'Bobcat': ('G4900570', 'Weber UT (Ogden)'), 'Bull': ('G4804530', 'Travis TX (Austin)'),
               'Cockatoo': ('G3601090', 'Tompkins NY (Ithaca)'), 'Eagle': ('G3400210', 'Mercer NJ (Trenton)'), 'Fox': ('G0400130', 'Maricopa AZ (Tempe)'),
               'Gator': ('G1200950', 'Orange FL (Orlando)'), 'Hog': ('G2700530', 'Hennepin MN (Minneapolis)'), 'Panther': ('G1200950', 'Orange FL (Orlando)'),
               'Peacock': ('G3400210', 'Mercer NJ (Princeton)'), 'Rat': ('G1100010', 'District of Columbia'), 'Swan': ('G0600850', 'Santa Clara CA (Mountain View)')}


def climate_zone(site, pts):
    if site in SITE_COUNTY:
        g, name = SITE_COUNTY[site]
        c = pd.read_parquet(f'{D}/proc/stock/county_climate_zones.parquet')
        return str(c[c.county == g].sort_values('n').cz.iloc[-1]), f'IECC 2004 zone of {name} (county {g}) from ResStock 2025R1 county table'
    w = pd.read_parquet(f'{P}/weather/{site}.parquet').T_out
    dm = w.resample('D').mean()
    hdd = float((18 - dm).clip(lower=0).sum() / 2); cdd = float((dm - 10).clip(lower=0).sum() / 2)
    z = 0 if cdd > 6000 else 1 if cdd > 5000 else 2 if cdd > 3500 else 3 if hdd <= 2000 else 4 if hdd <= 3000 else 5 if hdd <= 4000 else 6 if hdd <= 5000 else 7 if hdd <= 7000 else 8
    return f'{z}A', f'computed from 2016-17 site weather with ASHRAE 169 thermal thresholds (HDD18={hdd:.0f}, CDD10={cdd:.0f}); moisture regime A assumed (no dry summer)'


def spec():
    meta = pd.read_csv(f'{RAW}/metadata.csv')
    pts = json.load(open(f'{P}/site_points.json'))
    q = pd.read_parquet(f'{P}/meter_quality.parquet')
    cz = {s: climate_zone(s, pts) for s in pts}
    specs, rows = {}, []
    for _, m in meta.iterrows():
        s = m.site_id; p = pts[s]
        meters = [mm for mm in METERS if isinstance(m.get(mm), str) and m[mm] == 'Yes']
        qq = q[q.building_id == m.building_id]
        hv = None
        if 'chilledwater' in meters and ('steam' in meters or 'hotwater' in meters): hv = 'central plant chilled water + steam/hot water (campus)'
        elif 'chilledwater' in meters: hv = 'chilled water from central plant'
        elif 'steam' in meters or 'hotwater' in meters: hv = 'steam/hot water from central plant'
        sp = dict(id=m.building_id, site_id=s, lat=p['lat'], lon=p['lon'], climate_zone=cz[s][0],
                  primary_use=USE_MAP.get(m.primaryspaceusage, 'other'), primary_use_bdg2=m.primaryspaceusage, sub_use_bdg2=m.sub_primaryspaceusage,
                  floor_area_m2=float(m.sqm) if np.isfinite(m.sqm) else None,
                  n_floors=int(m.numberoffloors) if np.isfinite(m.numberoffloors) else None,
                  year_built=int(m.yearbuilt) if np.isfinite(m.yearbuilt) else None,
                  wall_U=None, roof_U=None, window_U=None, window_wwr=None, SHGC=None, infiltration_ach=None, thermal_mass=None,
                  internal_gains=None, occupancy_schedule=None, setpoints=None, ventilation=None,
                  system=None, hvac_hint=hv, heating_type_bdg2=m.heatingtype if isinstance(m.heatingtype, str) else None,
                  dhw=None, pv=({'kW_dc': None, 'note': 'solar meter present'} if 'solar' in meters else None), battery=None,
                  meters={mm: ('whole-building ' + mm) for mm in meters},
                  occupants=int(m.occupants) if np.isfinite(m.occupants) else None,
                  energystar_score=float(m.energystarscore) if pd.notna(pd.to_numeric(m.energystarscore, errors='coerce')) else None,
                  industry=m.industry if isinstance(m.industry, str) else None,
                  defaults_source={k: DEFAULT_SRC for k in ['wall_U', 'roof_U', 'window_U', 'window_wwr', 'SHGC', 'infiltration_ach', 'thermal_mass',
                                                            'internal_gains', 'occupancy_schedule', 'setpoints', 'ventilation', 'system', 'dhw']},
                  climate_zone_source=cz[s][1], coord_source=p['coord_source'])
        specs[m.building_id] = sp
        r = {k: v for k, v in sp.items() if not isinstance(v, dict)}
        r['meters'] = ','.join(meters)
        for mm in meters:
            x = qq[qq.meter == mm]
            if len(x):
                r[f'q_ok_{mm}'] = bool(x.q_ok.iloc[0]); r[f'q_ok_daily_{mm}'] = bool(x.q_ok_daily.iloc[0]); r[f'res_{mm}'] = x.native_res.iloc[0]
        rows.append(r)
    json.dump(specs, open(f'{P}/specs.json', 'w'), default=str)
    b = pd.DataFrame(rows); b.to_parquet(f'{P}/buildings.parquet', index=False)
    print(b.shape); print(b.groupby('site_id').climate_zone.first().to_string()); print(b.primary_use.value_counts().to_string())


if __name__ == '__main__':
    globals()[sys.argv[1]]()
