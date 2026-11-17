"""buildingdata: frozen real-world building data for the ML Ratchet buildings project (pandas / numpy / pyarrow only).

    import sys; sys.path.insert(0, 'shared/building')   # or the models' training/_paths.py
    import buildingdata as bd
    sp  = bd.splits()                                    # frozen splits (data/proc/splits.json)
    m   = bd.bdg2_meta(split='exam')                     # BDG2 buildings: metadata in the spec + quality per meter + split
    e   = bd.bdg2('electricity', split='dev')            # hourly kWh, UTC index, columns = buildings; bad hours NaN
    e   = bd.bdg2('steam', building='Hog_office_Lizzie', clean=False)   # raw values
    f   = bd.bdg2_flags('electricity', building='Hog_office_Lizzie')    # uint16 quality bitmask (FLAGS)
    q   = bd.bdg2_quality('electricity')                 # per building-meter quality table (q_ok, q_ok_daily, native_res...)
    s   = bd.bdg2_spec('Hog_office_Lizzie')              # building-spec dict (unknowns None, defaults_source)
    w   = bd.site_weather('Hog'); wm = bd.site_weather_meta('Hog')     # contract weather (UTC) + meta json
    r   = bd.resstock(columns=['bldg_id', 'in.vintage']) # whole-stock metadata (MODELLED)
    h   = bd.resstock_hourly(bldg_id=12345)              # hourly end uses of the stratified sample (MODELLED)
    c   = bd.comstock(); ch = bd.comstock_hourly(); sw = bd.stock_weather('G0800310')
    x   = bd.recs(split='exam'); y = bd.cbecs(split='dev')   # EIA surveys mapped to the spec (+ full=True for all codes)
    runs = bd.faults(split='exam'); ts = bd.fault_data('ers_rp1312', run_id='ERS_20070828_AHU-A')
    t   = bd.tariffs(); p = bd.tariff_prices('pge_e_tou_c', 2017, tz='America/Los_Angeles', utc=True)
    b   = bd.bill('pge_e_tou_c', load_kW)
    pa  = bd.tariff_arrays('pge_e_tou_c', times_utc, tz); r = bd.rtp_prices(times_utc)   # DST-safe; ERCOT RTP
    wp  = bd.wholesale_prices('ERCOT', 'HB_HOUSTON', '2023-01-01', '2024-01-01')   # via GridData (read-only)
    o   = bd.other('list'); bl = bd.baselines()

Conventions (BUILDING_BRIEF): UTC hour-beginning index `time_utc` for real meters and weather; kWh per hour (= mean kW),
degC, W/m2, m/s, Pa. NREL stock data are in local standard time (`time_lst`, hour-beginning). Fault trend logs are in the
facility's local time (`time_local`).
Rule: only `dev` may be used for calibration, gates and fitting; `exam` only for the final report. Simulated training may
use real inputs (weather) freely, and real-building metadata only from non-exam units (BDG2 dev; RECS/CBECS prior+dev).
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
GRID_CODE = _os.path.join(_os.path.dirname(BLDG_CODE), 'grid')                     # shared/grid (GridData)
import os, json, functools
import numpy as np, pandas as pd

ROOT = f'{BLDG_ROOT}/data'
PROC = f'{ROOT}/proc'
METERS = ['electricity', 'chilledwater', 'steam', 'hotwater', 'gas', 'water', 'irrigation', 'solar']
FLAGS = dict(MISSING=1, ZERO=2, ZERO_RUN=4, NEGATIVE=8, SPIKE=16, FLATLINE=32, INTERP=64, LEVEL_SHIFT=128,
             BDG2_OUTLIER=256, DST=512, SEASON_SIGN=1024, ZERO_RUN_INSEASON=2048)
_core = FLAGS['MISSING'] | FLAGS['NEGATIVE'] | FLAGS['SPIKE'] | FLAGS['FLATLINE'] | FLAGS['INTERP'] | FLAGS['DST']
BAD_MASK = {'electricity': _core | FLAGS['ZERO'] | FLAGS['ZERO_RUN'] | FLAGS['LEVEL_SHIFT'],
            'water': _core | FLAGS['ZERO_RUN'] | FLAGS['LEVEL_SHIFT'],
            'chilledwater': _core | FLAGS['ZERO_RUN_INSEASON'], 'steam': _core | FLAGS['ZERO_RUN_INSEASON'],
            'hotwater': _core | FLAGS['ZERO_RUN_INSEASON'], 'gas': _core | FLAGS['ZERO_RUN_INSEASON'],
            'irrigation': _core, 'solar': _core}
SITE_TZ = {'Bear': 'America/Los_Angeles', 'Bobcat': 'America/Denver', 'Bull': 'America/Chicago', 'Cockatoo': 'America/New_York',
           'Crow': 'America/Toronto', 'Eagle': 'America/New_York', 'Fox': 'America/Phoenix', 'Gator': 'America/New_York',
           'Hog': 'America/Chicago', 'Lamb': 'Europe/London', 'Moose': 'America/Toronto', 'Mouse': 'Europe/London',
           'Panther': 'America/New_York', 'Peacock': 'America/New_York', 'Rat': 'America/New_York', 'Robin': 'Europe/London',
           'Shrew': 'Europe/London', 'Swan': 'America/Los_Angeles', 'Wolf': 'Europe/Dublin'}


def _ts(x):
    if x is None: return None
    t = pd.Timestamp(x)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


@functools.lru_cache(maxsize=1)
def splits():
    """The frozen split file (DATA.md section 8)."""
    return json.load(open(f'{PROC}/splits.json'))


def _bdg2_ids(split):
    return None if split is None else set(splits()['bdg2'][split])


# ---------------------------------------------------------------- BDG2
def bdg2_meta(split=None):
    """One row per BDG2 building: spec fields (primary_use, floor_area_m2, year_built, climate_zone, lat/lon...), the
    meters it has, q_ok_<meter> / q_ok_daily_<meter> / res_<meter> per meter, and `split` (dev | exam)."""
    b = pd.read_parquet(f'{PROC}/bdg2/buildings.parquet')
    sp = splits()['bdg2']
    b['split'] = np.where(b.id.isin(sp['dev']), 'dev', 'exam')
    if split is not None: b = b[b.split == split]
    return b.reset_index(drop=True)


def bdg2(meter='electricity', building=None, split=None, clean=True, start=None, end=None, local=False):
    """Hourly meter readings (kWh per hour = mean kW; water/irrigation in litres), UTC hour-beginning index `time_utc`,
    one column per building. clean=True sets hours flagged bad for this meter (BAD_MASK) to NaN.
    building: id or list of ids; split: 'dev' | 'exam'. local=True: Series for one building indexed in its local time."""
    cols = None
    if building is not None:
        cols = [building] if isinstance(building, str) else list(building)
    elif split is not None:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(f'{PROC}/bdg2/meters/{meter}.parquet').schema_arrow.names
        cols = [c for c in names if c in _bdg2_ids(split)]
    x = pd.read_parquet(f'{PROC}/bdg2/meters/{meter}.parquet', columns=cols)
    if clean:
        f = pd.read_parquet(f'{PROC}/bdg2/meters/{meter}_flags.parquet', columns=list(x.columns))
        x = x.mask((f.to_numpy() & BAD_MASK[meter]) > 0)
    s, e = _ts(start), _ts(end)
    if s is not None: x = x[x.index >= s]
    if e is not None: x = x[x.index < e]
    if local:
        if x.shape[1] != 1: raise ValueError('local=True needs exactly one building')
        b = x.columns[0]; site = b.split('_')[0]
        y = x[b].copy(); y.index = y.index.tz_convert(SITE_TZ[site]); return y
    return x


def bdg2_flags(meter='electricity', building=None):
    """uint16 quality bitmask per hour (bits in FLAGS; DATA.md section 3)."""
    cols = None if building is None else ([building] if isinstance(building, str) else list(building))
    return pd.read_parquet(f'{PROC}/bdg2/meters/{meter}_flags.parquet', columns=cols)


def bdg2_quality(meter=None, split=None):
    """Per building-meter quality summary: fractions per flag, good_frac_2016/2017, annual totals, native_res
    (hourly | daily | monthly), breakpoints, yoy_ratio, eui_2017, q_ok_2016/2017, q_ok (hourly label, both years),
    q_ok_daily."""
    q = pd.read_parquet(f'{PROC}/bdg2/meter_quality.parquet')
    if meter is not None: q = q[q.meter == meter]
    if split is not None: q = q[q.building_id.isin(_bdg2_ids(split))]
    return q.reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _specs():
    return json.load(open(f'{PROC}/bdg2/specs.json'))


def bdg2_spec(building):
    """Building-spec dict (BUILDING_BRIEF contract). Unknown fields are None; defaults_source says where to get priors."""
    return dict(_specs()[building])


def bdg2_specs(split=None):
    ids = _bdg2_ids(split)
    return {k: dict(v) for k, v in _specs().items() if ids is None or k in ids}


def site_weather(site, start=None, end=None):
    """Hourly site weather in the contract (UTC): T_out, RH, dewpoint, GHI, DNI, DHI, wind_speed, wind_dir, pressure (Pa),
    cloud (0-1), precip_mm, plus GHI_clearsky, LW_down, pressure_sealevel, T_out_src, T_out_asos, T_out_nasa, RH_nasa."""
    w = pd.read_parquet(f'{PROC}/bdg2/weather/{site}.parquet')
    s, e = _ts(start), _ts(end)
    if s is not None: w = w[w.index >= s]
    if e is not None: w = w[w.index < e]
    return w


def site_weather_meta(site):
    return json.load(open(f'{PROC}/bdg2/weather/{site}.json'))


def site_points():
    return json.load(open(f'{PROC}/bdg2/site_points.json'))


def site_asos(site):
    """Raw IEM ASOS reports (UTC) of the station matched to the site (all candidates are kept in data/raw/asos)."""
    st = site_weather_meta(site)['asos_station']
    return pd.read_csv(f'{ROOT}/raw/asos/{st}_2016_2017.csv', parse_dates=['valid'], low_memory=False)


def tmy(site):
    """TMYx typical-year weather for a BDG2 site, if downloaded (DATA.md section 6); None otherwise."""
    f = f'{PROC}/tmy/{site}.parquet'
    return pd.read_parquet(f) if os.path.exists(f) else None


# ---------------------------------------------------------------- NREL stock (MODELLED)
def _stock(name, columns=None, filters=None):
    return pd.read_parquet(f'{PROC}/stock/{name}_meta.parquet', columns=columns, filters=filters)


def resstock(columns=None, filters=None):
    """ResStock 2025 R1 (AMY2018) baseline metadata + annual results, whole stock (549,971 dwelling models). MODELLED."""
    return _stock('resstock', columns, filters)


def comstock(columns=None, filters=None):
    """ComStock 2025 R3 (AMY2018) baseline metadata + annual results, national aggregate file: 149,272 rows = 91,464 unique
    models x state allocations (de-duplicate on bldg_id to count models). MODELLED."""
    return _stock('comstock', columns, filters)


def resstock_sample():
    return pd.read_parquet(f'{PROC}/stock/resstock_sample.parquet')


def comstock_sample():
    return pd.read_parquet(f'{PROC}/stock/comstock_sample.parquet')


def resstock_hourly(bldg_id=None, columns=None):
    """Hourly end-use profiles of the stratified ResStock sample (time_lst = local standard time, hour-beginning)."""
    f = None if bldg_id is None else [('bldg_id', 'in', [int(b) for b in np.atleast_1d(bldg_id)])]
    return pd.read_parquet(f'{PROC}/stock/resstock_hourly.parquet', filters=f, columns=columns)


def comstock_hourly(bldg_id=None, columns=None):
    f = None if bldg_id is None else [('bldg_id', 'in', [int(b) for b in np.atleast_1d(bldg_id)])]
    return pd.read_parquet(f'{PROC}/stock/comstock_hourly.parquet', filters=f, columns=columns)


def stock_weather(county):
    """NREL AMY2018 county weather used by the stock models (local standard time, hour-beginning)."""
    return pd.read_parquet(f'{PROC}/stock/weather/{county}.parquet')


# ---------------------------------------------------------------- EIA surveys (REAL)
def _survey(name, split, full):
    d = pd.read_parquet(f'{PROC}/eia/{name}{"_full" if full else ""}.parquet')
    if full:
        d['id'] = ('RECS_' + d.DOEID.astype(str)) if name.startswith('recs') else ('CBECS_' + d.PUBID.astype(str))
    sp = splits()[name]
    lab = pd.Series('exam', index=d.index)
    lab[d.id.isin(sp['prior'])] = 'prior'; lab[d.id.isin(sp['dev'])] = 'dev'
    d['split'] = lab.values
    return d[d.split == split].reset_index(drop=True) if split else d


def recs(split=None, full=False):
    """EIA RECS 2020 (v7) households mapped to the spec + annual energy by fuel (kWh) + weight. full=True: all public codes."""
    return _survey('recs2020', split, full)


def cbecs(split=None, full=False):
    """EIA CBECS 2018 buildings mapped to the spec + annual energy by fuel (kWh) + weight. full=True: all public codes."""
    return _survey('cbecs2018', split, full)


# ---------------------------------------------------------------- faults
def faults(dataset=None, split=None, gradable=None):
    """One row per fault run (unit-day): dataset, facility, unit, date, kind, fault_type (taxonomy()), severity,
    fault_scope, dt_s, split (dev | exam | sim_reference | noise_unlabelled | excluded), gradable."""
    r = pd.read_parquet(f'{PROC}/faults/runs.parquet')
    if dataset is not None: r = r[r.dataset == dataset]
    if split is not None: r = r[r.split == split]
    if gradable is not None: r = r[r.gradable == gradable]
    return r.reset_index(drop=True)


def fault_data(dataset, run_id=None):
    """Trend-log time series in the contract channels (SI, signals 0..1) + `fault` (0/1 per row; -1 = unlabelled)."""
    f = None if run_id is None else [('run_id', 'in', list(np.atleast_1d(run_id)))]
    return pd.read_parquet(f'{PROC}/faults/data/{dataset}.parquet', filters=f)


def taxonomy():
    return json.load(open(f'{PROC}/faults/taxonomy.json'))


# ---------------------------------------------------------------- other real datasets
def other(name='list'):
    """Other real datasets (DATA.md section 5). other('list') -> catalogue dict; other(<name>) -> its main frame."""
    cat = json.load(open(f'{PROC}/other/catalogue.json'))
    if name == 'list': return cat
    f = cat[name]['file']
    return pd.read_parquet(f'{PROC}/other/{f}')


# ---------------------------------------------------------------- tariffs and prices
@functools.lru_cache(maxsize=1)
def _tariffs():
    return json.load(open(f'{PROC}/tariffs/tariffs.json'))


def tariffs(key=None):
    """Selected URDB tariffs (structure: energy periods/tiers + 12x24 weekday/weekend schedules, TOU and flat demand
    charges, fixed charges). key=None: dict of all."""
    t = _tariffs()
    return t if key is None else t[key]


def tariff_prices(key, year=2017, tz=None, utc=False):
    """Hourly price frame for a year in LOCAL clock time (tz given -> tz-aware): energy_$per_kWh (tier 1 + adjustment;
    tiers ignored), energy_period, demand_period (-1 if none).
    DST: with tz the frame is correct on every hour (checked for all 19 tariffs, 7 site time zones, 2016-2019). Without
    tz the index is naive clock time: it has the nonexistent spring hour and one autumn 01:00, so it cannot be
    tz-localized (pandas raises), and BuildingSim reads a naive index as UTC (prices shifted by the UTC offset).
    utc=True (opt-in, needs tz): the same prices on the UTC hour-beginning index of the local year, computed per
    UTC hour from its local month / hour / weekday (ThermoPilot's price builder, tariff_arrays); every hour equals
    the tz frame's. This is the safe frame for simulate(prices=...)."""
    if utc:
        if tz is None:
            raise ValueError('tariff_prices(utc=True) needs tz')
        times = pd.date_range(pd.Timestamp(f'{year}-01-01', tz=tz).tz_convert('UTC'),
                              pd.Timestamp(f'{year + 1}-01-01', tz=tz).tz_convert('UTC'), freq='h', inclusive='left')
        a = tariff_arrays(key, times, tz)
        loc = times.tz_convert(tz)
        E = tariffs(key)['energy']
        m, h, we = loc.month.values - 1, loc.hour.values, loc.dayofweek.values >= 5
        per = np.where(we, np.array(E['weekend'])[m, h], np.array(E['weekday'])[m, h])
        return pd.DataFrame({'energy_$per_kWh': a['price'], 'energy_period': per, 'demand_period': a['dperiod']},
                            index=times.rename('time_utc'))
    t = tariffs(key)
    idx = pd.date_range(f'{year}-01-01', f'{year + 1}-01-01', freq='h', inclusive='left', tz=tz)
    m, h, we = idx.month - 1, idx.hour, idx.dayofweek >= 5
    E = t['energy']
    wd, wk = np.array(E['weekday']), np.array(E['weekend'])
    per = np.where(we, wk[m, h], wd[m, h])
    rate = np.array([p[0]['rate'] + p[0]['adj'] for p in E['periods']])
    out = pd.DataFrame({'energy_$per_kWh': rate[per], 'energy_period': per}, index=idx)
    DT = t['demand_tou']
    if DT['periods'] and DT['weekday']:
        dwd, dwk = np.array(DT['weekday']), np.array(DT['weekend'])
        out['demand_period'] = np.where(we, dwk[m, h], dwd[m, h])
    else:
        out['demand_period'] = -1
    return out


def bill(key, load_kW, tz=None):
    """Bill of an hourly load (kW, DatetimeIndex in local time) on a tariff: energy (tier 1 only), TOU demand charges
    (monthly max in each demand period), flat demand (monthly max), fixed charges. Returns dict of $ totals."""
    t = tariffs(key)
    load_kW = load_kW.astype(float)
    years = sorted(set(load_kW.index.year))
    pr = pd.concat([tariff_prices(key, y, tz=load_kW.index.tz) for y in years]).reindex(load_kW.index)
    energy = float((pr['energy_$per_kWh'] * load_kW).sum())
    dem = 0.0
    DT = t['demand_tou']
    if DT['periods']:
        for (y, mo, p), g in load_kW.groupby([load_kW.index.year, load_kW.index.month, pr.demand_period.values]):
            if p >= 0 and p < len(DT['periods']): dem += g.max() * DT['periods'][int(p)][0]['rate']
    DF = t['demand_flat']
    if DF['periods']:
        for (y, mo), g in load_kW.groupby([load_kW.index.year, load_kW.index.month]):
            p = DF['month_period'][mo - 1]
            if p is not None and int(p) < len(DF['periods']): dem += g.max() * DF['periods'][int(p)][0]['rate']
    fx = t['fixed']['first_meter'] or 0.0
    days = load_kW.index.normalize().nunique(); months = len(set(zip(load_kW.index.year, load_kW.index.month)))
    fixed = fx * days if (t['fixed']['units'] or '').endswith('day') else fx * months
    return dict(energy=energy, demand=dem, fixed=fixed, total=energy + dem + fixed)


def tariff_arrays(tariff, times, tz, rtp=None):
    """Per-step price arrays for a DatetimeIndex `times` (UTC or tz-aware, any step), DST-safe: every step takes the
    price of its own local month / hour / weekday; nothing is reindexed. Ported from ThermoPilot
    (training/tp/env.py price_arrays, v1.0). tariff: a URDB key (tariffs()), a dict {rates, weekday, weekend
    (12x24 period ids), demand_peak_rate, demand_hours (12x24 bool)}, or 'rtp...' (rtp_prices(times), or `rtp` =
    an hourly $/kWh Series). Returns dict of numpy arrays: price ($/kWh, tier 1 + adjustment), dperiod (TOU demand
    period id, -1 none), drate (TOU demand $/kW-month), frate (flat demand $/kW-month)."""
    loc = times.tz_convert(tz)
    n = len(times)
    if isinstance(tariff, str) and tariff.startswith('rtp'):
        p = rtp_prices(times) if rtp is None else rtp.reindex(times).ffill().bfill().values
        return dict(price=p, dperiod=np.full(n, -1), drate=np.zeros(n), frate=np.zeros(n))
    if isinstance(tariff, dict):
        m, h, we = loc.month.values - 1, loc.hour.values, loc.dayofweek.values >= 5
        wd, wk = np.array(tariff['weekday']), np.array(tariff['weekend'])
        per = np.where(we, wk[m, h], wd[m, h])
        price = np.array(tariff['rates'])[per]
        dh = np.array(tariff['demand_hours'])[m, h] & ~we
        drate = np.where(dh, tariff['demand_peak_rate'], 0.0)
        return dict(price=price, dperiod=np.where(dh, 0, -1), drate=drate, frate=np.zeros(n))
    t = tariffs(tariff)
    m, h, we = loc.month.values - 1, loc.hour.values, loc.dayofweek.values >= 5
    E = t['energy']
    ewd, ewk = np.array(E['weekday']), np.array(E['weekend'])
    eper = np.where(we, ewk[m, h], ewd[m, h])
    erate = np.array([p[0]['rate'] + p[0]['adj'] for p in E['periods']])
    D_ = t['demand_tou']
    if D_['periods'] and D_['weekday']:
        dwd, dwk = np.array(D_['weekday']), np.array(D_['weekend'])
        dp_ = np.where(we, dwk[m, h], dwd[m, h])
    else:
        dp_ = np.full(n, -1)
    pr = pd.DataFrame({'energy_$per_kWh': erate[eper], 'demand_period': dp_})
    dper = pr['demand_period'].values.astype(int)
    DT = t['demand_tou']
    drate = np.zeros(n)
    if DT['periods']:
        rates = np.array([p[0]['rate'] for p in DT['periods']])
        ok = (dper >= 0) & (dper < len(rates))
        drate[ok] = rates[dper[ok]]
    else:
        dper = np.full(n, -1)
    frate = np.zeros(n)
    DF = t['demand_flat']
    if DF['periods']:
        mp = DF['month_period']
        fr = np.array([DF['periods'][int(mp[mo])][0]['rate'] if mp[mo] is not None and int(mp[mo]) < len(DF['periods'])
                       else 0.0 for mo in range(12)])
        frate = fr[loc.month.values - 1]
    return dict(price=pr['energy_$per_kWh'].values.astype(float), dperiod=dper, drate=drate, frate=frate)


def bill_arrays(pa, load_kW, months):
    """(energy $, demand $) of a load on per-step arrays from tariff_arrays (ThermoPilot env.bill_arrays): energy =
    sum(price x load) (load in kWh per step, i.e. kW at 1 h); demand = per month the max load in each TOU demand
    period x its rate, plus the monthly max x the flat rate. months: one id per step (e.g. local year * 12 + month)."""
    e = float(np.sum(pa['price'] * load_kW))
    dem = 0.0
    for mo in np.unique(months):
        sel = months == mo
        dp = pa['dperiod'][sel]
        L = load_kW[sel]
        for p in np.unique(dp[dp >= 0]):
            s2 = dp == p
            dem += float(L[s2].max() * pa['drate'][sel][s2][0])
        if pa['frate'][sel].max() > 0:
            dem += float(L.max() * pa['frate'][sel][0])
    return e, dem


RTP_ADDER = 0.10      # $/kWh delivery + retail adder on the wholesale price (ThermoPilot's assumption, Griddy-style)
RTP_SHIFT_DAYS = 728  # weather year Y pairs with ERCOT prices of Y + 2 (104 weeks: weekdays stay aligned)


@functools.lru_cache(maxsize=4)
def _rtp_series(market, node):
    w = wholesale_prices(market, node)['price']
    return w[~w.index.duplicated()].sort_index()


def rtp_prices(times, market='ERCOT', node='HB_HOUSTON', shift_days=RTP_SHIFT_DAYS, adder=RTP_ADDER,
               clip=(-0.05, 3.0)):
    """Real-time retail prices for simulated weather years (ThermoPilot env.rtp_prices, v1.0): real hourly
    wholesale prices ($/MWh -> $/kWh, clipped) + adder, shifted forward by shift_days so the BDG2 weather years
    pair with years GridData has (ERCOT starts 2018: 2016 weather -> 2018 prices, 2017 -> 2019). times: UTC
    DatetimeIndex. Returns a numpy array ($/kWh)."""
    w = _rtp_series(market, node)
    t2 = times + pd.Timedelta(days=shift_days)
    p = w.reindex(t2).interpolate(limit_direction='both').values / 1000.0
    return np.clip(p, clip[0], clip[1]) + adder


def wholesale_prices(market='ERCOT', node='HB_HOUSTON', start=None, end=None, kind='rt'):
    """Real wholesale prices from GridData (read-only, no copy): ERCOT 2018-2026 hubs/load zones, CAISO 2023-08..2026."""
    import sys
    if GRID_CODE not in sys.path: sys.path.insert(0, GRID_CODE)   # griddata reads $ML_RATCHET_GRID itself
    import griddata as gd
    return gd.prices(market, node, start, end, kind=kind)


# ---------------------------------------------------------------- baselines
def baselines(name='summary'):
    """Baseline exam numbers: baselines() -> summary dict; baselines('<file stem>') -> per-unit table."""
    if name == 'summary': return json.load(open(f'{PROC}/baselines/summary.json'))
    return pd.read_parquet(f'{PROC}/baselines/{name}.parquet')


def priors(primary_use, climate_zone=None, vintage=None):
    """Envelope / HVAC / schedule priors for an unknown real building from ComStock 2025R3 (MODELLED stock): weighted
    distribution of in.* fields for the matching building-type group. Returns a DataFrame of value shares per field."""
    group = {'office': ['LargeOffice', 'MediumOffice', 'SmallOffice'], 'education': ['PrimarySchool', 'SecondarySchool'],
             'lodging': ['LargeHotel', 'SmallHotel'], 'retail': ['RetailStandalone', 'RetailStripmall'], 'healthcare': ['Hospital', 'Outpatient'],
             'warehouse': ['Warehouse'], 'food': ['FullServiceRestaurant', 'QuickServiceRestaurant', 'Grocery']}.get(primary_use)
    c = comstock(columns=None)
    if group: c = c[c['in.comstock_building_type'].astype(str).isin(group)]
    if climate_zone: c = c[c['in.as_simulated_ashrae_iecc_climate_zone_2006'].astype(str) == climate_zone]
    if vintage: c = c[c['in.vintage'].astype(str) == vintage]
    fields = ['in.hvac_system_type', 'in.wall_construction_type', 'in.window_type', 'in.window_to_wall_ratio_category',
              'in.number_of_stories', 'in.weekday_operating_hours..hr', 'in.tstat_htg_sp_f..f', 'in.tstat_clg_sp_f..f',
              'in.airtightness..m3_per_m2_h', 'in.heating_fuel']
    rows = []
    for f in fields:
        if f not in c: continue
        s = c.groupby(c[f].astype(str)).weight.sum(); s = s / s.sum()
        for v, w in s.sort_values(ascending=False).head(8).items(): rows.append(dict(field=f, value=v, share=round(float(w), 4)))
    return pd.DataFrame(rows)
