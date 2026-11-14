"""griddata: frozen real-world power-grid data for the ML Ratchet grid project (pandas / numpy / pyarrow only).

    import sys; sys.path.insert(0, f'{GRID_CODE}')
    import griddata as gd
    sp   = gd.splits()                                   # frozen dev/exam splits (data/proc/splits.json)
    d    = gd.eia930('ERCO', '2023-01-01', '2024-01-01')  # hourly demand, DA forecast, generation by fuel, interchange
    d    = gd.eia930('CISO', split='dev')                 # only dev years (2019, 2021)
    fl   = gd.fleet('ERCO')                               # list of generator-spec dicts (GRID_BRIEF contract)
    fl   = gd.fleet('ERCO', as_frame=True, plant_safe=True)   # PlantDetective: no CAMPD-derived values for exam plants
    u    = gd.campd_units(split='dev')                    # CAMPD Texas units (+ EIA links, BA, split)
    h    = gd.campd_unit('3470_WAP8', '2019-01-01', '2020-01-01')  # hourly gross load, heat input, CO2, op time
    ev   = gd.events('GB', split='exam')                   # frequency events (GB 1 s based; ERCO from ERCOT reports)
    meta, t, f = gd.event_trace('GB_20190809T155233')     # 1 s trace around one GB event (contract arrays)
    meta, t, f = gd.frequency('GB', '2019-08-09 15:40', '2019-08-09 16:10')
    p    = gd.prices('ERCOT', 'HB_HOUSTON', '2023-01-01', '2024-01-01', kind='rt')
    w    = gd.weather('ERCO', '2023-01-01', '2023-02-01')  # BA-level hourly weather (weights in points())
    net  = gd.network('case_ACTIVSg2000')                 # dict network (bus/gen/branch/gencost)
    b    = gd.baselines()                                  # baseline exam numbers (summary.json)

Conventions (GRID_BRIEF): UTC, hour-BEGINNING index named time_utc; MW, MWh, $/MWh, degC, m/s, W/m2.
Rule: only `dev` may be used for calibration, gates and fitting; `exam` is for the final report only.
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, json, functools, glob
import numpy as np, pandas as pd

ROOT = f'{GRID_ROOT}/data'
PROC = f'{ROOT}/proc'
RAW = f'{ROOT}/raw'
FOCUS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS', 'BPAT']


def _ts(x):
    if x is None: return None
    t = pd.Timestamp(x)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def _cut(df, start, end, col='time_utc'):
    s, e = _ts(start), _ts(end)
    idx = df.index if col is None else df[col]
    m = np.ones(len(df), bool)
    if s is not None: m &= (idx >= s)
    if e is not None: m &= (idx < e)
    return df[m]


@functools.lru_cache(maxsize=1)
def splits():
    """The frozen split file (see DATA.md section 9)."""
    return json.load(open(f'{PROC}/splits.json'))


def years(split):
    """Years of a time split: 'train_inputs' | 'dev' | 'exam' | 'reserve'."""
    return splits()['time'][split]


def _time_split(df, split, col=None):
    if split is None: return df
    idx = df.index if col is None else df[col]
    return df[np.isin(idx.year, years(split))]


# ---------------- EIA-930 ----------------
def eia930(ba, start=None, end=None, split=None, columns=None):
    """Hourly EIA-930 for one BA (index time_utc, hour-beginning UTC). Main columns: d (demand, EIA adjusted), df (BA
    day-ahead forecast), ng (net generation), ti (total interchange, + = export), ng_<FUEL> by fuel; d_rep/ng_rep/ti_rep
    (as reported); *_imp (EIA imputed); q_* quality flags; q_ok = demand usable as a label."""
    d = pd.read_parquet(f'{PROC}/eia930/ba/{ba}.parquet', columns=None if columns is None else ['time_utc'] + list(columns))
    d = _cut(d, start, end).set_index('time_utc')
    return _time_split(d, split)


def eia930_bas():
    """Per-BA coverage and quality summary (all BAs in the EIA-930 files)."""
    return json.load(open(f'{PROC}/eia930/quality_summary.json'))


def interchange(ba=None, diba=None, start=None, end=None):
    """Hourly interchange between directly connected BAs (MW, + = export from `ba` to `diba`)."""
    f = []
    if ba: f.append(('ba', '=', ba))
    if diba: f.append(('diba', '=', diba))
    d = pd.read_parquet(f'{PROC}/eia930/interchange.parquet', filters=f or None)
    return _cut(d, start, end)


def subregion(ba, start=None, end=None):
    """Hourly demand by EIA-930 subregion (e.g. ERCO weather zones, CISO PGAE/SCE/SDGE/VEA)."""
    d = pd.read_parquet(f'{PROC}/eia930/subregion.parquet', filters=[('ba', '=', ba)])
    return _cut(d, start, end)


# ---------------- fleet ----------------
def fleet(ba='ERCO', as_frame=False, plant_safe=False, existing_on=None):
    """Generator fleet in the GRID_BRIEF generator spec (one entry per EIA generator; `src` says where each filled
    value came from; defaults and their sources in fleet_defaults()).
    plant_safe=True: for PlantDetective EXAM plants, CAMPD-derived values (pmin, heat-rate curve, min up/down, observed
    ramps/starts) are replaced by values that use no CAMPD data. existing_on='YYYY-MM-DD': units online and not retired."""
    rows = json.load(open(f'{PROC}/fleet/fleet_{ba}.json'))
    if plant_safe:
        exam = set(splits()['plants']['exam'])
        for r in rows:
            if r['plant_id'] in exam and r.get('campd_linked'):
                r['pmin_MW'] = r.get('pmin_MW_nocampd'); r['heat_rate_curve'] = r.get('heat_rate_curve_nocampd')
                r['min_up_h'] = r.get('min_up_h_nocampd'); r['min_down_h'] = r.get('min_down_h_nocampd')
                r['ramp_obs_p99_MW_per_h'] = None; r['starts_per_year_obs'] = None
                r['src'] = {k: ('removed (plant_safe)' if 'CAMPD' in str(v) else v) for k, v in r['src'].items()}
    if existing_on is not None:
        t = str(pd.Timestamp(existing_on).date())
        rows = [r for r in rows if (r['online_date'] is None or r['online_date'] <= t) and (r['retired_date'] is None or r['retired_date'] > t)
                and r.get('status_latest') in ('existing', 'retired')]
    if as_frame:
        df = pd.DataFrame(rows)
        return df
    return rows


def fleet_defaults():
    return json.load(open(f'{PROC}/fleet/defaults.json'))


# ---------------- CAMPD ----------------
def campd_units(ba=None, split=None):
    """CAMPD Texas units: unit_id '<plant_id_epa>_<unit_id_epa>', names, type, fuel, EIA plant/generator links, BA,
    hours, plus PlantDetective split ('dev' | 'exam') by plant and the 2018-2022 operating characteristics."""
    u = pd.read_parquet(f'{PROC}/campd/units.parquet')
    ch = pd.read_parquet(f'{PROC}/campd/unit_characteristics.parquet').drop(columns=['plant_id_epa', 'unit_id_epa'])
    u = u.merge(ch, on='unit_id', how='left')
    sp = splits()['plants']
    u['split'] = np.where(u.plant_id_eia.isin(sp['dev']), 'dev', np.where(u.plant_id_eia.isin(sp['exam']), 'exam', None))
    if ba is not None: u = u[u.ba == ba]
    if split is not None: u = u[u.split == split]
    return u.reset_index(drop=True)


def campd_unit(unit_id, start=None, end=None):
    """Hourly CAMPD data for one unit (index time_utc): op_time [0-1 h], gross_load_MW, steam_klb_h, heat_input_MMBtu,
    co2_short_tons, nox_lbs, so2_lbs, hi_meas / co2_meas (measurement indicators)."""
    pe, ue = unit_id.split('_', 1)
    s, e = _ts(start), _ts(end)
    parts = []
    for f in sorted(glob.glob(f'{PROC}/campd/hourly/tx_*.parquet')):
        y = int(f[-12:-8])
        if s is not None and y < s.year - 1: continue   # local-standard-time files spill a few hours into the next UTC year
        if e is not None and y > e.year: continue
        parts.append(pd.read_parquet(f, filters=[('plant_id_epa', '=', int(pe)), ('unit_id_epa', '=', ue)]))
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if d.empty: return d
    d = _cut(d, start, end).set_index('time_utc').drop(columns=['plant_id_epa', 'unit_id_epa'])
    return d.sort_index()


# ---------------- frequency ----------------
def events(region='GB', split=None):
    """Frequency events. GB: detected in NESO 1 s data (largest 10 s fall <= -0.15 Hz), with 1 s metrics (f0, nadir,
    RoCoF), loss label (NESO report, or Elexon REMIT + B1610), NESO outturn inertia and generation mix.
    ERCO: ERCOT monthly Operations Reports (MW loss, delta f, min/max f, load, IRR share, inertia MW*s)."""
    if region == 'GB': d = pd.read_parquet(f'{PROC}/frequency/gb_events.parquet')
    elif region in ('ERCO', 'ERCOT'): d = pd.read_parquet(f'{PROC}/frequency/ercot_events.parquet'); region = 'ERCO'
    else: raise ValueError('region must be GB or ERCO')
    ev = splits()['events'].get(region, {})
    d['split'] = np.where(d.event_id.isin(ev.get('dev', [])), 'dev', np.where(d.event_id.isin(ev.get('exam', [])), 'exam',
                          np.where(d.event_id.isin(ev.get('reserve', [])), 'reserve', None)))
    if split is not None: d = d[d.split == split]
    return d.reset_index(drop=True)


def event_trace(event_id):
    """(meta, t [s from window start], f [Hz]) for a GB event window (1 s, ~10 min)."""
    p = f'{PROC}/frequency/GB/events/{event_id}'
    meta = json.load(open(p + '.json'))
    with np.load(p + '.npz') as z: return meta, z['t'], z['f']


def frequency(region='GB', start=None, end=None):
    """1 s system frequency (contract): returns (meta, t [s since start_utc], f [Hz], NaN = missing).
    GB full months stored: see frequency_months(); otherwise falls back to event windows covering the range."""
    if region != 'GB': raise ValueError('only GB 1 s frequency is available (see DATA.md)')
    s, e = _ts(start), _ts(end)
    out = []
    for f in sorted(glob.glob(f'{PROC}/frequency/GB/1s/*.npz')):
        with np.load(f) as z:
            t0 = pd.Timestamp(int(z['t0_unix']), unit='s', tz='UTC'); a = z['df_mHz']
        t1 = t0 + pd.Timedelta(seconds=len(a))
        if (e is not None and t0 >= e) or (s is not None and t1 <= s): continue
        i0 = 0 if s is None else max(0, int((s - t0).total_seconds())); i1 = len(a) if e is None else min(len(a), int((e - t0).total_seconds()))
        fv = np.where(a[i0:i1] == -32768, np.nan, 50.0 + a[i0:i1] / 1000.0)
        out.append(pd.Series(fv, index=t0 + pd.to_timedelta(np.arange(i0, i1), unit='s')))
    if not out:
        for f in sorted(glob.glob(f'{PROC}/frequency/GB/events/*.json')):
            m = json.load(open(f))
            with np.load(f[:-5] + '.npz') as z: tt, ff = z['t'], z['f']
            st = pd.Timestamp(m['start_utc'])
            idx = st + pd.to_timedelta(tt, unit='s')
            sr = pd.Series(ff, index=idx)
            if s is not None: sr = sr[sr.index >= s]
            if e is not None: sr = sr[sr.index < e]
            if len(sr): out.append(sr)
    if not out: return None, None, None
    x = pd.concat(out).sort_index(); x = x[~x.index.duplicated()]
    meta = dict(region='GB', nominal_hz=50.0, start_utc=str(x.index[0]), source='NESO system frequency data (1 s)')
    return meta, (x.index - x.index[0]).total_seconds().values, x.values


def frequency_months():
    return sorted(os.path.basename(f)[:7] for f in glob.glob(f'{PROC}/frequency/GB/1s/*.npz'))


def frequency_minutes(start=None, end=None):
    """Per-minute summary of GB 1 s frequency for every month 2018-01..2026-10 (min, max, mean, largest 10 s fall/rise)."""
    d = pd.read_parquet(f'{PROC}/frequency/GB/minute_summary.parquet')
    return _cut(d, start, end)


def gb_inertia(start=None, end=None):
    """NESO outturn and market-provided inertia (GVA.s) per settlement period (time_utc = period start)."""
    return _cut(pd.read_parquet(f'{PROC}/frequency/GB/inertia.parquet'), start, end)


def gb_genmix(start=None, end=None):
    """NESO historic GB generation mix, half-hourly MW by fuel (time_utc = period start)."""
    return _cut(pd.read_parquet(f'{PROC}/frequency/GB/genmix.parquet'), start, end)


# ---------------- prices ----------------
def prices(market='ERCOT', node='HB_HOUSTON', start=None, end=None, kind='rt'):
    """Settlement-point prices ($/MWh), index time_utc (interval beginning).
    ERCOT: kind 'da' (hourly DAM), 'rt' (hourly mean of 15-min RTM), 'rt15' (15-min RTM); hubs HB_* and zones LZ_*.
    CAISO: kind 'da' (hourly DAM LMP), 'rt15' (15-min RTPD LMP), 'rt' (hourly mean of RTPD); columns LMP, MCE, MCC, MCL.
    CAISO nodes: TH_NP15_GEN-APND, TH_SP15_GEN-APND, TH_ZP26_GEN-APND, DLAP_PGAE-APND, DLAP_SCE-APND, DLAP_SDGE-APND."""
    if market == 'ERCOT':
        f = {'da': 'ercot_dam', 'rt': 'ercot_rtm_hourly', 'rt15': 'ercot_rtm'}[kind]
        d = pd.read_parquet(f'{PROC}/prices/{f}.parquet', filters=[('node', '=', node)])
        return _cut(d, start, end).set_index('time_utc').drop(columns=['node'])
    if market == 'CAISO':
        f = {'da': 'caiso_dam', 'rt15': 'caiso_rtpd', 'rt': 'caiso_rtpd'}[kind]
        d = pd.read_parquet(f'{PROC}/prices/{f}.parquet', filters=[('node', '=', node)])
        d = _cut(d, start, end).set_index('time_utc').drop(columns=['node'])
        if kind == 'rt': d = d.resample('h').mean()
        return d
    raise ValueError('market must be ERCOT or CAISO')


# ---------------- weather ----------------
def weather(ba='ERCO', start=None, end=None):
    """BA-level hourly weather (index time_utc). NASA POWER (MERRA-2/CERES) weighted series: T2m_C, RH2m_pct, q2m_g_kg,
    ps_kPa (population points), wind50m_ms / wind10m_ms (wind-capacity points), ghi_Wm2 / ghi_clear_Wm2 (solar-capacity
    points); US only: T_asos_C, Td_asos_C, RH_asos_pct, wind_asos_ms (ASOS stations nearest the population points).
    ba='GB' for Great Britain."""
    d = pd.read_parquet(f'{PROC}/weather/{ba}.parquet')
    return _cut(d, start, end).set_index('time_utc')


def weather_points(ba=None):
    """Weather points and weights per BA (kind pop/wind/solar), plus the ASOS station chosen per population point."""
    p = json.load(open(f'{PROC}/weather/points.json'))
    return p if ba is None else p[ba]


def weather_point(pid, start=None, end=None):
    """NASA POWER hourly series at one weather point (see weather_points)."""
    d = pd.read_parquet(f'{RAW}/weather/power/{pid}.parquet')
    return _cut(d, start, end).set_index('time_utc')


# ---------------- networks ----------------
def network(name='case_ACTIVSg2000'):
    """Network dict: baseMVA, bus, gen, branch, gencost (MATPOWER units; see DATA.md section 8)."""
    return json.load(open(f'{PROC}/networks/{name}.json'))


def network_tables(name='case_ACTIVSg2000'):
    """(bus, gen, branch) DataFrames."""
    return tuple(pd.read_parquet(f'{PROC}/networks/{name}_{k}.parquet') for k in ('bus', 'gen', 'branch'))


def networks():
    return json.load(open(f'{PROC}/networks/index.json'))


# ---------------- baselines ----------------
def baselines(name='summary'):
    """'summary' -> dict of exam numbers; or a table name in proc/baselines (e.g. 'demand_exam_predictions',
    'heatrate_unit_years', 'battery_daily', 'frequency_events')."""
    if name == 'summary': return json.load(open(f'{PROC}/baselines/summary.json'))
    return pd.read_parquet(f'{PROC}/baselines/{name}.parquet')
