"""ThermoPilot environment: weather, tariffs, comfort bands, forecasts and episode bookkeeping on BuildingSim.

Everything runs at a 1 h step (dt = 1 h). An episode is one calendar month (local time) of one building, one weather
year and one tariff. The building is a BuildingSim spec run through bs.Simulator; the controller sets the heating and
cooling setpoints each hour.
"""
import sys, os, json, math, functools
from _paths import PROC, WORK   # BuildingSim / BuildingData on sys.path (was the sandbox bldg/ folder)
import numpy as np
import pandas as pd
import buildingsim as bs
import buildingdata as bd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(WORK, 'data')

# Gas prices: EIA "Natural Gas Prices" (dnav/ng/ng_pri_sum_dcu_nus_a), 2025 US annual average, nominal $/Mcf:
# residential 15.34, commercial 10.98. 1 Mcf ~ 1.036 MMBtu = 303.6 kWh. Tariffs are 2024-26 URDB rates, so 2025 gas.
GAS_RES = 15.34 / 303.6       # $/kWh of gas
GAS_COM = 10.98 / 303.6

RES_TARIFFS = ['pge_e_tou_c', 'sce_tou_d_prime', 'sce_tou_d_5_8', 'aps_r3', 'srp_e26', 'austin_res_tou', 'duke_fl_rt',
               'xcel_co_re_tou', 'rmp_ut_sched1_tou', 'xcel_mn_a04_tod', 'nyseg_sc12_tou', 'pseg_rs_tou_3p',
               'coned_sc1_rate3']
COM_TARIFFS = ['pge_b19_sec', 'sce_tou_gs3_b', 'ouc_gsd_tou', 'xcel_mn_a15_tod', 'pepco_gtlv']   # aps_e32_l excluded
# (APS E-32 L shows 0.57 $/kWh energy rates in URDB: a likely entry error, DATA.md section 10)

SITES_US = ['Bear', 'Swan', 'Bobcat', 'Bull', 'Cockatoo', 'Eagle', 'Fox', 'Gator', 'Panther', 'Hog', 'Rat', 'Peacock']
SITE_CZ = dict(Bear='3C', Swan='3C', Bobcat='5B', Bull='2A', Cockatoo='6A', Eagle='5A', Peacock='5A', Fox='2B',
               Gator='2A', Panther='2A', Hog='6A', Rat='4A')
EPW_CITIES = ['Fairbanks', 'Phoenix', 'Los.Angeles', 'San.Francisco', 'Boulder', 'Denver', 'Miami', 'Atlanta',
              'Chicago', 'Baltimore', 'Duluth', 'Minneapolis', 'Helena', 'New.York', 'El.Paso', 'Houston', 'Seattle']


# ------------------------------------------------------------------------------------------------ weather
@functools.lru_cache(maxsize=64)
def weather(source, year=None):
    """(DataFrame UTC, meta) in the contract. source: 'site:<BDG2 site>' (actual year, 2016 or 2017),
    'epw:<city>' (TMYx rebased to `year`), 'hue' (YVR station + NASA POWER irradiance), 'wpuq' (local station)."""
    kind, _, name = source.partition(':')
    if kind == 'site':
        w = bd.site_weather(name); m = bd.site_weather_meta(name)
        w = w[['T_out', 'RH', 'dewpoint', 'GHI', 'DNI', 'DHI', 'wind_speed', 'wind_dir', 'pressure', 'cloud']].copy()
        w = w.interpolate(limit_direction='both')
        return w, dict(lat=m['lat'], lon=m['lon'], elev=m['elev'], tz=bd.SITE_TZ[name])
    if kind == 'epw':
        df, m = bs.read_epw(bs.epw_path(name), year=year or 2016)
        return df, m
    if kind == 'hue':
        h = pd.read_parquet(os.path.join(PROC, 'other', 'hue_weather.parquet'))
        h = h[h.station == 'YVR'].set_index('time_utc').sort_index()
        h = h[~h.index.duplicated()]
        frames = []
        for f in ('power_2016.json', 'power_2017.json'):
            j = json.load(open(os.path.join(DATA, 'raw', 'nasa_yvr', f)))['properties']['parameter']
            d = pd.DataFrame({k: pd.Series(v) for k, v in j.items()})
            d.index = pd.to_datetime(d.index, format='%Y%m%d%H', utc=True)
            frames.append(d)
        p = pd.concat(frames); p = p[~p.index.duplicated()].sort_index().replace(-999.0, np.nan)
        idx = pd.date_range('2016-01-01', '2018-01-02', freq='h', tz='UTC')
        T = h['T_out'].reindex(idx).interpolate(limit_direction='both')
        RH = h['RH'].reindex(idx).interpolate(limit_direction='both').clip(5, 100)
        Pa = h['pressure'].reindex(idx).interpolate(limit_direction='both')
        df = pd.DataFrame(dict(T_out=T, RH=RH, pressure=Pa, GHI=p['ALLSKY_SFC_SW_DWN'].reindex(idx).fillna(0).clip(0),
                               DNI=p['ALLSKY_SFC_SW_DNI'].reindex(idx).fillna(0).clip(0),
                               DHI=p['ALLSKY_SFC_SW_DIFF'].reindex(idx).fillna(0).clip(0),
                               wind_speed=p['WS10M'].reindex(idx).interpolate(limit_direction='both'),
                               wind_dir=180.0), index=idx)
        return df, dict(lat=49.19, lon=-123.18, elev=4.0, tz='America/Vancouver', source='YVR (HUE) + NASA POWER')
    if kind == 'wpuq':
        w = pd.read_parquet(os.path.join(PROC, 'other', 'wpuq_weather_hourly.parquet'))
        idx = pd.date_range('2018-01-01', '2021-01-01', freq='h', tz='UTC', inclusive='left')
        w = w[~w.index.duplicated()].reindex(idx)
        df = pd.DataFrame(dict(T_out=w['T_out'].interpolate(limit_direction='both'),
                               RH=w['RH'].interpolate(limit_direction='both').clip(5, 100),
                               pressure=(w['pressure_raw'].interpolate(limit_direction='both') *
                                         (100.0 if w['pressure_raw'].median() < 2000 else 1.0)),
                               GHI=w['GHI'].fillna(0).clip(0),
                               wind_speed=w['wind_speed'].interpolate(limit_direction='both').clip(0, 40),
                               wind_dir=w['wind_dir'].interpolate(limit_direction='both')), index=idx)
        return df, dict(lat=52.10, lon=9.36, elev=70.0, tz='Europe/Berlin', source='WPuQ local weather (Hameln)')
    raise ValueError(source)


# ------------------------------------------------------------------------------------------------ tariffs
@functools.lru_cache(maxsize=None)
def _tariff(key):
    return bd.tariffs(key)


def synth_tariff(seed, demand=False):
    """A random TOU / flat tariff for training only (dict in the same shape the episode code uses)."""
    r = np.random.default_rng(seed)
    base = float(r.uniform(0.07, 0.25))
    kind = r.choice(['flat', 'tou2', 'tou3'], p=[0.15, 0.55, 0.30])
    sched = np.zeros((12, 24), int); rates = [base]
    if kind != 'flat':
        h0 = int(r.integers(12, 19)); L = int(r.integers(2, 7))
        ratio = float(r.uniform(1.4, 4.0))
        rates.append(base * ratio)
        months = np.arange(12) if r.random() < 0.5 else np.array([5, 6, 7, 8]) if r.random() < 0.6 else np.array([0, 1, 11, 5, 6, 7, 8])
        for mo in months:
            sched[mo, h0:min(h0 + L, 24)] = 1
        if kind == 'tou3':
            rates.append(base * (1 + ratio) / 2)
            for mo in months:
                for h in list(range(max(h0 - 3, 0), h0)) + list(range(min(h0 + L, 24), min(h0 + L + 2, 24))):
                    if sched[mo, h] == 0: sched[mo, h] = 2
    wk = sched.copy() if r.random() < 0.3 else np.zeros((12, 24), int)
    dem = float(r.uniform(5, 20)) if (demand or r.random() < 0.15) else 0.0
    dh = sched > 0
    if dem > 0 and not dh.any():
        dh = np.zeros((12, 24), bool); dh[:, 16:20] = True
    return dict(key=f'synth{seed}', rates=rates, weekday=sched.tolist(), weekend=wk.tolist(), demand_peak_rate=dem,
                demand_hours=dh.tolist(), fixed=0.0)


def price_arrays(tariff, times, tz, rtp=None):
    """Per-step arrays for UTC `times`: energy price $/kWh, demand period id (-1 none), TOU demand rate $/kW-month,
    flat demand rate $/kW-month. tariff: URDB key, synth dict, or 'rtp' (with rtp = hourly $/kWh Series)."""
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
    t = _tariff(tariff)
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


RTP_ADDER = 0.10     # $/kWh delivery + retail adder on top of the wholesale price (assumption, Griddy-style pass-through)
RTP_SHIFT_DAYS = 728  # weather year Y uses ERCOT prices of year Y + 2 (104 weeks: weekdays stay aligned)


@functools.lru_cache(maxsize=4)
def _ercot():
    w = bd.wholesale_prices('ERCOT', 'HB_HOUSTON')['price']
    return w[~w.index.duplicated()].sort_index()


def rtp_prices(times):
    """Real ERCOT HB_HOUSTON real-time hourly prices ($/MWh -> $/kWh, clipped to [-0.05, 3]) + RTP_ADDER,
    shifted forward by RTP_SHIFT_DAYS so 2016 weather pairs with 2018 prices and 2017 weather with 2019 prices."""
    w = _ercot()
    t2 = times + pd.Timedelta(days=RTP_SHIFT_DAYS)
    p = w.reindex(t2).interpolate(limit_direction='both').values / 1000.0
    return np.clip(p, -0.05, 3.0) + RTP_ADDER


def bill(tariff, load_kW, tz, gas_kWh=0.0, gas_price=GAS_RES):
    """$ bill of an hourly electricity load (UTC Series) + gas. URDB: bd.bill (tier-1 energy, TOU + flat demand,
    fixed). Synthetic / RTP: energy + peak demand (no fixed)."""
    if isinstance(tariff, str) and not tariff.startswith('rtp'):
        loc = load_kW.copy(); loc.index = loc.index.tz_convert(tz)
        loc = loc[~loc.index.duplicated()]
        b = bd.bill(tariff, loc)
    else:
        pa = price_arrays(tariff, load_kW.index, tz, rtp=getattr(bill, '_rtp', None)) if not isinstance(tariff, str) else None
        b = dict(energy=float('nan'), demand=0.0, fixed=0.0, total=float('nan'))
    b['gas'] = gas_kWh * gas_price
    b['total_with_gas'] = b['total'] + b['gas']
    return b


def bill_arrays(pa, load_kW, months):
    """Bill from per-step price arrays (any tariff kind): energy + TOU demand (monthly max per period) + flat demand.
    months: array of (year*12+month) per step."""
    e = float(np.sum(pa['price'] * load_kW))
    dem = 0.0
    for mo in np.unique(months):
        sel = months == mo
        dp = pa['dperiod'][sel]; L = load_kW[sel]
        for p in np.unique(dp[dp >= 0]):
            s2 = dp == p
            dem += float(L[s2].max() * pa['drate'][sel][s2][0])
        if pa['frate'][sel].max() > 0:
            dem += float(L.max() * pa['frate'][sel][0])
    return e, dem


# ------------------------------------------------------------------------------------------------ forecasts
class Forecaster:
    """Weather forecasts with error that grows with lead time, issued every 6 h (like NWP cycles).
    Temperature: additive error, sd 0.6 + 1.6 (1 - exp(-L/24)) K, AR(1) in lead (rho 0.95 per h).
    Irradiance: multiplicative log-normal error, sd 0.15 + 0.25 (1 - exp(-L/12)), AR(1) rho 0.9.
    perfect=True returns the truth. The current hour's temperature is always observed."""

    def __init__(self, T, G, H, seed, perfect=False, scale=1.0):
        self.T, self.G, self.H, self.perfect = np.asarray(T, float), np.asarray(G, float), H, perfect
        n = len(T)
        nc = n // 6 + 2
        L = np.arange(H + 7)
        r = np.random.default_rng(seed)
        def ar(rho, size):
            z = r.standard_normal(size)
            out = np.empty(size)
            out[:, 0] = z[:, 0]
            for j in range(1, size[1]):
                out[:, j] = rho * out[:, j - 1] + math.sqrt(1 - rho * rho) * z[:, j]
            return out
        self.eT = scale * (0.6 + 1.6 * (1 - np.exp(-L / 24.0)))[None, :] * ar(0.95, (nc, H + 7))
        self.eG = scale * (0.15 + 0.25 * (1 - np.exp(-L / 12.0)))[None, :] * ar(0.9, (nc, H + 7))

    def get(self, k):
        n = len(self.T)
        idx = np.minimum(np.arange(k, k + self.H), n - 1)
        T, G = self.T[idx].copy(), self.G[idx].copy()
        if not self.perfect:
            c = k // 6
            lead = idx - 6 * c
            lead = np.minimum(lead, self.eT.shape[1] - 1)
            T += self.eT[c, lead]
            G *= np.exp(self.eG[c, lead] - 0.5 * self.eG[c, lead] ** 2)
            T[0] = self.T[k]
        return T, G


# ------------------------------------------------------------------------------------------------ comfort bands
def band_schedule(sim, pref):
    """Per-step comfort band [lo, hi] (thermostat air temperature, C) and the state 0 home-awake, 1 asleep, 2 away
    (homes) / 0 occupied, 2 unoccupied (commercial). pref: dict(lo, hi, w_sleep, w_away)."""
    sc = sim.sched
    n = sim.n
    g = lambda k, d: float(pref.get(k, pref.get(d, 0.0)))
    wsh, wsc, wah, wac = g('w_sleep_h', 'w_sleep'), g('w_sleep_c', 'w_sleep'), g('w_away_h', 'w_away'), g('w_away_c', 'w_away')
    if sim.res:
        asleep = np.asarray(sc['asleep'][:n]) > 0.5
        away = (np.asarray(sc['away'][:n]) > 0.5) & ~asleep
        state = np.where(asleep, 1, np.where(away, 2, 0))
        wh = np.where(asleep, wsh, np.where(away, wah, 0.0))
        wc = np.where(asleep, wsc, np.where(away, wac, 0.0))
    else:
        occ = np.asarray(sim.hvac_sched[:n]) > 0.5
        state = np.where(occ, 0, 2)
        wh = np.where(occ, 0.0, wah); wc = np.where(occ, 0.0, wac)
    return pref['lo'] - wh, pref['hi'] + wc, state


def local_calendar(times, tz):
    loc = times.tz_convert(tz)
    return dict(hour=loc.hour.values, wd=(loc.dayofweek.values < 5), month=loc.month.values,
                ym=loc.year.values * 12 + loc.month.values, doy=loc.dayofyear.values)
