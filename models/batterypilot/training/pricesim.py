"""BatteryPilot price simulator: GridSim fleets + real inputs -> hourly day-ahead and real-time price paths.

Wraps GridSim (unmodified) in this folder, as the Phase 2 brief asks.  Per scenario-year:
 1. Real inputs of one year (EIA-930 demand, its day-ahead demand forecast, wind, solar, hydro, nuclear, net imports;
    NASA POWER population-weighted temperature), scaled by the scenario (demand growth, solar/wind build-out).
 2. Thermal fleet = GridSim's compiled fleet (real ERCOT/CAISO fleet of that year, or a generated fleet), with
    - Markov forced outages + planned outages (GridSim `availability(mode='random')`),
    - gas-turbine ambient derate (output falls ~0.6 %/K for CTs, 0.4 %/K for CCs above the 35 C summer rating),
    - a cold-weather outage fraction f_cold(T) (gas supply freeze-offs and frozen instruments; fitted on Uri, dev 2021).
 3. System storage (a fleet of batteries) flattens the thermal net load each day (valley filling / peak shaving).
 4. Energy price lambda = the marginal block of the merit-order stack (pmin blocks + 3 incremental heat-rate
    segments per cluster, daily fuel prices).  Slow-start units needed for the day's peak stay at least at pmin;
    when the net load falls below that floor, wind/solar set the price (their negative offer).
 5. Scarcity: an operating-reserve demand curve like ERCOT's ORDC.  Reserve R = available thermal capacity -
    thermal output + storage discharge headroom.  Adder = (VOLL - lambda) * LOLP(R - X), LOLP = 1 - Phi((R - X -
    mu)/sigma).  mu, sigma fitted on the dev years (2019, 2021).  RT price = min(lambda + adder, cap).
 6. Day-ahead: the same with forecast inputs known the day before (real EIA-930 demand forecast, wind/solar forecast
    errors, outages as known at noon the day before) and the EXPECTED adder under the forecast uncertainty
    (risk-neutral convergence: DA ~ E[RT]).
 7. GridSim physics 1.2 only ($GRIDSIM_PHYSICS=1.2; BatteryPilot v1.3): GridSim's real-time scarcity the day-ahead
    market cannot see (gridsim.market) is added to RT, and its hour/month expectation to DA.  ERCOT only.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr

import os  # noqa: E402
import _paths  # noqa: E402  (GRID/gridsim and GRID on sys.path)
from gridsim.fleet import (prepare_fleet, merge_combined_cycles, fuel_prices_daily, calibration,  # noqa: E402
                           calibration_path, physics_version)
from gridsim.dispatch import CompiledFleet, availability  # noqa: E402
from gridsim import market  # noqa: E402  (physics 1.2: RT scarcity the DA cannot see; zeros under 1.0 / 1.1)

FLEETS = {'ERCO': os.path.join(_paths.GRID, 'data', 'proc', 'fleet', 'fleet_ERCO.parquet'),
          'CISO': os.path.join(_paths.GRID, 'data', 'proc', 'fleet', 'fleet_CISO.parquet')}
TZ = {'ERCO': 'America/Chicago', 'CISO': 'America/Los_Angeles'}
SLOW = ('coal_steam', 'gas_cc', 'gas_steam', 'biomass', 'other', 'geothermal')
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_W = GH_W / GH_W.sum()

# default physics / scarcity parameters (calibrated values are loaded from data/calibration.json when present)
DEFAULTS = dict(mu=0.0, sigma=2000.0, X=2000.0, voll=9000.0, cap=9000.0, floor=-250.0,
                ren_offer=-5.0, ct_derate=0.006, cc_derate=0.004, T_rating=35.0,
                cold_fmax=0.0, cold_T50=-6.0, cold_w=2.0, cold_tau_h=36.0,
                wind_fc_sd=0.07, solar_fc_sd=0.05, fc_rho=0.97, outage_sd_frac=0.02, markup=0.0,
                floor_reserve=3000.0, rt_noise_sd=0.0, rt_noise_rho=0.7, floor_share=1.0, ramp_k=0.0,
                da_sd_scale=1.0, scar_scale=1.0, scar_fit='ERCO')


def ercot_dispatch():
    """GridSim's dev-2019 ERCOT dispatch calibration (coal must-run, coal offer adder) for the active GridSim physics
    version ($GRIDSIM_PHYSICS; gridsim.fleet.calibration()).  Physics 1.0 gives the released -6.62 $/MWh.  The adder
    was fitted with the version's own outage defaults, so it must come from the same version; no silent default."""
    v = physics_version()
    ed = calibration(v).get('ercot_dispatch') or {}
    if 'coal_steam' not in ed.get('offer_adders', {}):
        raise RuntimeError(f'GridSim physics {v}: no ercot_dispatch.offer_adders.coal_steam in {calibration_path(v)}')
    return v, float(ed['offer_adders']['coal_steam']), list(ed.get('must_run', ['coal_steam']))


def load_inputs(ba, year):
    """Real hourly inputs for a BA-year (UTC index): d, df, wind, solar, hydro, nuc, imp, T."""
    import griddata as gd
    s, e = pd.Timestamp(f'{year}-01-01', tz='UTC'), pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
    idx = pd.date_range(s, e, freq='h', inclusive='left')
    y = gd.eia930(ba, s, e)
    y = (y.set_index('time_utc') if 'time_utc' in y else y)
    y = y[~y.index.duplicated()].reindex(idx)
    num = lambda c: pd.to_numeric(y[c], errors='coerce').astype(float).interpolate(limit_direction='both').fillna(0)  # noqa
    f = pd.DataFrame(index=idx)
    f['d'] = num('d')
    f['df'] = pd.to_numeric(y['df'], errors='coerce').astype(float)
    bad = f.df.isna() | ((f.df - f.d).abs() > 0.25 * f.d)
    f.loc[bad, 'df'] = np.nan
    f['df'] = f.df.interpolate(limit_direction='both').fillna(f.d)
    f['wind'] = num('ng_WND').clip(lower=0)
    f['solar'] = num('ng_SUN').clip(lower=0)
    f['hydro'] = num('ng_WAT').clip(lower=0)
    f['nuc'] = num('ng_NUC').clip(lower=0)
    f['imp'] = -num('ti')
    w = gd.weather(ba, s, e)
    f['T'] = w['T2m_C'].reindex(idx).astype(float).interpolate(limit_direction='both')
    if ba == 'CISO':
        # CISO's hydro and 'other' (geothermal, biomass, batteries before 2024-07) are taken as fixed injections
        f['hydro'] = f.hydro + num('ng_OTH').clip(0, None) * 0.0
    return f


class Engine:
    """Compiled thermal stack of one fleet for one year."""

    def __init__(self, ba='ERCO', year=2019, fleet=None, retire_frac=0.0, seed=0):
        self.ba = ba
        src = FLEETS[ba] if fleet is None else fleet
        u = prepare_fleet(src, start=f'{year}-01-01', end=f'{year}-12-31',
                          hr_source='spec' if fleet is not None else 'eia923')
        u = u[~u.tech.isin(['nuclear', 'hydro_res', 'hydro_ror', 'pumped', 'battery', 'wind', 'solar_pv'])]
        if retire_frac > 0:
            rng = np.random.default_rng(seed)
            th = u.index[u.tech.isin(['coal_steam', 'gas_steam', 'gas_ct', 'gas_cc'])]
            drop = rng.choice(th, int(retire_frac * len(th)), replace=False)
            u = u.drop(drop)
        u = merge_combined_cycles(u)
        self.units = u
        self.cf = CompiledFleet(u)
        self.year = year
        cf = self.cf
        self.slow = np.isin(cf.tech, SLOW)
        self.is_ct = np.isin(cf.tech, ['gas_ct', 'gas_ice', 'oil'])
        self.is_cc = cf.tech == 'gas_cc'
        self.is_gas = np.isin(cf.tech, ['gas_cc', 'gas_ct', 'gas_steam', 'gas_ice'])
        # GridSim's ERCOT dispatch calibration of the active physics version (was hard-coded: must-run coal, -6.62)
        # (CISO: no must-run, no adder, as released)
        self.physics = physics_version()
        self.must_run = np.zeros(cf.C, bool); self.coal_adder = 0.0
        if ba == 'ERCO':
            _, self.coal_adder, must = ercot_dispatch()
            self.must_run = np.isin(cf.tech, must)

    def fuel(self, times, gas_scale=1.0, mode='eia923'):
        cf = self.cf
        days = pd.date_range(times[0].tz_localize(None).normalize(), times[-1].tz_localize(None).normalize(), freq='D')
        fp = fuel_prices_daily(cf.thermal_units, days[0], days[-1], mode=mode)
        fpc = np.zeros((len(days), cf.C))
        for j, c in enumerate(cf.clusters):
            w = cf.thermal_units.loc[c['members'], 'pmax_MW'].values
            fpc[:, j] = fp[c['members']].values @ w / w.sum()
        gas = np.isin(cf.tech, ['gas_cc', 'gas_ct', 'gas_steam', 'gas_ice'])
        fpc[:, gas] *= gas_scale
        di = np.searchsorted(days.values, times.tz_localize(None).normalize().values)
        return fpc[di]                                    # (T, C) $/MMBtu

    def blocks(self, fprice, coal_adder=0.0, carbon=0.0):
        """Block costs (T, C, K+1) and widths per unit (C, K+1): block 0 = pmin at average cost.
        coal_adder: GridSim's dev-2019 coal offer adder for ERCOT (gridsim.fleet.calibration() of the physics
        version: -6.62 $/MWh in 1.0, data/calibration.json; 1.1 in data/calibration_v1_1.json)."""
        cf = self.cf
        pmin = np.maximum(cf.pmin, 1e-3)
        add = cf.vom + np.where(cf.tech == 'coal_steam', coal_adder, 0.0)
        c0 = cf.Fmin[None, :] / pmin[None, :] * fprice + add[None, :]
        ck = cf.ihr[None, :, :] * fprice[:, :, None] + add[None, :, None]
        cost = np.concatenate([c0[:, :, None], ck], 2)
        if carbon:
            hr = np.concatenate([(cf.Fmin / pmin)[:, None], cf.ihr], 1)          # MMBtu/MWh per block
            cost = cost + carbon * (cf.co2[:, None] / 1000.0 * hr)[None, :, :]    # $/t x t/MWh
        wid = np.concatenate([cf.pmin[:, None], cf.widths], 1)
        return cost, wid


def ordc_lolp(R, X, mu, sigma):
    return np.where(R <= X, 1.0, 1.0 - ndtr((R - X - mu) / sigma))


def storage_flatten(L, day_id, Ps, Es, eta=0.85):
    """Daily valley-filling / peak-shaving of net load L by a storage fleet (power Ps MW, energy Es MWh stored).
    Returns modified L and the storage discharge headroom (MW) available each hour."""
    if Ps <= 0 or Es <= 0:
        return L.copy(), np.zeros_like(L)
    Ln = L.copy(); head = np.zeros_like(L)
    ec, ed = np.sqrt(eta), np.sqrt(eta)
    for d in np.unique(day_id):
        i = np.where(day_id == d)[0]
        x = L[i]
        # discharge top hours down to level hd delivering at most Es*ed; charge bottom hours up to hc storing the same
        def take(level):  # energy delivered above level
            return np.minimum(Ps, np.maximum(x - level, 0)).sum()
        def fill(level):
            return np.minimum(Ps, np.maximum(level - x, 0)).sum()
        lo, hi = x.min(), x.max()
        a, b = lo, hi
        for _ in range(30):
            m = 0.5 * (a + b)
            if take(m) > Es * ed: a = m
            else: b = m
        hd = b; Ed = take(hd)
        a, b = lo, hi
        for _ in range(30):
            m = 0.5 * (a + b)
            if fill(m) * ec > Ed / ed: b = m
            else: a = m
        hc = a
        if hc >= hd:          # flat day: no arbitrage
            head[i] = Ps * 0.5
            continue
        dis = np.minimum(Ps, np.maximum(x - hd, 0)); ch = np.minimum(Ps, np.maximum(hc - x, 0))
        Ln[i] = x - dis + ch
        # SOC-limited discharge headroom: rough (charged energy available all day, before the discharge hours)
        head[i] = np.maximum(Ps - dis, 0) * 0.6
    return Ln, head


def stack_price(eng: Engine, L, cap_units, cost, wid, ren_av, ren_offer, floor, floor_reserve, day_id,
                floor_share=1.0):
    """Merit-order price for thermal net load L (T,), unit counts available cap_units (T, C) and derated widths.
    Returns lambda (T,), thermal output (T,), available thermal capacity (T,), curtailment flag."""
    T, C, B = cost.shape
    capb = cap_units[:, :, None] * wid[None, :, :]                 # (T, C, B) MW per block
    cost2 = cost.reshape(T, C * B); cap2 = capb.reshape(T, C * B)
    o = np.argsort(cost2, 1, kind='stable')
    cs = np.take_along_axis(cost2, o, 1); cp = np.take_along_axis(cap2, o, 1)
    cum = np.cumsum(cp, 1)
    total = cum[:, -1]
    # floor: pmin of slow units needed for the day's peak (+ reserve)
    slow_pmin = (cap_units * eng.cf.pmin[None, :] * eng.slow[None, :])
    M = np.zeros(T)
    for d in np.unique(day_id):
        i = np.where(day_id == d)[0]
        k = i[np.argmax(L[i])]
        need = L[k] + floor_reserve
        j = np.searchsorted(cum[k], need)
        sel = o[k, :min(j + 1, C * B)] // B
        used = np.zeros(C, bool); used[np.unique(sel)] = True
        used |= eng.must_run
        M[i] = floor_share * (slow_pmin[k] * used).sum()
    gen = np.clip(np.maximum(L, M), 0, total)
    j = np.array([np.searchsorted(cum[t], gen[t]) for t in range(T)])
    j = np.minimum(j, C * B - 1)
    lam = cs[np.arange(T), j]
    low = L < M
    over = M - L
    lam = np.where(low, np.where(over <= ren_av, ren_offer, floor), lam)
    return lam, gen, total, low


def fundamentals(eng: Engine, inp: pd.DataFrame, scen: dict, P: dict, seed=0):
    """Expensive part of a scenario-year (does not depend on the ORDC / noise parameters): energy prices and
    reserves in real time and day ahead.  Returns a dict of arrays."""
    P = {**DEFAULTS, **P, **{k: v for k, v in scen.items() if k in DEFAULTS}}
    rng = np.random.default_rng(seed)
    times = inp.index
    T = len(times)
    loc = times.tz_convert(TZ[eng.ba])
    day_id = ((loc.normalize().tz_localize(None) - pd.Timestamp('2000-01-01')).days).values
    ds, ws, ss = scen.get('demand_scale', 1.0), scen.get('wind_scale', 1.0), scen.get('solar_scale', 1.0)
    d = inp.d.values * ds
    df = inp.df.values * ds
    wind = inp.wind.values * ws
    solar = inp.solar.values * ss
    fixed = inp.hydro.values + inp.nuc.values + inp.imp.values * scen.get('imports_scale', 1.0)
    Tamb = inp['T'].values

    def ar1(sd, rho):
        e = np.zeros(T); z = rng.normal(0, sd * np.sqrt(1 - rho ** 2), T); e[0] = rng.normal(0, sd)
        for t in range(1, T):
            e[t] = rho * e[t - 1] + z[t]
        return e
    wcap = max(wind.max(), 1.0); scap = max(solar.max(), 1.0)
    wind_f = np.clip(wind + ar1(P['wind_fc_sd'], P['fc_rho']) * wcap, 0, wcap)
    solar_f = np.clip(solar + ar1(P['solar_fc_sd'], P['fc_rho']) * scap * (solar > 0.02 * scap), 0, scap)
    cf = eng.cf
    navail, _ = availability(cf, times, mode='random', seed=seed)
    fc = P['cold_fmax'] * ndtr((P['cold_T50'] - Tamb) / P['cold_w'])
    if P['cold_fmax'] > 0:
        a = 1.0 / P['cold_tau_h']; s = np.zeros(T)
        for t in range(T):
            s[t] = max(fc[t], (1 - a) * (s[t - 1] if t else 0.0))
        fc = s
    dT = P['T_rating'] - Tamb
    der = np.ones((T, cf.C))
    der[:, eng.is_ct] = np.clip(1 + P['ct_derate'] * dT, 0.8, 1.15)[:, None]
    der[:, eng.is_cc] = np.clip(1 + P['cc_derate'] * dT, 0.85, 1.10)[:, None]
    der *= (1 - fc)[:, None]
    cap_units = navail * der
    fprice = eng.fuel(times, scen.get('gas_scale', 1.0))
    cost, wid = eng.blocks(fprice, eng.coal_adder, scen.get('carbon', 0.0))
    cost = cost * (1 + P['markup'])
    Ps = scen.get('storage_MW', 0.0); Es = Ps * scen.get('storage_h', 2.0)
    L = d - wind - solar - fixed
    Ln, sthead = storage_flatten(L, day_id, Ps, Es)
    lam, gen, total, low = stack_price(eng, Ln, cap_units, cost, wid, wind + solar, P['ren_offer'], P['floor'],
                                       P['floor_reserve'], day_id, P['floor_share'])
    R = total - gen + sthead
    lag = 12 + loc.hour.values
    idx = np.clip(np.arange(T) - lag, 0, T - 1)
    cap_da = navail[idx] * der
    Lf = df - wind_f - solar_f - fixed
    Lfn, sth_f = storage_flatten(Lf, day_id, Ps, Es)
    lam_f, gen_f, total_f, _ = stack_price(eng, Lfn, cap_da, cost, wid, wind_f + solar_f, P['ren_offer'],
                                           P['floor'], P['floor_reserve'], day_id, P['floor_share'])
    Rf = total_f - gen_f + sth_f
    ramp = np.abs(np.diff(np.r_[Ln[0], Ln])) / 1000.0
    return dict(times=times, ba=eng.ba, lam=lam, R=R, lam_da=lam_f, R_da=Rf, ramp=ramp, d=d, df=df, T=Tamb,
                day=day_id,
                hour=loc.hour.values, dow=loc.dayofweek.values, month=loc.month.values, doy=loc.dayofyear.values,
                cold=fc, wind=wind, solar=solar)


def prices(F: dict, P: dict, scen: dict = None, seed=0):
    """Cheap part: ORDC scarcity adder, day-ahead expectation and real-time noise -> DataFrame (rt, da, ...)."""
    P = {**DEFAULTS, **P, **{k: v for k, v in (scen or {}).items() if k in DEFAULTS}}
    rng = np.random.default_rng(seed + 7919)
    lam, R, lam_f, Rf = F['lam'], F['R'], F['lam_da'], F['R_da']
    T = len(lam)
    adder = np.maximum(P['voll'] - lam, 0) * ordc_lolp(R, P['X'], P['mu'], P['sigma'])
    rt = lam + adder
    if P['rt_noise_sd'] > 0:
        rho = P['rt_noise_rho']; e = np.zeros(T)
        z = rng.standard_t(3, T) * P['rt_noise_sd'] * np.sqrt(1 - rho ** 2) / np.sqrt(3.0)
        z = z * (1 + P['ramp_k'] * F['ramp'])
        for t in range(1, T):
            e[t] = rho * e[t - 1] + z[t]
        rt = rt + e * np.maximum(np.abs(lam), 5.0)
    rt = np.clip(rt, P['floor'], P['cap'])
    # DA: expected adder over the empirical distribution of (R_rt - R_da) at the same local hour (self-consistent
    # forecast uncertainty; risk-neutral DA = E[RT]); da_sd_scale shrinks/widens it
    err = (R - Rf) * P['da_sd_scale']
    hrs = F['hour']
    ea = np.zeros(T)
    nk = 40
    for h in range(24):
        ih = np.where(hrs == h)[0]
        if len(ih) == 0:
            continue
        # condition on the forecast reserve level (5 bins per hour): errors are larger/positive when R_da is low
        q = np.quantile(Rf[ih], [0.2, 0.4, 0.6, 0.8])
        bins = np.searchsorted(q, Rf[ih])
        for b in range(5):
            jb = ih[bins == b]
            if len(jb) == 0:
                continue
            pool = err[jb]
            smp = pool[rng.integers(0, len(pool), nk)]
            Rx = Rf[jb][:, None] + smp[None, :]
            ea[jb] = (np.maximum(P['voll'] - lam_f[jb], 0)[:, None] *
                      ordc_lolp(Rx, P['X'], P['mu'], P['sigma'])).mean(1)
    if market.enabled():
        # GridSim physics 1.2: unforecast RT scarcity (gridsim.market, fitted on ERCOT 2018/2020/2022 prices; zeros
        # for CISO), driven by the real day-ahead demand-forecast miss of the input year; the DA prices in only its
        # hour/month expectation.  Its own random stream: every other draw is as in 1.1.
        # The default fit ('ERCO') also reads the DA price's own shape (rank within the day, level against the past
        # week; before this premium); 'ERCO_first_fit' (frozen set v5) did not.
        ba = F.get('ba', 'ERCO')
        derr = (F['d'] - F['df']) / np.maximum(F['df'], 1.0)
        pz = market.params(P['scar_fit']) if ba == 'ERCO' else None
        da0 = np.clip(lam_f + ea, P['floor'], P['cap'])
        sc = market.rt_scarcity(F['times'], ba, seed=seed, demand_err=derr, scale=P['scar_scale'], params_=pz,
                                da=da0)
        rt = np.clip(rt + sc, P['floor'], P['cap'])
        ea = ea + market.expected(F['times'], ba, scale=P['scar_scale'], params_=pz, da=da0)
    da = np.clip(lam_f + ea, P['floor'], P['cap'])
    out = pd.DataFrame(dict(rt=rt, da=da, lam=lam, adder=adder, R=R, R_da=Rf, d=F['d'], df=F['df'], T=F['T'],
                            day=F['day'], hour=F['hour'], dow=F['dow'], month=F['month'], doy=F['doy']),
                       index=F['times'])
    return out


def simulate(eng, inp, scen, P, seed=0):
    return prices(fundamentals(eng, inp, scen, P, seed), P, scen, seed)
