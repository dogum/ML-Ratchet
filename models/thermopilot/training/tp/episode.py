"""Episodes: one building x one month of weather x one tariff, run hour by hour through bs.Simulator, and the rule
controllers (fixed, setback, pre-cool). Every strategy is scored by the same simulator with the same metrics."""
import copy, time
import numpy as np
import pandas as pd
import buildingsim as bs
from . import env
from .mpc import Model, MPC

LAM_PER_150M2 = 3.0      # comfort weight $ per K h outside the band, per 150 m2 of floor (min 1 $/K h)


def lam_for(spec):
    return max(1.0, float(spec['floor_area_m2']) / 150.0) * LAM_PER_150M2


def prep_spec(spec, pref):
    """Thermostat twin: occupant band as the schedule default, no natural ventilation, no PV/battery."""
    s = copy.deepcopy(spec)
    s['ventilation'] = dict(s.get('ventilation') or {}, natural_ventilation=False)
    s['pv'] = None; s['battery'] = None
    sp = dict(s.get('setpoints') or {})
    sp.update(heat=pref['lo'], cool=pref['hi'], setback_heat=None, setback_cool=None, schedule='constant', recovery_h=0.0)
    s['setpoints'] = sp
    return s


def _remaining_days(c):
    """Per step: days left in the local month (incl. today) that have this step's demand period / any flat demand."""
    n = len(c['price'])
    day = c['ym'] * 40 + (c['doy'] % 40)
    drem = np.ones(n); frem = np.ones(n)
    for ym in np.unique(c['ym']):
        sel = np.flatnonzero(c['ym'] == ym)
        dd = c['doy'][sel]
        for p in np.unique(c['dperiod'][sel]):
            if p < 0:
                continue
            days = np.unique(dd[c['dperiod'][sel] == p])
            for i in sel[c['dperiod'][sel] == p]:
                drem[i] = np.sum(days >= c['doy'][i])
        days = np.unique(dd)
        for i in sel:
            frem[i] = np.sum(days >= c['doy'][i])
    return drem, frem


class Episode:
    def __init__(self, spec_true, spec_nom, wsrc, year, month, tariff, pref, seed=0, rtp=None, months=1,
                 fc_scale=1.0, H=24, day=1, days=None):
        df, meta = env.weather(wsrc, year)
        self.tz = tz = meta['tz']
        start = pd.Timestamp(year=year, month=month, day=day, tz=tz)
        end = start + (pd.DateOffset(months=months) if days is None else pd.Timedelta(days=days))
        self.start, self.end = start.tz_convert('UTC'), end.tz_convert('UTC')
        self.pref, self.tariff, self.wsrc = pref, tariff, wsrc
        self.spec_true = prep_spec(spec_true, pref)
        self.spec_nom = prep_spec(spec_nom, pref)
        self.wx = (df, meta)
        self.sim = None
        self.msim = bs.Simulator(self.spec_nom, self.wx, self.start, self.end + pd.Timedelta(hours=H + 2), dt='1h',
                                 warmup_days=0, record_trend=False)
        self.model = Model(self.msim)
        if spec_true is spec_nom:
            self.mtrue = self.model
        else:
            self.tsim = bs.Simulator(self.spec_true, self.wx, self.start, self.end + pd.Timedelta(hours=H + 2), dt='1h',
                                     warmup_days=0, record_trend=False)
            self.mtrue = Model(self.tsim)
        times = self.msim.times
        self.n = int(round((self.end - self.start) / pd.Timedelta(hours=1)))
        pa = env.price_arrays(tariff, times, tz, rtp=rtp)
        lo, hi, state = env.band_schedule(self.msim, pref)
        cal = env.local_calendar(times, tz)
        self.ctx = dict(price=pa['price'], dperiod=pa['dperiod'], drate=pa['drate'], frate=pa['frate'], lo=lo, hi=hi,
                        state=state, pref_lo=pref['lo'], pref_hi=pref['hi'], **cal)
        self.pa = pa
        self.ctx['drem'], self.ctx['frem'] = _remaining_days(self.ctx)
        self.lam = lam_for(self.spec_nom)
        self.gas_price = env.GAS_RES if self.model.res else env.GAS_COM
        T, G = self.model.To, self.model.ghi
        self.fc_true = env.Forecaster(T, G, H, seed, perfect=True)
        self.fc_noisy = env.Forecaster(T, G, H, seed, perfect=False, scale=fc_scale)
        self.H = H
        self.peak_hours = self._peak_hours()
        self.fixed_charge = 0.0
        if isinstance(tariff, str) and not tariff.startswith('rtp'):
            t = env._tariff(tariff)
            fx = t['fixed']['first_meter'] or 0.0
            ndays = self.n / 24.0
            self.fixed_charge = float(fx * ndays if (t['fixed']['units'] or '').endswith('day') else fx * months)

    def _peak_hours(self):
        """Per local month: weekday hours whose price equals that month's weekday maximum (if TOU)."""
        c = self.ctx; out = {}
        for mo in np.unique(c['month']):
            sel = (c['month'] == mo) & c['wd']
            p = c['price'][sel]; h = c['hour'][sel]
            if len(p) == 0 or p.max() <= p.min() + 1e-6:
                out[mo] = set(range(15, 19)) if str(self.tariff).startswith('rtp') else set()
            else:
                ph = pd.Series(p).groupby(h).mean()
                out[mo] = set(ph[ph >= ph.max() - 1e-6].index)
        return out

    def new_sim(self):
        self.sim = bs.Simulator(self.spec_true, self.wx, self.start, self.end, dt='1h', record_trend=False)
        return self.sim

    def run(self, strategy, record=None):
        """strategy(k, obs, info_prev) -> action dict. Returns metrics dict (and the per-step log if asked)."""
        sim = self.new_sim()
        obs = sim.reset()
        info = None
        k = 0
        acts = []
        t0 = time.time()
        while obs is not None:
            a = strategy(k, obs, info)
            a = dict(heat_sp=float(a['heat_sp']), cool_sp=float(a['cool_sp']))
            if record is not None:
                record(k, obs, info, a)
            acts.append((a['heat_sp'], a['cool_sp']))
            obs, info = sim.step(a)
            k += 1
        if record is not None:
            record(k, None, info, None)
        wall = time.time() - t0
        out = sim.finish()
        m = self.metrics(out)
        m['wall_s'] = wall
        m['actions'] = np.array(acts, np.float32)
        return m, out

    def metrics(self, out):
        ts = out['ts']
        n = len(ts)
        c = self.ctx
        el = ts['meter_electricity_kW'].values
        gas = ts['meter_gas_kW'].values
        T = ts['T_zone'].values
        lo, hi, st = c['lo'][:n], c['hi'][:n], c['state'][:n]
        pa = {k: v[:n] for k, v in self.pa.items()}
        e_cost, d_cost = env.bill_arrays(pa, el, c['ym'][:n])
        g_cost = float(gas.sum()) * self.gas_price
        # deviations the HVAC cannot address do not count: overheating in a house without cooling (occupants
        # open windows; natural ventilation is off in the twins), cold in a house without heating
        under = np.maximum(lo - T, 0) * (self.mtrue.kind != 'none')
        over = np.maximum(T - hi, 0) * (self.mtrue.cool is not None)
        dev = under + over
        pmv = ts['PMV'].values
        home = st == 0
        hv = (ts['heating_electricity_kW'] + ts['cooling_electricity_kW'] + ts['fans_electricity_kW'] +
              ts['pumps_electricity_kW']).values
        pk = np.isin(c['hour'][:n], list(set().union(*self.peak_hours.values()))) & c['wd'][:n]
        bill = e_cost + d_cost + g_cost
        DH = float(dev.sum())
        mode = ts['hvac_mode'].values
        on = (np.abs(ts['Q_hvac_sens_kW'].values) > 0.05).astype(int)
        hsp, csp = ts['heat_sp'].values, ts['cool_sp'].values
        days = n / 24.0
        return dict(DH_under=float(under.sum()), DH_over=float(over.sum()), bill=bill, energy_cost=e_cost, demand_cost=d_cost, gas_cost=g_cost, kWh=float(el.sum()),
                    hvac_kWh=float(hv.sum()), gas_kWh=float(gas.sum()), peak_kW=float(el.max()),
                    peak_window_kW=float(el[pk].max()) if pk.any() else float('nan'),
                    DH=DH, DH_home=float(dev[home].sum()), pmv_h=float((np.abs(pmv[home]) > 0.5).sum()),
                    ppd_home=float(ts['PPD'].values[home].mean()) if home.any() else float('nan'),
                    J=bill + self.lam * DH, lam=self.lam, n=n, ledger_ok=bool(out['ledger']['ok']),
                    mode_switch_per_day=float(np.sum(mode[1:] != mode[:-1]) / days),
                    onoff_per_day=float(np.sum(np.abs(np.diff(on))) / days),
                    sp_changes_per_day=float(np.sum((np.abs(np.diff(hsp)) > 0.05) | (np.abs(np.diff(csp)) > 0.05)) / days),
                    fixed=self.fixed_charge)

    # ---------------------------------------------------------------- rule strategies
    def fixed(self):
        lo, hi = self.pref['lo'], self.pref['hi']
        return lambda k, obs, info: dict(heat_sp=lo, cool_sp=hi)

    def setback(self):
        c = self.ctx
        return lambda k, obs, info: dict(heat_sp=float(c['lo'][k]), cool_sp=float(c['hi'][k]))

    def setback_std(self):
        """Standard (ENERGY STAR-style) setback, whatever the occupant tolerates: heating -4 K asleep/away
        (unoccupied), cooling +2 K asleep, +4 K away. Comfort is still scored against the occupant's band."""
        c, lo, hi = self.ctx, self.pref['lo'], self.pref['hi']
        st = c['state']
        def f(k, obs, info):
            s_ = st[k]
            return dict(heat_sp=lo - (4.0 if s_ > 0 else 0.0), cool_sp=hi + (2.0 if s_ == 1 else 4.0 if s_ == 2 else 0.0))
        return f

    def precool_p1(self):
        """Phase 1 rule (BuildingData baseline): cool to hi - 2 K in the 2 h before the weekday peak window,
        hi + 2 K during it (outside the occupant band: costs comfort)."""
        lo, hi = self.pref['lo'], self.pref['hi']
        c, P = self.ctx, self.peak_hours

        def f(k, obs, info):
            ph = P.get(c['month'][k], set())
            cs = hi
            if ph and c['wd'][k]:
                h = c['hour'][k]; p0 = min(ph)
                if h in ph:
                    cs = hi + 2.0
                elif h in ((p0 - 2) % 24, (p0 - 1) % 24):
                    cs = hi - 2.0
            return dict(heat_sp=lo, cool_sp=cs)
        return f

    def precool(self):
        """Simple in-band pre-cool rule: on weekdays with a priced peak window, cool to max(lo + 0.5, hi - 2) in
        the 3 h before the window, then let the zone float up to the band edge during it. Otherwise follow the
        occupant band (like setback)."""
        c, P = self.ctx, self.peak_hours
        lo0, hi0 = self.pref['lo'], self.pref['hi']

        def f(k, obs, info):
            ph = P.get(c['month'][k], set())
            hs, cs = float(c['lo'][k]), float(c['hi'][k])
            if ph and c['wd'][k]:
                h = c['hour'][k]; p0 = min(ph)
                if h in ((p0 - 3) % 24, (p0 - 2) % 24, (p0 - 1) % 24):
                    cs = max(lo0 + 0.5, hi0 - 2.0)
                    hs = min(hs, cs - 0.5)
            return dict(heat_sp=hs, cool_sp=cs)
        return f

    def has_demand(self):
        return bool((self.ctx['drate'][:self.n] > 0).any() or (self.ctx['frate'][:self.n] > 0).any())

    def peaks(self, el):
        """Monthly peaks per demand period (and 'flat') of an hourly electricity series."""
        c = self.ctx; n = len(el); out = {}
        for i in range(n):
            if c['dperiod'][i] >= 0:
                k = (c['ym'][i], int(c['dperiod'][i])); out[k] = max(out.get(k, 0.0), float(el[i]))
            if c['frate'][i] > 0:
                k = (c['ym'][i], 'flat'); out[k] = max(out.get(k, 0.0), float(el[i]))
        return out

    def oracle_floor(self, fracs=(0.8, 0.9, 1.0)):
        """Perfect-foresight demand handling (uses exact re-simulation, i.e. rollback): run the oracle DP once,
        then re-run it with the monthly peak target set to f x that run's peaks (charged at the full demand rate
        above the target) and keep the target with the lowest J. Returns (peak_floor or None, J)."""
        m, out = self.run(self.mpc(True).act)
        self._oracle_best = m
        if not self.has_demand():
            return None, m['J']
        P1 = self.peaks(out['ts']['meter_electricity_kW'].values)
        best = (None, m['J'])
        for f in fracs:
            fl = {k: f * v for k, v in P1.items()}
            mf, _ = self.run(self.mpc(True, peak_floor=fl).act)
            if mf['J'] < best[1] - 1e-9:
                best = (fl, mf['J'])
                self._oracle_best = mf
        return best

    def mpc(self, oracle=False, peak_floor=None):
        """oracle: perfect-foresight bound (true house model, true weather, true state). Otherwise the classical
        MPC: nominal model, forecast with error, observer state estimate."""
        m = MPC(self.mtrue if oracle else self.model, self.ctx, self.fc_true if oracle else self.fc_noisy, self.lam,
                H=self.H, gas_price=self.gas_price, perfect_state=oracle, peak_floor=peak_floor)
        return m
