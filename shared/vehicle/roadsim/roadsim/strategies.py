"""Energy-management and driving strategies.

A hybrid strategy sees, each second: the wheel torque/speed the driver asks for, the battery/engine state and the
trip so far, and picks one of the candidate controls the architecture can score (engine power, or engine torque fraction
for P2). All strategies share the candidate evaluation in powertrain.py, so they differ only in *how they choose*.

Rule-based:      series_thermostat, series_power_follower, parallel_rule (electric assist / load levelling),
                 powersplit_rule (Prius-style engine on/off with an optimal operating line)
Optimisation:    ecms (constant equivalence factor s), aecms (SOC-feedback s), dp (offline, see dp.py),
                 mpc (receding-horizon DP on a short preview, see dp.py)
PHEV:            phev_cdcs (charge-depleting then charge-sustaining), phev_blended (SOC reference falls with distance)
Conventional:    conv_rule (load/speed shift schedule), conv_opt (BSFC-optimal gear), conv_stopstart
BEV:             bev_regen_series (max regen), bev_regen_parallel (fixed regen share), bev_one_pedal
"""
import math
import numpy as np

STRATEGIES = {
    'conv': ['conv_rule', 'conv_opt', 'conv_stopstart'],
    'hev_series': ['series_thermostat', 'series_power_follower', 'ecms', 'aecms', 'dp', 'mpc'],
    'phev_series': ['phev_cdcs', 'phev_blended', 'series_thermostat', 'series_power_follower', 'ecms', 'aecms', 'dp', 'mpc'],
    'hev_powersplit': ['powersplit_rule', 'ecms', 'aecms', 'dp', 'mpc'],
    'phev_powersplit': ['phev_cdcs', 'phev_blended', 'powersplit_rule', 'ecms', 'aecms', 'dp', 'mpc'],
    'hev_parallel_p2': ['parallel_rule', 'ecms', 'aecms', 'dp', 'mpc'],
    'bev': ['bev_regen_series', 'bev_regen_parallel', 'bev_one_pedal'],
}
DEFAULT = {'conv': 'conv_rule', 'hev_series': 'aecms', 'phev_series': 'phev_cdcs',
           'hev_powersplit': 'aecms', 'phev_powersplit': 'phev_cdcs', 'hev_parallel_p2': 'aecms',
           'bev': 'bev_regen_series'}


def list_strategies(architecture=None):
    if architecture is None:
        return dict(STRATEGIES)
    return list(STRATEGIES[architecture])


class Strategy:
    name = 'base'
    is_hybrid = False
    regen_blend = None      # override of the vehicle's brake blending

    def __init__(self, **params):
        self.p = params

    def setup(self, veh, pt, trip, st):
        self.veh = veh; self.pt = pt; self.trip = trip
        b = veh.battery
        if b is not None:
            self.soc_lo, self.soc_hi = b.soc_lo, b.soc_hi
            self.soc_mid = 0.5 * (b.soc_lo + b.soc_hi)

    def step_end(self, st, plan, ctx):
        pass


# ------------------------------------------------------------------ conventional / BEV (configuration-type strategies)
class ConvStrategy(Strategy):
    def __init__(self, name='conv_rule', **p):
        super().__init__(**p); self.name = name
        self.shift = 'opt' if name == 'conv_opt' else 'rule'
        self.idle_stop = name == 'conv_stopstart'


class BEVStrategy(Strategy):
    def __init__(self, name='bev_regen_series', **p):
        super().__init__(**p); self.name = name
        self.regen_blend = {'bev_regen_series': 'series', 'bev_regen_parallel': 'parallel', 'bev_one_pedal': 'one_pedal'}[name]


# ------------------------------------------------------------------ hybrid base
class HybridStrategy(Strategy):
    is_hybrid = True
    min_on_s = 4.0
    min_off_s = 2.0

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        e = veh.engine
        self.Pe_max = e.P_max
        self.start_W = e.fuel_W(e.w_idle, 0.0) * 1.5      # start cost ~1.5 s of idle fuel, spread over the start second
        self.soc_target = self.p.get('soc_target', self.soc_mid)
        from . import calib
        self.cat_warm_s = self.p.get('cat_warmup_s', calib.get('cat_warmup_s'))
        self.t_cat = 0.0

    def grid(self, ctx):
        if hasattr(self.pt, 'default_fracs'):
            return self.pt.default_fracs(self.p.get('n_ctrl', 12))
        return self.pt.ctrl_grid(self.p.get('n_ctrl', 14))

    def forced_on(self, st, ctx):
        """engine must run: catalyst warm-up after a cold start, cabin heat with a cold coolant, or SOC far below window."""
        if ctx.get('need_heat'):
            return True
        if st['T_eng'] < 35.0 and self.t_cat < self.cat_warm_s and ctx['moving']:
            return True
        if st['soc'] < self.soc_lo - 0.02:
            return True
        return False

    def chem_power(self, C, st):
        b = self.veh.battery
        I, V = b.current_vec(C['P_batt'], st['soc'], st['T_bat'], st['v_rc'])
        return b.ocv(st['soc']) * I

    def pick_closest(self, C, target_ctrl, ctrl, st, want_on=None):
        ok = C['ok'].copy()
        if want_on is not None and (ok & (C['on'] == want_on)).any():
            ok &= C['on'] == want_on
        if not ok.any():
            return -1
        d = np.abs(np.asarray(ctrl, float) - target_ctrl)
        d = np.where(ok, d, np.inf)
        return int(np.argmin(d))

    def step_end(self, st, plan, ctx):
        if plan.get('eng_on') and st['T_eng'] < 60.0:
            self.t_cat += ctx['dt']


class ECMS(HybridStrategy):
    """Equivalent consumption minimisation: J = P_fuel + s * P_batt,chemical  (Paganelli 2002; Sciarretta 2004)."""
    name = 'ecms'

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        self.s0 = self.p.get('s', 2.6)
        self.s = self.s0

    def s_value(self, st, ctx):
        # soft SOC-window penalty (Sciarretta 2004): s grows as SOC approaches the lower bound, shrinks near the upper
        x = (st['soc'] - self.soc_mid) / (0.5 * (self.soc_hi - self.soc_lo))
        return self.s0 * (1.0 - 0.6 * x ** 3) if abs(x) < 1.5 else self.s0 * (1.0 - 0.6 * math.copysign(3.375, x))

    def candidates(self, st, ctx):
        return self.grid(ctx)

    def select(self, C, ctrl, st, ctx):
        s = self.s_value(st, ctx)
        self.s = s
        J = C['P_fuel'] + s * self.chem_power(C, st)
        on = C['on']
        if not st['eng_on']:
            J = J + np.where(on, self.start_W, 0.0)
        ok = C['ok'].copy()
        if self.forced_on(st, ctx) and (ok & on).any():
            ok &= on
        if st['eng_on'] and st['t_on'] < self.min_on_s and (ok & on).any():
            ok &= on
        if (not st['eng_on']) and st['t_off'] < self.min_off_s and (ok & ~on).any():
            ok &= ~on
        if not ok.any():
            return -1
        return int(np.argmin(np.where(ok, J, np.inf)))


class AECMS(ECMS):
    """Adaptive ECMS: s = s0 + Kp (SOC_ref - SOC) + Ki * integral  (Onori, Serrao & Rizzoni 2010)."""
    name = 'aecms'

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        self.Kp = self.p.get('Kp', 12.0); self.Ki = self.p.get('Ki', 0.02); self.I = 0.0

    def soc_ref(self, st, ctx):
        return self.soc_target

    def s_value(self, st, ctx):
        e = self.soc_ref(st, ctx) - st['soc']
        self.I = min(max(self.I + e * ctx['dt'], -30.0), 30.0)
        return max(0.5, self.s0 + self.Kp * e + self.Ki * self.I)


class PHEVBlended(AECMS):
    """blended PHEV: SOC reference falls linearly with distance from the start SOC to the CS level at trip end."""
    name = 'phev_blended'

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        self.soc0 = st['soc']; self.D = max(trip['distance_m'], 1.0)
        self.soc_cs = self.p.get('soc_cs', self.soc_lo + 0.05)
        self.Kp = self.p.get('Kp', 6.0)

    def soc_ref(self, st, ctx):
        f = min(ctx['dist'] / self.D, 1.0)
        return self.soc0 - (self.soc0 - self.soc_cs) * f


class PHEVCDCS(AECMS):
    """charge-depleting (EV first; engine only when the battery/motor cannot meet demand) then charge-sustaining A-ECMS."""
    name = 'phev_cdcs'

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        self.soc_cs = self.p.get('soc_cs', self.soc_lo + 0.05)
        self.soc_target = self.soc_cs
        self.cd = st['soc'] > self.soc_cs + 0.01

    def select(self, C, ctrl, st, ctx):
        if self.cd and st['soc'] <= self.soc_cs:
            self.cd = False
        if self.cd:
            ok = C['ok'] & ~C['on']
            if ok.any() and not ctx.get('need_heat'):
                return int(np.argmax(ok))
            # blended assist: cheapest engine operating point that makes the demand feasible
            J = C['P_fuel'] + 0.3 * self.chem_power(C, st)
            return int(np.argmin(np.where(C['ok'], J, np.inf))) if C['ok'].any() else -1
        return AECMS.select(self, C, ctrl, st, ctx)


# ------------------------------------------------------------------ rule-based
class SeriesThermostat(HybridStrategy):
    """engine on at its most efficient point when SOC falls below soc_on, off above soc_off (thermostat / bang-bang)."""
    name = 'series_thermostat'

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        w = self.soc_hi - self.soc_lo
        self.soc_on = self.p.get('soc_on', self.soc_lo + 0.25 * w)
        self.soc_off = self.p.get('soc_off', self.soc_lo + 0.75 * w)
        self.on = False

    def candidates(self, st, ctx):
        if st['soc'] < self.soc_on: self.on = True
        elif st['soc'] > self.soc_off: self.on = False
        P = max(self.veh.engine.P_best, 0.0)
        need = ctx['P_dem_el'] - 0.9 * ctx['p_dis']      # demand beyond what the battery can deliver
        on = self.on or self.forced_on(st, ctx) or need > 0
        if on:
            P = min(max(P, need / 0.85), self.Pe_max * 0.98)
            return np.array([P])
        return np.array([0.0])

    def select(self, C, ctrl, st, ctx):
        return 0 if C['ok'][0] else -1


class SeriesPowerFollower(HybridStrategy):
    """engine power follows the electrical demand plus a SOC correction; on/off thresholds with hysteresis."""
    name = 'series_power_follower'

    def candidates(self, st, ctx):
        Pm = self.Pe_max
        corr = (self.soc_target - st['soc']) / max(self.soc_hi - self.soc_lo, 0.05) * 2.0 * 0.4 * Pm
        P = ctx['P_dem_el'] + corr
        on_th, off_th = self.p.get('P_on_frac', 0.15) * Pm, self.p.get('P_off_frac', 0.08) * Pm
        if st['eng_on']:
            on = P > off_th or st['soc'] < self.soc_lo or st['t_on'] < self.min_on_s
        else:
            on = P > on_th or st['soc'] < self.soc_lo
        on = on or self.forced_on(st, ctx)
        if not on:
            return np.array([0.0])
        return np.array([min(max(P, 0.12 * Pm), 0.98 * Pm)])

    def select(self, C, ctrl, st, ctx):
        return 0 if C['ok'][0] else -1


class PowerSplitRule(HybridStrategy):
    """Prius-style: EV at low power and speed; otherwise engine on the optimal operating line at
    P_e = P_demand + SOC-correction; hysteresis on the on/off power threshold."""
    name = 'powersplit_rule'

    def candidates(self, st, ctx):
        Pm = self.Pe_max
        P_chg = (self.soc_target - st['soc']) / 0.1 * 0.15 * Pm
        P = ctx['P_dem_mech'] + P_chg + ctx['P_aux_el']
        v_ev = self.p.get('v_ev_max', 19.0)
        on_th, off_th = self.p.get('P_on_frac', 0.15) * Pm, self.p.get('P_off_frac', 0.08) * Pm
        u = ctx['u']
        if st['eng_on']:
            on = P > off_th or u > v_ev - 2.0 or st['t_on'] < self.min_on_s
        else:
            on = P > on_th or u > v_ev
        on = (on or self.forced_on(st, ctx)) and ctx['P_dem_mech'] > -1.0
        if not on:
            return np.array([0.0])
        return np.array([min(max(P, 0.10 * Pm), 0.98 * Pm)])

    def select(self, C, ctrl, st, ctx):
        return 0 if C['ok'][0] else -1


class ParallelRule(HybridStrategy):
    """P2 electric assist / load levelling: EV at low power and speed; otherwise engine torque held in an efficient band
    (35-85 % of T_max) shifted by a SOC correction; the motor fills the gap (assist) or absorbs it (charging)."""
    name = 'parallel_rule'

    def candidates(self, st, ctx):
        Pm = self.Pe_max
        u = ctx['u']
        P = ctx['P_dem_mech']
        ev_P = self.p.get('P_ev_frac', 0.12) * Pm
        v_ev = self.p.get('v_ev_max', 14.0)
        # EV power threshold shrinks as SOC falls below target (charge sustaining)
        ev_P *= min(max((st['soc'] - self.soc_lo) / max(self.soc_target - self.soc_lo, 0.02), 0.0), 1.5)
        if st['eng_on']:
            ev = P < 0.7 * ev_P and u < v_ev - 1.0 and st['t_on'] >= self.min_on_s
        else:
            ev = P < ev_P and u < v_ev
        ev = ev and st['soc'] > self.soc_lo + 0.02 and not self.forced_on(st, ctx)
        if ev:
            return np.array([0.0])
        corr = (self.soc_target - st['soc']) / 0.1 * 0.15
        # desired torque fraction: demand / Tmax at shaft + SOC correction, kept inside [0.35, 0.85]
        fr = ctx.get('T_in_frac', 0.5) + corr
        return np.array([min(max(fr, 0.35), 0.85 if ctx.get('T_in_frac', 0) < 0.85 else 1.0), 1.0])

    def select(self, C, ctrl, st, ctx):
        if C['ok'][0]:
            return 0
        if len(C['ok']) > 1 and C['ok'][1]:
            return 1
        return -1


STRATEGY_CLASSES = {
    'ecms': ECMS, 'aecms': AECMS, 'phev_blended': PHEVBlended, 'phev_cdcs': PHEVCDCS,
    'series_thermostat': SeriesThermostat, 'series_power_follower': SeriesPowerFollower,
    'powersplit_rule': PowerSplitRule, 'parallel_rule': ParallelRule,
}


def make_strategy(arch, strategy=None):
    """strategy: None | name | {'strategy'|'name': name, 'params': {...}} | Strategy instance."""
    if isinstance(strategy, Strategy):
        return strategy
    params = {}
    if isinstance(strategy, dict):
        params = dict(strategy.get('params', {}) or {})
        strategy = strategy.get('strategy') or strategy.get('name')
    name = strategy or DEFAULT[arch]
    if name not in STRATEGIES[arch]:
        raise ValueError('strategy %r not available for %s; choose from %s' % (name, arch, STRATEGIES[arch]))
    if arch == 'conv':
        return ConvStrategy(name, **params)
    if arch == 'bev':
        return BEVStrategy(name, **params)
    if name in ('dp', 'mpc'):
        from .dp import DPStrategy, MPCStrategy
        return (DPStrategy if name == 'dp' else MPCStrategy)(**params)
    return STRATEGY_CLASSES[name](**params)
