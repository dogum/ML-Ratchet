"""HybridPilot v2.1: a learned, causal hybrid energy-management policy (numpy only), built on RoadSim physics v3.1.

Each second of traction the controller
  1. builds 66 causal features (vehicle parameters, battery/engine state, recent history of the speed trace,
     optionally the next 1-60 s of speed and grade from navigation, optionally the remaining trip distance),
  2. runs a 2x96 tanh MLP -> an equivalence factor s (fuel joules per battery joule) and P(engine on),
  3. buses and mild P1 hybrids: s is replaced by a constant prior (series bus 2.8, P2 bus 2.6, mild 2.8; v2.1),
     then SOC feedback  s <- s + Kp dz(SOC_ref - SOC) + Ki * integral, dz = the error beyond a +-0.03 dead band (v2.1;
     plug-ins with a known trip length and dtg: SOC_ref falls linearly with distance to the charge-sustaining level),
  4. picks, among the candidate controls the powertrain can deliver, the one minimising
         P_fuel + s * P_battery,chemical  (+ engine start fuel, + start_pen, when starting from off),
     with the platform's minimum engine on/off times (RoadSim v2+: 8 s / 3 s) and its forced-on rules (catalyst
     light-off, warm-up, cabin heat, production speed rules; plug-ins in charge depletion: RoadSim's CD rules).

Use it two ways:
  - with your own powertrain model: `HybridPilotCore.decide(...)` takes candidate arrays (fuel power, battery
    chemical power, engine-on flag, feasibility) and returns the chosen index;
  - inside RoadSim (Phase 1 simulator): `make_roadsim_strategy()` returns a strategy object for roadsim.simulate.
The feature code is vendored here (no roadsim needed for features or the network).
"""
import math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, 'weights', 'hybridpilot_v2.npz')
G = 9.80665
BINS = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 60)]
N_FEAT = 66
# min_on_s / min_off_s None = the simulator's own minimum on / off times (RoadSim v2: 8 s / 3 s)
# v2.1 (15 Nov 2026, RoadSim physics v3.1): proportional SOC feedback with a +-0.03 dead band, and a constant equivalence
# factor in place of the network's s for buses and mild P1 hybrids (fitted on training-distribution simulations).
DEFAULTS = dict(onoff='ecms', Kp=12.0, Ki=0.02, preview_s=0, nav=False, min_on_s=None, min_off_s=None, s_scale=1.0, s_bias=0.0,
                start_pen=0.0, dtg=False, Kp_dtg=6.0, fb_norm=False, fb_tau=0.0, fb_dead=0.03, fb_dead_mode='soft', cubic=0.0,
                s_prior={'bus_hev_series': (1.0, 2.8), 'bus_hev_parallel_p2': (1.0, 2.6), 'mild': (1.0, 2.8)})
V20 = dict(fb_dead=0.0, s_prior=None)       # HybridPilotCore(**V20) / make_roadsim_strategy(**V20): the v2.0 decision rule


class MLP:
    """x -> (x - mu)/sd -> [Linear, tanh] x 2 -> Linear(2) = (log(s + 0.05), logit P(on))."""

    def __init__(self, path=WEIGHTS):
        z = np.load(path)
        self.p = {k: np.asarray(z[k], np.float64) for k in z.files}
        self.nl = len([k for k in self.p if k.startswith('W')])

    def __call__(self, x):
        h = (np.asarray(x, np.float64) - self.p['mu']) / self.p['sd']
        for i in range(self.nl):
            h = h @ self.p[f'W{i}'] + self.p[f'b{i}']
            if i < self.nl - 1:
                h = np.tanh(h)
        return h

    def s_pon(self, x):
        y = self(x)
        s = math.exp(min(max(y[0], -5.0), 3.0)) - 0.05
        return s, 1.0 / (1.0 + math.exp(-min(max(y[1], -30.0), 30.0)))


# ------------------------------------------------------------------------------------------------ features
def _ema(x, tau):
    from scipy.signal import lfilter
    a = 1.0 / tau
    y, _ = lfilter([a], [1, -(1 - a)], x, zi=[(1 - a) * x[0]])
    return y


def vehicle_features(vp):
    """vp: dict(arch, hd, mass, P_eng_max [W], P_mot_max [W, traction motors], E_batt [J], CdA, Crr, soc_lo, soc_hi)."""
    a = vp['arch']; m = vp['mass']
    return np.array([a in ('hev_series', 'phev_series'), a in ('hev_powersplit', 'phev_powersplit'), a == 'hev_parallel_p2',
                     a.startswith('phev'), bool(vp['hd']), math.log(m / 1500.0), vp['P_eng_max'] / m / 100.0,
                     vp['P_mot_max'] / m / 100.0, math.log(vp['E_batt'] / 3.6e6), vp['CdA'] / m * 1e3, vp['Crr'] * 100.0,
                     vp['soc_hi'] - vp['soc_lo']], float)


def road_power(vp, trip):
    """wheel power demand over each second of the speed trace (same road-load model as RoadSim's DP requirements):
    m_eff a + 1/2 rho CdA (v + headwind)|v + headwind| + m g cos(th) Crr(T) (1 + Crr_v u) + parasitic + m g sin(th)."""
    v = np.asarray(trip['v'], float)
    gr = np.asarray(trip.get('grade', np.zeros_like(v)), float)
    gm = np.concatenate([0.5 * (gr[1:] + gr[:-1]), gr[-1:]]); th = np.arctan(gm)
    N = len(v)
    ws = np.asarray(trip.get('wind_speed', np.zeros(N)), float); wd = np.asarray(trip.get('wind_dir', np.zeros(N)), float)
    hd = np.asarray(trip.get('heading', np.zeros(N)), float)
    wh = ws * np.cos(np.radians(wd - hd))
    T = np.clip(np.asarray(trip.get('T_amb', np.full(N, 20.0)), float), -30, 45)
    Kt = vp.get('tire_K_t', 0.008)
    f_T = (1 + Kt * (20 - 25)) / np.maximum(1 + Kt * (T - 25), 0.4)
    crr = vp['Crr'] * f_T
    rho = np.asarray(trip['rho_air'], float) if 'rho_air' in trip else 101325.0 / (287.05 * (T + 273.15))
    u = 0.5 * (v[:-1] + v[1:]); rv = u + wh[:-1]
    F = (vp['m_eff'] * np.diff(v) + 0.5 * rho[:-1] * vp['CdA'] * rv * np.abs(rv)
         + vp['mass'] * G * np.cos(th[:-1]) * crr[:-1] * (1 + vp.get('Crr_v', 0.0) * u) + vp.get('parasitic_N', 0.0)
         + vp['mass'] * G * np.sin(th[:-1]))
    stopped = (v[:-1] < 1e-3) & (v[1:] < 1e-3)
    F = np.where(stopped, 0.0, F)
    return np.concatenate([F * u, [0.0]])


class TripContext:
    """per-trip arrays: causal history (EMAs over the past) and preview windows over the future speed trace.
    Only the first `preview_s` seconds ahead are ever read (see `preview`)."""

    def __init__(self, vp, trip):
        v = np.asarray(trip['v'], float); N = len(v)
        P = road_power(vp, trip); Pe = vp['P_eng_max']
        Ppos = np.maximum(P, 0) / Pe; Pneg = np.minimum(P, 0) / Pe; stop = (v < 0.5).astype(float)
        self.N = N; self.v = v; self.a = np.concatenate([np.diff(v), [0.0]])
        self.grade = np.asarray(trip.get('grade', np.zeros(N)), float)
        self.T_amb = np.asarray(trip.get('T_amb', np.full(N, 20.0)), float)
        self.ema30_v = _ema(v, 30.0) / 30.0; self.ema300_v = _ema(v, 300.0) / 30.0
        self.ema60_P = _ema(Ppos, 60.0); self.ema300_P = _ema(Ppos, 300.0); self.ema120_stop = _ema(stop, 120.0)
        d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]))]); self.D = d[-1]
        pad = 61
        cs = lambda x, f: np.concatenate([[0.0], np.cumsum(np.concatenate([x, np.full(pad, f)]))])
        self.cs = dict(Ppos=cs(Ppos, 0.0), Pneg=cs(Pneg, 0.0), v=cs(v / 30.0, 0.0), stop=cs(stop, 1.0))

    def preview(self, k, L):
        out = np.zeros(25)
        for i, (a, b) in enumerate(BINS):
            if L >= b:
                n = b - a + 1
                for j, q in enumerate(('Ppos', 'Pneg', 'v', 'stop')):
                    c = self.cs[q]; out[4 * i + j] = (c[k + b + 1] - c[k + a]) / n
                out[20 + i] = 1.0
        return out


def features(vf, tc, vp, st, ctx, preview_s=0, nav=False, soc_ref=None, cat_pending=0.0):
    """st: soc, eng_on, t_on, t_off, T_eng, T_bat;  ctx: k, u, dist, P_dem_mech, p_dis, p_chg, P_aux_el, need_heat, T_in_frac."""
    lo, hi = vp['soc_lo'], vp['soc_hi']; mid = 0.5 * (lo + hi); half = 0.5 * (hi - lo); Pe = vp['P_eng_max']
    k = min(ctx['k'], tc.N - 1); soc = st['soc']
    ref = soc_ref if soc_ref is not None else ((lo + 0.05) if vp['arch'].startswith('phev') else mid)
    x = np.empty(N_FEAT)
    x[0:12] = vf
    x[12:26] = [(soc - mid) / half, (soc - lo) / (hi - lo), (soc - ref) * 10.0, float(st['eng_on']), min(st['t_on'], 60.0) / 60.0,
                min(st['t_off'], 60.0) / 60.0, st['T_eng'] / 100.0, float(tc.T_amb[k]) / 40.0, st['T_bat'] / 40.0,
                float(bool(ctx.get('need_heat'))), float(cat_pending), ctx['p_dis'] / Pe, ctx['p_chg'] / Pe, ctx['P_aux_el'] / Pe]
    x[26:31] = [ctx['u'] / 30.0, tc.a[k], tc.grade[k] * 10.0, ctx['P_dem_mech'] / Pe, ctx.get('T_in_frac', 0.0)]
    x[31:37] = [tc.ema30_v[k], tc.ema300_v[k], tc.ema60_P[k], tc.ema300_P[k], tc.ema120_stop[k], math.log1p(ctx['dist'] / 1e3)]
    x[37:62] = tc.preview(k, preview_s)
    E_av = max(soc - ref, 0.0) * vp['E_batt'] / 3.6e6
    if nav:
        rem = max(tc.D - ctx['dist'], 0.0) / 1e3; e_km = 0.12 * vp['mass'] / 1500.0
        x[62:66] = [1.0, math.log1p(rem), min(E_av / (rem * e_km + 0.05), 5.0), E_av / (vp['mass'] / 1000.0)]
    else:
        x[62:66] = [0.0, 0.0, 0.0, E_av / (vp['mass'] / 1000.0)]
    return x


# ------------------------------------------------------------------------------------------------ decision
class HybridPilotCore:
    """model-agnostic decision: give it candidate controls scored by YOUR powertrain model."""

    def __init__(self, net=None, **params):
        self.net = net or MLP(); self.p = dict(DEFAULTS); self.p.update(params)
        self.I = 0.0          # integral of (SOC_ref - SOC) over the trip, s (|I| <= 30, as RoadSim's A-ECMS)
        self.ef = None        # low-passed SOC error (fb_tau)

    def equivalence(self, x, soc, soc_ref, phev, plan_ref=None, window=None, prior_key=None, dt=1.0):
        """plan_ref: the distance-to-go SOC reference (plug-ins with dtg and a known trip length); feedback both ways.
        window: (soc_lo, soc_hi), needed by fb_norm and cubic. prior_key: 'mild' | 'bus_<arch>' | arch, for s_prior."""
        p = self.p
        s, p_on = self.net.s_pon(x)
        s = max(s * p['s_scale'] + p['s_bias'], 0.0)
        if p['s_prior'] and prior_key in p['s_prior']:
            w, s0 = p['s_prior'][prior_key]
            s = (1.0 - w) * s + w * s0
        if p['cubic'] and not phev:
            mid, half = 0.5 * (window[0] + window[1]), 0.5 * (window[1] - window[0])
            z = (soc - mid) / half
            s = s * (1.0 - p['cubic'] * (z ** 3 if abs(z) < 1.5 else math.copysign(3.375, z)))
        if plan_ref is not None:
            return max(s + p['Kp_dtg'] * (plan_ref - soc), 0.0), p_on
        e = soc_ref - soc
        if (p['Kp'] or p['Ki']) and (not phev or e > -0.03):
            if p['fb_norm']:
                e = e / (0.5 * (window[1] - window[0]))
            if p['Ki']:
                self.I = min(max(self.I + e * dt, -30.0), 30.0)
            ep = e
            if p['fb_tau']:
                self.ef = e if self.ef is None else self.ef + min(dt / p['fb_tau'], 1.0) * (e - self.ef)
                ep = self.ef
            if p['fb_dead']:         # 'soft': the band is subtracted; 'hard': full Kp e outside the band, 0 inside
                ep = (ep if abs(ep) > p['fb_dead'] else 0.0) if p['fb_dead_mode'] == 'hard' else                     math.copysign(max(abs(ep) - p['fb_dead'], 0.0), ep)
            s = max(s + p['Kp'] * ep + p['Ki'] * self.I, 0.0)
        return s, p_on

    def decide(self, x, soc, soc_ref, phev, P_fuel, P_chem, on, ok, eng_on, t_on, t_off, start_W, forced_on=False,
               min_on_s=8.0, min_off_s=3.0, plan_ref=None, window=None, prior_key=None):
        """P_fuel, P_chem [W], on, ok: arrays over candidate controls. Returns (index or -1, s, p_on)."""
        s, p_on = self.equivalence(x, soc, soc_ref, phev, plan_ref, window, prior_key)
        if self.p['min_on_s'] is not None: min_on_s = self.p['min_on_s']
        if self.p['min_off_s'] is not None: min_off_s = self.p['min_off_s']
        on = np.asarray(on, bool); ok = np.asarray(ok, bool).copy(); sel = ok.copy()
        if forced_on and (ok & on).any(): sel &= on
        if eng_on and t_on < min_on_s and (ok & on).any(): sel &= on
        if (not eng_on) and t_off < min_off_s and (ok & ~on).any(): sel &= ~on
        if not sel.any(): sel = ok
        if not sel.any(): return -1, s, p_on
        J = np.asarray(P_fuel, float) + s * np.asarray(P_chem, float)
        if not eng_on: J = J + np.where(on, start_W + self.p['start_pen'], 0.0)
        return int(np.argmin(np.where(sel, J, np.inf))), s, p_on


def vehicle_params_from_roadsim(veh):
    b = veh.battery
    return dict(arch=veh.arch, hd=bool(veh.hd), mass=veh.mass, m_eff=veh.m_eff, CdA=veh.CdA, Crr=veh.Crr, Crr_v=veh.Crr_v,
                parasitic_N=veh.parasitic_N, tire_K_t=veh.tire_K_t, P_eng_max=veh.engine.P_max,
                P_mot_max=sum(m.P_max for r, m in veh.motors.items() if r in ('traction', 'MG2', 'P2')),
                E_batt=b.E_nom, soc_lo=b.soc_lo, soc_hi=b.soc_hi)


def make_roadsim_strategy(**params):
    """a strategy object for roadsim.simulate(spec, trip, strategy=make_roadsim_strategy())."""
    from roadsim.strategies import HybridStrategy, PHEVCDCS

    class HybridPilotRoadSim(HybridStrategy):
        name = 'hybridpilot'

        def setup(self, veh, pt, trip, st):
            super().setup(veh, pt, trip, st)
            self.core = HybridPilotCore(**params)
            if self.core.p['min_on_s'] is not None: self.min_on_s = self.core.p['min_on_s']
            if self.core.p['min_off_s'] is not None: self.min_off_s = self.core.p['min_off_s']
            if ((getattr(veh, 'control', None) or {}).get('params') or {}).get('engine_on_moving'):
                self.p['engine_on_moving'] = True       # mild (P0/P1) twin: no electric-only driving
            self.vp = vehicle_params_from_roadsim(veh); self.vf = vehicle_features(self.vp)
            self.tc = TripContext(self.vp, trip)
            self.phev = veh.arch.startswith('phev')
            self.soc_cs = veh.battery.soc_lo + 0.05
            self.ref = self.soc_cs if self.phev else self.soc_mid
            self.cd = self.phev and st['soc'] > self.soc_cs + 0.01
            self.soc0 = st['soc']
            self.dtg = bool(self.core.p['dtg']) and self.phev and bool(self.core.p['nav'])
            self.trace = []
            self.window = (veh.battery.soc_lo, veh.battery.soc_hi)
            self.prior_key = ('mild' if self.p.get('engine_on_moving') else
                              ('bus_' + veh.arch if veh.cls == 'bus' else veh.arch))

        def forced_on(self, st, ctx):
            """platform rules; plug-ins in charge depletion follow RoadSim's charge-depleting rules (as phev_cdcs)."""
            if self.phev and self.o.get('phev_cd_rules'):
                if self.cd and st['soc'] <= self.soc_cs:
                    self.cd = False
                if self.cd:
                    return PHEVCDCS.cd_engine_wanted(self, st, ctx) or st['soc'] < self.soc_lo - 0.02
            return HybridStrategy.forced_on(self, st, ctx)

        def plan_ref(self, ctx):
            if not self.dtg:
                return None
            f = min(ctx['dist'] / max(self.tc.D, 1.0), 1.0)
            return max(self.soc_cs, self.soc0 - (self.soc0 - self.soc_cs) * f)

        def grid(self, ctx):
            if hasattr(self.pt, 'default_fracs'):
                return self.pt.default_fracs(20)
            return self.dd_augment(self.pt.ctrl_grid(20), ctx)

        def candidates(self, st, ctx):
            return self.grid(ctx)

        def select(self, C, ctrl, st, ctx):
            cat = float(st['T_eng'] < 35.0 and self.t_cat < self.cat_warm_s)
            x = features(self.vf, self.tc, self.vp, st, ctx, self.core.p['preview_s'], self.core.p['nav'], self.ref, cat)
            on = C['on'] & (C['P_fuel'] > 0)
            j, s, p_on = self.core.decide(x, st['soc'], self.ref, self.phev, C['P_fuel'], self.chem_power(C, st), on, C['ok'],
                                          st['eng_on'], st['t_on'], st['t_off'], self.start_W, self.forced_on(st, ctx),
                                          self.min_on_s, self.min_off_s, self.plan_ref(ctx), self.window, self.prior_key)
            self.trace.append((ctx['k'], s, p_on))
            return j

    return HybridPilotRoadSim()
