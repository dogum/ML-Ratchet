"""HybridPilot policy: causal features, the learned strategy (numpy MLP -> equivalence factor s and engine-on
probability), and the DP teacher used to label states (imitation + DAgger).

Decision each traction second (same candidate controls every RoadSim strategy scores):
    s, p_on = net(features)          features = vehicle parameters, state, recent history, optional preview, optional nav
    mode    = on/off with hysteresis on p_on (+ the forced-on rules every causal strategy obeys)
    action  = argmin over candidates in that mode of  P_fuel + s * P_batt,chemical  (+ start fuel when starting)
The DP teacher (optionally with the physics' minimum on/off times) gives, at any visited (time, SOC, engine state), the optimal action and the local costate
s* = -dJ/dSOC / (V_oc Q) of its cost-to-go table; the network regresses s* and classifies DP's on/off choice.
"""
import _paths as _P  # portable paths (see _paths.py)
import sys, math
sys.path.insert(0, _P.VEHICLE + '/roadsim' + '')
import numpy as np
from roadsim.strategies import HybridStrategy, PHEVCDCS
from roadsim.dp import _stage_table, _interp_rows, _interp_all, requirements
from roadsim.constants import G
from roadsim.options import resolve as resolve_options
from roadsim.powertrain import ARCH_CLASSES
from roadsim import aging as _aging
from hp_env import PHYS

BINS = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 60)]
N_FEAT = 66
FEAT_NAMES = (['arch_series', 'arch_split', 'arch_p2', 'phev', 'hd', 'log_mass', 'Pe_per_kg', 'Pm_per_kg', 'log_EkWh', 'CdA_per_t',
               'Crr', 'soc_width'] +
              ['x_soc', 'soc_frac', 'soc_minus_ref', 'eng_on', 't_on', 't_off', 'T_eng', 'T_amb', 'T_bat', 'need_heat', 'cat_pending',
               'p_dis', 'p_chg', 'P_aux'] +
              ['u', 'a', 'grade', 'P_dem', 'T_in_frac'] +
              ['ema30_v', 'ema300_v', 'ema60_P', 'ema300_P', 'ema120_stop', 'log_dist'] +
              [f'prev{i}_{q}' for i in range(5) for q in ('Ppos', 'Pneg', 'v', 'stop')] + [f'prev{i}_mask' for i in range(5)] +
              ['nav_mask', 'nav_log_rem', 'nav_E_ratio', 'E_avail_per_t'])
assert len(FEAT_NAMES) == N_FEAT


def _ema(x, tau):
    a = 1.0 / tau
    y = np.empty_like(x); acc = x[0]
    for i in range(len(x)):
        acc += a * (x[i] - acc); y[i] = acc
    return y


def _ema_fast(x, tau):
    # exact recursive EMA via scipy's lfilter (fast); falls back to python loop
    try:
        from scipy.signal import lfilter
        a = 1.0 / tau
        y, _ = lfilter([a], [1, -(1 - a)], x, zi=[(1 - a) * x[0]])
        return y
    except Exception:
        return _ema(x, tau)


def vehicle_features(veh):
    arch = veh.arch; b = veh.battery; e = veh.engine
    Pm = sum(m.P_max for r, m in veh.motors.items() if r in ('traction', 'MG2', 'P2'))
    m = veh.mass
    return np.array([arch in ('hev_series', 'phev_series'), arch in ('hev_powersplit', 'phev_powersplit'), arch == 'hev_parallel_p2',
                     arch.startswith('phev'), veh.hd, math.log(m / 1500.0), e.P_max / m / 100.0, Pm / m / 100.0,
                     math.log(b.E_nom / 3.6e6), veh.CdA / m * 1e3, veh.Crr * 100.0, b.soc_hi - b.soc_lo], float)


class TripContext:
    """per-trip arrays (computed once at setup): causal history and preview windows of the speed trace's power demand."""

    def __init__(self, veh, trip):
        v = np.asarray(trip['v'], float); N = len(v)
        F, u, stopped = requirements(veh, trip, tire_warm=1.0)
        P = np.concatenate([F * u, [0.0]])                      # demand over second k -> k+1, stored at k
        Pe = veh.engine.P_max
        Ppos = np.maximum(P, 0) / Pe; Pneg = np.minimum(P, 0) / Pe
        stop = (v < 0.5).astype(float)
        self.N = N
        self.v = v
        self.a = np.concatenate([np.diff(v), [0.0]])
        self.grade = np.asarray(trip['grade'], float)
        self.ema30_v = _ema_fast(v, 30.0) / 30.0
        self.ema300_v = _ema_fast(v, 300.0) / 30.0
        self.ema60_P = _ema_fast(Ppos, 60.0)
        self.ema300_P = _ema_fast(Ppos, 300.0)
        self.ema120_stop = _ema_fast(stop, 120.0)
        d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]))])
        self.dist = d
        self.D = d[-1]
        # cumulative sums for window means (padded with a parked car after the trip end)
        pad = 61
        def cs(x, fill):
            return np.concatenate([[0.0], np.cumsum(np.concatenate([x, np.full(pad, fill)]))])
        self.cs = dict(Ppos=cs(Ppos, 0.0), Pneg=cs(Pneg, 0.0), v=cs(v / 30.0, 0.0), stop=cs(stop, 1.0))

    def preview(self, k, L):
        out = np.zeros(25)
        for i, (a, b) in enumerate(BINS):
            if L >= b:
                n = b - a + 1
                for j, q in enumerate(('Ppos', 'Pneg', 'v', 'stop')):
                    c = self.cs[q]
                    out[4 * i + j] = (c[k + b + 1] - c[k + a]) / n
                out[20 + i] = 1.0
        return out


def make_features(vf, tc, veh, st, ctx, strat):
    """causal feature vector at decision time (uses: vehicle, state, the past trace, L s of preview, nav if allowed)."""
    b = veh.battery; Pe = veh.engine.P_max
    k = min(ctx['k'], tc.N - 1)
    lo, hi = b.soc_lo, b.soc_hi; mid = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
    soc = st['soc']
    ref = strat.soc_ref_cs
    cat = float(st['T_eng'] < 35.0 and strat.t_cat < strat.cat_warm_s)
    x = np.empty(N_FEAT)
    x[0:12] = vf
    x[12:26] = [(soc - mid) / half, (soc - lo) / (hi - lo), (soc - ref) * 10.0, float(st['eng_on']), min(st['t_on'], 60.0) / 60.0,
                min(st['t_off'], 60.0) / 60.0, st['T_eng'] / 100.0, tc_T(strat, k) / 40.0, st['T_bat'] / 40.0,
                float(bool(ctx.get('need_heat'))), cat, ctx['p_dis'] / Pe, ctx['p_chg'] / Pe, ctx['P_aux_el'] / Pe]
    x[26:31] = [ctx['u'] / 30.0, tc.a[k], tc.grade[k] * 10.0, ctx['P_dem_mech'] / Pe, ctx.get('T_in_frac', 0.0)]
    x[31:37] = [tc.ema30_v[k], tc.ema300_v[k], tc.ema60_P[k], tc.ema300_P[k], tc.ema120_stop[k], math.log1p(ctx['dist'] / 1e3)]
    x[37:62] = tc.preview(k, strat.preview_s)
    E_av = max(soc - ref, 0.0) * b.E_nom / 3.6e6                # kWh above the charge-sustaining level
    if strat.nav:
        rem = max(tc.D - ctx['dist'], 0.0) / 1e3
        e_km = 0.12 * veh.mass / 1500.0                           # rough kWh/km scale
        x[62:66] = [1.0, math.log1p(rem), min(E_av / (rem * e_km + 0.05), 5.0), E_av / (veh.mass / 1000.0)]
    else:
        x[62:66] = [0.0, 0.0, 0.0, E_av / (veh.mass / 1000.0)]
    return x


def tc_T(strat, k):
    return float(strat.trip['T_amb'][k])


class MLP:
    """numpy MLP: x -> standardise -> [Linear, tanh] x L -> Linear(2): (y_s, on_logit). s = exp(y_s) - 0.05."""

    def __init__(self, params):
        self.p = {k: np.asarray(v, np.float64) for k, v in params.items()}
        self.nl = len([k for k in self.p if k.startswith('W')])

    def __call__(self, x):
        h = (x - self.p['mu']) / self.p['sd']
        for i in range(self.nl):
            h = h @ self.p[f'W{i}'] + self.p[f'b{i}']
            if i < self.nl - 1:
                h = np.tanh(h)
        return h

    @staticmethod
    def load(path):
        z = np.load(path)
        return MLP({k: z[k] for k in z.files})


def costate_label(sol, k1, soc, branch, b, T_bat):
    """s* = -dJ_{k+1}/dSOC / (V_oc * Q): fuel-energy value of one joule of battery chemical energy at this state."""
    s = sol['s']; ds = s[1] - s[0]
    J = sol['J'][k1, branch]
    h = 2.0 * ds
    lo, hi = s[0] + h, s[-1] - h
    x = min(max(soc, lo), hi)
    g = (_interp_rows(J, s, np.array([x + h]), 50.0 * b.E_nom)[0] - _interp_rows(J, s, np.array([x - h]), 50.0 * b.E_nom)[0]) / (2 * h)
    return -g / (b.ocv(soc) * b.Q)


def dp_backward(veh, stage, soc_final, soc_bounds=None, n_soc=121, terminal_weight=12.0, dt=1.0, start_J=None, s_end=None):
    """roadsim.dp.dp_solve's Bellman recursion on a precomputed stage table (so several terminal SOCs share one table).
    No minimum on/off times (the v1 teacher)."""
    bat = veh.battery
    grid, PF, PB, OK, ON = stage[:5]
    if start_J is None:
        e = veh.engine
        start_J = e.fuel_W(e.w_idle, 0.0) * 1.5 * dt
    lo, hi = soc_bounds or (bat.soc_lo, bat.soc_hi)
    s = np.linspace(lo, hi, n_soc)
    Tb = stage_T(stage)
    Voc = bat.ocv.vec(s); R = np.array([bat.r0(x, max(Tb, 15.0)) for x in s])
    Nm = PF.shape[0]
    E = bat.E_nom
    Jtab = np.zeros((Nm + 1, 2, n_soc), np.float64)
    if s_end is None:
        Jtab[Nm, :, :] = terminal_weight * np.maximum(soc_final - s, 0.0) * E
    else:   # linear terminal value of stored charge (charge keeps its value after the trip ends)
        Jtab[Nm, :, :] = s_end * (soc_final - s) * E + terminal_weight * np.maximum(lo + 0.02 - s, 0.0) * E
    sw = np.where(ON, start_J, 0.0)
    for k in range(Nm - 1, -1, -1):
        P = PB[k][None, :]
        disc = np.maximum(Voc[:, None] ** 2 - 4 * R[:, None] * P, 0.0)
        I = (Voc[:, None] - np.sqrt(disc)) / (2 * R[:, None])
        sn = s[:, None] - I * dt / bat.Q
        on = ON[k]
        Jn = np.where(on[None, :], _interp_rows(Jtab[k + 1, 1], s, sn, penalty=50.0 * E),
                      _interp_rows(Jtab[k + 1, 0], s, sn, penalty=50.0 * E))
        base = PF[k][None, :] * dt + Jn
        base = np.where(OK[k][None, :], base, np.inf)
        Jtab[k, 1] = np.min(base, axis=1)
        Jtab[k, 0] = np.min(base + sw[k][None, :], axis=1)
    return dict(J=Jtab, s=s, grid=grid, soc_final=soc_final, start_J=start_J, timed=False)


def stage_T(stage):
    return stage[5] if len(stage) > 5 else 20.0


def make_pt(veh, options=PHYS):
    """the powertrain model exactly as roadsim.simulate sets it up for these physics options (DP stage tables must see
    the same physics as the forward runs: start energy, P2 converter, brake blending, direct drive ...)."""
    opt = resolve_options(options, veh.spec)
    for m in veh.motors.values():
        m.set_options(opt)
    if veh.battery is not None:
        _aging.apply(veh.battery, opt)
    if veh.engine is not None:
        veh.engine._x2 = None
    pt = ARCH_CLASSES[veh.arch](veh)
    pt.opt = opt
    pt.regen_share = float((opt.get('regen_share') or {}).get(veh.arch, 1.0))
    return pt


def make_stage(veh, pt, trip, n_ctrl=20):
    """(grid, P_fuel, P_batt, ok, on, T_amb at start, P_batt in a start step) along the trace; pt from make_pt."""
    g, PF, PB, OK, ON, PBS = _stage_table(veh, pt, trip, None, n_ctrl)
    return (g, PF, PB, OK, ON, float(trip['T_amb'][0]), PBS)


def timers(veh, options=PHYS):
    """minimum engine on / off steps the physics imposes on every hybrid strategy (v2: 8 s / 3 s)."""
    o = resolve_options(options, veh.spec)
    n_on = max(1, int(round(o['hybrid_min_on_s']))) if o.get('hybrid_min_on_s') is not None else 4
    n_off = max(1, int(round(o['hybrid_min_off_s']))) if o.get('hybrid_min_off_s') is not None else 2
    return n_on, n_off


def dp_backward_timed(veh, stage, soc_final, n_on, n_off, start_J=None, soc_bounds=None, n_soc=121, terminal_weight=12.0,
                      dt=1.0, s_end=None):
    """roadsim.dp.dp_solve_timed's recursion (state = engine mode x time in mode x SOC) on a precomputed stage table, so
    several terminal SOCs share one table. start_J: fuel energy per start (default the forward runs' start fuel; no
    virtual start penalty, so this is the fuel optimum under the minimum on/off times). s_end: linear terminal value of
    stored charge instead of the target penalty (teacher labels)."""
    bat = veh.battery
    grid, PF, PB, OK, ON = stage[:5]
    PBS = stage[6] if len(stage) > 6 else None
    if start_J is None:
        e = veh.engine
        start_J = e.fuel_W(e.w_idle, 0.0) * 1.5 * dt
    lo, hi = soc_bounds or (bat.soc_lo, bat.soc_hi)
    s = np.linspace(lo, hi, n_soc)
    Tb = stage_T(stage)
    Voc = bat.ocv.vec(s); R = np.array([bat.r0(x, max(Tb, 15.0)) for x in s])
    Nm = PF.shape[0]
    E = bat.E_nom
    pen = 50.0 * E
    S = n_on + n_off
    on_next = np.minimum(np.arange(n_on) + 1, n_on - 1)
    off_next = n_on + np.minimum(np.arange(n_off) + 1, n_off - 1)
    Jtab = np.zeros((Nm + 1, S, n_soc), np.float64)
    if s_end is None:
        Jtab[Nm, :, :] = terminal_weight * np.maximum(soc_final - s, 0.0) * E
    else:
        Jtab[Nm, :, :] = s_end * (soc_final - s) * E + terminal_weight * np.maximum(lo + 0.02 - s, 0.0) * E

    def snext(P):
        disc = np.maximum(Voc[:, None] ** 2 - 4 * R[:, None] * P[None, :], 0.0)
        I = (Voc[:, None] - np.sqrt(disc)) / (2 * R[:, None])
        return s[:, None] - I * dt / bat.Q

    for k in range(Nm - 1, -1, -1):
        V = _interp_all(Jtab[k + 1], s, snext(PB[k]), pen)
        stg = (PF[k] * dt)[None, None, :]
        m_on = OK[k] & ON[k]; m_off = OK[k] & ~ON[k]
        any_on = m_on.any(); any_off = m_off.any()
        stay_on = np.where(m_on[None, None, :], stg + V[on_next], np.inf).min(axis=2)
        go_off = np.where(m_off[None, :], stg[0] + V[n_on], np.inf).min(axis=1)
        Jon = stay_on.copy()
        free_on = np.zeros(n_on, bool); free_on[-1] = True
        if not any_on:
            free_on[:] = True
        Jon[free_on] = np.minimum(Jon[free_on], go_off[None, :])
        stay_off = np.where(m_off[None, None, :], stg + V[off_next], np.inf).min(axis=2)
        Vs = V[0] if PBS is None else _interp_all(Jtab[k + 1][:1], s, snext(PBS[k]), pen)[0]
        go_on = np.where(m_on[None, :], stg[0] + Vs + start_J, np.inf).min(axis=1)
        Joff = stay_off.copy()
        free_off = np.zeros(n_off, bool); free_off[-1] = True
        if not any_off:
            free_off[:] = True
        Joff[free_off] = np.minimum(Joff[free_off], go_on[None, :])
        Jtab[k, :n_on] = Jon
        Jtab[k, n_on:] = Joff
    return dict(J=Jtab, s=s, grid=grid, soc_final=soc_final, start_J=start_J, n_on=n_on, n_off=n_off,
                on_next=on_next, off_next=off_next, timed=True)


def timed_rows(sol, st, dt=1.0):
    """(value row after an on choice, value row after an off choice, current mode still locked by its minimum time)."""
    n_on, n_off = sol['n_on'], sol['n_off']
    if st['eng_on']:
        a = min(max(int(round(st['t_on'] / dt)), 1), n_on) - 1
        return int(sol['on_next'][a]), n_on, a < n_on - 1
    bb = min(max(int(round(st['t_off'] / dt)), 1), n_off) - 1
    return 0, int(sol['off_next'][bb]), bb < n_off - 1


def dp_select(sol, C, st, ctx, b, start_W, mask=None, start_pen=0.0):
    """DP closed-loop choice at the current state (as roadsim.dp.DPStrategy.select / select_timed)
    -> (index, on, k+1, value row of the chosen action). mask: candidates the platform allows (forced-on rules)."""
    I, V = b.current_vec(C['P_batt'], st['soc'], st['T_bat'], st['v_rc'])
    sn = st['soc'] - I * ctx['dt'] / b.Q
    k = min(ctx['k'] + 1, sol['J'].shape[0] - 1)
    on = C['on'] & (C['P_fuel'] > 0)
    ok = C['ok'].copy()
    if not ok.any():
        return -1, False, k, 0
    if mask is not None and (ok & mask).any():
        ok &= mask
    if sol.get('timed'):
        r_on, r_off, locked = timed_rows(sol, st, ctx['dt'])
        if locked:
            if st['eng_on'] and (ok & on).any():
                ok &= on
            elif (not st['eng_on']) and (ok & ~on).any():
                ok &= ~on
        rows = _interp_all(sol['J'][k][[r_on, r_off]], sol['s'], sn[None, :], 50.0 * b.E_nom)[:, 0, :]
        Jn = np.where(on, rows[0], rows[1])
    else:
        r_on, r_off = 1, 0
        Jn = np.where(on, _interp_rows(sol['J'][k, 1], sol['s'], sn, penalty=50.0 * b.E_nom),
                      _interp_rows(sol['J'][k, 0], sol['s'], sn, penalty=50.0 * b.E_nom))
    cost = C['P_fuel'] * ctx['dt'] + Jn
    if not st['eng_on']:
        cost = cost + np.where(on, start_W * ctx['dt'] + start_pen, 0.0)
    j = int(np.argmin(np.where(ok, cost, np.inf)))
    return j, bool(on[j]), k, (r_on if on[j] else r_off)


class HybridPilot(HybridStrategy):
    """the learned causal policy (and, with a teacher, the data-collection strategy).

    params: net (MLP) | None, preview_s, nav, th_on, th_off, s_scale, s_bias, Kp, Ki, guard_k, min_on_s, min_off_s (default:
            the physics' minimum on/off times, v2 8 s / 3 s), start_pen (J, virtual start cost: drivability term),
            dtg (plug-ins with nav: SOC reference falls linearly with distance to the charge-sustaining level at the
            trip end), Kp_dtg; teacher: dict(sol=DP solution) for labelling, act: 'student' | 'teacher' | 'mix', beta,
            explore, rng
    """
    name = 'hybridpilot'

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        p = self.p
        ctl = (getattr(veh, 'control', None) or {}).get('params') or {}
        if ctl.get('engine_on_moving'):          # mild (P0/P1) twin: no electric-only driving (as RoadSim's strategies)
            self.p['engine_on_moving'] = True
        self.net = p.get('net')
        self.preview_s = int(p.get('preview_s', 0))
        self.nav = bool(p.get('nav', False))
        self.th_on = p.get('th_on', 0.5); self.th_off = p.get('th_off', 0.5)
        self.s_scale = p.get('s_scale', 1.0); self.s_bias = p.get('s_bias', 0.0)
        self.guard_k = p.get('guard_k', 0.0)
        self.Kp = p.get('Kp', 0.0)
        self.Ki = p.get('Ki', 0.0); self.I = 0.0      # integral SOC feedback (as A-ECMS: Ki 0.02 1/s, |integral| <= 30)
        # (15 Nov 2026 options, all off by default = the v2.0 rule)
        self.fb_norm = bool(p.get('fb_norm', False))   # feedback error in half-windows: e = (ref - SOC) / (window / 2)
        self.fb_tau = float(p.get('fb_tau', 0.0))      # proportional term on the SOC error low-passed over fb_tau s
        self.fb_dead = float(p.get('fb_dead', 0.0))    # proportional term only beyond this |e| (half-windows if fb_norm)
        self.cubic = float(p.get('cubic', 0.0))        # ECMS soft-window factor: s *= 1 - cubic x^3, x = (SOC - mid) / half
        self.s_prior = p.get('s_prior')                # {arch or 'bus' / 'mild': (weight, s0)}: s <- (1 - w) s + w s0
        self.ef = None
        # (15 Nov 2026, round 2; off by default) per-class feedback settings: {prior_key() or 'phev': {Kp, Ki, fb_dead}}
        cp = p.get('cls_fb') or {}
        if cp:
            v = veh
            key = 'mild' if p.get('engine_on_moving') else (('bus_' + v.arch) if v.cls == 'bus' else v.arch)
            over = cp.get(key) or (cp.get('phev') if v.arch.startswith('phev') else None) or {}
            self.Kp = over.get('Kp', p.get('Kp', 0.0)); self.Ki = over.get('Ki', p.get('Ki', 0.0))
            self.fb_dead = float(over.get('fb_dead', self.fb_dead))
        self.onoff = p.get('onoff', 'net')
        self.pon_w = p.get('pon_w', 0.0)
        self.min_on_s = p.get('min_on_s', self.min_on_s)
        self.min_off_s = p.get('min_off_s', self.min_off_s)
        self.start_pen_hp = float(p.get('start_pen', 0.0))
        self.n_ctrl = p.get('n_ctrl', 20)
        self.vf = vehicle_features(veh)
        self.tc = TripContext(veh, trip)
        b = veh.battery
        self.phev = veh.arch.startswith('phev')
        self.soc_cs = b.soc_lo + 0.05
        self.soc_ref_cs = self.soc_cs if self.phev else self.soc_mid
        self.cd = self.phev and st['soc'] > self.soc_cs + 0.01
        self.soc0 = st['soc']
        self.dtg = bool(p.get('dtg', False)) and self.phev and self.nav
        self.Kp_dtg = p.get('Kp_dtg', self.Kp)
        self.teacher = p.get('teacher')
        self.act = p.get('act', 'student')
        self.beta = p.get('beta', 0.0)
        self.rng = p.get('rng') or np.random.default_rng(0)
        self.explore = p.get('explore', 0.0)
        self.expl_left = 0
        self.log = [] if p.get('record') else None
        self.trace = [] if p.get('trace') else None
        self.mode_on = False

    def forced_on(self, st, ctx):
        """the platform's forced-on rules; plug-ins in charge depletion use RoadSim's charge-depleting rules (as
        phev_cdcs: the engine joins only above a power share of MG2, a road speed, or for heat)."""
        if self.phev and self.o.get('phev_cd_rules'):
            if self.cd and st['soc'] <= self.soc_cs:
                self.cd = False
            if self.cd:
                return PHEVCDCS.cd_engine_wanted(self, st, ctx) or st['soc'] < self.soc_lo - 0.02
        return HybridStrategy.forced_on(self, st, ctx)

    def grid(self, ctx):
        if hasattr(self.pt, 'default_fracs'):
            return self.pt.default_fracs(self.n_ctrl)
        return self.dd_augment(self.pt.ctrl_grid(self.n_ctrl), ctx)

    def candidates(self, st, ctx):
        return self.grid(ctx)

    def policy_out(self, x):
        y = self.net(x)
        s = math.exp(min(max(y[0], -5.0), 3.0)) - 0.05
        p_on = 1.0 / (1.0 + math.exp(-min(max(y[1], -30.0), 30.0)))
        return s, p_on

    def soc_ref(self, ctx):
        if self.dtg:
            f = min(ctx['dist'] / max(self.tc.D, 1.0), 1.0)
            return max(self.soc_cs, self.soc0 - (self.soc0 - self.soc_cs) * f)
        return self.soc_ref_cs

    def student_choice(self, C, st, ctx, x):
        s, p_on = self.policy_out(x)
        s = max(s * self.s_scale + self.s_bias, 0.0)
        if self.s_prior:
            w_s0 = self.s_prior.get(self.prior_key())
            if w_s0:
                s = (1.0 - w_s0[0]) * s + w_s0[0] * w_s0[1]
        if self.cubic and not self.phev:
            b = self.veh.battery
            z = (st['soc'] - self.soc_mid) / (0.5 * (b.soc_hi - b.soc_lo))
            s = s * (1.0 - self.cubic * (z ** 3 if abs(z) < 1.5 else math.copysign(3.375, z)))
        if self.dtg:          # distance-to-go plan: feedback both ways around the depletion line
            s = max(s + self.Kp_dtg * (self.soc_ref(ctx) - st['soc']), 0.0)
        elif self.Kp or self.Ki:         # SOC feedback (as in A-ECMS); PHEV only below its CS level
            e = self.soc_ref_cs - st['soc']
            if not self.phev or e > -0.03:
                if self.fb_norm:
                    b = self.veh.battery
                    e = e / (0.5 * (b.soc_hi - b.soc_lo))
                if self.Ki:
                    self.I = min(max(self.I + e * ctx['dt'], -30.0), 30.0)
                ep = e
                if self.fb_tau:
                    a = min(ctx['dt'] / self.fb_tau, 1.0)
                    self.ef = e if self.ef is None else self.ef + a * (e - self.ef)
                    ep = self.ef
                if self.fb_dead:     # 'soft': the band is subtracted; 'hard': full Kp e outside the band, 0 inside
                    ep = (ep if abs(ep) > self.fb_dead else 0.0) if self.p.get('fb_dead_mode') == 'hard' else                         math.copysign(max(abs(ep) - self.fb_dead, 0.0), ep)
                s = max(s + self.Kp * ep + self.Ki * self.I, 0.0)
        if self.guard_k:
            b = self.veh.battery
            z = (st['soc'] - self.soc_ref_cs) / (0.5 * (b.soc_hi - b.soc_lo))
            if z < -0.6: s += self.guard_k * (-0.6 - z)
            elif z > 0.8: s = max(s - self.guard_k * (z - 0.8), 0.0)
        on_c = C['on'] & (C['P_fuel'] > 0)
        ok = C['ok'].copy()
        forced = self.forced_on(st, ctx)
        if st['eng_on']:
            want = not (p_on < self.th_off and st['t_on'] >= self.min_on_s)
        else:
            want = p_on > self.th_on and st['t_off'] >= self.min_off_s
        want = want or forced
        if self.onoff == 'ecms':      # on/off left to the s-weighted cost (with start fuel), as ECMS does
            sel = ok.copy()
            if forced and (ok & on_c).any(): sel &= on_c
            if st['eng_on'] and st['t_on'] < self.min_on_s and (ok & on_c).any(): sel &= on_c
            if (not st['eng_on']) and st['t_off'] < self.min_off_s and (ok & ~on_c).any(): sel &= ~on_c
        else:
            sel = ok & (on_c == want)
        if not sel.any():
            sel = ok
        if not sel.any():
            return -1, s, p_on
        J = C['P_fuel'] + s * self.chem_power(C, st)
        if not st['eng_on']:
            J = J + np.where(on_c, self.start_W + self.start_pen_hp / ctx['dt'], 0.0)
        if self.onoff == 'ecms' and self.pon_w:     # the on/off head as a soft prior on the cost
            J = J + np.where(on_c, self.pon_w * (0.5 - p_on) * 0.05 * self.Pe_max, 0.0)
        return int(np.argmin(np.where(sel, J, np.inf))), s, p_on

    def prior_key(self):
        v = self.veh
        if self.p.get('engine_on_moving'):
            return 'mild'
        if v.cls == 'bus':
            return 'bus_' + v.arch
        return v.arch

    def platform_mask(self, C, st, ctx):
        """candidates the platform's rules allow (forced on) -- the teacher acts inside them too."""
        on_c = C['on'] & (C['P_fuel'] > 0)
        return on_c if self.forced_on(st, ctx) else None

    def select(self, C, ctrl, st, ctx):
        x = make_features(self.vf, self.tc, self.veh, st, ctx, self)
        b = self.veh.battery
        jt = None
        if self.teacher is not None:
            sol = self.teacher['sol']
            jt, on_t, k1, row = dp_select(sol, C, st, ctx, b, self.start_W, mask=self.platform_mask(C, st, ctx),
                                          start_pen=self.teacher.get('start_pen', 0.0))
            if jt >= 0 and self.log is not None:
                s_t = costate_label(sol, k1, st['soc'], row, b, st['T_bat'])
                self.log.append((x.astype(np.float32), s_t, float(on_t), ctx['k']))
        use_teacher = self.act == 'teacher' or (self.act == 'mix' and self.rng.random() < self.beta) or self.net is None
        if use_teacher and jt is not None and (jt >= 0 or self.net is None):
            j = jt
        else:
            j, s, p_on = self.student_choice(C, st, ctx, x)
            if self.trace is not None:
                self.trace.append((ctx['k'], s, p_on))
        # exploration: short bursts of a random feasible control (visits off-trajectory SOC / engine states)
        if self.explore > 0:
            if self.expl_left > 0:
                self.expl_left -= 1
                okc = np.flatnonzero(C['ok'])
                if len(okc):
                    j = int(okc[self.expl_jit % len(okc)]) if self.expl_mode == 'fixed' else int(self.rng.choice(okc))
            elif self.rng.random() < self.explore:
                self.expl_left = int(self.rng.integers(3, 25)); self.expl_mode = 'fixed' if self.rng.random() < 0.5 else 'rand'
                self.expl_jit = int(self.rng.integers(0, 100))
        return j


def run_policy(spec, trip, soc0, temps, params, record=True):
    from roadsim import simulate
    strat = HybridPilot(**params)
    out = simulate(spec, trip, strategy=strat, init_soc=soc0, init_temps=temps, record=record, options=PHYS)
    return out, strat


def teacher_solution(veh, stage, trip_len, phev, soc0=None, timed=False):
    """DP teacher. HEV: two passes - a first solve to the window middle gives the trip's typical costate s_end; the
    second solve values end-of-trip charge linearly at s_end (so labels near the trip end keep a causal meaning).
    PHEV: deplete to the CS level (soc_lo + 0.05) by the trip end (the trip length is known to DP); leftover charge has
    no value. timed: DP with the physics' minimum on/off times (v2: 8 s / 3 s), the platform every strategy runs on."""
    b = veh.battery
    if timed:
        n_on, n_off = timers(veh)
        solve = lambda **kw: dp_backward_timed(veh, stage, n_on=n_on, n_off=n_off, **kw)
        row_on = n_on - 1
    else:
        solve = lambda **kw: dp_backward(veh, stage, **kw)
        row_on = 1
    if phev:
        return solve(soc_final=b.soc_lo + 0.05)
    mid = 0.5 * (b.soc_lo + b.soc_hi)
    sol = solve(soc_final=mid)
    # costate along the middle of the window, first 80 % of the trip
    N = sol['J'].shape[0]
    ks = np.linspace(0, int(0.8 * (N - 1)), 12).astype(int)
    vals = [costate_label(sol, k, mid, row_on, b, 20.0) for k in ks]
    s_end = float(np.clip(np.median(vals), 1.0, 5.0))
    sol2 = solve(soc_final=mid, s_end=s_end)
    sol2['s_end'] = s_end
    return sol2
