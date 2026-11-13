"""HybridPilot policy: causal features, the learned strategy (numpy MLP -> equivalence factor s and engine-on
probability), and the DP teacher used to label states (imitation + DAgger).

Decision each traction second (same candidate controls every RoadSim strategy scores):
    s, p_on = net(features)          features = vehicle parameters, state, recent history, optional preview, optional nav
    mode    = on/off with hysteresis on p_on (+ the forced-on rules every causal strategy obeys)
    action  = argmin over candidates in that mode of  P_fuel + s * P_batt,chemical  (+ start fuel when starting)
The DP teacher gives, at any visited (time, SOC, engine state), the optimal action and the local costate
s* = -dJ/dSOC / (V_oc Q) of its cost-to-go table; the network regresses s* and classifies DP's on/off choice.
"""
import _paths as _P  # portable paths (see _paths.py)
import sys, math
sys.path.insert(0, _P.VEHICLE + '/roadsim' + '')
import numpy as np
from roadsim.strategies import HybridStrategy
from roadsim.dp import stage_table, _interp_rows, requirements
from roadsim.constants import G

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
    """roadsim.dp.dp_solve's Bellman recursion on a precomputed stage table (so several terminal SOCs share one table)."""
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
    return dict(J=Jtab, s=s, grid=grid, soc_final=soc_final, start_J=start_J)


def stage_T(stage):
    return stage[5] if len(stage) > 5 else 20.0


def make_stage(veh, pt, trip, n_ctrl=20):
    g, PF, PB, OK, ON = stage_table(veh, pt, trip, None, n_ctrl)
    return (g, PF, PB, OK, ON, float(trip['T_amb'][0]))


def dp_select(sol, C, st, ctx, b, start_W):
    """DP closed-loop choice at the current state (as roadsim.dp.DPStrategy.select) -> (index, on, costs)."""
    I, V = b.current_vec(C['P_batt'], st['soc'], st['T_bat'], st['v_rc'])
    sn = st['soc'] - I * ctx['dt'] / b.Q
    k = min(ctx['k'] + 1, sol['J'].shape[0] - 1)
    on = C['on'] & (C['P_fuel'] > 0)
    Jn = np.where(on, _interp_rows(sol['J'][k, 1], sol['s'], sn, penalty=50.0 * b.E_nom),
                  _interp_rows(sol['J'][k, 0], sol['s'], sn, penalty=50.0 * b.E_nom))
    cost = C['P_fuel'] * ctx['dt'] + Jn
    if not st['eng_on']:
        cost = cost + np.where(on, start_W * ctx['dt'], 0.0)
    ok = C['ok']
    if not ok.any():
        return -1, False, k
    j = int(np.argmin(np.where(ok, cost, np.inf)))
    return j, bool(on[j]), k


class HybridPilot(HybridStrategy):
    """the learned causal policy (and, with a teacher, the data-collection strategy).

    params: net (MLP) | None, preview_s, nav, th_on, th_off, s_scale, s_bias, guard_k, min_on_s, min_off_s,
            teacher: dict(sol=DP solution) for labelling, act: 'student' | 'teacher' | 'mix', beta, explore, rng
    """
    name = 'hybridpilot'

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        p = self.p
        self.net = p.get('net')
        self.preview_s = int(p.get('preview_s', 0))
        self.nav = bool(p.get('nav', False))
        self.th_on = p.get('th_on', 0.5); self.th_off = p.get('th_off', 0.5)
        self.s_scale = p.get('s_scale', 1.0); self.s_bias = p.get('s_bias', 0.0)
        self.guard_k = p.get('guard_k', 0.0)
        self.Kp = p.get('Kp', 0.0)
        self.onoff = p.get('onoff', 'net')
        self.pon_w = p.get('pon_w', 0.0)
        self.min_on_s = p.get('min_on_s', 4.0); self.min_off_s = p.get('min_off_s', 2.0)
        self.n_ctrl = p.get('n_ctrl', 20)
        self.vf = vehicle_features(veh)
        self.tc = TripContext(veh, trip)
        b = veh.battery
        self.soc_ref_cs = (b.soc_lo + 0.05) if veh.arch.startswith('phev') else self.soc_mid
        self.teacher = p.get('teacher')
        self.act = p.get('act', 'student')
        self.beta = p.get('beta', 0.0)
        self.rng = p.get('rng') or np.random.default_rng(0)
        self.explore = p.get('explore', 0.0)
        self.expl_left = 0
        self.log = [] if p.get('record') else None
        self.trace = [] if p.get('trace') else None
        self.mode_on = False

    def grid(self, ctx):
        if hasattr(self.pt, 'default_fracs'):
            return self.pt.default_fracs(self.n_ctrl)
        return self.pt.ctrl_grid(self.n_ctrl)

    def candidates(self, st, ctx):
        return self.grid(ctx)

    def policy_out(self, x):
        y = self.net(x)
        s = math.exp(min(max(y[0], -5.0), 3.0)) - 0.05
        p_on = 1.0 / (1.0 + math.exp(-min(max(y[1], -30.0), 30.0)))
        return s, p_on

    def student_choice(self, C, st, ctx, x):
        s, p_on = self.policy_out(x)
        s = max(s * self.s_scale + self.s_bias, 0.0)
        if self.Kp:     # SOC feedback (as in A-ECMS); PHEV only below its CS level
            e = self.soc_ref_cs - st['soc']
            if not self.veh.arch.startswith('phev') or e > -0.03:
                s = max(s + self.Kp * e, 0.0)
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
            J = J + np.where(on_c, self.start_W, 0.0)
        if self.onoff == 'ecms' and self.pon_w:     # the on/off head as a soft prior on the cost
            J = J + np.where(on_c, self.pon_w * (0.5 - p_on) * 0.05 * self.Pe_max, 0.0)
        return int(np.argmin(np.where(sel, J, np.inf))), s, p_on

    def select(self, C, ctrl, st, ctx):
        x = make_features(self.vf, self.tc, self.veh, st, ctx, self)
        b = self.veh.battery
        jt = None
        if self.teacher is not None:
            sol = self.teacher['sol']
            jt, on_t, k1 = dp_select(sol, C, st, ctx, b, self.start_W)
            if jt >= 0 and self.log is not None:
                s_t = costate_label(sol, k1, st['soc'], int(on_t), b, st['T_bat'])
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
    out = simulate(spec, trip, strategy=strat, init_soc=soc0, init_temps=temps, record=record)
    return out, strat


def teacher_solution(veh, stage, trip_len, phev, soc0=None):
    """DP teacher. HEV: two passes - a first solve to the window middle gives the trip's typical costate s_end; the
    second solve values end-of-trip charge linearly at s_end (so labels near the trip end keep a causal meaning).
    PHEV: deplete to the CS level (soc_lo + 0.05) by the trip end; leftover charge has no value."""
    b = veh.battery
    if phev:
        return dp_backward(veh, stage, soc_final=b.soc_lo + 0.05)
    mid = 0.5 * (b.soc_lo + b.soc_hi)
    sol = dp_backward(veh, stage, soc_final=mid)
    # costate along the middle of the window, first 80 % of the trip
    N = sol['J'].shape[0]
    ks = np.linspace(0, int(0.8 * (N - 1)), 12).astype(int)
    vals = [costate_label(sol, k, mid, 1, b, 20.0) for k in ks]
    s_end = float(np.clip(np.median(vals), 1.0, 5.0))
    sol2 = dp_backward(veh, stage, soc_final=mid, s_end=s_end)
    sol2['s_end'] = s_end
    return sol2
