"""Dynamic programming (offline, known cycle) and short-horizon MPC for hybrid energy management.

DP (Bellman 1957; Sundstrom & Guzzella 2009 'dpm' practice): state = SOC on a grid, control = the architecture's
candidate grid (engine power, or P2 engine-torque fraction), stage cost = fuel energy, terminal cost penalises ending
below the target SOC. The trace is assumed to be followed exactly (backward-facing requirement), the engine warm,
the battery at the start temperature. The resulting cost-to-go J_k(SOC) is then used as a closed-loop policy inside the
forward simulation: each second the control minimising fuel*dt + J_k+1(SOC_next) is applied (robust to small deviations).

MPC: the same Bellman recursion on a receding preview window (H seconds of the known future trace) from the current SOC,
with an ECMS-type terminal value s * E_batt(SOC_ref - SOC); only the first control is applied.
"""
import math
import numpy as np
from .constants import G
from .strategies import HybridStrategy
from .powertrain import ETA_DCDC


def requirements(veh, trip, tire_warm=1.0):
    """wheel force and mean speed needed to follow the trace exactly (same road-load model as sim.py, no PI)."""
    v = trip['v']; N = len(v)
    gr = trip['grade']; gm = np.concatenate([0.5 * (gr[1:] + gr[:-1]), gr[-1:]])
    th = np.arctan(gm)
    rel = np.radians(trip['wind_dir'] - trip['heading'])
    wh = trip['wind_speed'] * np.cos(rel)
    Tc = np.clip(trip['T_amb'], -30, 45); Kt = veh.tire_K_t
    f_T = (1 + Kt * (20 - 25)) / np.maximum(1 + Kt * (Tc - 25), 0.4)
    crr = veh.Crr * f_T * (1 + veh.tire_warm_amp * (1 - tire_warm))
    u = 0.5 * (v[:-1] + v[1:])
    rv = u + wh[:-1]
    F = (veh.m_eff * np.diff(v) + 0.5 * trip['rho_air'][:-1] * veh.CdA * rv * np.abs(rv)
         + veh.mass * G * np.cos(th[:-1]) * crr[:-1] * (1 + veh.Crr_v * u) + veh.parasitic_N + veh.mass * G * np.sin(th[:-1]))
    stopped = (v[:-1] < 1e-3) & (v[1:] < 1e-3)
    F = np.where(stopped, 0.0, F)
    return F, u, stopped


def _aux_estimate(veh, T_amb):
    cab = veh.cabin
    q = cab.demand(cab.T_heat_set if T_amb < cab.heat_on_below else cab.T_cool_set, T_amb, 0.0)
    el = 0.0
    if q > 0 and cab.source == 'heat_pump': el = q / cab.hp_cop(T_amb)
    elif q > 0 and cab.source == 'ptc': el = q / 0.95
    elif q < 0: el = -q / cab.ac_cop(T_amb)
    return veh.aux_base / ETA_DCDC + el + (cab.blower_W if q != 0 else 0.0)


def stage_table(veh, pt, trip, strat, n_ctrl=20):
    """per-step candidate table: (P_fuel [N-1, J], P_batt [N-1, J], ok [N-1, J]) assuming perfect tracking."""
    return _stage_table(veh, pt, trip, strat, n_ctrl)[:5]


def _stage_table(veh, pt, trip, strat, n_ctrl=20):
    """stage_table plus P_batt for the step in which the engine starts (option hybrid_start_energy: cranking energy),
    or None when starts cost no electrical energy."""
    F, u, stopped = requirements(veh, trip)
    o = getattr(pt, 'opt', None) or {}
    start_el = bool(o.get('hybrid_start_energy'))
    bat = veh.battery
    Tb = float(trip['T_amb'][0])
    p_dis, p_chg = bat.limits(0.5 * (bat.soc_lo + bat.soc_hi), max(Tb, 15.0), 0.0)
    st = dict(T_eng=90.0, soc=0.5 * (bat.soc_lo + bat.soc_hi), T_bat=Tb, v_rc=0.0, gear=0, eng_on=True, t_on=99, t_off=99)
    Nm = len(F)
    if hasattr(pt, 'default_fracs'):
        grid = pt.default_fracs(n_ctrl)
    else:
        grid = pt.ctrl_grid(n_ctrl)
        if o.get('series_direct_drive') and hasattr(pt, 'dd_speed'):
            grid = np.concatenate([grid, -grid[grid > 1.0]])     # the same powers through the lock-up clutch
    J = len(grid)
    PF = np.zeros((Nm, J)); PB = np.zeros((Nm, J)); OK = np.zeros((Nm, J), bool); ON = np.zeros((Nm, J), bool)
    PBS = np.zeros((Nm, J)) if start_el else None
    st0 = dict(st, eng_on=False)
    for k in range(Nm):
        Paux = _aux_estimate(veh, float(trip['T_amb'][k]))
        env = dict(P_aux_el=Paux, p_dis=p_dis, p_chg=p_chg, stopped=bool(stopped[k]), need_heat=False)
        w_w = u[k] / veh.r
        if stopped[k]:
            PB[k, :] = Paux; OK[k, 0] = True
            continue
        if F[k] < 0:
            cap = pt.regen_force_cap(u[k], st, Paux, Tb, st['soc'], 0.0)
            Freg = min(-F[k], cap)
            if veh.arch == 'hev_parallel_p2':
                env['gear'] = pt._gear_for_regen(w_w * pt.FD)
            C = pt.evaluate(-Freg * veh.r, w_w, np.array([0.0]), st, env)
            if not C['ok'][0]:
                C = pt.evaluate(0.0, w_w, np.array([0.0]), st, env)
            PB[k, :] = C['P_batt'][0]; OK[k, 0] = True
            continue
        if veh.arch == 'hev_parallel_p2':
            T_o = pt.to_box_out(F[k] * veh.r)
            g = pt.conv.choose_gear(max(T_o, 0.0), w_w * pt.FD, st, max(F[k] * u[k], 0.0))
            st['gear'] = g; env['gear'] = g
        C = pt.evaluate(F[k] * veh.r, w_w, grid, st, env)
        if not C['ok'].any():
            # infeasible demand: take the max-capability control (trace miss handled in forward sim)
            C['ok'][-1] = True
        PF[k] = C['P_fuel']; PB[k] = C['P_batt']; OK[k] = C['ok']; ON[k] = C['on'] & (C['P_fuel'] > 0)
        if start_el:
            PBS[k] = pt.evaluate(F[k] * veh.r, w_w, grid, dict(st0, gear=st['gear']), env)['P_batt']
    if start_el:
        for k in range(Nm):
            if not ON[k].any():
                PBS[k] = PB[k]
    return grid, PF, PB, OK, ON, PBS


def _interp_rows(J, s_grid, s_next, penalty=None):
    """linear interpolation of J (n_s,) at s_next; outside the grid the edge value plus a steep finite penalty
    (an infinite boundary would creep inward one grid cell per step and swamp the table)."""
    n = len(s_grid)
    ds = s_grid[1] - s_grid[0]
    f = (s_next - s_grid[0]) / ds
    fc = np.clip(f, 0.0, n - 1.000001)
    i = fc.astype(np.int64)
    w = fc - i
    val = J[i] * (1 - w) + J[i + 1] * w
    if penalty is None:
        penalty = 1e9
    out = np.maximum(-f, 0.0) + np.maximum(f - (n - 1), 0.0)
    return val + penalty * out * ds


def dp_solve(veh, pt, trip, strat=None, n_soc=121, n_ctrl=20, soc_final=None, soc_bounds=None, terminal_weight=12.0, dt=1.0,
             start_J=None):
    """state = (engine on/off, SOC); an engine start costs start_J of fuel energy (same as the forward simulation)."""
    bat = veh.battery
    grid, PF, PB, OK, ON = stage_table(veh, pt, trip, strat, n_ctrl)
    if start_J is None:
        e = veh.engine
        start_J = e.fuel_W(e.w_idle, 0.0) * 1.5 * dt
    lo, hi = soc_bounds or (bat.soc_lo, bat.soc_hi)
    s = np.linspace(lo, hi, n_soc)
    Tb = float(trip['T_amb'][0])
    Voc = bat.ocv.vec(s); R = np.array([bat.r0(x, max(Tb, 15.0)) for x in s])
    Nm = PF.shape[0]
    if soc_final is None:
        soc_final = 0.5 * (lo + hi)
    E = bat.E_nom
    Jtab = np.zeros((Nm + 1, 2, n_soc), np.float64)
    Jtab[Nm, :, :] = terminal_weight * np.maximum(soc_final - s, 0.0) * E
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
        Jtab[k, 1] = np.min(base, axis=1)                                          # engine was on
        Jtab[k, 0] = np.min(base + np.where(on, start_J, 0.0)[None, :], axis=1)    # engine was off: starts cost fuel
    return dict(J=Jtab, s=s, grid=grid, soc_final=soc_final, start_J=start_J)


def _interp_all(Jrows, s_grid, s_next, penalty):
    """_interp_rows for several value rows at once: Jrows (S, n_s), s_next (n_s, J) -> (S, n_s, J)."""
    n = len(s_grid)
    ds = s_grid[1] - s_grid[0]
    f = (s_next - s_grid[0]) / ds
    fc = np.clip(f, 0.0, n - 1.000001)
    i = fc.astype(np.int64)
    w = fc - i
    out = np.maximum(-f, 0.0) + np.maximum(f - (n - 1), 0.0)
    return Jrows[:, i] * (1 - w) + Jrows[:, i + 1] * w + (penalty * out * ds)[None]


def dp_solve_timed(veh, pt, trip, n_soc=121, n_ctrl=20, soc_final=None, soc_bounds=None, terminal_weight=12.0, dt=1.0,
                   start_J=None, n_on=1, n_off=1):
    """DP with minimum engine on/off times (option dp_min_on_off). State = (mode, time in mode capped at n_on / n_off
    steps, SOC): the engine may stop only after n_on steps on and start only after n_off steps off, except when the
    step has no feasible candidate in the current mode (braking, stops, demand beyond EV capability), as in the forward
    strategies. A start costs start_J (start fuel + the virtual start penalty) and, with hybrid_start_energy, its
    cranking energy from the battery. n_on = n_off = 1 is dp_solve with these start costs."""
    bat = veh.battery
    grid, PF, PB, OK, ON, PBS = _stage_table(veh, pt, trip, None, n_ctrl)
    if start_J is None:
        e = veh.engine
        start_J = e.fuel_W(e.w_idle, 0.0) * 1.5 * dt
    lo, hi = soc_bounds or (bat.soc_lo, bat.soc_hi)
    s = np.linspace(lo, hi, n_soc)
    Tb = float(trip['T_amb'][0])
    Voc = bat.ocv.vec(s); R = np.array([bat.r0(x, max(Tb, 15.0)) for x in s])
    Nm = PF.shape[0]
    if soc_final is None:
        soc_final = 0.5 * (lo + hi)
    E = bat.E_nom
    pen = 50.0 * E
    S = n_on + n_off
    on_next = np.minimum(np.arange(n_on) + 1, n_on - 1)
    off_next = n_on + np.minimum(np.arange(n_off) + 1, n_off - 1)
    Jtab = np.zeros((Nm + 1, S, n_soc), np.float64)
    Jtab[Nm, :, :] = terminal_weight * np.maximum(soc_final - s, 0.0) * E

    def snext(P):
        disc = np.maximum(Voc[:, None] ** 2 - 4 * R[:, None] * P[None, :], 0.0)
        I = (Voc[:, None] - np.sqrt(disc)) / (2 * R[:, None])
        return s[:, None] - I * dt / bat.Q

    for k in range(Nm - 1, -1, -1):
        V = _interp_all(Jtab[k + 1], s, snext(PB[k]), pen)                     # (S, n_s, J)
        stage = (PF[k] * dt)[None, None, :]
        m_on = OK[k] & ON[k]; m_off = OK[k] & ~ON[k]
        any_on = m_on.any(); any_off = m_off.any()
        # from an on state: stay on (timer advances) or stop (only once the minimum on time is served)
        stay_on = np.where(m_on[None, None, :], stage + V[on_next], np.inf).min(axis=2)          # (n_on, n_s)
        go_off = np.where(m_off[None, :], stage[0] + V[n_on], np.inf).min(axis=1)                  # (n_s,)
        Jon = stay_on.copy()
        free_on = np.zeros(n_on, bool); free_on[-1] = True
        if not any_on:
            free_on[:] = True
        Jon[free_on] = np.minimum(Jon[free_on], go_off[None, :])
        # from an off state: stay off or start (only once the minimum off time is served)
        stay_off = np.where(m_off[None, None, :], stage + V[off_next], np.inf).min(axis=2)       # (n_off, n_s)
        Vs = V[0] if PBS is None else _interp_all(Jtab[k + 1][:1], s, snext(PBS[k]), pen)[0]
        go_on = np.where(m_on[None, :], stage[0] + Vs + start_J, np.inf).min(axis=1)
        Joff = stay_off.copy()
        free_off = np.zeros(n_off, bool); free_off[-1] = True
        if not any_off:
            free_off[:] = True
        Joff[free_off] = np.minimum(Joff[free_off], go_on[None, :])
        Jtab[k, :n_on] = Jon
        Jtab[k, n_on:] = Joff
    return dict(J=Jtab, s=s, grid=grid, soc_final=soc_final, start_J=start_J, n_on=n_on, n_off=n_off,
                on_next=on_next, off_next=off_next)


class DPStrategy(HybridStrategy):
    """closed-loop policy from the DP cost-to-go on the known trip."""
    name = 'dp'
    min_on_s = 0.0
    min_off_s = 0.0

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        self.cat_warm_s = 0.0
        sf = self.p.get('soc_final', st['soc'])
        self.timed = bool(self.o.get('dp_min_on_off'))
        if self.timed:
            n_on = max(1, int(round(self.min_on_s))); n_off = max(1, int(round(self.min_off_s)))   # dt = 1 s steps
            self.sol = dp_solve_timed(veh, pt, trip, n_soc=self.p.get('n_soc', 121), n_ctrl=self.p.get('n_ctrl', 20),
                                      soc_final=sf, start_J=self.start_W + self.start_pen, n_on=n_on, n_off=n_off)
        else:
            self.sol = dp_solve(veh, pt, trip, n_soc=self.p.get('n_soc', 121), n_ctrl=self.p.get('n_ctrl', 20), soc_final=sf)
        self.k = 0

    def grid(self, ctx):
        return self.sol['grid']

    def candidates(self, st, ctx):
        self.k = ctx['k']
        return self.sol['grid']

    def forced_on(self, st, ctx):
        return bool(ctx.get('need_heat'))

    def select_timed(self, C, st, ctx):
        b = self.veh.battery; sol = self.sol
        I, V = b.current_vec(C['P_batt'], st['soc'], st['T_bat'], st['v_rc'])
        sn = st['soc'] - I * ctx['dt'] / b.Q
        k = min(ctx['k'] + 1, sol['J'].shape[0] - 1)
        on = C['on'] & (C['P_fuel'] > 0)
        ok = C['ok']
        if not ok.any():
            return -1
        any_on = (ok & on).any(); any_off = (ok & ~on).any()
        n_on, n_off = sol['n_on'], sol['n_off']
        allowed = ok.copy()
        if st['eng_on']:
            a = min(max(int(round(st['t_on'] / ctx['dt'])), 1), n_on) - 1
            r_on, r_off = sol['on_next'][a], n_on
            if a < n_on - 1 and any_on:
                allowed &= on
        else:
            bb = min(max(int(round(st['t_off'] / ctx['dt'])), 1), n_off) - 1
            r_on, r_off = 0, sol['off_next'][bb]
            if bb < n_off - 1 and any_off:
                allowed &= ~on
        rows = _interp_all(sol['J'][k][[r_on, r_off]], sol['s'], sn[None, :], 50.0 * b.E_nom)[:, 0, :]
        cost = C['P_fuel'] * ctx['dt'] + np.where(on, rows[0], rows[1])
        if not st['eng_on']:
            cost = cost + np.where(on, self.start_W * ctx['dt'] + self.start_pen, 0.0)
        return int(np.argmin(np.where(allowed, cost, np.inf)))

    def select(self, C, ctrl, st, ctx):
        if self.timed:
            return self.select_timed(C, st, ctx)
        b = self.veh.battery
        I, V = b.current_vec(C['P_batt'], st['soc'], st['T_bat'], st['v_rc'])
        sn = st['soc'] - I * ctx['dt'] / b.Q
        k = min(ctx['k'] + 1, self.sol['J'].shape[0] - 1)
        on = C['on'] & (C['P_fuel'] > 0)
        Jn = np.where(on, _interp_rows(self.sol['J'][k, 1], self.sol['s'], sn, penalty=50.0 * b.E_nom),
                      _interp_rows(self.sol['J'][k, 0], self.sol['s'], sn, penalty=50.0 * b.E_nom))
        cost = C['P_fuel'] * ctx['dt'] + Jn
        if not st['eng_on']:
            cost = cost + np.where(on, self.start_W * ctx['dt'], 0.0)
        ok = C['ok']
        if not ok.any():
            return -1
        return int(np.argmin(np.where(ok, cost, np.inf)))


class MPCStrategy(HybridStrategy):
    """receding-horizon DP over an H-second preview with ECMS-type terminal value."""
    name = 'mpc'

    def setup(self, veh, pt, trip, st):
        super().setup(veh, pt, trip, st)
        self.H = int(self.p.get('horizon_s', 15))
        self.s_eq = self.p.get('s', 2.6)
        self.grid_, self.PF, self.PB, self.OK, _ = stage_table(veh, pt, trip, self, self.p.get('n_ctrl', 14))
        self.ns = self.p.get('n_soc', 31)
        self.half = self.p.get('soc_halfwidth', 0.03)

    def grid(self, ctx):
        return self.grid_

    def candidates(self, st, ctx):
        return self.grid_

    def select(self, C, ctrl, st, ctx):
        b = self.veh.battery
        k = ctx['k']; Nm = self.PF.shape[0]
        k1 = min(k + 1, Nm); k2 = min(k + 1 + self.H, Nm)
        s = np.linspace(st['soc'] - self.half, st['soc'] + self.half, self.ns)
        Voc = b.ocv.vec(s); R = b.r0(st['soc'], st['T_bat'])
        ref = self.soc_target
        s_eq = max(0.5, self.s_eq + self.p.get('Kp', 12.0) * (ref - st['soc']))   # SOC feedback as in A-ECMS
        Jn = s_eq * (ref - s) * b.E_nom            # terminal value of stored energy (fuel-equivalent)
        for kk in range(k2 - 1, k1 - 1, -1):
            P = self.PB[kk][None, :]
            disc = np.maximum(Voc[:, None] ** 2 - 4 * R * P, 0.0)
            I = (Voc[:, None] - np.sqrt(disc)) / (2 * R)
            sn = s[:, None] - I * ctx['dt'] / b.Q
            c = self.PF[kk][None, :] * ctx['dt'] + _interp_rows(Jn, s, sn, penalty=50.0 * b.E_nom)
            c = np.where(self.OK[kk][None, :], c, np.inf)
            Jn = np.min(c, axis=1)
        I, V = b.current_vec(C['P_batt'], st['soc'], st['T_bat'], st['v_rc'])
        sn = st['soc'] - I * ctx['dt'] / b.Q
        cost = C['P_fuel'] * ctx['dt'] + _interp_rows(Jn, s, sn, penalty=50.0 * b.E_nom)
        if not st['eng_on']:
            cost = cost + np.where(C['on'], self.start_W + self.start_pen / ctx['dt'], 0.0)
        ok = C['ok'].copy()
        if self.forced_on(st, ctx) and (ok & C['on']).any():
            ok &= C['on']
        if not ok.any():
            return -1
        return int(np.argmin(np.where(ok, cost, np.inf)))
