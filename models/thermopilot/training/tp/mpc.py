"""Control-oriented model of a BuildingSim building and the dynamic-programming MPC built on it.

Thermal part: the ISO 13790 5R1C zone that BuildingSim itself integrates is linear in the mass temperature and
the HVAC heat, so for every hour we take its exact one-step map (vectorised calls of the simulator's own
`zone.solve`) from the NOMINAL spec, evaluated with FORECAST outdoor temperature and solar gain:
    T_air = A + At * T_mass + Aq * q,      T_mass' = C + Ct * T_mass + Cq * q.
Infiltration is linearised at the middle of the comfort band; natural ventilation is off in all twins.
Plant part: the nominal unit's own equipment curves (heat-pump capacity/EIR with defrost, DX capacity/EIR at the
current indoor humidity, cycling losses), furnace/strip/baseboard efficiencies, fan power, the commercial
outdoor-air conductance of a constant-volume fan. The economizer is not modelled (MPC model error).
The MPC: backward DP over a 0.25 K mass-temperature grid, 30 h horizon, hourly replanning, actions = a menu of
(heating, cooling) setpoints relative to the comfort band; cost = energy $ + gas $ + demand-charge increments over
the running monthly peak + lambda x degree-hours outside the band.
"""
import math
import numpy as np
from buildingsim import psychro as ps
from buildingsim import curves as cv
from buildingsim.zone import CP_AIR

# action menu, per side: 0 off, 1 the current band edge (wider when asleep/away), then fixed offsets from the
# OCCUPIED band edge (lo + 0..3 K for heating, hi - 0..3 K for cooling) so pre-conditioning works in any state
DH = [None, 'edge', 0.0, 1.0, 2.0, 3.0]
DC = [None, 'edge', 0.0, -1.0, -2.0, -3.0]
ACTIONS = [(i, j) for i in range(len(DH)) for j in range(len(DC))]


def setpoints(lo_k, hi_k, lo, hi, ih, ic):
    """Absolute (heat_sp, cool_sp) for menu indices; enforces cool >= heat + 0.5."""
    d = DH[ih]
    h = -100.0 if d is None else (lo_k if d == 'edge' else max(lo + d, lo_k))
    d = DC[ic]
    c = 100.0 if d is None else (hi_k if d == 'edge' else min(hi + d, hi_k))
    if c < h + 0.5:
        c = h + 0.5
    return h, c


def _plf(f, c):
    return c[0] + f * (c[1] + f * c[2])


def part_power(P_on, f, speed):
    if speed == 'single':
        return P_on * f / (cv.DX_PLF[0] + f * cv.DX_PLF[1])
    f = np.clip(f, 0.0, 1.0)
    if speed == 'variable':
        hi = P_on * f * (0.75 + 0.25 * f)
        f3 = np.maximum(f / 0.3, 1e-9)
        lo = P_on * 0.3 * (0.75 + 0.25 * 0.3) * f3 / _plf(np.minimum(f3, 1.0), cv.DX_PLF)
        P = np.where(f >= 0.3, hi, lo)
    elif speed == 'two':
        P = P_on * f / _plf(f, cv.DX_PLF_TWO)
    else:
        P = P_on * f / _plf(f, cv.DX_PLF)
    return np.where(f > 0, P, 0.0)


USE_NUMBA = True
from .dp_fast import dp_sweep

SUPPORTED = ('hp', 'furnace', 'elec', 'baseboard', 'none')


class Model:
    """Control model of one (nominal) building over a time window, from a bs.Simulator that is never stepped."""

    def __init__(self, sim):
        self.sim = sim
        s = sim.system
        if type(s).__name__ != 'UnitarySystem' or s.heat_kind not in SUPPORTED:
            raise ValueError(f'unsupported system {type(s).__name__}/{getattr(s, "heat_kind", None)}')
        self.z = sim.zone
        self.n = sim.n
        self.res = sim.res
        self.To = np.asarray(sim.T_out)
        self.Wo = np.asarray(sim.W_out)
        self.th_em = np.asarray(sim._th_em)
        self.pi = np.asarray(sim._phi_int)
        self.psol = np.asarray(sim._phi_sol)
        self.CwU2 = np.asarray(sim.CwU2)
        self.inf_fac = np.asarray(sim.inf_fac)
        self.rho = np.asarray(sim.rho)
        self.m_v = np.asarray(sim.m_v)
        self.T_v = np.asarray(sim.T_v)
        self.ela, self.Cs, self.Qexh2 = sim.ela, sim.Cs, sim.Q_exh ** 2
        self.ghi = np.asarray(sim.wx['ghi'], float)
        self.base_W = np.asarray(sim._base_elec_list()) + np.asarray(sim.pv_W)     # non-HVAC electricity (no PV)
        self.occ = np.asarray(sim.hvac_on) > 0.5
        self.P = float(np.median(sim.wx['P']))
        # plant
        self.kind = s.heat_kind
        self.fan_cont = s.fan_mode == 'continuous_occupied'
        self.m = s.m
        self.P_fan = s.P_fan
        self.gx = s.m * CP_AIR * s.oa_frac
        self.backup = float(s.backup or 0.0)
        h = s.heat or {}
        self.Qheat = float(h.get('Q', 0.0) or 0.0)
        self.eff = float(h.get('eff', 1.0) or 1.0)
        self.bb_fuel = h.get('fuel', 'elec')
        self.hp = h.get('coil') if self.kind == 'hp' else None
        self.cool = s.cool['coil'] if (s.cool is not None and s.cool.get('kind') == 'dx') else None
        self.Tg = np.arange(-40.0, 50.01, 0.5)
        if self.hp is not None:
            Qh, Ph = [], []
            for T in self.Tg:
                W = float(ps.w_from_rh(T, 80.0, self.P))
                q_, p_, _ = self.hp.full(T, W)
                Qh.append(q_); Ph.append(p_)
            self.Qh_tab, self.Ph_tab = np.array(Qh), np.array(Ph)
            self.hp_speed = self.hp.speed
        self.cool_speed = self.cool.speed if self.cool is not None else 'single'
        # descriptors
        z = self.z
        Hve = float(np.mean(self.rho * self.ela * np.sqrt(self.Cs * 10 + self.CwU2) * 1e-3 * self.inf_fac) * CP_AIR
                    + np.mean(self.m_v) * CP_AIR)
        self.UA = Hve + 1.0 / (1.0 / z.H_tr_is + 1.0 / (z.H_tr_w + 1.0 / (1.0 / z.H_tr_ms + 1.0 / z.H_tr_em)))
        self.C = z.C_m
        self.tau_h = self.C / self.UA / 3600.0

    # ---------------------------------------------------------------- one-step zone maps
    def coeffs(self, idx, To, G, Ta):
        z = self.z
        dTo = To - self.To[idx]
        ratio = np.where(self.ghi[idx] > 20, np.clip(G / np.maximum(self.ghi[idx], 1e-6), 0, 3), 1.0)
        psol = self.psol[idx] * ratio
        th_em = self.th_em[idx] + dTo
        Qi = self.ela * np.sqrt(self.Cs * np.abs(Ta - To) + self.CwU2[idx]) * 1e-3 * self.inf_fac[idx]
        if self.Qexh2 > 0:
            Qi = np.sqrt(Qi * Qi + self.Qexh2)
        H_inf = self.rho[idx] * Qi * CP_AIR
        H_v = self.m_v[idx] * CP_AIR
        H_ve = H_inf + H_v + 1e-6
        th_sup = (H_inf * To + H_v * self.T_v[idx]) / H_ve
        pi = self.pi[idx]
        zero = np.zeros_like(To)
        r0 = z.solve(zero, To, th_em, H_ve, th_sup, pi, psol, 0.0, 0.0, 0.0)
        r1 = z.solve(zero + 1.0, To, th_em, H_ve, th_sup, pi, psol, 0.0, 0.0, 0.0)
        rq = z.solve(zero, To, th_em, H_ve, th_sup, pi, psol, 1000.0, 0.0, 0.0)
        return dict(A=r0[3], At=r1[3] - r0[3], Aq=(rq[3] - r0[3]) * 1e-3,
                    C=r0[0], Ct=r1[0] - r0[0], Cq=(rq[0] - r0[0]) * 1e-3)

    def capacities(self, To, W_in, T_in):
        """Per-step heating capacity (W), HP full-load (Q, P), cooling sensible capacity and full power."""
        H = len(To)
        if self.kind == 'hp':
            Qh = np.interp(To, self.Tg, self.Qh_tab); Ph = np.interp(To, self.Tg, self.Ph_tab)
            Qmax = Qh + self.backup
        elif self.kind == 'none':
            Qh = Ph = np.zeros(H); Qmax = np.zeros(H)
        else:
            Qh = Ph = np.zeros(H); Qmax = np.full(H, self.Qheat)
        if self.cool is not None:
            Qs = np.empty(H); Pc = np.empty(H)
            cache = self.__dict__.setdefault('_ccache', {})
            for j in range(H):
                key = (int(round(T_in[j] * 4)), int(round(W_in * 2e4)), int(round(To[j] * 4)))
                v = cache.get(key)
                if v is None:
                    _, _, Qt, q_s, p_ = self.cool.full(key[0] / 4.0, key[1] / 2e4, key[2] / 4.0, self.m)
                    v = cache[key] = (q_s, p_)
                Qs[j], Pc[j] = v
        else:
            Qs = Pc = np.zeros(H)
        return Qh, Ph, Qmax, Qs, Pc


class MPC:
    """Receding-horizon DP controller. forecaster.get(k) -> (T_out, GHI) over H steps. perfect_state: use the
    simulator's true mass temperature (oracle); otherwise a dead-beat observer through the model."""

    def __init__(self, model, ctx, forecaster, lam, H=30, gas_price=0.05, perfect_state=False, peak_floor=None):
        self.M, self.ctx, self.fc, self.lam, self.H = model, ctx, forecaster, lam, H
        # peak_floor: {(ym, period or 'flat'): kW} known monthly peak target (perfect-foresight two-pass oracle);
        # with it, exceeding the floor is charged the full demand rate (no amortisation)
        self.peak_floor = peak_floor
        self.gas_price = gas_price
        self.perfect_state = perfect_state
        self.tm_hat = None
        self.peak = {}
        self.last = None
        self.aih = np.array([a[0] for a in ACTIONS]); self.aic = np.array([a[1] for a in ACTIONS])
        self.lo0, self.hi0 = float(ctx['pref_lo']), float(ctx['pref_hi'])

    def _sp(self, lo, hi):
        """(H, NA) heating and cooling setpoints."""
        H = len(lo)
        hs = np.empty((H, len(DH))); cs = np.empty((H, len(DC)))
        for i, d in enumerate(DH):
            hs[:, i] = -100.0 if d is None else (lo if d == 'edge' else np.maximum(self.lo0 + d, lo))
        for i, d in enumerate(DC):
            cs[:, i] = 100.0 if d is None else (hi if d == 'edge' else np.minimum(self.hi0 + d, hi))
        hsp = hs[:, self.aih]; csp = cs[:, self.aic]
        csp = np.maximum(csp, hsp + 0.5)
        return hsp, csp

    def observe(self, k, obs, info_prev):
        M = self.M
        if self.perfect_state or k == 0 or info_prev is None:
            self.tm_hat = float(obs['T_mass'])
            return
        kk = np.array([k - 1])
        lo, hi = self.ctx['lo'][kk], self.ctx['hi'][kk]
        c = M.coeffs(kk, M.To[kk], M.ghi[kk], 0.5 * (lo + hi))
        q = info_prev['q_hvac_kW'] * 1000.0
        Ta = info_prev['T_zone']
        # dead-beat: T_mass(k-2) from the measured air temperature, then propagate one step
        tm_prev = (Ta - c['A'][0] - c['Aq'][0] * q) / c['At'][0] if abs(c['At'][0]) > 1e-3 else self.tm_hat
        tm_prev = float(np.clip(tm_prev, -10, 50))
        self.tm_hat = float(c['C'][0] + c['Ct'][0] * tm_prev + c['Cq'][0] * q)

    def update_peak(self, k, elec_kW):
        ctx = self.ctx
        key = (ctx['ym'][k], int(ctx['dperiod'][k]))
        if ctx['dperiod'][k] >= 0:
            self.peak[key] = max(self.peak.get(key, 0.0), elec_kW)
        keyf = (ctx['ym'][k], 'flat')
        if ctx['frate'][k] > 0:
            self.peak[keyf] = max(self.peak.get(keyf, 0.0), elec_kW)

    def plan(self, k, W_air, return_values=False):
        M, ctx, H = self.M, self.ctx, self.H
        idx = np.minimum(np.arange(k, k + H), M.n - 1)
        To, G = self.fc.get(k)
        lo, hi = ctx['lo'][idx], ctx['hi'][idx]
        if self.fc.perfect:
            # perfect forecast: the one-step maps of each hour never change, compute them once for the episode
            if getattr(self, '_call', None) is None:
                ia = np.arange(M.n)
                la, ha = ctx['lo'][np.minimum(ia, len(ctx['lo']) - 1)], ctx['hi'][np.minimum(ia, len(ctx['hi']) - 1)]
                self._call = M.coeffs(ia, M.To[ia], M.ghi[ia], 0.5 * (la + ha))
            c = {k_: (v[idx] if np.ndim(v) else v) for k_, v in self._call.items()}
        else:
            c = M.coeffs(idx, To, G, 0.5 * (lo + hi))
        T_in = hi if M.res else (M.gx / max(M.m * CP_AIR, 1e-9)) * To + (1 - M.gx / max(M.m * CP_AIR, 1e-9)) * hi
        Qh, Ph, Qmax, Qs, Pc = M.capacities(To, W_air, T_in)
        hsp, csp = self._sp(lo, hi)
        price, drate, frate = ctx['price'][idx], ctx['drate'][idx], ctx['frate'][idx]
        # demand charges are paid once a month on the monthly maximum: an exceedance over the running peak is
        # amortised over (window hours in the horizon / 3) and the days left in the month that still have the window
        pf = self.peak_floor or {}
        if self.peak_floor is None:
            nwin = max(int(np.sum(drate > 0)), 1)
            drate = drate / (max(1.0, nwin / 3.0) * np.maximum(ctx['drem'][idx], 1))
            nfl = max(int(np.sum(frate > 0)), 1)
            frate = frate / (max(1.0, nfl / 3.0) * np.maximum(ctx['frem'][idx], 1))
        pref = np.array([max(self.peak.get((ctx['ym'][i], int(ctx['dperiod'][i])), 0.0),
                             pf.get((ctx['ym'][i], int(ctx['dperiod'][i])), 0.0)) for i in idx])
        fref = np.array([max(self.peak.get((ctx['ym'][i], 'flat'), 0.0), pf.get((ctx['ym'][i], 'flat'), 0.0)) for i in idx])
        # demand charge: exceedance over the running monthly peak, over the running monthly peak
        # (the charge is paid once per month, not once per hour)
        dp = ctx['dperiod'][idx]
        # (paid once per month: each hour's exceedance is charged in full, which over-counts several exceeding
        # hours and so errs towards peak shaving)
        base = M.base_W[idx]
        occ = M.occ[idx]
        tm0 = float(np.clip(self.tm_hat, -5, 45))
        g0 = math.floor(min(tm0, lo.min()) - 3.0)
        grid = np.arange(g0, max(tm0, hi.max()) + 3.01, 0.25)
        dg = 0.25
        NG = len(grid)
        if USE_NUMBA:
            kc = {'hp': 0, 'furnace': 1}.get(M.kind, None)
            if kc is None:
                kc = (2 if M.bb_fuel != 'elec' else 3) if M.kind == 'baseboard' else 4
            spd = lambda s_: {'single': 0, 'variable': 1, 'two': 2}.get(s_, 3)
            first = dp_sweep(grid.astype(np.float64), tm0, c['A'], c['At'], np.broadcast_to(c['Aq'], (H,)).astype(np.float64),
                             c['C'], c['Ct'], np.broadcast_to(c['Cq'], (H,)).astype(np.float64),
                             (occ & M.fan_cont).astype(np.bool_), float(M.gx), float(M.P_fan),
                             np.ascontiguousarray(hsp, np.float64), np.ascontiguousarray(csp, np.float64),
                             np.asarray(To, np.float64), np.asarray(Qmax, np.float64), np.asarray(Qh, np.float64),
                             np.asarray(Ph, np.float64), np.asarray(Qs, np.float64), np.asarray(Pc, np.float64), kc,
                             spd(getattr(M, 'hp_speed', 'single')), spd(M.cool_speed), float(M.Qheat), float(M.eff),
                             np.asarray(base, np.float64), np.asarray(lo, np.float64), np.asarray(hi, np.float64),
                             np.asarray(price, np.float64), float(self.gas_price), float(self.lam),
                             np.asarray(drate, np.float64), np.asarray(frate, np.float64), pref.astype(np.float64),
                             fref.astype(np.float64), np.array(cv.DX_PLF, np.float64), np.array(cv.DX_PLF_TWO, np.float64))
            a = int(np.argmin(first))
            return (a, first) if return_values else a
        V = np.zeros(NG)
        first = None
        for j in range(H - 1, -1, -1):
            Tm = grid[:, None] if j > 0 else np.array([[tm0]])
            Aair = c['A'][j] + c['At'][j] * Tm
            b = c['Aq'][j]
            fc_ = M.fan_cont and occ[j]
            gx = M.gx if fc_ else 0.0
            q0 = M.P_fan if fc_ else 0.0
            Tb = (Aair + b * (gx * To[j] + q0)) / (1 + b * gx)
            be = b / (1 + b * gx)
            hs, cs = hsp[j][None, :], csp[j][None, :]
            heat = Tb < hs
            cool = (Tb > cs) & (Qs[j] > 0)
            Qreq_h = np.where(heat, (hs - Tb) / be, 0.0)
            Qreq_c = np.where(cool, (Tb - cs) / be, 0.0)
            Qhd = np.minimum(Qreq_h, Qmax[j])
            Qcd = np.minimum(Qreq_c, Qs[j])
            T = Tb + be * (Qhd - Qcd)
            qtot = (gx * (To[j] - T) + q0 if fc_ else 0.0) + Qhd - Qcd
            Tm1 = c['C'][j] + c['Ct'][j] * Tm + c['Cq'][j] * qtot
            # equipment power (W)
            gas = 0.0
            if M.kind == 'hp':
                f = np.where(Qh[j] > 0, np.minimum(Qhd, Qh[j]) / max(Qh[j], 1e-9), 0.0)
                Pel = part_power(Ph[j], f, M.hp_speed) + np.maximum(Qhd - Qh[j], 0.0)
                run_h = np.where(Qhd > Qh[j], 1.0, f) if Qh[j] > 0 else np.where(Qhd > 0, 1.0, 0.0)
            elif M.kind == 'furnace':
                run_h = Qhd / max(M.Qheat, 1e-9)
                gas = Qhd / M.eff
                Pel = 0.004 * M.Qheat * run_h
            elif M.kind == 'baseboard' and M.bb_fuel != 'elec':
                run_h = np.zeros_like(Qhd)
                gas = Qhd / M.eff
                Pel = 0.01 * Qhd
            elif M.kind == 'baseboard':
                run_h = np.zeros_like(Qhd)
                Pel = Qhd
            else:
                run_h = Qhd / max(M.Qheat, 1e-9)
                Pel = Qhd
            fcl = np.where(Qs[j] > 0, Qcd / max(Qs[j], 1e-9), 0.0)
            Pel = Pel + part_power(Pc[j], fcl, M.cool_speed)
            if fc_:
                Pel = Pel + M.P_fan
            else:
                Pel = Pel + M.P_fan * np.clip(run_h + fcl, 0, 1)
            elec = (base[j] + Pel) / 1000.0
            dev = np.maximum(lo[j] - T, 0.0) + np.maximum(T - hi[j], 0.0)
            cost = price[j] * elec + self.gas_price * gas / 1000.0 + self.lam * dev
            if drate[j] > 0:
                cost = cost + drate[j] * np.maximum(elec - pref[j], 0.0)
            if frate[j] > 0:
                cost = cost + frate[j] * np.maximum(elec - fref[j], 0.0)
            pos = np.clip((Tm1 - g0) / dg, 0, NG - 1.000001)
            i0 = pos.astype(int)
            w = pos - i0
            Q = cost + V[i0] * (1 - w) + V[i0 + 1] * w
            if j > 0:
                V = Q.min(axis=1)
            else:
                first = Q[0]
        a = int(np.argmin(first))
        if return_values:
            return a, first
        return a

    def act(self, k, obs, info_prev):
        if info_prev is not None and k > 0:
            self.update_peak(k - 1, info_prev['electricity_kW'])
        self.observe(k, obs, info_prev)
        a = self.plan(k, obs['W_zone'])
        self.last = a
        ih, ic = ACTIONS[a]
        h, c = setpoints(self.ctx['lo'][k], self.ctx['hi'][k], self.lo0, self.hi0, ih, ic)
        return dict(heat_sp=h, cool_sp=c)
