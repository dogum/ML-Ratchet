"""Powertrain architectures.

Each architecture turns a requested wheel force F (N, + traction, - braking) at mid-step vehicle speed u (m/s) into a
*plan*: the force it actually delivers at the wheels plus every power flow of the step (W):

    P_fuel      fuel chemical power                    P_eng   engine brake power (shaft)
    P_trans     transmission/TC/clutch/final-drive loss P_mot   motor+inverter losses (sum)
    P_batt      battery terminal power (+ discharge)    P_aux   auxiliaries + HVAC at their supply point
    P_regen     electrical power recovered by braking (info)

Power balance per step:  P_eng + P_batt = P_aux + P_trans + P_mot + F*u   (checked in sim.py via the ledger).

Hybrid architectures expose `evaluate(T_w, w_w, ctrl, st)` which scores an array of candidate controls at once; the
strategy (strategies.py) picks one and `finalize` turns it into a plan.
"""
import math
import numpy as np

ETA_ALT = 0.65      # alternator efficiency (conventional 12 V supply)
ETA_DCDC = 0.93     # HV -> 12 V DC/DC efficiency


def _spin(veh):
    """gearbox spin-loss coefficients (Nm at input = c0 + c1 w) and mesh efficiency by type."""
    t = veh.ttype
    Tp = veh.engine.T_peak if veh.engine is not None else 300.0
    sc = max(1.0, Tp / 250.0) ** 0.7
    if t == 'AT':   return 1.6 * sc, 0.0025 * sc, 0.975
    if t == 'DCT':  return 0.9 * sc, 0.0015 * sc, 0.98
    if t == 'AMT':  return 1.0 * sc, 0.0015 * sc, 0.985
    if t == 'CVT':  return 2.5 * sc, 0.0040 * sc, 0.95
    return 0.3, 0.0005, 0.985


class _Base:
    def __init__(self, veh):
        self.v = veh
        self.r = veh.r; self.FD = veh.FD; self.eta_dl = veh.eta_dl
        self.eng = veh.engine
        self.bat = veh.battery

    def to_box_out(self, T_w):
        """wheel torque -> torque at final-drive input (signed)"""
        if T_w >= 0.0:
            return T_w / (self.FD * self.eta_dl)
        return T_w * self.eta_dl / self.FD

    def from_box_out(self, T_o):
        if T_o >= 0.0:
            return T_o * self.FD * self.eta_dl
        return T_o * self.FD / self.eta_dl

    def regen_capacity(self, u, st, P_aux):
        return 0.0


# =============================================================================== conventional
class ConvPT(_Base):
    def __init__(self, veh, shift='rule'):
        super().__init__(veh)
        self.c0, self.c1, self.eta_g = _spin(veh)
        self.ng = len(veh.gears)
        self.shift = shift
        e = self.eng
        gas = e.fuel == 'gasoline'
        self.w_up_lo = (1500 if gas else 1300) * math.pi / 30
        if veh.hd: self.w_up_lo = 1250 * math.pi / 30
        self.w_up_hi = min(e.w_rated * 0.92, e.w_max * 0.9)
        self.w_launch = max(e.w_idle * 1.25, (1100 if not veh.hd else 900) * math.pi / 30)
        self.w_lock = max(e.w_min_load, (1050 if gas else 950) * math.pi / 30)
        if veh.hd: self.w_lock = 950 * math.pi / 30
        tr = veh.trans
        self.cvt_range = tr.get('ratio_range', [2.6, 0.4]) if veh.ttype == 'CVT' else None
        self.f_neutral_idle = tr.get('neutral_idle_factor', 0.5)
        self.f_dl = 1.0

    def reset(self, st):
        st['gear'] = 0; st['eng_on'] = True; st['tc_locked'] = False; st['t_in_gear'] = 99

    def tc_idle_torque(self, w_t):
        """pump torque of the converter with the engine at idle and the turbine at w_t (creep load), reduced by a
        neutral-idle allowance (modern automatics partly release the forward clutch at a stop)."""
        tc = self.v.tc
        if tc is None:
            return 0.0
        e = self.eng
        sr = min(max(w_t / e.w_idle, 0.0), 1.0)
        return self.f_neutral_idle * (e.w_idle / tc.K_stall) ** 2 * max(0.0, 1.0 - sr * sr)

    # --- gearbox helpers (input <-> output) for gear ratio g
    def box_in(self, T_o, w_in, g):
        """output torque -> input torque including spin loss (scaled by the driveline-oil viscosity factor)"""
        sp = (self.c0 + self.c1 * w_in) * self.f_dl
        if T_o >= 0.0:
            return T_o / (g * self.eta_g) + sp
        return T_o * self.eta_g / g + sp

    def box_out(self, T_in, w_in, g):
        T = T_in - (self.c0 + self.c1 * w_in) * self.f_dl
        if T >= 0.0:
            return T * g * self.eta_g
        return T * g / self.eta_g

    def _cvt_ratio(self, w_o, P_in):
        e = self.eng
        rmax, rmin = self.cvt_range
        w_t = max(e.ool_w(min(max(P_in, 0.0), e.P_max)), (1250 if not self.v.hd else 1000) * math.pi / 30)
        if w_o <= 1e-6:
            return rmax
        return min(max(w_t / w_o, rmin), rmax)

    def choose_gear(self, T_o, w_o, st, P_req):
        """rule-based shift schedule by pedal (load) and engine speed, with kick-down; or BSFC-optimal ('opt')."""
        v = self.v; e = self.eng; gears = v.gears; ng = self.ng
        g = st['gear']
        pedal = min(max(P_req / e.P_max, 0.0), 1.0)
        w_up = self.w_up_lo + pedal * (self.w_up_hi - self.w_up_lo)
        w_dn = max(e.w_min_load, 0.55 * w_up)
        if self.shift == 'opt' and T_o > 0:
            best, bf = g, float('inf')
            for i in range(ng):
                w = w_o * gears[i]
                if w < e.w_min_load or w > 0.95 * e.w_max:
                    continue
                Ti = self.box_in(T_o, w, gears[i])
                if Ti > 0.9 * e.t_max(w):
                    continue
                f = e.fuel_W(w, Ti) * (1.0 if i == g else 1.03)
                if f < bf: bf, best = f, i
            if bf < float('inf'):
                return best
        w = w_o * gears[g]
        # overspeed protection: upshift whenever the engine would exceed ~97 % of its maximum speed
        while g < ng - 1 and w > 0.97 * e.w_max:
            g += 1; w = w_o * gears[g]
        # upshift while the next gear keeps the engine above the downshift line and has torque reserve
        while g < ng - 1:
            wn = w_o * gears[g + 1]
            if w > w_up and wn >= w_dn and (T_o <= 0 or self.box_in(T_o, wn, gears[g + 1]) < 0.85 * e.t_max(wn)):
                g += 1; w = wn
            else:
                break
        # downshift if lugging or not enough torque
        while g > 0:
            need = T_o > 0 and self.box_in(T_o, w, gears[g]) > 0.92 * e.t_max(max(w, e.w_idle))
            if (w < w_dn and w_o > 0) or need:
                wn = w_o * gears[g - 1]
                if wn > 0.97 * e.w_max:
                    break
                g -= 1; w = wn
            else:
                break
        return g

    def plan(self, F, u, st, env):
        v = self.v; e = self.eng; r = self.r
        P_aux = env['P_aux_mech']
        self.f_dl = env.get('f_dl', 1.0)
        w_w = u / r; T_w = F * r
        T_o = self.to_box_out(T_w); w_o = w_w * self.FD
        stopped = env['stopped']
        Teng = st['T_eng']
        if stopped:
            g = 0
            if e.idle_stop and st['T_eng'] > 40.0 and not env.get('need_heat', False):
                return dict(F=0.0, P_fuel=0.0, P_eng=0.0, w_e=0.0, T_e=0.0, eng_on=False, gear=g, P_trans=0.0, P_mot=0.0,
                            P_batt=0.0, P_aux=0.0, aux_unserved=P_aux, limited=False)
            w_e = e.w_idle
            T_tc = self.tc_idle_torque(0.0)       # in Drive at a stop the converter loads the engine (stall torque)
            T_e = P_aux / w_e + T_tc
            Pf = e.fuel_cold(w_e, T_e, Teng)
            return dict(F=0.0, P_fuel=Pf, P_eng=T_e * w_e, w_e=w_e, T_e=T_e, eng_on=True, gear=g, P_trans=T_tc * w_e, P_mot=0.0,
                        P_batt=0.0, P_aux=P_aux, limited=False)
        t = v.ttype
        if t == 'CVT':
            gr = self._cvt_ratio(w_o, T_o * w_o + P_aux)
            g = 0
        else:
            g = self.choose_gear(T_o, w_o, st, max(F * u, 0.0))
            gr = v.gears[g]
        w_in = w_o * gr
        T_in = self.box_in(T_o, w_in, gr)
        limited = False
        # ---- coupling: torque converter (AT, CVT) or launch clutch (DCT/AMT/manual)
        if t in ('AT', 'CVT'):
            locked = w_in >= self.w_lock and (g >= 1 or t == 'CVT' or w_in > 1.4 * self.w_lock)
        else:
            locked = w_in >= self.w_launch * 0.98
        P_J = 0.0
        if locked:
            w_e = w_in
            if w_e > e.w_max:
                w_e = e.w_max
            # engine + transmission-input inertia accelerate with the wheels in the engaged gear
            T_J = e.J * env.get('a', 0.0) / r * self.FD * gr
            T_req = T_in + P_aux / w_e + T_J
            Tmax = e.t_max(w_e)
            Tmin = e.t_min(w_e)
            if T_req > Tmax:            # full load: traction limited
                limited = True
                T_e = Tmax
                Pf = e.fuel_cold(w_e, T_e, Teng)
            elif T_req <= Tmin:         # deceleration fuel cut-off: motoring torque, friction brakes do the rest
                T_e = Tmin
                Pf = 0.0
            else:                       # part load (incl. light negative torque on a trailing throttle)
                T_e = T_req
                Pf = e.fuel_cold(w_e, T_e, Teng)
            T_in_d = T_e - P_aux / w_e - T_J
            if not limited and T_req > Tmin:
                T_in_d = T_in
            P_J = T_J * w_e
            T_o_d = self.box_out(T_in_d, w_in, gr)
            P_trans_box = T_in_d * w_in - T_o_d * w_o
            P_cpl = T_in_d * (w_e - w_in)
        else:
            if T_in > 0.0:
                if t in ('AT', 'CVT') and v.tc is not None:
                    w_p, T_p = v.tc.solve(max(w_in, 0.0), T_in)
                    w_e = max(w_p, e.w_idle)
                    T_req = T_p + P_aux / w_e
                    Tmax = e.t_max(w_e)
                    if T_req > Tmax:
                        # full throttle with the converter slipping: engine/converter equilibrium
                        limited = True
                        w_e, T_p, T_t = v.tc.stall_equilibrium(max(w_in, 0.0), e, P_aux)
                        T_in_d = min(T_t, T_in)
                        T_e = T_p + P_aux / w_e
                    else:
                        T_e = T_req; T_in_d = T_in
                    P_cpl = T_p * w_e - T_in_d * w_in
                else:   # slipping launch clutch
                    w_e = self.w_launch
                    T_req = T_in + P_aux / w_e
                    Tmax = e.t_max(w_e)
                    if T_req > Tmax:
                        limited = True; T_e = Tmax; T_in_d = Tmax - P_aux / w_e
                    else:
                        T_e = T_req; T_in_d = T_in
                    P_cpl = T_in_d * (w_e - w_in)
                Pf = e.fuel_cold(w_e, T_e, Teng)
            else:   # decoupled coasting / braking at low speed: engine idles (converter creep load dissipated)
                w_e = e.w_idle; T_in_d = 0.0
                T_tc = self.tc_idle_torque(w_in) if t in ('AT', 'CVT') else 0.0
                T_e = P_aux / w_e + T_tc; P_cpl = T_tc * w_e
                Pf = e.fuel_cold(w_e, T_e, Teng)
            T_o_d = self.box_out(T_in_d, w_in, gr)
            P_trans_box = T_in_d * w_in - T_o_d * w_o
        T_w_d = self.from_box_out(T_o_d)
        P_fd = T_o_d * w_o - T_w_d * w_w
        F_d = T_w_d / r
        if F >= 0.0 and F_d > F:      # never deliver more traction than asked (spin loss rounding)
            F_d = F
        return dict(F=F_d, P_fuel=Pf, P_eng=T_e * w_e, w_e=w_e, T_e=T_e, eng_on=True, gear=g, gear_ratio=gr,
                    P_trans=P_trans_box + P_cpl + P_fd + P_J + (T_w_d / r - F_d) * u, P_mot=0.0, P_batt=0.0, P_aux=P_aux,
                    limited=limited or F_d < F - 1e-6 * abs(F) - 1e-9)


# =============================================================================== electrified base
class _ElecBase(_Base):
    def regen_force_cap(self, u, st, P_aux, T_bat, soc, v_rc):
        """maximum regenerative braking force (N) from motor, battery acceptance and the stability / speed limits."""
        v = self.v
        rg = v.regen
        p_dis, p_chg = self.bat.limits(soc, T_bat, v_rc)
        w_m, gmot, Tlim = self._regen_motor(u)
        if w_m <= 0:
            return 0.0
        F_mot = Tlim * gmot * self.FD / (self.eta_dl * self.r) if gmot else 0.0
        F_bat = (p_chg + P_aux) / max(u, 0.1) * 1.08       # mech-side power slightly above electrical (losses)
        F_stab = rg.get('max_decel_g', 0.3) * v.mass * 9.80665
        vmin = rg.get('min_speed_mps', 2.0)
        fade = min(max((u - 0.5 * vmin) / (0.5 * vmin + 1e-9), 0.0), 1.0)
        return max(0.0, min(F_mot, F_bat, F_stab) * fade)

    def _regen_motor(self, u):
        raise NotImplementedError


# =============================================================================== BEV
class BEVPT(_ElecBase):
    def __init__(self, veh):
        super().__init__(veh)
        self.m = veh.motors.get('traction') or list(veh.motors.values())[0]

    def reset(self, st):
        st['gear'] = 0; st['eng_on'] = False

    def _regen_motor(self, u):
        w_m = u / self.r * self.FD
        return w_m, 1.0, self.m.t_lim(w_m)

    def plan(self, F, u, st, env):
        m = self.m
        w_m = u / self.r * self.FD
        T_w = F * self.r
        T_m = self.to_box_out(T_w)
        lim = m.t_lim(w_m)
        limited = False
        if T_m > lim:
            T_m = lim; limited = True
        elif T_m < -lim:
            T_m = -lim; limited = True
        P_aux = env['P_aux_el']
        P_el = m.elec(w_m, T_m) if w_m > 0 else 0.0
        P_batt = P_el + P_aux
        p_dis = env['p_dis']
        if P_batt > p_dis:
            # scale torque down to the battery's discharge limit (bisection on the monotone map)
            lo, hi = 0.0, T_m
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if m.elec(w_m, mid) + P_aux > p_dis: hi = mid
                else: lo = mid
            T_m = lo; P_el = m.elec(w_m, T_m); P_batt = P_el + P_aux; limited = True
        T_w_d = self.from_box_out(T_m)
        F_d = T_w_d / self.r
        loss = P_el - T_m * w_m
        P_fd = T_m * w_m - T_w_d * (u / self.r)
        return dict(F=F_d, P_fuel=0.0, P_eng=0.0, w_e=0.0, T_e=0.0, eng_on=False, gear=0, P_trans=P_fd, P_mot=loss,
                    P_batt=P_batt, P_aux=P_aux, limited=limited, w_m=w_m, T_m=T_m, P_m_el=P_el)


# =============================================================================== hybrid helpers
class _HybridBase(_ElecBase):
    def reset(self, st):
        st['gear'] = 0; st['eng_on'] = False; st['t_on'] = 0.0; st['t_off'] = 99.0

    def engine_fuel_vec(self, w, T, on, cold):
        f = self.eng.fuel_W_vec(w, T) * cold
        return np.where(on, f, 0.0)


# =============================================================================== series (and EREV)
class SeriesPT(_HybridBase):
    def __init__(self, veh):
        super().__init__(veh)
        self.m = veh.motors['traction']; self.gen = veh.motors['generator']

    def _regen_motor(self, u):
        w_m = u / self.r * self.FD
        return w_m, 1.0, self.m.t_lim(w_m)

    def ctrl_grid(self, n=14):
        e = self.eng
        Pmin = max(0.12 * e.P_max, 3000.0)
        return np.concatenate([[0.0], np.linspace(Pmin, e.P_max * 0.98, n)])

    def evaluate(self, T_w, w_w, ctrl, st, env):
        """ctrl = engine brake power (W); 0 = engine off."""
        e = self.eng; m = self.m; gen = self.gen
        T_w = np.asarray(T_w, float); P_e = np.asarray(ctrl, float)
        T_w, P_e = np.broadcast_arrays(T_w, P_e)
        w_m = w_w * self.FD
        T_m = np.where(T_w >= 0, T_w / (self.FD * self.eta_dl), T_w * self.eta_dl / self.FD)
        lim = m.t_lim(w_m)
        ok = np.abs(T_m) <= lim + 1e-9
        P_m = m.elec_vec(np.full_like(T_m, w_m), T_m) if w_m > 0 else np.zeros_like(T_m)
        on = P_e > 1.0
        w_e = np.where(on, e.ool_w.vec(P_e), 0.0)
        T_e = np.where(on, P_e / np.maximum(w_e, 1.0), 0.0)
        ok &= ~on | (T_e <= e.tmax_curve.vec(w_e) * 1.001)
        ok &= ~on | (w_e <= gen.w_max)
        Tg = -T_e
        ok &= np.abs(Tg) <= gen.t_lim_vec(w_e) + 1e-9
        P_g = np.where(on, gen.elec_vec(w_e, Tg), 0.0)
        P_batt = P_m + P_g + env['P_aux_el']
        P_fuel = np.where(on, e.fuel_cold_vec(w_e, T_e, st['T_eng']), 0.0)
        ok &= (P_batt <= env['p_dis'] + 1e-6) & (P_batt >= -env['p_chg'] - 1e-6)
        P_mloss = (P_m - T_m * w_m) + np.where(on, P_g - Tg * w_e, 0.0)
        P_fd = T_m * w_m - T_w * w_w
        return dict(ok=ok, P_fuel=P_fuel, P_batt=P_batt, P_e=P_e, w_e=w_e, T_e=T_e, on=on, P_mot=P_mloss, P_trans=P_fd,
                    T_w=T_w, T_m=T_m, w_m=np.full_like(T_m, w_m), P_m_el=P_m, T_g=Tg, w_g=w_e, P_g_el=P_g)

    def max_ctrl(self):
        return self.eng.P_max * 0.98


# =============================================================================== power split (Prius-like eCVT)
class SplitPT(_HybridBase):
    def __init__(self, veh):
        super().__init__(veh)
        tr = veh.trans
        self.rho = tr.get('sun_teeth', 30) / tr.get('ring_teeth', 78)
        self.k2 = tr.get('mg2_ratio', 2.636)
        self.mg1 = veh.motors['MG1']; self.mg2 = veh.motors['MG2']

    def _regen_motor(self, u):
        w_r = u / self.r * self.FD
        return w_r * self.k2, self.k2, self.mg2.t_lim(w_r * self.k2)

    def ctrl_grid(self, n=14):
        e = self.eng
        Pmin = max(0.08 * e.P_max, 3000.0)
        return np.concatenate([[0.0], np.linspace(Pmin, e.P_max * 0.98, n)])

    def evaluate(self, T_w, w_w, ctrl, st, env):
        """ctrl = engine brake power (W); 0 = engine unfueled (stopped, or spun by MG1 when the vehicle is fast)."""
        e = self.eng; rho = self.rho; k2 = self.k2; mg1 = self.mg1; mg2 = self.mg2
        T_w = np.asarray(T_w, float); P_e = np.asarray(ctrl, float)
        T_w, P_e = np.broadcast_arrays(T_w, P_e)
        w_r = w_w * self.FD
        T_r = np.where(T_w >= 0, T_w / (self.FD * self.eta_dl), T_w * self.eta_dl / self.FD)
        w1max = mg1.w_max * 0.97
        w_lo = max((w_r - rho * w1max) / (1 + rho), 0.0)
        w_hi = (w_r + rho * w1max) / (1 + rho)
        on = P_e > 1.0
        w_ool = e.ool_w.vec(P_e)
        w_on = np.clip(w_ool, max(w_lo, e.w_min_load), max(min(w_hi, e.w_max), max(w_lo, e.w_min_load)))
        w_spin = w_lo
        w_e = np.where(on, w_on, w_spin)
        T_e = np.where(on, P_e / np.maximum(w_e, 1.0), np.where(w_spin > 0, e.tct_curve.vec(w_spin), 0.0))
        ok = ~on | (T_e <= e.tmax_curve.vec(w_e) * 1.001)
        ok &= w_e <= e.w_max * 1.001
        w_s = (1 + 1 / rho) * w_e - w_r / rho
        T1 = -T_e * rho / (1 + rho)
        T2 = (T_r - T_e / (1 + rho)) / k2
        w2 = w_r * k2
        ok &= np.abs(w_s) <= mg1.w_max * 1.0001
        ok &= np.abs(T1) <= mg1.t_lim_vec(np.abs(w_s)) + 1e-9
        ok &= np.abs(T2) <= mg2.t_lim(w2) + 1e-9
        P1 = mg1.elec_vec(w_s, T1); P2 = mg2.elec_vec(np.full_like(T2, w2), T2)
        P_batt = P1 + P2 + env['P_aux_el']
        ok &= (P_batt <= env['p_dis'] + 1e-6) & (P_batt >= -env['p_chg'] - 1e-6)
        P_fuel = np.where(on, e.fuel_cold_vec(w_e, T_e, st['T_eng']), 0.0)
        P_mloss = (P1 - T1 * w_s) + (P2 - T2 * w2)
        P_fd = T_r * w_r - T_w * w_w
        return dict(ok=ok, P_fuel=P_fuel, P_batt=P_batt, P_e=T_e * w_e, w_e=w_e, T_e=T_e, on=on, P_mot=P_mloss,
                    P_trans=P_fd, T_w=T_w, T1=T1, w_s=w_s, T2=T2, w2=np.full_like(T2, w2), P_m_el=P2, P_g_el=P1)

    def max_ctrl(self):
        return self.eng.P_max * 0.98


# =============================================================================== parallel P2
class P2PT(_HybridBase):
    """engine -K0- P2 motor -[launch clutch / TC]- gearbox - final drive. Gear from the shift schedule."""

    def __init__(self, veh):
        super().__init__(veh)
        self.m = veh.motors.get('P2') or list(veh.motors.values())[0]
        self.conv = ConvPT(veh)           # reuse gearbox + shift logic
        self.c0, self.c1, self.eta_g = self.conv.c0, self.conv.c1, self.conv.eta_g
        self.J_m = 0.04 * (self.m.P_max / 50e3) ** 0.5

    def reset(self, st):
        _HybridBase.reset(self, st)
        st['gear'] = 0

    def _regen_motor(self, u):
        w_o = u / self.r * self.FD
        g = self._gear_for_regen(w_o)
        w_in = w_o * self.v.gears[g]
        return w_in, self.v.gears[g] * self.eta_g, self.m.t_lim(w_in)

    def _gear_for_regen(self, w_o):
        gears = self.v.gears
        for i in range(len(gears) - 1, -1, -1):
            if w_o * gears[i] >= 1300 * math.pi / 30:
                return i
        return 0

    def ctrl_grid(self, n=12):
        return None   # depends on shaft speed; built in evaluate via fractions

    def evaluate(self, T_w, w_w, ctrl, st, env):
        """ctrl = engine torque fraction of T_max at shaft speed (0 = engine off / EV, K0 open)."""
        e = self.eng; m = self.m; v = self.v
        T_w = np.asarray(T_w, float); fr = np.asarray(ctrl, float)
        T_w, fr = np.broadcast_arrays(T_w, fr)
        w_o = w_w * self.FD
        T_o = np.where(T_w >= 0, T_w / (self.FD * self.eta_dl), T_w * self.eta_dl / self.FD)
        Tref = float(np.max(T_o)) if T_o.size else 0.0
        g = env.get('gear')
        self.conv.f_dl = env.get('f_dl', 1.0)
        if g is None:
            g = self.conv.choose_gear(max(Tref, 0.0), w_o, st, max(Tref * w_o, 0.0))
        gr = v.gears[g]
        w_in = w_o * gr
        T_in = np.where(T_o >= 0, T_o / (gr * self.eta_g), T_o * self.eta_g / gr) + (self.c0 + self.c1 * w_in) * env.get('f_dl', 1.0)
        on = fr > 1e-6
        alpha = env.get('a', 0.0) / self.r * self.FD * gr
        J = np.where(on, e.J, 0.0) + self.J_m
        T_in = T_in + J * alpha             # rotating inertia (engine when coupled + motor rotor) supplied by the shaft
        P_J = J * alpha * w_in
        # engine can only couple above its minimum loaded speed; below that it may slip-launch at w_launch
        slip = on & (w_in < e.w_min_load)
        w_e = np.where(on, np.maximum(w_in, e.w_min_load), 0.0)
        Tmax = e.tmax_curve.vec(np.maximum(w_e, 1.0))
        T_e = np.where(on, fr * Tmax, 0.0)
        # with slipping K0/launch clutch the engine torque passes through, motor provides the rest at w_in
        T_m = T_in - T_e
        ok = np.abs(T_m) <= m.t_lim(w_in) + 1e-9
        ok &= w_in <= m.w_max
        ok &= ~on | (w_e <= e.w_max)
        P_m = m.elec_vec(np.full_like(T_m, w_in), T_m) if w_in > 0 else T_m * 0.0
        P_batt = P_m + env['P_aux_el']
        ok &= (P_batt <= env['p_dis'] + 1e-6) & (P_batt >= -env['p_chg'] - 1e-6)
        P_fuel = np.where(on, e.fuel_cold_vec(w_e, T_e, st['T_eng']), 0.0)
        P_slip = np.where(slip, T_e * (w_e - w_in), 0.0)
        P_box = (T_in - J * alpha) * w_in - T_o * w_o
        P_fd = T_o * w_o - T_w * w_w
        return dict(ok=ok, P_fuel=P_fuel, P_batt=P_batt, P_e=T_e * w_e, w_e=w_e, T_e=T_e, on=on, P_mot=P_m - T_m * w_in + 0.0 * P_J,
                    P_trans=P_box + P_fd + P_slip + P_J, T_w=T_w, T_m=T_m, w_m=np.full_like(T_m, w_in), gear=g, P_m_el=P_m)

    def default_fracs(self, n=12):
        return np.concatenate([[0.0], np.linspace(0.15, 1.0, n)])


ARCH_CLASSES = {'conv': ConvPT, 'bev': BEVPT, 'hev_series': SeriesPT, 'phev_series': SeriesPT,
                'hev_powersplit': SplitPT, 'phev_powersplit': SplitPT, 'hev_parallel_p2': P2PT}
