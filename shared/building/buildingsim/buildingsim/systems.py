"""HVAC systems: the zone-side terminal behaviour and equipment energy.

Every system exposes, per time step, the heat it can give the zone in a linear "conductance" form
q = g (T_s - T_zone) + q0 for its base state (e.g. fan on with minimum outdoor air) and for each capacity stage
(economizer, compressor, backup strips, reheat). The thermostat (thermostat_solve) finds the stage and the
fraction within it that holds the setpoint, or runs the last stage flat out and lets the zone float. Because both
the zone (T = a + b q) and every stage are linear in T, the result is exact, not iterated. finish() then works out
what the equipment did (runtime fractions, damper and valve positions, coil loads, part-load efficiency, fuel and
electricity) and books it in the ledgers.

Conventions: q > 0 heats the zone [W]; return air is the zone air of the previous step (the usual one-step system
lag), and the resulting difference is booked explicitly in the air-side ledger as `return_lag`.
"""
import math

from . import curves as cv
from . import psychro as ps
from .faults import charge_multipliers
from .plant import ChillerPlant, BoilerPlant, District
from .zone import CP_AIR

REC = ('E_heat_elec', 'E_heat_gas', 'E_heat_dist', 'E_cool_elec', 'E_cool_dist', 'E_fans', 'E_pumps', 'E_rej',
       'Q_heat', 'Q_cool_sens', 'Q_cool_lat', 'plr_heat', 'plr_cool', 'cop_heat', 'cop_cool', 'T_mix', 'T_sup',
       'T_ret', 'oa_frac', 'fan_frac', 'cool_cmd', 'heat_cmd', 'reheat_cmd', 'P_fan', 'm_sup', 'unmet_W',
       'E_heat_backup', 'hvac_on', 'ahu_heating_mode')

LED = ('to_zone', 'coil_heat', 'coil_cool_sens', 'fan_heat', 'oa_sens', 'return_lag', 'supply_lag', 'direct',
       'in_elec', 'in_fuel', 'in_dist', 'from_air_water', 'from_ambient', 'to_air_water', 'to_outdoors',
       'coil_latent_kg')


def stage_T(a, b, st):
    g, Ts, q0 = st
    return (a + b * (g * Ts + q0)) / (1.0 + b * g)


def stage_q(st, T):
    return st[0] * (st[1] - T) + st[2]


def thermostat_solve(a, b, base, hsp, csp, heat_fn, cool_fn):
    """Single zone. Returns (q, T, mode, stage, frac). mode 0 base, 1 heating, -1 cooling."""
    Tb = stage_T(a, b, base)
    if Tb < hsp:
        stages = heat_fn()
        if stages:
            q_req = (hsp - a) / b
            prev = stage_q(base, hsp)
            for i, st in enumerate(stages):
                if st is None:                       # ideal (unlimited)
                    return q_req, hsp, 1, i, 1.0
                qm = stage_q(st, hsp)
                if q_req <= qm + 1e-9:
                    f = (q_req - prev) / (qm - prev) if qm > prev + 1e-9 else 1.0
                    return q_req, hsp, 1, i, min(max(f, 0.0), 1.0)
                prev = qm
            st = stages[-1]
            T = stage_T(a, b, st)
            return stage_q(st, T), T, 1, len(stages) - 1, 1.0
    elif Tb > csp:
        stages = cool_fn()
        if stages:
            q_req = (csp - a) / b
            prev = stage_q(base, csp)
            for i, st in enumerate(stages):
                if st is None:
                    return q_req, csp, -1, i, 1.0
                qm = stage_q(st, csp)
                if q_req >= qm - 1e-9:
                    f = (prev - q_req) / (prev - qm) if prev > qm + 1e-9 else 1.0
                    return q_req, csp, -1, i, min(max(f, 0.0), 1.0)
                prev = qm
            st = stages[-1]
            T = stage_T(a, b, st)
            return stage_q(st, T), T, -1, len(stages) - 1, 1.0
    return stage_q(base, Tb), Tb, 0, -1, 0.0


class System:
    inj = 'air'
    has_air = False

    def __init__(self, sim):
        self.sim = sim
        n = sim.n
        self.rec = {k: [0.0] * n for k in REC}
        self.led = {k: 0.0 for k in LED}
        self.dt = sim.dt
        self.F = sim.F

    def plan(self, k):
        return (0.0, 0.0, 0.0)

    def heat_stages(self):
        return []

    def cool_stages(self):
        return []

    def finish(self, k, q, T, mode, stage, frac):
        return 0.0, 0.0

    def _book_direct(self, q):
        self.led['to_zone'] += q * self.dt
        self.led['direct'] += q * self.dt


class NoSystem(System):
    pass


class IdealSystem(System):
    """Unlimited heating and cooling to the setpoint; energy = load (meter 'ideal'). Outdoor air, if any, enters
    the zone directly at outdoor conditions (ventilation term of the zone)."""

    def __init__(self, sim, latent=False):
        super().__init__(sim)
        self.latent = latent

    def heat_stages(self):
        return [None]

    def cool_stages(self):
        return [None]

    def finish(self, k, q, T, mode, stage, frac):
        r = self.rec
        if q > 0:
            r['Q_heat'][k] = q
            r['E_heat_dist'][k] = q
            self.led['in_dist'] += q * self.dt
            self.led['to_air_water'] += q * self.dt
        elif q < 0:
            r['Q_cool_sens'][k] = -q
            r['E_cool_dist'][k] = -q
            self.led['from_air_water'] += -q * self.dt
            self.led['to_outdoors'] += -q * self.dt
            self.led['in_dist'] += 0.0
        r['hvac_on'][k] = 1.0 if q != 0 else 0.0
        self._book_direct(q)
        # bookkeeping of ideal cooling: the removed heat leaves; thermal meter reads it
        if q < 0:
            self.led['in_dist'] -= 0.0
        return 0.0, 0.0


# ----------------------------------------------------------------------------------------------- DX components
class DXCoil:
    """Single-coil DX cooling (EnergyPlus Coil:Cooling:DX:SingleSpeed formulation) with the bypass-factor /
    apparatus-dew-point latent model; A0 is set from the rated SHR."""

    def __init__(self, Q_rated, COP_rated, m_rated, SHR=0.78, speed='single', P=ps.P_ATM):
        self.Qr = Q_rated
        self.COPr = COP_rated
        self.mr = m_rated
        self.speed = speed
        self.P = P
        T_in = 26.67
        W_in = _w_from_db_wb(T_in, 19.44, P)
        h_in = ps.enthalpy_s(T_in, W_in)
        h_out = h_in - Q_rated / m_rated
        T_out = T_in - SHR * Q_rated / (m_rated * CP_AIR)
        W_out = ps.H_FG and (h_out - ps.CP_A * T_out) / (ps.H_FG + ps.CP_V * T_out)
        # extend the line in -> out to the saturation curve
        lo, hi = 1.0, min(50.0, (T_in + 40.0) / max(T_in - T_out, 1e-3))
        if W_out < W_in:
            hi = min(hi, W_in / (W_in - W_out) * 0.999)
        for _ in range(60):
            s = 0.5 * (lo + hi)
            T = T_in + s * (T_out - T_in)
            W = W_in + s * (W_out - W_in)
            if W > ps.w_sat_s(T, P):
                hi = s
            else:
                lo = s
        BF = max(1.0 - 1.0 / s, 1e-3)
        self.A0 = -math.log(BF) * m_rated
        self.rated_SHR = SHR

    def full(self, T_in, W_in, T_odb, m, cap_mult=1.0, eir_mult=1.0, dT_cond=0.0):
        """Steady full-load operation. Returns (T_out, W_out, Q_tot, Q_sens, P_elec)."""
        twb = ps.wetbulb_s(T_in, W_in, self.P)
        To = T_odb + dT_cond
        ff = m / self.mr
        cap = self.Qr * cv.dx_cool_capft(twb, To) * cv.quad(cv.DX_CAP_FFF, ff) * cap_mult
        eir = cv.dx_cool_eirft(twb, To) * cv.quad(cv.DX_EIR_FFF, ff) * eir_mult / self.COPr
        P_el = cap * eir
        h_in = ps.enthalpy_s(T_in, W_in)
        BF = math.exp(-self.A0 / m)
        h_out = h_in - cap / m
        h_adp = h_in - cap / (m * (1.0 - BF))
        T_adp = ps.t_sat_from_h_s(h_adp, self.P)
        W_adp = ps.w_sat_s(T_adp, self.P)
        if W_adp < W_in:
            W_out = BF * W_in + (1 - BF) * W_adp
            T_out = (h_out - ps.H_FG * W_out) / (ps.CP_A + ps.CP_V * W_out)
            Q_s = m * CP_AIR * (T_in - T_out)
            if Q_s >= cap or W_out >= W_in:
                W_out, Q_s = W_in, cap
                T_out = T_in - cap / (m * CP_AIR)
        else:
            W_out, Q_s = W_in, cap
            T_out = T_in - cap / (m * CP_AIR)
        return T_out, W_out, cap, Q_s, P_el

    def part_power(self, P_on, f):
        if f <= 0:
            return 0.0
        if self.speed == 'variable':
            if f >= 0.3:
                return P_on * cv.vs_power_frac(f)
            return P_on * cv.vs_power_frac(0.3) * (f / 0.3) / cv.quad(cv.DX_PLF, f / 0.3)
        if self.speed == 'two':
            return P_on * f / cv.quad(cv.DX_PLF_TWO, f)
        return P_on * f / cv.quad(cv.DX_PLF, f)


def _w_from_db_wb(t, twb, P=ps.P_ATM):
    ws = ps.w_sat_s(twb, P)
    return ((2501.0 - 2.326 * twb) * ws - 1.006 * (t - twb)) / (2501.0 + 1.86 * t - 4.186 * twb)


class HPHeating:
    """Air-source heat-pump heating (EnergyPlus Coil:Heating:DX:SingleSpeed curves, timed reverse-cycle defrost
    after the EnergyPlus Engineering Reference: frost multipliers 0.909 - 107.33 dW (capacity) and
    0.90 - 36.45 dW (input), defrost fraction 0.058333, indoor defrost load 0.01 t (7.222 - T_o) Q_r / 1.01667,
    defrost input with a constant EIR of 0.3)."""

    def __init__(self, Q_rated, COP_rated, speed='single', min_T=-17.8, P=ps.P_ATM):
        self.Qr, self.COPr, self.speed, self.min_T, self.P = Q_rated, COP_rated, speed, min_T, P

    def full(self, T_odb, W_odb):
        if T_odb < self.min_T:
            return 0.0, 0.0, 0.0
        capft = cv.cold_climate_capft(T_odb) if self.speed == 'variable' else cv.hp_heat_capft(T_odb)
        Q = self.Qr * capft
        P = Q * cv.hp_heat_eirft(T_odb) / self.COPr
        Qdef = 0.0
        if T_odb < 5.0:
            Tc = 0.82 * T_odb - 8.589
            dw = max(1e-6, W_odb - ps.w_sat_s(Tc, self.P))
            tf = 0.058333
            Q = Q * (0.909 - 107.33 * dw)
            P = P * (0.90 - 36.45 * dw)
            Qdef = 0.01 * tf * (7.222 - T_odb) * self.Qr / 1.01667
            P += 0.3 * self.Qr / 1.01667 * tf
        return max(Q - Qdef, 0.0), P, Qdef

    def part_power(self, P_on, f):
        if f <= 0:
            return 0.0
        if self.speed == 'variable':
            if f >= 0.3:
                return P_on * cv.vs_power_frac(f)
            return P_on * cv.vs_power_frac(0.3) * (f / 0.3) / cv.quad(cv.DX_PLF, f / 0.3)
        if self.speed == 'two':
            return P_on * f / cv.quad(cv.DX_PLF_TWO, f)
        return P_on * f / cv.quad(cv.DX_PLF, f)


# ----------------------------------------------------------------------------------------- unitary air systems
class UnitarySystem(System):
    """One air unit per zone: PTAC/PTHP, packaged RTU, split AC + furnace, air- and ground-source heat pumps,
    baseboard + window AC. Constant air volume; the fan runs continuously while occupied (commercial) or cycles
    with the compressor/burner (residential, PTAC, night cycle)."""
    has_air = True

    def __init__(self, sim, cfg):
        super().__init__(sim)
        self.c = cfg
        self.m = cfg['m_dot']
        self.P_fan = cfg['P_fan']
        self.dTf = self.P_fan / (self.m * CP_AIR) if self.m > 0 else 0.0
        self.cool = cfg.get('cool')
        self.heat_kind = cfg['heat_kind']
        self.heat = cfg.get('heat')
        self.backup = cfg.get('backup_W', 0.0)
        self.fan_mode = cfg['fan_mode']
        self.oa_frac = cfg.get('oa_frac', 0.0)
        self.econ = cfg.get('economizer', False)
        self.hl = cfg.get('econ_high_limit', 21.0)
        self.gshp = cfg.get('gshp')
        if self.gshp:
            self.T_loop = self.gshp['T_ground']
        self._ctx = None

    # ---- per-step planning
    def plan(self, k):
        sim, F = self.sim, self.F
        occ = sim.hvac_on[k] > 0.5
        T_out, W_out = sim.T_out[k], sim.W_out[k]
        T_ret, W_ret = sim.T_air, sim.W_air
        fan_cont = occ and self.fan_mode == 'continuous_occupied'
        x_min = self.oa_frac if occ else 0.0
        stuck = F.get('oa_damper_stuck', k)
        leak = F.get('oa_damper_leak', k)
        if leak is not None:
            x_min = max(x_min, leak)
        if stuck is not None:
            x_min = stuck
        m = self.m
        la = F.get('low_airflow', k)
        cf = F.get('coil_fouling', k)
        mult = 1.0
        if la is not None:
            mult *= (1 - la)
        if cf is not None:
            mult *= (1 - 0.5 * cf)
        m *= mult
        P_fan = self.P_fan * (1 - 0.3 * (1 - mult))
        dTf = P_fan / (m * CP_AIR) if m > 0 else 0.0
        T_mix = x_min * T_out + (1 - x_min) * T_ret
        W_mix = x_min * W_out + (1 - x_min) * W_ret
        self._ctx = c = dict(k=k, occ=occ, fan_cont=fan_cont, x_min=x_min, T_out=T_out, W_out=W_out, T_ret=T_ret,
                             W_ret=W_ret, m=m, P_fan=P_fan, dTf=dTf, T_mix=T_mix, W_mix=W_mix, stuck=stuck)
        # implicit return air (return = the zone air of THIS step): a constant-volume unit gives the zone
        # q = m cp x (T_out - T) + P_fan + coil, i.e. the stage form (g = m cp x, Ts = T_out, q0 = P_fan + coil)
        gx = m * CP_AIR * x_min
        c['gx'] = gx
        if fan_cont and m > 0:
            c['base'] = (gx, T_out, P_fan)
        else:
            c['base'] = (0.0, 0.0, 0.0)
        cs = F.get('cooling_valve_stuck', k)
        c['forced'] = None
        if cs is not None and fan_cont and m > 0 and self.cool is not None and self.cool['kind'] == 'dx':
            # compressor stuck running at fraction cs whenever the fan runs (contactor / control fault)
            capm, eirm = charge_multipliers(F.get('refrigerant_charge', k) or 0.0)
            T_o, W_o, Qt, Qs, P = self.cool['coil'].full(T_mix, W_mix, T_out, m, capm, eirm)
            c['forced'] = (cs, T_o, W_o, Qt, Qs, P)
            c['base'] = (gx, T_out, P_fan - cs * Qs)
        return c['base']

    def heat_stages(self):
        c, sim, k = self._ctx, self.sim, self._ctx['k']
        if self.heat_kind == 'none':
            return []
        if sim.F.get('schedule_override', k) == 4:
            return []
        m = c['m']
        gx, To, Pf = c['gx'], c['T_out'], c['P_fan']
        q0f = Pf - (c['forced'][0] * c['forced'][4] if c.get('forced') else 0.0)   # stuck compressor underneath
        st = []
        if self.heat_kind == 'baseboard':
            c['h_on'] = [(self.heat['Q'], 0.0, 0.0)]
            b = c['base']
            return [(b[0], b[1], b[2] + self.heat['Q'])]
        if self.heat_kind in ('furnace', 'elec', 'boiler', 'district'):
            Q = self.heat['Q']
            c['h_on'] = [(Q, 0.0, 0.0)]
            return [(gx, To, q0f + Q)]
        if self.heat_kind == 'hp':
            Qh, Ph, Qdef = self.heat['coil'].full(c['T_out'], c['W_out'])
            ch = self.F.get('refrigerant_charge', k)
            if ch:
                capm, eirm = charge_multipliers(ch)
                Qh, Ph = Qh * capm, Ph * capm * eirm
            c['hp'] = (Qh, Ph, Qdef)
            stages = []
            ons = []
            if Qh > 0:
                stages.append((gx, To, q0f + Qh))
                ons.append((Qh, Ph, 0.0))
            if self.backup > 0:
                stages.append((gx, To, q0f + Qh + self.backup))
                ons.append((Qh, Ph, self.backup))
            c['h_on'] = ons
            return stages
        if self.heat_kind == 'gshp':
            capm, eirm = cv.gshp_heat(self.T_loop)
            Q = self.heat['Q'] * capm
            P = Q * eirm / self.heat['COP']
            c['hp'] = (Q, P, 0.0)
            c['h_on'] = [(Q, P, 0.0)] + ([(Q, P, self.backup)] if self.backup > 0 else [])
            stages = [(gx, To, q0f + Q)]
            if self.backup > 0:
                stages.append((gx, To, q0f + Q + self.backup))
            return stages
        return st

    def cool_stages(self):
        c, sim, k = self._ctx, self.sim, self._ctx['k']
        if self.cool is None or sim.F.get('schedule_override', k) == 4:
            return []
        F = self.F
        stages = []
        m = c['m']
        econ_ok = False
        if self.econ and c['fan_cont'] and c['stuck'] is None and F.get('economizer_disabled', k) is None \
                and not c.get('forced'):
            hl = F.get('economizer_high_limit', k)
            hl = self.hl if hl is None else hl
            To_s = c['T_out'] + (F.get('bias_T_out', k) or 0.0)
            Tr_s = c['T_ret'] + (F.get('bias_T_ret', k) or 0.0)
            econ_ok = To_s < Tr_s and To_s < hl
        c['econ_ok'] = econ_ok
        x_c = 1.0 if econ_ok else c['x_min']
        if econ_ok:
            stages.append((m * CP_AIR, c['T_out'], c['P_fan']))
        T_in = x_c * c['T_out'] + (1 - x_c) * c['T_ret']
        W_in = x_c * c['W_out'] + (1 - x_c) * c['W_ret']
        c['x_c'] = x_c
        c['Tin_c'], c['Win_c'] = T_in, W_in
        if self.cool['kind'] == 'gshp':
            capm, eirm = cv.gshp_cool(self.T_loop)
            capm_, eirm_ = 1.0, 1.0
            coil = self.cool['coil']
            T_o, W_o, Qt, Qs, P = coil.full(T_in, W_in, 35.0, m, capm, eirm)
        else:
            capm, eirm = charge_multipliers(F.get('refrigerant_charge', k) or 0.0)
            if F.get('coil_fouling', k):
                capm *= (1 - F.get('coil_fouling', k))
            dTc = 8.0 * (F.get('condenser_fouling', k) or 0.0)
            T_o, W_o, Qt, Qs, P = self.cool['coil'].full(T_in, W_in, c['T_out'], m, capm, eirm, dTc)
        c['c_on'] = (T_o, W_o, Qt, Qs, P)
        stages.append((m * CP_AIR * x_c, c['T_out'], c['P_fan'] - Qs))
        return stages

    # ---- after the zone solve
    def finish(self, k, q, T, mode, stage, frac):
        c, r, L, dt = self._ctx, self.rec, self.led, self.dt
        m, dTf, P_fan = c['m'], c['dTf'], c['P_fan']
        T_out, T_ret = c['T_out'], c['T_ret']
        stuck_cool = self.F.get('cooling_valve_stuck', k)
        # states: list of (weight, x, coil_sens(+heat), coil_lat_W, W_sup, fan_on)
        states = []
        E = dict(he=0.0, hg=0.0, ce=0.0, bu=0.0)
        qh = qc = ql = 0.0
        plr_h = plr_c = 0.0
        if mode == 0 and c.get('forced'):
            p, T_o, W_o, Qt, Qs, P_on = c['forced']
            states.append((p, c['x_min'], -Qs, W_o, True))
            states.append((1 - p, c['x_min'], 0.0, c['W_mix'], True))
            P = self.cool['coil'].part_power(P_on, p)
            E['ce'] += P
            L['in_elec'] += P * dt
            L['from_air_water'] += p * Qt * dt
            L['to_outdoors'] += (p * Qt + P) * dt
            L['coil_latent_kg'] += p * m * (c['W_mix'] - W_o) * dt
            qc, ql, plr_c = p * Qs, p * (Qt - Qs), p
        elif mode == 0:
            if c['fan_cont']:
                states.append((1.0, c['x_min'], 0.0, c['W_mix'], True))
        elif mode == 1:
            ons = c['h_on']
            on = ons[stage]
            if self.heat_kind == 'baseboard':
                Q = q - (stage_q(c['base'], T) if c['fan_cont'] else 0.0)
                if c['fan_cont']:
                    states.append((1.0, c['x_min'], 0.0, c['W_mix'], True))
                self._heat_device(k, Q, Q / self.heat['Q'] if self.heat['Q'] > 0 else 0, E)
                qh = Q
                plr_h = Q / self.heat['Q'] if self.heat['Q'] else 0.0
                self.led['direct'] += Q * dt
                self.led['to_zone'] += Q * dt
                return self._close(k, q, T, states, E, qh, qc, ql, plr_h, plr_c, c, direct=Q)
            if stage == 0:
                f = frac
                prev_on = None
            else:
                f = frac
                prev_on = ons[stage - 1]
            # heat delivered by the unit (W, time averaged)
            Q_unit = q - ((1 - (f if stage == 0 else 0)) * 0.0)
            if stage == 0:
                Qh_avg = f * (on[0] + on[2])
                run = f
            else:
                Qh_avg = prev_on[0] + prev_on[2] + f * (on[0] + on[2] - prev_on[0] - prev_on[2])
                run = 1.0
            del Q_unit
            fan_run = 1.0 if c['fan_cont'] else run
            if c.get('forced'):
                p_, T_o_, W_o_, Qt_, Qs_, P_on_ = c['forced']
                qh_f = Qh_avg
                states.append((p_, c['x_min'], -Qs_ + qh_f, W_o_, True))
                states.append((1 - p_, c['x_min'], qh_f, c['W_mix'], True))
                Pc = self.cool['coil'].part_power(P_on_, p_)
                E['ce'] += Pc
                L['in_elec'] += Pc * dt
                L['from_air_water'] += p_ * Qt_ * dt
                L['to_outdoors'] += (p_ * Qt_ + Pc) * dt
                L['coil_latent_kg'] += p_ * m * (c['W_mix'] - W_o_) * dt
                qc, ql, plr_c = p_ * Qs_, p_ * (Qt_ - Qs_), p_
            else:
                # supply humidity unchanged by heating
                states.append((fan_run, c['x_min'], Qh_avg / max(fan_run, 1e-9), c['W_mix'], True))
            self._heat_device(k, Qh_avg, run if stage == 0 else 1.0, E, f_stage=f, stage=stage)
            qh = Qh_avg
            plr_h = run if stage == 0 else 1.0
        else:
            econ = c.get('econ_ok', False)
            mech_idx = 1 if econ else 0
            if econ and stage == 0:
                # economizer modulating, no compressor
                den = m * CP_AIR * (T_out - T)
                x = (q - P_fan) / den if abs(den) > 1e-9 else 1.0
                x = min(max(x, c['x_min']), 1.0)
                Wm = x * c['W_out'] + (1 - x) * c['W_ret']
                states.append((1.0, x, 0.0, Wm, True))
            else:
                T_o, W_o, Qt, Qs, P_on = c['c_on']
                f = frac if stage == mech_idx else 1.0
                if c.get('forced'):
                    f = c['forced'][0] + f * (1 - c['forced'][0])     # runtime from the stuck fraction up
                if stuck_cool is not None and c['fan_cont']:
                    pass
                x_c = c['x_c']
                Wm_c = c['Win_c']
                # 'off' part of the cycle: previous stage (econ at x=1 or base at x_min, or fan off)
                if econ:
                    off_state = (1.0 - f, 1.0, 0.0, c['W_out'], True)
                elif c['fan_cont']:
                    off_state = (1.0 - f, c['x_min'], 0.0, c['W_mix'], True)
                else:
                    off_state = None
                Ql_on = Qt - Qs
                states.append((f, x_c, -Qs, W_o, True))
                if off_state is not None and off_state[0] > 0:
                    states.append(off_state)
                qc, ql = f * Qs, f * Ql_on
                P = self.cool['coil'].part_power(P_on, f)
                E['ce'] += P
                L['in_elec'] += P * dt
                L['from_air_water'] += f * Qt * dt
                L['to_outdoors'] += (f * Qt + P) * dt
                L['coil_latent_kg'] += f * m * (Wm_c - W_o) * dt
                if self.gshp:
                    self._gshp_loop(f * Qt + P, k)
                plr_c = f
                r['cop_cool'][k] = Qt / P_on if P_on > 0 else 0.0
        return self._close(k, q, T, states, E, qh, qc, ql, plr_h, plr_c, c)

    def _heat_device(self, k, Q, run, E, f_stage=None, stage=0):
        L, dt = self.led, self.dt
        hk = self.heat_kind
        if Q <= 0:
            return
        if hk == 'baseboard' and self.heat.get('fuel', 'elec') != 'elec':
            hk = 'boiler' if self.heat['fuel'] in ('boiler', 'gas') else self.heat['fuel']
        if hk in ('furnace', 'boiler'):
            fuel = Q / self.heat['eff']
            if hk == 'boiler':     # hot-water coil / baseboard: circulator pump instead of an inducer
                P_pump = 0.01 * Q
                E['hg'] += fuel
                E['pump'] = E.get('pump', 0.0) + P_pump
                L['in_fuel'] += fuel * dt
                L['in_elec'] += P_pump * dt
                L['to_air_water'] += Q * dt
                L['to_outdoors'] += (fuel - Q + P_pump) * dt
                return
            P_ind = 0.004 * self.heat['Q'] * run
            E['hg'] += fuel
            E['he'] += P_ind
            L['in_fuel'] += fuel * dt
            L['in_elec'] += P_ind * dt
            L['to_air_water'] += Q * dt
            L['to_outdoors'] += (fuel - Q + P_ind) * dt
        elif hk == 'district':
            P_pump = 0.01 * Q
            E['hd'] = E.get('hd', 0.0) + Q
            E['pump'] = E.get('pump', 0.0) + P_pump
            L['in_dist'] += Q * dt
            L['in_elec'] += P_pump * dt
            L['to_air_water'] += Q * dt
            L['to_outdoors'] += P_pump * dt
        elif hk in ('elec', 'baseboard'):
            E['he'] += Q
            L['in_elec'] += Q * dt
            L['to_air_water'] += Q * dt
        elif hk in ('hp', 'gshp'):
            ons = self._ctx['h_on']
            Qh, Ph, _ = self._ctx['hp']
            if stage == 0 and ons and ons[0][0] > 0:
                f = Q / Qh if Qh > 0 else 0.0
                P = self.heat['coil'].part_power(Ph, f) if hk == 'hp' else Ph * f / cv.quad(cv.DX_PLF, f)
                Qb = 0.0
            else:
                # compressor full (if available) + strips for the rest
                Qhp = Qh if (ons and ons[0][0] > 0) else 0.0
                P = Ph if Qhp > 0 else 0.0
                Qb = max(Q - Qhp, 0.0)
                Q = Qhp + Qb
            E['he'] += P + Qb
            E['bu'] += Qb
            L['in_elec'] += (P + Qb) * dt
            L['to_air_water'] += Q * dt
            L['from_ambient'] += (Q - Qb - P) * dt
            self.rec['cop_heat'][self._ctx['k']] = (Q - Qb) / P if P > 0 else 0.0
            if hk == 'gshp':
                self._gshp_loop(-(Q - Qb - P), self._ctx['k'])

    def _gshp_loop(self, Q_to_loop, k):
        g = self.gshp
        # lumped loop: C dT/dt = Q_to_loop - UA (T - T_ground)
        Tg = self.sim.T_ground_deep[k]
        self.T_loop += (Q_to_loop - g['UA'] * (self.T_loop - Tg)) * self.dt / g['C']

    def _close(self, k, q, T, states, E, qh, qc, ql, plr_h, plr_c, c, direct=0.0):
        """Air-side ledger, fan energy, records. Returns (m_sup avg, W_sup avg) for the moisture balance.
        Exact identity per state: w m cp (Ts - T) = w coil + w P_fan + w m x cp (T_out - T) + w m (1-x) cp (T_ret - T)."""
        r, L, dt = self.rec, self.led, self.dt
        m, dTf, P_fan = c['m'], c['dTf'], c['P_fan']
        T_out, T_ret = c['T_out'], T          # implicit coupling: return air = zone air of this step
        m_sup = mw = fan_run = coil = xs = 0.0
        for (w, x, coil_s, Wsup, fan) in states:
            if w <= 0:
                continue
            fan_run += w
            m_sup += w * m
            mw += w * m * Wsup
            coil += w * coil_s
            xs += w * x
            L['oa_sens'] += w * m * x * CP_AIR * (T_out - T) * dt
            L['return_lag'] += w * m * (1 - x) * CP_AIR * (T_ret - T) * dt
        Pf = P_fan * fan_run
        L['fan_heat'] += Pf * dt
        if coil >= 0:
            L['coil_heat'] += coil * dt
        else:
            L['coil_cool_sens'] += -coil * dt
        L['to_zone'] += (q - direct) * dt
        L['in_elec'] += Pf * dt
        r['E_heat_elec'][k] = E['he']
        r['E_heat_gas'][k] = E['hg']
        r['E_heat_dist'][k] = E.get('hd', 0.0)
        r['E_pumps'][k] = E.get('pump', 0.0)
        r['E_cool_elec'][k] = E['ce']
        r['E_heat_backup'][k] = E['bu']
        r['E_fans'][k] = Pf
        r['Q_heat'][k] = qh
        r['Q_cool_sens'][k] = qc
        r['Q_cool_lat'][k] = ql
        r['plr_heat'][k] = plr_h
        r['plr_cool'][k] = plr_c
        r['fan_frac'][k] = fan_run
        r['P_fan'][k] = Pf
        r['m_sup'][k] = m_sup
        r['hvac_on'][k] = 1.0 if (fan_run > 0 or qh > 0) else 0.0
        r['T_ret'][k] = T_ret
        if fan_run > 0:
            x_avg = xs / fan_run
            r['oa_frac'][k] = x_avg
            r['T_mix'][k] = x_avg * T_out + (1 - x_avg) * T_ret
            r['T_sup'][k] = T + (q - direct) / (m_sup * CP_AIR)
        else:
            r['T_mix'][k] = T_ret
            r['T_sup'][k] = T_ret
        r['cool_cmd'][k] = plr_c
        r['heat_cmd'][k] = plr_h
        return m_sup, (mw / m_sup if m_sup > 0 else 0.0)


# ------------------------------------------------------------------------------------------------- VAV systems
class VAVSystem(System):
    """VAV air handler with economizer, preheat and chilled-water (or DX) cooling coil, supply-air temperature
    control (with outdoor reset), VAV boxes with reheat; plant = chiller/tower/boiler or district."""
    has_air = True

    def __init__(self, sim, cfg):
        super().__init__(sim)
        self.c = cfg
        self.m_max = cfg['m_max']
        self.m_min = cfg['min_frac'] * self.m_max
        # dual-maximum boxes: heating airflow up to what the design heating load needs at the reheat limit,
        # capped at 50 % of the cooling maximum; any remaining design heating comes from perimeter radiation
        # (hot water from the same plant), as in older buildings with VAV + perimeter heat
        T_rh = cfg.get('T_reheat_max', 35.0)
        self.m_heat = min(max(self.m_min, cfg.get('Q_heat_zone', 0.0) / (CP_AIR * (T_rh - 21.0))), 0.5 * self.m_max)
        self.m_heat = max(self.m_heat, self.m_min)
        self.Q_perim = max(0.0, cfg.get('Q_heat_zone', 0.0) - self.m_heat * CP_AIR * (T_rh - 21.0))
        self.ahu_mode = 'cooling'
        self.last_mode = 0
        self.P_fan_d = cfg['P_fan']
        self.fan_curve = cv.VAV_FAN if cfg.get('static_reset', True) else cv.VAV_FAN_NO_RESET
        self.m_oa = cfg['m_oa']
        self.econ = cfg.get('economizer', True)
        self.hl = cfg.get('econ_high_limit', 21.0)
        self.T_rh_max = cfg.get('T_reheat_max', 35.0)
        self.sat = cfg.get('sat', 12.8)
        self.sat_reset = cfg.get('sat_reset', True)
        self.cool_plant = cfg['cool_plant']     # ChillerPlant / District('chw') / None
        self.heat_plant = cfg['heat_plant']     # BoilerPlant / District('steam') / 'elec'
        self.reheat_elec = cfg.get('reheat_elec', False)
        self.Qcc_d = cfg['Q_cc_design']
        self.T_chw = cfg.get('T_chw', 6.67)
        self.cc_lim = 1.0
        self.T_sa_prev = self.sat
        self.m_prev = self.m_max * 0.5
        self.dTf_prev = self.P_fan_d / (self.m_max * CP_AIR)
        self._ctx = None
        self.unmet_steps = 0

    def _sat(self, k):
        s = self.sat
        if self.sat_reset:
            To = self.sim.T_out[k]
            if To <= 15.6:
                s = self.sat + 2.8
            elif To < 21.1:
                s = self.sat + 2.8 * (21.1 - To) / 5.5
        off = self.F.get('sat_setpoint_offset', k)
        if off is not None:
            s += off
        b = self.F.get('bias_T_sup', k)       # biased supply-air sensor: the loop holds sensor = setpoint
        if b is not None:
            s -= b
        return s

    def plan(self, k):
        sim, F = self.sim, self.F
        occ = sim.hvac_on[k] > 0.5
        so = F.get('schedule_override', k)
        T_sa = self._sat(k)
        T_ret = sim.T_air
        # a stuck cooling valve / fouled coil can make the supply air temperature float (lagged estimate)
        stuck = F.get('cooling_valve_stuck', k)
        if stuck is not None and self.m_prev > 0 and stuck < 0.999:
            T_sa = max(T_sa, self._sa_with_stuck(k, stuck, T_ret, occ))
        T_sa = max(T_sa, self.T_sa_prev) if self.cc_lim < 1.0 else T_sa
        # AHU heating mode (zone called for heat last step): economizer and cooling coil locked out, supply air =
        # mixed air (+ preheat to >= sat) + fan heat; predicted with last step's flow (difference -> supply_lag)
        if self.last_mode == 1:
            self.ahu_mode = 'heating'
        elif self.last_mode == -1:
            self.ahu_mode = 'cooling'
        if self.ahu_mode == 'heating' and occ:
            x = min(self.m_oa / max(self.m_prev, 1e-6), 1.0)
            stk = F.get('oa_damper_stuck', k)
            if stk is not None:
                x = stk
            Tm = x * sim.T_out[k] + (1 - x) * T_ret
            T_sa = max(Tm, self.sat) + self.dTf_prev
        vb = F.get('vav_box_stuck', k)
        c = dict(k=k, occ=occ, T_sa=T_sa, T_ret=T_ret, W_ret=sim.W_air, vb=vb, so=so)
        self._ctx = c
        if so == 4:
            c['base'] = (0.0, 0.0, 0.0)
            c['q_leak'] = 0.0
            return c['base']
        rvl = F.get('reheat_valve_leak', k)
        # a leaking reheat valve heats whatever air passes the box (only while the box has flow)
        c['q_leak'] = q_leak = (rvl or 0.0) * self.c['Q_rh_design'] if (occ or vb is not None) else 0.0
        if vb is not None and (occ or so in (1, 3)):
            c['base'] = (vb * self.m_max * CP_AIR, T_sa, q_leak)
        elif occ:
            c['base'] = (self.m_min * CP_AIR, T_sa, q_leak)
        else:
            c['base'] = (0.0, 0.0, 0.0)
        return c['base']

    def _sa_with_stuck(self, k, pos, T_ret, occ):
        sim = self.sim
        x = self.m_oa / self.m_prev if occ else 0.0
        x = min(max(x, 0.0), 1.0)
        Tm = x * sim.T_out[k] + (1 - x) * T_ret
        eps = 0.8 * pos / (pos + 0.2) if pos > 0 else 0.0
        return Tm - eps * max(Tm - self.T_chw, 0.0) + self.dTf_prev

    def heat_stages(self):
        c = self._ctx
        if c['so'] == 4:
            return []
        if c['vb'] is not None and (c['occ']):
            m = c['vb'] * self.m_max
        else:
            m = self.m_heat
        ql = c.get('q_leak', 0.0)
        st = [(m * CP_AIR, self.T_rh_max, ql)]
        if self.Q_perim > 0:
            st.append((m * CP_AIR, self.T_rh_max, self.Q_perim + ql))
        return st

    def cool_stages(self):
        c = self._ctx
        if c['so'] == 4 or self.cool_plant is None:
            return []
        if c['vb'] is not None and c['occ']:
            return []
        la = self.F.get('low_airflow', c['k'])
        return [(self.m_max * (1 - (la or 0.0)) * CP_AIR, c['T_sa'], c.get('q_leak', 0.0))]

    def finish(self, k, q, T, mode, stage, frac):
        c, r, L, dt, sim, F = self._ctx, self.rec, self.led, self.dt, self.sim, self.F
        T_sa = c['T_sa']
        occ = c['occ']
        T_out, W_out = sim.T_out[k], sim.W_out[k]
        T_ret, W_ret = T, c['W_ret']          # return air = zone air of this step (zone solved first)
        Q_rh = 0.0
        q_leak = c.get('q_leak', 0.0)
        if mode == 0:
            g = c['base'][0]
            m = g / CP_AIR
            Ts_z = T_sa
            Q_rh = q_leak if m > 0 else 0.0
        elif mode == -1:
            base_m = c['base'][0] / CP_AIR
            m_top = self.m_max * (1 - (F.get('low_airflow', k) or 0.0))
            m = base_m + frac * (m_top - base_m)
            Ts_z = T_sa
            Q_rh = q_leak if m > 0 else 0.0
        else:
            m = self.heat_stages()[0][0] / CP_AIR
            Q_per = 0.0
            if stage == 1:
                Q_per = q - m * CP_AIR * (self.T_rh_max - T)
                q_air = q - Q_per
            else:
                q_air = q
            c['Q_perim'] = Q_per
            Ts_z = T + q_air / (m * CP_AIR)
            if occ or c['vb'] is not None:
                Q_rh = m * CP_AIR * (Ts_z - T_sa)
            else:
                Q_rh = None   # night heating: the AHU heats return air directly
        if m <= 1e-9:
            self._off(k, T_ret)
            return 0.0, 0.0
        # ---- AHU
        ff = m / self.m_max
        cf = F.get('coil_fouling', k)
        P_fan = self.P_fan_d * max(cv.cubic(self.fan_curve, max(ff, 0.2)), 0.05) * (1 + 0.5 * (cf or 0.0))
        dTf = P_fan / (m * CP_AIR)
        x_min = min(self.m_oa / m, 1.0) if (occ and c['so'] != 4) else 0.0
        leak = F.get('oa_damper_leak', k)
        if leak is not None:
            x_min = max(x_min, leak)
        stuck = F.get('oa_damper_stuck', k)
        night_heat = Q_rh is None
        T_c = (Ts_z if night_heat else T_sa) - dTf     # coil leaving target
        heating_mode = self.ahu_mode == 'heating' and occ and not night_heat
        if heating_mode:
            T_c = self.sat                               # preheat only protects the minimum supply temperature
        econ_ok = False
        if stuck is not None:
            x = stuck
        else:
            x = x_min
            if self.econ and occ and not heating_mode and F.get('economizer_disabled', k) is None:
                hl = F.get('economizer_high_limit', k)
                hl = self.hl if hl is None else hl
                To_s = T_out + (F.get('bias_T_out', k) or 0.0)
                Tr_s = T_ret + (F.get('bias_T_ret', k) or 0.0)
                Tm_bias = F.get('bias_T_mix', k) or 0.0
                if To_s < Tr_s and To_s < hl:
                    econ_ok = True
                    den = T_out - T_ret
                    x = (T_c - Tm_bias - T_ret) / den if abs(den) > 1e-9 else 1.0
                    x = min(max(x, x_min), 1.0)
        T_mix = x * T_out + (1 - x) * T_ret
        W_mix = x * W_out + (1 - x) * W_ret
        # coils in air-flow order: mixed air -> preheat coil -> cooling coil -> fan (draw-through).
        # Each coil holds the coil-leaving target T_c; a leaking / stuck coil is fought by the other one
        # (simultaneous heating and cooling, the classic signature).
        W_c = W_mix
        cvs = F.get('cooling_valve_stuck', k)
        cvl = F.get('cooling_valve_leak', k) or 0.0
        hvl = F.get('heating_valve_leak', k) or 0.0
        hvs = F.get('heating_valve_stuck', k)
        T_adp = self.T_chw + 3.5 + 5.0 * (cf or 0.0)
        mc = m * CP_AIR
        have_cc = self.cool_plant is not None
        # what the cooling coil does regardless of control (stuck or leaking valve): an air-side effectiveness
        # eps = 0.8 L / (L + 0.2) toward the chilled-water temperature (L = open fraction / leak), never below T_chw
        Lf = cvs if cvs is not None else cvl
        eps = (0.8 * Lf / (Lf + 0.2)) if (have_cc and Lf and Lf > 0) else 0.0
        def cc_forced(Tin):
            return eps * mc * max(Tin - self.T_chw, 0.0)
        # preheat: holds the cooling-coil leaving air >= T_c, also against a coil it cannot shut
        if hvs is not None:
            Q_ph = hvs * self.c['Q_ph_design']
        else:
            T_need = (T_c - eps * self.T_chw) / (1.0 - eps) if eps > 0 and T_c > self.T_chw else T_c
            need = max(T_need - T_mix, 0.0) * mc
            Q_ph = max(need, hvl * self.c['Q_ph_design'])
        T_ph = T_mix + Q_ph / mc
        if cvs is not None or heating_mode or not have_cc:
            Q_cc_s = cc_forced(T_ph)
        else:
            Q_cc_s = max(max(T_ph - T_c, 0.0) * mc, cc_forced(T_ph))
        T_after_cc = T_ph - Q_cc_s / mc
        if Q_cc_s > 0:
            # latent: ADP line from the coil-entering state toward the coil apparatus dew point
            if T_ph - T_adp > 0.1 and T_after_cc > T_adp:
                BF = (T_after_cc - T_adp) / (T_ph - T_adp)
                W_adp = ps.w_sat_s(T_adp)
                if W_adp < W_mix:
                    W_c = BF * W_mix + (1 - BF) * W_adp
        T_sup_actual = T_after_cc + dTf          # (preheat is already inside T_ph -> T_after_cc)
        # supply lag: difference between the supply temperature the zone solve used and the actual one
        T_used = Ts_z if night_heat else T_sa
        sup_lag = m * CP_AIR * (T_sup_actual - T_used)
        Q_lat = m * (W_mix - W_c) * ps.H_FG_ROOM
        Q_cc_tot = Q_cc_s + Q_lat
        self.cc_lim = 1.0
        # reheat (box)
        rvl = F.get('reheat_valve_leak', k)
        Q_rh = Q_rh or 0.0
        if night_heat:
            Q_rh = 0.0
        Q_per = c.get('Q_perim', 0.0)
        # ---- plants
        E_ce = E_cd = E_he = E_hg = E_hd = E_pump = E_rej = 0.0
        if isinstance(self.cool_plant, ChillerPlant):
            self.cool_plant.fouling = 5.0 * ((F.get('chiller_fouling', k) or 0.0) +
                                             (F.get('condenser_fouling', k) or 0.0))
        if isinstance(self.heat_plant, BoilerPlant):
            self.heat_plant.fouling = F.get('boiler_fouling', k) or 0.0
        if Q_cc_tot > 0 and self.cool_plant is not None:
            if isinstance(self.cool_plant, ChillerPlant):
                res = self.cool_plant.step(Q_cc_tot, T_out, sim.T_wb[k], dt)
                if res:
                    E_ce = res['P_chiller']
                    E_pump += res['P_pumps']
                    E_rej = res['P_tower']
                    L['in_elec'] += (res['P_chiller'] + res['P_pumps'] + res['P_tower']) * dt
                    L['from_air_water'] += Q_cc_tot * dt
                    L['to_outdoors'] += (Q_cc_tot + res['P_chiller'] + res['P_pumps'] + res['P_tower']) * dt
                    if res['unmet'] > 0:
                        self.cc_lim = 0.9
                        self.unmet_steps += 1
                        r['unmet_W'][k] += res['unmet']
                    r['cop_cool'][k] = res['cop']
                    r['plr_cool'][k] = res['plr']
            elif isinstance(self.cool_plant, District):
                res = self.cool_plant.step(Q_cc_tot)
                if res:
                    E_cd = res['meter']
                    E_pump += res['P_pump']
                    L['in_elec'] += res['P_pump'] * dt
                    L['in_dist'] += 0.0
                    L['from_air_water'] += Q_cc_tot * dt
                    L['to_outdoors'] += (Q_cc_tot + res['P_pump']) * dt
            elif isinstance(self.cool_plant, DXCoil):
                # packaged VAV: DX coil holds the coil-leaving temperature, power from its curves
                twb_in = ps.wetbulb_s(T_mix, W_mix)
                capft = cv.dx_cool_capft(twb_in, T_out)
                eirft = cv.dx_cool_eirft(twb_in, T_out)
                capm, eirm = charge_multipliers(F.get('refrigerant_charge', k) or 0.0)
                dTc = 8.0 * (F.get('condenser_fouling', k) or 0.0)
                cap = self.cool_plant.Qr * cv.dx_cool_capft(twb_in, T_out + dTc) * capm
                plr = min(Q_cc_tot / cap, 1.0) if cap > 0 else 1.0
                if Q_cc_tot > cap:
                    self.cc_lim = 0.9
                    r['unmet_W'][k] += Q_cc_tot - cap
                P_on = cap * cv.dx_cool_eirft(twb_in, T_out + dTc) * eirm / self.cool_plant.COPr
                E_ce = self.cool_plant.part_power(P_on, plr) if self.cool_plant.speed != 'single' else \
                    P_on * plr / cv.quad(cv.DX_PLF_TWO, plr)
                del capft, eirft
                L['in_elec'] += E_ce * dt
                L['from_air_water'] += Q_cc_tot * dt
                L['to_outdoors'] += (Q_cc_tot + E_ce) * dt
                r['plr_cool'][k] = plr
                r['cop_cool'][k] = Q_cc_tot / E_ce if E_ce > 0 else 0.0
        Q_hw = Q_ph + Q_rh + Q_per
        if Q_hw > 0:
            hp = self.heat_plant
            if self.reheat_elec and (Q_rh + Q_per) > 0:
                E_he += Q_rh + Q_per
                L['in_elec'] += (Q_rh + Q_per) * dt
                L['to_air_water'] += (Q_rh + Q_per) * dt
                Q_hw = Q_ph
            if Q_hw > 0:
                if isinstance(hp, BoilerPlant):
                    res = hp.step(Q_hw, T_out)
                    if res:
                        if hp.fuel == 'electric':
                            E_he += res['fuel']
                            L['in_elec'] += res['fuel'] * dt
                        else:
                            E_hg += res['fuel']
                            L['in_fuel'] += res['fuel'] * dt
                        E_pump += res['P_pump']
                        L['in_elec'] += res['P_pump'] * dt
                        L['to_air_water'] += Q_hw * dt
                        L['to_outdoors'] += (res['fuel'] + res['P_pump'] - Q_hw) * dt
                        r['plr_heat'][k] = res['plr']
                elif isinstance(hp, District):
                    res = hp.step(Q_hw)
                    if res:
                        E_hd += res['meter']
                        E_pump += res['P_pump']
                        L['in_dist'] += res['meter'] * dt
                        L['in_elec'] += res['P_pump'] * dt
                        L['to_air_water'] += Q_hw * dt
                        L['to_outdoors'] += (res['meter'] + res['P_pump'] - Q_hw) * dt
                else:   # electric coils
                    E_he += Q_hw
                    L['in_elec'] += Q_hw * dt
                    L['to_air_water'] += Q_hw * dt
        # ---- air-side ledger (exact): q = coils + fan + OA + return lag + supply lag
        L['to_zone'] += q * dt
        L['coil_heat'] += (Q_ph + Q_rh) * dt
        L['direct'] += Q_per * dt
        L['coil_cool_sens'] += Q_cc_s * dt
        L['fan_heat'] += P_fan * dt
        L['oa_sens'] += m * x * CP_AIR * (T_out - T) * dt
        L['return_lag'] += m * (1 - x) * CP_AIR * (T_ret - T) * dt
        L['supply_lag'] += -sup_lag * dt
        L['in_elec'] += P_fan * dt
        L['coil_latent_kg'] += m * (W_mix - W_c) * dt
        self.T_sa_prev = T_sup_actual
        self.last_mode = mode
        self.m_prev = m
        self.dTf_prev = dTf
        r['E_cool_elec'][k] = E_ce
        r['E_cool_dist'][k] = E_cd
        r['E_heat_elec'][k] = E_he
        r['E_heat_gas'][k] = E_hg
        r['E_heat_dist'][k] = E_hd
        r['E_pumps'][k] = E_pump
        r['E_rej'][k] = E_rej
        r['E_fans'][k] = P_fan
        r['P_fan'][k] = P_fan
        r['Q_heat'][k] = Q_ph + Q_rh + Q_per
        r['Q_cool_sens'][k] = Q_cc_s
        r['Q_cool_lat'][k] = Q_lat
        r['T_mix'][k] = T_mix
        r['T_sup'][k] = T_sup_actual
        r['T_ret'][k] = T_ret
        r['oa_frac'][k] = x
        r['fan_frac'][k] = ff
        r['m_sup'][k] = m
        r['cool_cmd'][k] = min(Q_cc_tot / self.Qcc_d, 1.0) if self.Qcc_d > 0 else 0.0
        r['heat_cmd'][k] = min(Q_ph / self.c['Q_ph_design'], 1.0) if self.c['Q_ph_design'] > 0 else 0.0
        r['reheat_cmd'][k] = min(Q_rh / self.c['Q_rh_design'], 1.0) if self.c['Q_rh_design'] > 0 else 0.0
        r['hvac_on'][k] = 1.0
        r['ahu_heating_mode'][k] = 1.0 if heating_mode else 0.0
        # moisture: the zone receives the supply humidity (heating does not change W)
        return m, W_c

    def _off(self, k, T_ret):
        self.last_mode = 0
        r = self.rec
        r['T_ret'][k] = T_ret
        r['T_mix'][k] = T_ret
        r['T_sup'][k] = T_ret
        self.m_prev = self.m_max * 0.3


# ------------------------------------------------------------------------------------ zonal / hydronic systems
class ZonalSystem(System):
    """Fan coils (4-pipe), radiant floors (heat into the mass node), electric baseboards; outdoor air from a
    dedicated outdoor-air unit enters the zone directly (zone ventilation term, energy booked here)."""

    def __init__(self, sim, cfg):
        super().__init__(sim)
        self.c = cfg
        self.inj = cfg.get('inj', 'air')
        self.Qh = cfg['Q_heat']
        self.Qc = cfg.get('Q_cool', 0.0)
        self.P_fan = cfg.get('P_fan', 0.0)
        self.heat_src = cfg['heat_src']          # BoilerPlant / District / 'elec' / ('hp', COP_rated)
        self.cool_plant = cfg.get('cool_plant')
        self.shr = cfg.get('shr', 0.75)
        self._ctx = None

    def plan(self, k):
        self._ctx = dict(k=k)
        return (0.0, 0.0, 0.0)

    def heat_stages(self):
        if self.F.get('schedule_override', self._ctx['k']) == 4:
            return []
        return [(0.0, 0.0, self.Qh)]

    def cool_stages(self):
        if self.cool_plant is None or self.Qc <= 0:
            return []
        return [(0.0, 0.0, -self.Qc)]

    def finish(self, k, q, T, mode, stage, frac):
        """q is the terminal output to the zone (coil + fan heat for fan coils)."""
        r, L, dt, sim = self.rec, self.led, self.dt, self.sim
        T_out = sim.T_out[k]
        E_he = E_hg = E_hd = E_ce = E_cd = E_p = E_rej = 0.0
        Pf = 0.0
        m_sup = W_sup = 0.0
        if q > 0:
            if self.P_fan > 0:
                Pf = self.P_fan * min(q / self.Qh, 1.0)
            Qc = q - Pf                       # coil heat
            r['Q_heat'][k] = Qc
            r['plr_heat'][k] = q / self.Qh
            src = self.heat_src
            if src == 'elec':
                E_he = Qc
                L['in_elec'] += Qc * dt
                L['to_air_water'] += Qc * dt
            elif isinstance(src, tuple) and src[0] == 'hp':
                cop = max(1.0, src[1] / cv.hp_heat_eirft(T_out)) if T_out > -25 else 1.0
                P = Qc / cop
                E_he = P
                L['in_elec'] += P * dt
                L['from_ambient'] += (Qc - P) * dt
                L['to_air_water'] += Qc * dt
                r['cop_heat'][k] = cop
            elif isinstance(src, BoilerPlant):
                res = src.step(Qc, T_out)
                if res:
                    if src.fuel == 'electric':
                        E_he = res['fuel']
                        L['in_elec'] += res['fuel'] * dt
                    else:
                        E_hg = res['fuel']
                        L['in_fuel'] += res['fuel'] * dt
                    E_p = res['P_pump']
                    L['in_elec'] += E_p * dt
                    L['to_air_water'] += Qc * dt
                    L['to_outdoors'] += (res['fuel'] + E_p - Qc) * dt
            elif isinstance(src, District):
                res = src.step(Qc)
                if res:
                    E_hd, E_p = res['meter'], res['P_pump']
                    L['in_dist'] += E_hd * dt
                    L['in_elec'] += E_p * dt
                    L['to_air_water'] += Qc * dt
                    L['to_outdoors'] += (E_hd + E_p - Qc) * dt
            L['direct'] += Qc * dt
        elif q < 0:
            if self.P_fan > 0:
                Pf = self.P_fan * min(-q / self.Qc, 1.0)
            Qs = -q + Pf                      # coil sensible removal
            Ql = 0.0
            W = sim.W_air
            if self.inj == 'air' and self.P_fan > 0:
                m_fc = self.c['m_fc']
                m_sup = m_fc * min(-q / self.Qc, 1.0)
                W_adp = ps.w_sat_s(self.c.get('T_adp', 10.0))
                Ql = max(0.0, min(Qs * (1 - self.shr) / self.shr, m_sup * (W - W_adp) * ps.H_FG_ROOM))
                W_sup = W - Ql / (ps.H_FG_ROOM * m_sup) if m_sup > 0 else W
            r['Q_cool_sens'][k] = Qs
            r['Q_cool_lat'][k] = Ql
            r['plr_cool'][k] = -q / self.Qc
            cp = self.cool_plant
            Qt = Qs + Ql
            if isinstance(cp, ChillerPlant):
                res = cp.step(Qt, T_out, sim.T_wb[k], dt)
                if res:
                    E_ce, E_p, E_rej = res['P_chiller'], res['P_pumps'], res['P_tower']
                    r['cop_cool'][k] = res['cop']
            elif isinstance(cp, District):
                res = cp.step(Qt)
                if res:
                    E_cd, E_p = res['meter'], res['P_pump']
            L['in_elec'] += (E_ce + E_p + E_rej) * dt
            L['from_air_water'] += Qt * dt
            L['to_outdoors'] += (Qt + E_ce + E_p + E_rej) * dt
            L['coil_latent_kg'] += Ql / ps.H_FG_ROOM * dt
            L['direct'] += -Qs * dt
        L['to_zone'] += q * dt
        L['fan_heat'] += Pf * dt
        L['in_elec'] += Pf * dt
        r['E_heat_elec'][k] = E_he
        r['E_heat_gas'][k] = E_hg
        r['E_heat_dist'][k] = E_hd
        r['E_cool_elec'][k] = E_ce
        r['E_cool_dist'][k] = E_cd
        r['E_pumps'][k] = E_p
        r['E_rej'][k] = E_rej
        r['E_fans'][k] = Pf
        r['P_fan'][k] = Pf
        r['hvac_on'][k] = 1.0 if q != 0 else 0.0
        r['T_ret'][k] = sim.T_air
        r['T_mix'][k] = sim.T_air
        r['T_sup'][k] = sim.T_air
        return m_sup, W_sup
