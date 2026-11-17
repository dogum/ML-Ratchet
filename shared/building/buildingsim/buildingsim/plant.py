"""Central plant: electric chillers (DOE-2 EIR model) with cooling towers or air-cooled condensers, hot-water
boilers, district steam / chilled water, pumps.

Chiller (DOE-2 / EnergyPlus Chiller:Electric:EIR): available capacity Q_avail = Q_ref CAPFT(T_chw, T_cond);
P = (Q_avail / COP_ref) EIRFT(T_chw, T_cond) EIRFPLR(PLR); below the minimum unloading ratio the compressor
false-loads (power at PLR_min). Condenser heat = evaporator load + compressor power (exact).
Cooling tower: Braun, Klein & Mitchell (1989) effectiveness model of a counterflow tower (Merkel assumptions):
the air-side effectiveness with the saturation-enthalpy slope c_s, NTU = c (m_w / m_a)^(1 + n) with n = -0.6 and
c fitted at the design point (wet bulb 25.6 C, range 5.6 K, approach 3.9 K). The fan modulates (cube law) to hold
the condenser-water setpoint (max(21 C, T_wb + 3 K)); at full speed the leaving temperature floats.
Boiler: nominal efficiency x normalised part-load curve (non-condensing) or the condensing efficiency vs
return-water temperature; hot-water supply reset with outdoor temperature.
Pumps: design power from W per L/s (DOE prototypes ~19 W/gpm = 300 W per L/s); variable speed uses the 90.1
Appendix G curve. Pump heat goes into the water (adds to chilled-water load, offsets boiler load).
"""
import math

from . import curves as cv
from . import psychro as ps

CP_W = 4186.0
W_PER_LPS = 301.0           # pump power at design per L/s of flow (19 W/gpm)


class ChillerPlant:
    def __init__(self, Q_design_W, cop=5.5, condenser='water', T_chw=6.67, n_chillers=None):
        self.Q_tot = max(Q_design_W, 1000.0)
        self.n = n_chillers or (2 if self.Q_tot > 1.0e6 else 1)
        self.Q_ref = self.Q_tot / self.n
        self.cop = cop
        self.condenser = condenser
        self.T_chw = T_chw
        self.flow_chw = self.Q_tot / (CP_W * 5.56)                  # kg/s at 10 F dT
        self.P_chwp = W_PER_LPS * self.flow_chw                       # constant primary
        self.P_cwp = W_PER_LPS * self.Q_tot * 1.2 / (CP_W * 5.56) if condenser == 'water' else 0.0
        self.m_w = self.Q_tot * 1.2 / (CP_W * 5.56)
        self.T_cw = 29.44
        self.fouling = 0.0          # condenser fouling fault: +K on condenser temperature
        self.cap_derate = 1.0
        if condenser == 'water':
            self.tower = CoolingTower(self.Q_tot * 1.25, self.m_w)

    def step(self, Q_load, T_out, T_wb, dt):
        """Q_load: chilled-water coil load [W] (>= 0). Returns dict of W rates."""
        if Q_load <= 1.0:
            return None
        Q_evap = Q_load + self.P_chwp                     # pump heat into the water
        if self.condenser == 'water':
            T_cond = self.T_cw
        else:
            T_cond = T_out + 0.0
        T_cond_eff = T_cond + self.fouling
        if self.condenser == 'air':
            T_eval = T_cond_eff - (cv.CH_RATED['T_cond_air'] - cv.CH_RATED['T_cond_water'])
        else:
            T_eval = T_cond_eff
        capft = cv.ch_capft(self.T_chw, T_eval)
        eirft = cv.ch_eirft(self.T_chw, T_eval)
        n_on = 1 if (self.n == 1 or Q_evap <= self.Q_ref * capft * self.cap_derate) else self.n
        Q_avail = n_on * self.Q_ref * capft * self.cap_derate
        unmet = max(0.0, Q_evap - Q_avail)
        Q_served = Q_evap - unmet
        plr = Q_served / Q_avail if Q_avail > 0 else 0.0
        plr_e = max(plr, cv.CH_MIN_PLR)
        P = Q_avail / self.cop * eirft * cv.quad(cv.CH_EIR_FPLR, plr_e)
        Q_false = (plr_e - plr) * Q_avail                 # hot-gas bypass false load (heat into condenser)
        Q_cond = Q_served + P
        P_cwp = self.P_cwp * n_on / self.n
        P_tower = 0.0
        if self.condenser == 'water':
            Q_rej = Q_cond + P_cwp
            self.T_cw, P_tower = self.tower.step(Q_rej, T_wb)
        else:
            Q_rej = Q_cond
        return dict(P_chiller=P, P_pumps=self.P_chwp + P_cwp, P_tower=P_tower, Q_evap=Q_served, Q_cond=Q_cond,
                    Q_rej=Q_rej, unmet=unmet, plr=plr, cop=Q_served / P if P > 0 else 0.0, T_cond=T_cond,
                    Q_false=Q_false, Q_pump_heat=self.P_chwp)


class CoolingTower:
    """Counterflow tower, Braun et al. (1989) effectiveness model; variable-speed fan."""

    def __init__(self, Q_design_W, m_w, wb_d=25.6, range_d=5.56, approach_d=3.89, LG=1.2):
        self.m_w = m_w
        self.m_a_d = m_w / LG
        self.P_fan_d = 0.0105 * Q_design_W        # ~ 0.012 kW per kW rejected (typical axial-fan towers)
        self.n = -0.6
        T_lv = wb_d + approach_d
        T_in = T_lv + range_d
        Q = m_w * CP_W * range_d
        # fit c so that the design point closes
        lo, hi = 0.05, 20.0
        for _ in range(60):
            c = 0.5 * (lo + hi)
            if self._Q(T_in, T_lv, wb_d, self.m_a_d, c) < Q:
                lo = c
            else:
                hi = c
        self.c = c
        self.T_set_min = 21.0

    def _eps(self, m_a, c, cs):
        NTU = c * (self.m_w / m_a) ** (1 + self.n)
        m_star = m_a * cs / (self.m_w * CP_W)
        if abs(1 - m_star) < 1e-6:
            return NTU / (1 + NTU)
        e = math.exp(-NTU * (1 - m_star))
        return (1 - e) / (1 - m_star * e)

    def _Q(self, T_in, T_lv, wb, m_a, c):
        h_a = ps.enthalpy_s(wb, ps.w_sat_s(wb))
        h_wi = ps.enthalpy_s(T_in, ps.w_sat_s(T_in))
        h_wo = ps.enthalpy_s(T_lv, ps.w_sat_s(T_lv))
        cs = (h_wi - h_wo) / max(T_in - T_lv, 1e-6)
        return self._eps(m_a, c, cs) * m_a * (h_wi - h_a)

    def step(self, Q_rej, T_wb):
        """Returns (leaving water temperature, fan power)."""
        rng = Q_rej / (self.m_w * CP_W)
        T_set = max(self.T_set_min, T_wb + 3.0)
        # air flow needed to hold T_set
        T_in = T_set + rng
        if self._Q(T_in, T_set, T_wb, self.m_a_d * 0.15, self.c) >= Q_rej:
            sp = 0.15
            # free convection / min speed already enough: leaving floats below set (approx: hold at set)
            return T_set, self.P_fan_d * sp ** 3
        if self._Q(T_in, T_set, T_wb, self.m_a_d, self.c) >= Q_rej:
            lo, hi = 0.15, 1.0
            for _ in range(14):
                sp = 0.5 * (lo + hi)
                if self._Q(T_in, T_set, T_wb, self.m_a_d * sp, self.c) >= Q_rej:
                    hi = sp
                else:
                    lo = sp
            sp = hi
            return T_set, self.P_fan_d * sp ** 3
        # full speed: find leaving temperature that rejects Q_rej
        lo, hi = T_set, T_set + 25.0
        for _ in range(16):
            T = 0.5 * (lo + hi)
            if self._Q(T + rng, T, T_wb, self.m_a_d, self.c) >= Q_rej:
                hi = T
            else:
                lo = T
        return hi, self.P_fan_d


class BoilerPlant:
    def __init__(self, Q_design_W, eff=0.80, kind='noncondensing', fuel='gas'):
        self.Q = max(Q_design_W, 1000.0)
        self.eff = eff
        self.kind = kind
        self.fuel = fuel
        self.flow = self.Q / (CP_W * 11.1)
        self.P_pump_d = W_PER_LPS * self.flow
        self.fouling = 0.0     # efficiency fault (fractional drop)

    def supply_temp(self, T_out):
        if self.kind == 'condensing':
            hi, lo = 60.0, 43.0
        else:
            hi, lo = 82.2, 60.0
        if T_out <= -6.7:
            return hi
        if T_out >= 10.0:
            return lo
        return hi + (lo - hi) * (T_out + 6.7) / 16.7

    def step(self, Q_load, T_out):
        if Q_load <= 1.0:
            return None
        plr_flow = min(Q_load / self.Q, 1.0)
        P_pump = self.P_pump_d * max(cv.cubic(cv.VS_PUMP, max(plr_flow, 0.25)), 0.02)
        Q_boiler = max(Q_load - P_pump, 0.0)
        plr = min(Q_boiler / self.Q, 1.0)
        unmet = max(0.0, Q_boiler - self.Q)
        Q_served = Q_boiler - unmet
        if self.fuel == 'electric':
            eta = 0.98
        elif self.kind == 'condensing':
            T_ret = self.supply_temp(T_out) - 11.1 * max(plr, 0.3) ** 0.5
            eta = cv.condensing_eff_factor(T_ret) * (1.0 + 0.02 * (1 - plr))
        else:
            eta = self.eff * cv.quad(cv.BOILER_PLR, max(plr, 0.1)) / cv.quad(cv.BOILER_PLR, 1.0)
        eta *= (1.0 - self.fouling)
        fuel = Q_served / eta
        return dict(fuel=fuel, P_pump=P_pump, Q_served=Q_served, unmet=unmet, eta=eta, plr=plr,
                    loss=fuel - Q_served, Q_pump_heat=P_pump)


class District:
    """District steam / hot water / chilled water: the building meter reads the delivered thermal energy divided
    by an in-building distribution efficiency; secondary pumps are electric."""

    def __init__(self, Q_design_W, eff=1.0, kind='steam'):
        self.Q = max(Q_design_W, 1000.0)
        self.eff = eff
        self.kind = kind
        dT = 5.56 if kind == 'chw' else 11.1
        self.P_pump_d = W_PER_LPS * self.Q / (CP_W * dT) * (0.5 if kind == 'steam' else 1.0)

    def step(self, Q_load):
        if Q_load <= 1.0:
            return None
        plr = min(Q_load / self.Q, 1.0)
        P_pump = self.P_pump_d * max(cv.cubic(cv.VS_PUMP, max(plr, 0.25)), 0.02)
        if self.kind == 'chw':
            Q_meter = (Q_load + P_pump) / self.eff
        else:
            Q_meter = max(Q_load - P_pump, 0.0) / self.eff
        return dict(meter=Q_meter, P_pump=P_pump, loss=Q_meter - (Q_load + P_pump if self.kind == 'chw'
                                                                     else max(Q_load - P_pump, 0.0)))
