"""Subsystem models: engine, electric machine, battery, torque converter, cabin/HVAC.

Every model works on Python floats (for the per-second loop) and most also have numpy `*_vec` versions used when a
hybrid strategy scores many candidate operating points at once.

Sources (details in README.md):
- Engine fuel maps: US EPA ALPHA engine map packages (NVFEL/NCAT), resized at constant BMEP (ALPHA practice).
  Heavy-duty diesel: Willans-line map with Chen-Flynn-type friction (Heywood 2018, ch. 13), peak BTE ~44-45 %.
- Motor/inverter losses: US EPA ALPHA e-machine packages (ORNL/SwRI test data), scaled by P_max, w_max, T_max.
- Battery OCV: NMC811/graphite and LFP/graphite half-cell curves (Chen 2020; Prada 2013, as in PyBaMM),
  NiMH typical cell curve; resistance Arrhenius in temperature.
- Torque converter: K-factor / torque-ratio model (Gillespie 1992; Hrovat & Tobler 1985).
"""
import math
import numpy as np
from .maps import Grid2, Interp1, load_maps
from .constants import T0K
from . import calib

# ---------------------------------------------------------------- engine
_WILLANS_CACHE = {}


def willans_map(kind='hd_diesel'):
    """Parametric fuel-power map per litre of displacement for a heavy-duty diesel (or a generic SI fallback).

    fuel_W = (P_brake + P_friction) / eta_indicated(load, speed).
    FMEP [bar] = a + b n + c n^2 with n = rpm/1000 (Chen-Flynn form). Indicated efficiency falls at light load
    (heat transfer) and slightly at very high speed. Returns a dict shaped like the ALPHA-derived maps.
    """
    if kind in _WILLANS_CACHE:
        return _WILLANS_CACHE[kind]
    if kind == 'hd_diesel':
        rpm_pk0, rpm_pk1, rpm_rated, rpm_max, rpm_idle = 1000., 1400., 1800., 2100., 600.
        bmep_max = 21.0     # bar at peak torque (e.g. 15 L, 2500 Nm)
        eta0, a, b, c, n_best = 0.478, 0.55, 0.30, 0.14, 1.15
        fuel = 'diesel'
    else:   # generic SI naturally aspirated gasoline
        rpm_pk0, rpm_pk1, rpm_rated, rpm_max, rpm_idle = 3500., 4500., 6000., 6500., 650.
        bmep_max = 11.5
        eta0, a, b, c, n_best = 0.385, 0.75, 0.25, 0.10, 2.5
        fuel = 'gasoline'
    w = lambda r: r * math.pi / 30
    # full-load torque per litre (Nm/L): T = BMEP * Vd / (4 pi), Vd = 1e-3 m^3 per L
    Tpk = bmep_max * 1e5 * 1e-3 / (4 * math.pi)
    rpm_pts = np.array([0, rpm_idle, 0.5 * (rpm_idle + rpm_pk0), rpm_pk0, rpm_pk1, rpm_rated, rpm_max, rpm_max + 50])
    T_pts = Tpk * np.array([0, 0.55, 0.85, 1.0, 1.0, 0.0, 0.0, 0.0])
    P_rated = 0.80 * Tpk * w(rpm_rated)  # power at rated with 80% of peak torque
    T_pts[5] = P_rated / w(rpm_rated)
    T_pts[6] = 0.6 * T_pts[5]
    wft = w(rpm_pts); tft = T_pts
    wmax = w(rpm_max + 50)
    NW, NT = 48, 48
    wg = np.linspace(0, wmax, NW)
    tmin, tmax = -0.25 * Tpk, 1.05 * Tpk
    tg = np.linspace(tmin, tmax, NT)
    TT, WW = np.meshgrid(tg, wg, indexing='ij')
    n = WW * 30 / math.pi / 1000.0
    fmep = (a + b * n + c * n * n) * 1e5                      # Pa
    T_fric = fmep * 1e-3 / (4 * math.pi)                      # Nm per L
    load = np.clip((TT + T_fric) / (Tpk + T_fric), 0.0, 1.2)
    eta = eta0 * (1 - 0.30 * (1 - load) ** 3) * (1 - 0.04 * ((n - n_best) / n_best) ** 2)
    fuelW = np.where(TT + T_fric > 0, (TT + T_fric) * WW / np.maximum(eta, 0.05), 0.0)
    # closed throttle (motoring) torque = -friction torque
    wct = wg; tct = -(a + b * (wg * 30 / math.pi / 1000) + c * (wg * 30 / math.pi / 1000) ** 2) * 1e5 * 1e-3 / (4 * math.pi)
    out = dict(fuelW_per_L=fuelW, wft=wft, tft_per_L=tft, wct=wct, tct_per_L=tct, w_max=wmax,
               t_grid_per_L=[tmin, tmax, NT], w_grid=[0.0, wmax, NW], idle_radps=w(rpm_idle), fuel=fuel,
               label='Willans-line parametric %s map' % kind)
    _WILLANS_CACHE[kind] = out
    return out


def _map_source(map_id):
    if map_id.startswith('willans'):
        return willans_map('hd_diesel' if 'diesel' in map_id else 'si')
    maps, meta = load_maps()
    m = meta['engines'][map_id]
    k = 'eng_' + map_id
    return dict(fuelW_per_L=maps[k + '_fuelW_per_L'], wft=maps[k + '_wft'], tft_per_L=maps[k + '_tft_per_L'],
                wct=maps[k + '_wct'], tct_per_L=maps[k + '_tct_per_L'], w_max=m['w_max'], t_grid_per_L=m['t_grid_per_L'],
                w_grid=m['w_grid'], idle_radps=m['idle_radps'], fuel=m['fuel'], label=m['label'])


class Engine:
    """Map-based engine. Resized from its template map by an effective displacement so that the rated power matches the
    spec (constant BMEP, optional speed scaling by max_rpm). Fuel power is multiplied by a cold-start factor that decays
    as the lumped engine/coolant temperature rises."""

    def __init__(self, spec):
        self.spec = spec
        mid = spec['bsfc_map']
        src = _map_source(mid)
        self.map_id = mid
        self.label = src['label']
        wft = np.asarray(src['wft'], float); tft = np.asarray(src['tft_per_L'], float)
        P_perL = float(np.max(wft * tft))
        w_rated_map = float(wft[np.argmax(wft * tft)])
        s_w = 1.0
        if spec.get('max_rpm'):
            s_w = (spec['max_rpm'] * math.pi / 30) / float(np.max(wft[tft > 0]))
        P_spec = spec['max_power_kW'] * 1e3
        d_eff = P_spec / (P_perL * s_w)
        self.d_eff, self.s_w = d_eff, s_w
        self.disp = spec.get('displacement_L', d_eff)
        self.eff_scale = spec.get('eff_scale', 1.0)
        t0, t1, nt = src['t_grid_per_L']; w0, w1, nw = src['w_grid']
        F = np.asarray(src['fuelW_per_L'], float).T * d_eff * s_w / self.eff_scale     # (nw, nt) on (w, T)
        self.fuel_grid = Grid2(w0 * s_w, w1 * s_w, t0 * d_eff, t1 * d_eff, F)
        self.w_max = float(np.max(wft[tft > 0])) * s_w
        self.w_rated = w_rated_map * s_w
        wg = np.linspace(0, self.w_max, 96)
        self.tmax_curve = Interp1(0, self.w_max, np.interp(wg, wft * s_w, tft * d_eff))
        self.tct_curve = Interp1(0, self.w_max, np.interp(wg, np.asarray(src['wct'], float) * s_w,
                                                           np.asarray(src['tct_per_L'], float) * d_eff))
        self.T_peak = float(np.max(tft)) * d_eff
        self.P_max = P_spec
        self.w_idle = spec.get('idle_rpm', src['idle_radps'] * 30 / math.pi) * math.pi / 30
        self.fuel = spec.get('fuel', src['fuel'])
        self.w_min_load = max(self.w_idle * 1.15, spec.get('min_load_rpm', 1050 if self.fuel == 'gasoline' else 900) * math.pi / 30 * s_w)
        if self.T_peak > 1500:   # heavy-duty diesel lugs lower
            self.w_min_load = max(self.w_idle * 1.1, 850 * math.pi / 30)
        # warm-up model
        wu = spec.get('warmup_model', {}) or {}
        self.T_hot = wu.get('T_hot', 88.0)
        # cold-start penalty = extra friction torque (viscous oil, cold bearings) that fades linearly as the
        # coolant/oil warm from 20 C to T_hot; Nm per litre of displacement at 20 C
        self.c_fric = wu.get('cold_fric_Nm_per_L', calib.get('cold_fric_gasoline') if self.fuel == 'gasoline' else calib.get('cold_fric_diesel'))
        self.C_th = wu.get('C_J_K', 12e3 * self.disp + 12e3)
        self.hA = wu.get('hA_W_K', 8.0 * self.disp + 10.0)
        self.f_coolant = wu.get('f_coolant', 0.45)
        self._g20 = 1.0
        self._g20 = self._visc_g(20.0)
        self.idle_stop = bool(spec.get('idle_stop', False))
        # rotating inertia of crank/flywheel + transmission input (kg m2); Gillespie (1992) mass-factor practice
        self.J = spec.get('inertia_kgm2', (0.08 + 0.06 * self.disp) if self.T_peak < 1500 else 0.25 * self.disp)
        self._build_ool()

    # fuel power (W) for a running engine, warm
    def fuel_W(self, w, T):
        return self.fuel_grid(w, T)

    def fuel_W_vec(self, w, T):
        return self.fuel_grid.vec(w, T)

    def t_max(self, w):
        return self.tmax_curve(w)

    def t_min(self, w):
        return self.tct_curve(w)

    def _visc_g(self, T):
        # oil viscosity ratio nu(T)/nu(T_hot) (Arrhenius fit: ~12x between 90 C and 20 C for 0W/5W-30 oils);
        # film friction taken to scale with sqrt(viscosity)
        r = math.exp(3776.0 * (1.0 / (T + T0K) - 1.0 / (self.T_hot + T0K)))
        return math.sqrt(r) - 1.0

    def cold_dT(self, T_eng):
        if T_eng >= self.T_hot:
            return 0.0
        return self.c_fric * self.disp * self._visc_g(max(T_eng, -40.0)) / self._g20

    def fuel_cold(self, w, T, T_eng):
        if T_eng >= self.T_hot:
            return self.fuel_grid(w, T)
        return self.fuel_grid(w, T + self.cold_dT(T_eng))

    def fuel_cold_vec(self, w, T, T_eng):
        return self.fuel_grid.vec(w, T + self.cold_dT(T_eng))

    def thermal_step(self, T_eng, P_fuel, P_brake, T_amb, Q_heater, dt, running):
        q_in = self.f_coolant * max(P_fuel - P_brake, 0.0) if running else 0.0
        T_new = T_eng + dt * (q_in - self.hA * (T_eng - T_amb) - Q_heater) / self.C_th
        return min(T_new, self.T_hot + 2.0)   # thermostat + radiator hold the coolant near T_hot

    def _build_ool(self):
        """Optimal operating line: for each brake power, the speed with minimum fuel power (warm)."""
        P = np.linspace(0, self.P_max, 64)
        wg = np.linspace(self.w_min_load, self.w_max * 0.98, 160)
        tm = self.tmax_curve.vec(wg)
        w_ool = np.zeros_like(P); f_ool = np.zeros_like(P)
        for i, p in enumerate(P):
            T = p / wg
            ok = T <= 0.985 * tm
            if not ok.any():
                j = int(np.argmax(tm * wg)); w_ool[i] = wg[j]; f_ool[i] = self.fuel_W(wg[j], tm[j]); continue
            f = np.where(ok, self.fuel_grid.vec(wg, T), np.inf)
            j = int(np.argmin(f)); w_ool[i] = wg[j]; f_ool[i] = f[j]
        w_ool = np.maximum.accumulate(w_ool)             # monotone speed with power (smooth operation)
        f_ool = np.array([self.fuel_W(w, p / w) for w, p in zip(w_ool, P)])
        self.ool_w = Interp1(0, self.P_max, w_ool)
        self.ool_f = Interp1(0, self.P_max, f_ool)
        eff = np.where(f_ool > 0, P / np.maximum(f_ool, 1e-9), 0)
        self.P_best = float(P[int(np.argmax(eff))])   # most efficient power on the OOL
        self.eta_best = float(np.max(eff))


# ---------------------------------------------------------------- electric machine
class Motor:
    """Electric machine + inverter, losses from an ALPHA map scaled to (P_max, T_max, w_max)."""

    def __init__(self, spec):
        maps, meta = load_maps()
        mid = spec.get('eff_map', 'ipm_generic_150')
        m = meta['motors'][mid]
        k = 'mot_' + mid
        self.role = spec.get('role', 'traction')
        self.P_max = spec['max_power_kW'] * 1e3
        self.w_max = spec.get('max_rpm', m['w_max'] * 30 / math.pi) * math.pi / 30
        self.T_max = spec.get('max_torque_Nm', self.P_max / (0.3 * self.w_max))
        self.loss_scale = spec.get('loss_scale', calib.get('motor_loss_scale'))
        L = maps[k + '_loss_norm'].astype(float) * self.P_max * self.loss_scale
        self.loss_grid = Grid2(0.0, self.w_max, -self.T_max, self.T_max, L)
        wl = maps[k + '_wlim_norm'].astype(float); tl = maps[k + '_tlim_norm'].astype(float)
        wg = np.linspace(0, self.w_max, 64)
        tlim = np.minimum(np.interp(wg / self.w_max, wl, tl) * self.T_max, self.P_max / np.maximum(wg, 1e-6))
        tlim = np.minimum(tlim, self.T_max)
        self.tlim = Interp1(0, self.w_max, tlim)
        self.map_id = mid
        # at exactly zero torque the inverter stops switching: only iron + windage losses remain (~40 % of the
        # map's zero-torque loss, which includes inverter switching). Used for a freewheeling MG1 / idle P2 motor.
        self.freewheel_frac = spec.get('freewheel_frac', 0.4)

    def t_lim(self, w):
        if w > self.w_max:
            return 0.0
        return self.tlim(w)

    def t_lim_vec(self, w):
        return np.where(w > self.w_max * 1.0001, 0.0, self.tlim.vec(np.minimum(w, self.w_max)))

    def loss(self, w, T):
        if w < 0: w = -w; T = -T
        if T == 0.0:
            return self.freewheel_frac * self.loss_grid(w, 0.0)
        return self.loss_grid(w, T)

    def loss_vec(self, w, T):
        w = np.asarray(w, float); T = np.asarray(T, float)
        s = np.where(w < 0, -1.0, 1.0)
        L = self.loss_grid.vec(w * s, T * s)
        return np.where(T == 0.0, self.freewheel_frac * L, L)

    def elec(self, w, T):
        """electrical power at the DC bus (W, + = consumed) for shaft torque T at speed w."""
        return T * w + self.loss(w, T)

    def elec_vec(self, w, T):
        return T * w + self.loss_vec(w, T)


# ---------------------------------------------------------------- battery
def _ocv_nmc(s):
    # NMC811 / graphite (Chen et al. 2020, J. Electrochem. Soc. 167 080534; PyBaMM Chen2020 parameter set)
    x = 0.9084 - (0.9084 - 0.2661) * s
    y = 0.0279 + (0.9014 - 0.0279) * s
    Up = (-0.8090 * x + 4.4875 - 0.0428 * np.tanh(18.5138 * (x - 0.5542)) - 17.7326 * np.tanh(15.7890 * (x - 0.3117))
          + 17.5842 * np.tanh(15.9308 * (x - 0.3120)))
    Un = (1.9793 * np.exp(-39.3631 * y) + 0.2482 - 0.0909 * np.tanh(29.8538 * (y - 0.1234))
          - 0.04478 * np.tanh(14.9159 * (y - 0.2769)) - 0.0205 * np.tanh(30.4444 * (y - 0.6103)))
    return Up - Un


def _ocv_lfp(s):
    # LFP (Afshar 2017) / graphite (Chen 2020), as PyBaMM Prada2013 and CellSense
    x = 0.97 - 0.95 * s
    y = 0.02 + 0.80 * s
    Up = 3.4077 - 0.020269 * x + 0.5 * np.exp(-150 * x) - 0.9 * np.exp(-30 * (1 - x))
    Un = (1.9793 * np.exp(-39.3631 * y) + 0.2482 - 0.0909 * np.tanh(29.8538 * (y - 0.1234))
          - 0.04478 * np.tanh(14.9159 * (y - 0.2769)) - 0.0205 * np.tanh(30.4444 * (y - 0.6103)))
    return Up - Un


def _ocv_nimh(s):
    # typical NiMH cell open-circuit curve (1.22 V empty .. 1.42 V full)
    return 1.30 + 0.06 * (s - 0.5) * 2 + 0.05 * np.tanh((s - 0.95) * 20) - 0.07 * np.exp(-s * 15)


OCV_FUN = {'NMC': _ocv_nmc, 'NCA': _ocv_nmc, 'LFP': _ocv_lfp, 'NiMH': _ocv_nimh, 'LMO': _ocv_nmc}


class Battery:
    """Pack-level Thevenin model: OCV(SOC) - R0(SOC,T) I - V_rc, one RC branch, lumped thermal mass.

    Sign: I > 0 discharge. Losses = chemical power (OCV I) - terminal power (V I) = I^2 R0 + V_rc I.
    """

    def __init__(self, spec):
        chem = spec.get('chemistry', 'NMC')
        self.chem = chem
        self.E_nom = spec['capacity_kWh'] * 3.6e6
        self.V_nom = spec['nominal_V']
        s = np.linspace(0, 1, 101)
        ocv = OCV_FUN.get(chem, _ocv_nmc)(s)
        ocv_mean = float(np.trapezoid(ocv, s))
        self.ocv = Interp1(0.0, 1.0, ocv / ocv_mean * self.V_nom)       # pack OCV with mean = nominal V
        self.Ah = self.E_nom / 3600.0 / self.V_nom
        self.Q = self.Ah * 3600.0
        kind = spec.get('kind', 'power' if spec['capacity_kWh'] < 4 else 'energy')
        k = 0.012 if kind == 'power' else 0.04          # h, R0 = k V^2 / Wh (calibrated on Prius NiMH, 60 kWh EV packs)
        self.R0_25 = spec.get('R0_ohm') or k * self.V_nom ** 2 / (spec['capacity_kWh'] * 1e3)
        self.R1 = 0.6 * self.R0_25
        self.tau = spec.get('tau_s', 40.0)
        self.Ea = spec.get('Ea_J_mol', 22000.0)
        w = spec.get('soc_window', [0.1, 0.95])
        self.soc_lo, self.soc_hi = float(w[0]), float(w[1])
        self.v_min = self.ocv(0.0) * 0.92
        self.v_max = self.ocv(1.0) * 1.03
        pmax = spec.get('max_power_kW')
        self.P_dis_max = pmax * 1e3 if pmax else (25.0 if kind == 'power' else 3.0) * spec['capacity_kWh'] * 1e3
        self.P_chg_max = spec.get('max_charge_kW', self.P_dis_max / 1e3 * (0.8 if kind == 'power' else 0.6)) * 1e3
        th = spec.get('thermal', {}) or {}
        self.C_th = th.get('C_J_K', 1000.0 * spec['capacity_kWh'] * 1e3 / 150.0)   # ~ pack mass (150 Wh/kg) x 1 kJ/kgK
        self.hA = th.get('hA_W_K', 2.0 + 0.4 * spec['capacity_kWh'])
        self.heater_W = th.get('heater_W', 0.0)
        self.cooler_W = th.get('cooler_W', 0.0)
        self.T_heat_on = th.get('T_heat_on', 0.0)
        self.T_cool_on = th.get('T_cool_on', 35.0)

    def r0(self, soc, T):
        f = math.exp(self.Ea / 8.314 * (1.0 / (T + T0K) - 1.0 / 298.15))
        if soc < 0.2:
            f *= 1.0 + 1.5 * (0.2 - soc)
        return self.R0_25 * f

    def limits(self, soc, T, v_rc):
        """terminal power limits (W): (max discharge, max charge >0)."""
        V = self.ocv(soc) - v_rc
        R = self.r0(soc, T)
        # current limited by voltage window
        I_dis = max((V - self.v_min) / R, 0.0)
        p_dis = min(V * I_dis - R * I_dis * I_dis, V * V / (4 * R) * 0.95, self.P_dis_max)
        I_chg = max((self.v_max - V) / R, 0.0)
        p_chg = min(V * I_chg + R * I_chg * I_chg, self.P_chg_max)
        # temperature derating (Li-ion charging below 0 C) and SOC taper
        if T < 0.0:
            p_chg *= max(0.1, 1.0 + T / 20.0)
            p_dis *= max(0.3, 1.0 + T / 40.0)
        if soc > self.soc_hi:
            p_chg *= max(0.0, 1.0 - (soc - self.soc_hi) / 0.03)
        if soc < self.soc_lo:
            p_dis *= max(0.0, 1.0 - (self.soc_lo - soc) / 0.03)
        return p_dis, p_chg

    def current(self, P, soc, T, v_rc):
        """current for terminal power P (W, + discharge)."""
        V = self.ocv(soc) - v_rc
        R = self.r0(soc, T)
        disc = V * V - 4.0 * R * P
        if disc < 0.0:
            disc = 0.0
        return (V - math.sqrt(disc)) / (2.0 * R)

    def current_vec(self, P, soc, T, v_rc):
        V = self.ocv(soc) - v_rc
        R = self.r0(soc, T)
        disc = np.maximum(V * V - 4.0 * R * P, 0.0)
        return (V - np.sqrt(disc)) / (2.0 * R), V

    def ocv_vec(self, soc):
        return self.ocv.vec(soc)


# ---------------------------------------------------------------- torque converter
class TorqueConverter:
    """K-factor model. Capacity factor K = w_pump / sqrt(T_pump); torque ratio TR = T_turbine / T_pump vs speed ratio SR.
    Given turbine speed and torque, SR follows from w_t^2/T_t = SR^2 K(SR)^2 / TR(SR), inverted by table lookup."""

    def __init__(self, T_peak, stall_rpm=2300.0):
        self.K_stall = stall_rpm * math.pi / 30 / math.sqrt(T_peak)
        sr = np.linspace(0.0, 0.97, 200)
        tr = np.where(sr < 0.85, 2.0 - sr / 0.85, 1.0)
        K = self.K_stall * (1.0 + 1.2 * (sr / 0.9) ** 10)
        g = sr ** 2 * K ** 2 / tr
        self.g = g; self.sr = sr; self.tr = tr
        self.g_max = float(g[-1])

    def stall_equilibrium(self, w_t, eng, P_aux):
        """full-throttle operating point with the converter slipping: engine speed w_e such that the pump absorbs the
        engine's full torque, w_e = K(SR) sqrt(T_max(w_e) - P_aux/w_e), SR = w_t / w_e. Returns (w_e, T_pump, T_turbine)."""
        w_e = max(eng.w_idle * 1.5, w_t * 1.05)
        for _ in range(25):
            T_p = max(eng.t_max(w_e) - P_aux / w_e, 1.0)
            SR = min(w_t / w_e, 0.97)
            K = self.K_stall * (1.0 + 1.2 * (SR / 0.9) ** 10)
            w_new = K * math.sqrt(T_p)
            if w_new < w_t / 0.97:
                w_new = w_t / 0.97
            w_e = 0.5 * w_e + 0.5 * min(w_new, eng.w_max)
        T_p = max(eng.t_max(w_e) - P_aux / w_e, 1.0)
        SR = min(w_t / w_e, 0.97)
        TR = 2.0 - SR / 0.85 if SR < 0.85 else 1.0
        return w_e, T_p, TR * T_p

    def solve(self, w_t, T_t):
        """returns (w_pump, T_pump) for turbine (w_t >= 0, T_t > 0)."""
        x = w_t * w_t / T_t
        if x >= self.g_max:
            SR = 0.97
        else:
            SR = float(np.interp(x, self.g, self.sr))
        TR = 2.0 - SR / 0.85 if SR < 0.85 else 1.0
        T_p = T_t / TR
        if SR < 1e-3:
            w_p = self.K_stall * math.sqrt(T_p)
        else:
            w_p = w_t / SR
        return w_p, T_p


# ---------------------------------------------------------------- cabin / HVAC
class Cabin:
    """Lumped cabin: C dT/dt = Q_hvac + UA (T_amb - T) + Q_solar + Q_people.

    Heating sources: engine coolant (ICE; free heat, limited by coolant temperature), PTC resistive (COP 0.95) or heat
    pump (COP falling with ambient temperature, PTC backup). Cooling: vapour-compression A/C, COP falling with ambient.
    """

    def __init__(self, spec, heat_source):
        self.UA = spec.get('UA_W_K', 120.0)
        self.C = spec.get('C_J_K', 9.0e4)
        self.T_heat_set = spec.get('T_heat_set', 21.0)
        self.T_cool_set = spec.get('T_cool_set', 22.0)
        self.heat_on_below = spec.get('heat_on_below_C', 16.0)
        self.cool_on_above = spec.get('cool_on_above_C', 22.0)
        self.Q_heat_max = spec.get('heat_max_W', 6000.0)
        self.Q_cool_max = spec.get('cool_max_W', 5000.0)
        self.blower_W = spec.get('blower_W', 150.0)
        self.A_solar = spec.get('solar_area_m2', 0.45)
        self.Q_people = spec.get('occupant_W', 100.0)
        self.tau = spec.get('pulldown_s', 400.0)
        self.source = heat_source        # 'engine' | 'ptc' | 'heat_pump'
        self.hx_UA = spec.get('heater_core_W_K', 160.0)
        self.enabled = spec.get('enabled', True)

    @staticmethod
    def hp_cop(T_amb):
        # EV heat-pump COP vs ambient: ~3 at 10 C, ~2.2 at 0 C, ~1.4 at -10 C (e.g. Zhang et al. 2018; SAE 2019-01-1063)
        return min(3.5, max(1.0, 1.0 + 0.08 * (T_amb + 15.0)))

    @staticmethod
    def ac_cop(T_amb):
        # automotive R134a/R1234yf A/C COP falling with condenser (ambient) temperature
        return min(3.5, max(1.5, 3.3 - 0.06 * (T_amb - 25.0)))

    def demand(self, T_cab, T_amb, solar):
        """thermal power needed (W, + heat, - cool) to hold / approach the set point."""
        if not self.enabled:
            return 0.0
        q_gain = self.UA * (T_amb - T_cab) + solar * self.A_solar + self.Q_people
        if T_amb < self.heat_on_below:
            q = self.UA * (self.T_heat_set - T_amb) - solar * self.A_solar - self.Q_people + self.C * (self.T_heat_set - T_cab) / self.tau
            return min(max(q, 0.0), self.Q_heat_max)
        if T_amb > self.cool_on_above or T_cab > self.T_cool_set + 6.0:
            q = -(self.UA * (T_amb - self.T_cool_set) + solar * self.A_solar + self.Q_people) + self.C * (self.T_cool_set - T_cab) / self.tau
            return max(min(q, 0.0), -self.Q_cool_max)
        return 0.0

    def step(self, T_cab, T_amb, solar, Q, dt):
        return T_cab + dt * (Q + self.UA * (T_amb - T_cab) + solar * self.A_solar + self.Q_people) / self.C
