"""The time loop: simulate(spec, weather, start, end, dt, controller, faults, prices, init, rng).

Per step: boundary conditions and gains (precomputed arrays) -> infiltration from the current indoor/outdoor
difference -> zone linear response T_air = a + b q (two evaluations of the 5R1C equations) -> setpoints (schedule,
setback, faults, controller) -> HVAC plan (base state + capacity stages) -> exact thermostat solve -> zone advance
-> HVAC finish (equipment energy, coil loads, air-side and equipment ledgers) -> moisture balance -> battery.
"""
import math
import time

import numpy as np
import pandas as pd
import pvlib

from . import psychro as ps
from . import schedules as sch
from . import weather as wxm
from .comfort import pmv_ppd
from .faults import compile_faults, FaultView
from .ledger import build_ledger
from .sizing import design_loads, build_system
from .spec import fill_spec
from .systems import thermostat_solve, REC
from .zone import (CP_AIR, Zone5R1C, geometry, boundary, window_solar, infiltration_coeffs)


OPTIONS = {
    'holidays': "public-holiday calendar: 'auto' (country from the weather tz) or 'US' / 'CA' / 'UK' / 'IE'; "
                "absent = v1.0 US calendar (LoadCast wrapper)",
}


def check_options(options):
    """Opt-in Phase 2 flags that act inside the time loop (README, 'Phase 2 wrappers'). None or {} = v1.0."""
    opts = dict(options or {})
    bad = sorted(set(opts) - set(OPTIONS))
    if bad:
        raise ValueError(f'unknown options {bad}; known: {sorted(OPTIONS)}')
    return opts


def _prepare_kw(opts, meta):
    if opts.get('holidays') is None:
        return {}
    from .holidays import resolve
    c = resolve(opts['holidays'], meta.get('tz', 'UTC'))
    return {} if c is None else dict(holidays=c)


def _weather_args(weather, spec):
    if isinstance(weather, str):
        return wxm.read_epw(weather)
    if isinstance(weather, tuple):
        return weather
    if isinstance(weather, dict) and 'df' in weather:
        return weather['df'], weather['meta']
    if isinstance(weather, pd.DataFrame):
        meta = dict(weather.attrs.get('meta', {}))
        for k in ('lat', 'lon', 'elev', 'tz'):
            if k not in meta and spec.get(k) is not None:
                meta[k] = spec[k]
        meta.setdefault('elev', 0.0)
        meta.setdefault('tz', 'UTC')
        if 'lat' not in meta:
            raise ValueError('weather meta needs lat/lon (pass (df, meta) or set df.attrs["meta"])')
        return weather, meta
    raise TypeError('weather: EPW path, (DataFrame, meta) tuple, or DataFrame with attrs["meta"]')


class Simulator:
    def __init__(self, spec, weather, start=None, end=None, dt='1h', controller=None, faults=None, prices=None,
                 init=None, rng=None, warmup_days=7, record_trend=True, sensor_noise=None, options=None):
        t0 = time.time()
        self.options = opts = check_options(options)
        df, meta = _weather_args(weather, spec)
        s0 = dict(spec)
        if s0.get('lat') is None:
            s0['lat'], s0['lon'] = meta['lat'], meta['lon']
        self.spec = s = fill_spec(s0)
        pv = s.get('pv')
        self.wx = wx = wxm.prepare(df, meta, start, end, dt, orientation=s['geometry']['orientation_deg'],
                                   pv=[(pv['tilt'], pv['azimuth'])] if pv else None,
                                   **_prepare_kw(opts, meta))
        self.n = n = wx['n']
        self.dt = dt_s = wx['dt_s']
        self.times = wx['times']
        self.rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        self.geo = geo = geometry(s)
        self.bnd = bnd = boundary(s, geo, wx)
        self.phi_sol_facade, self.phi_sol = window_solar(s, geo, wx)
        self.phi_sol = np.asarray(self.phi_sol, float) * np.ones(n)
        if s.get('zone_model', '5R1C') == 'multinode':
            from .zone_multinode import ZoneMultiNode
            self.zone = zone = ZoneMultiNode(s, geo, bnd, n, dt_s)
        else:
            self.zone = zone = Zone5R1C(s, geo, bnd, n, dt_s)
        self.multinode = getattr(zone, 'kind', '') == 'multinode'
        th_em = zone.w_wall * bnd['T_wall'] + zone.w_roof * bnd['T_roof'] + zone.w_gnd * bnd['T_ground']
        self.th_em = th_em
        # faults
        self.fault_arrays, self.fault_labels = compile_faults(faults, self.times)
        self.F = FaultView(self.fault_arrays)
        # schedules
        sc = sch.build(s['occupancy_schedule'], wx['cal'])
        so = self.fault_arrays.get('schedule_override')
        hv = sc['hvac'].copy()
        light = sc['light'].copy()
        if so is not None:
            hv = np.where(np.isin(so, (1, 3)), 1.0, hv)
            light = np.where(np.isin(so, (2, 3)), np.maximum(light, 0.9), light)
        self.sched = sc
        self.hvac_sched = hv
        res = s['primary_use'].startswith('residential')
        self.res = res
        A = geo['A_f']
        ig = s['internal_gains']
        if res:
            ig['people_W_m2'] = ig['occupants'] * ig['people'] / A
        p_tot = ig['people_W_m2'] * A * sc['occ']
        lat_frac = ig['people_latent_frac']
        self.P_light = ig['lighting'] * A * light
        self.P_plug = ig['plug'] * A * sc['plug'] + ig['process'] * A
        cs = np.clip(sc['occ'] / max(sc['occ'].max(), 1e-6), 0, 1)
        cook = ig['cooking'] * A * cs if ig['cooking'] else np.zeros(n)
        if ig.get('cooking_kWh_yr'):        # annual appliance gas (ranges, dryers) spread on occupancy
            cook = cook + ig['cooking_kWh_yr'] * 1000.0 / 8760.0 * cs / max(cs.mean(), 1e-6)
        self.cook = cook
        to_space = ig['cooking_to_space']
        dark = wx['zenith'] > 90
        self.P_ext = ig['exterior_lighting'] * A * dark
        # DHW (vectorised)
        self._dhw(s, sc, wx)
        phi_int = p_tot * (1 - lat_frac) + self.P_light + self.P_plug + cook * to_space * 2 / 3 + self.dhw_to_zone
        self.G_lat = p_tot * lat_frac / ps.H_FG_ROOM + cook * to_space / 3 / ps.H_FG_ROOM     # kg/s
        self.phi_int = phi_int
        # ventilation
        vent = s['ventilation']
        n_people_peak = ig['people_W_m2'] * A / max(ig['people'], 1.0) if ig['people'] else 0.0
        self.V_oa_design = (vent['L_s_person'] * n_people_peak + vent['L_s_m2'] * A) / 1000.0
        sysn = s['system']
        self.oa_through_system = sysn in ('ptac', 'packaged_rtu', 'vav_chiller_boiler', 'district_steam_chw') or \
            (sysn in ('heat_pump_air', 'heat_pump_ground', 'split_ac_furnace') and not res)
        self.erv_eff = vent['erv_eff'] if vent['erv'] else 0.0
        self.rho = (ps.rho_dry_per_m3(wx['T_out'], wx['W_out'], wx['P'])).tolist()
        self.ela, self.Cs, CwU2 = infiltration_coeffs(s, geo, wx)
        self.ela *= float(s.get('infiltration_factor') or 1.0)
        self.Cw0 = float(CwU2[0] / max(wx['wind'][0] ** 2, 1e-9)) if wx['wind'][0] > 0 else \
            float(np.median(CwU2 / np.maximum(wx['wind'] ** 2, 1e-9)))
        if s.get('infiltration_model') == 'constant':
            # fixed air change rate (BESTEST-style): Q = ACH V / 3600, independent of weather
            self.ela, self.Cs = float(s['infiltration_ach']) * geo['volume'] / 3.6, 0.0
            CwU2 = np.ones(n)
            self.Cw0 = 0.0
        self.CwU2 = CwU2.tolist()
        # natural ventilation (window opening): cooling season (30-day mean T_out >= 15 C), dry enough outside
        # (W < 0.0115, as OS-HPXML), and later in the loop: zone warmer than outdoors + 1 K and above mid-deadband
        self.nv_area = 0.0
        if vent.get('natural_ventilation') and sum(geo['win'].values()) > 0:
            rm = pd.Series(wx['T_out']).rolling(int(30 * 86400 / dt_s), min_periods=1).mean().values
            self.nv_ok = ((rm >= 15.0) & (wx['W_out'] < 0.0115)).tolist()
            self.nv_area = float(vent['nv_open_frac']) * sum(geo['win'].values()) * 1e4 * 0.5   # cm2 x C_d
        else:
            self.nv_ok = None
        inf_fac = np.ones(n)
        if not res and sysn not in ('none', 'ideal', 'baseboard_resistance', 'radiant', 'fan_coil',
                                    'hydronic_baseboard', 'wall_furnace'):
            inf_fac = np.where(hv > 0.5, 0.25, 1.0)
        if s.get('infiltration_schedule') == 'constant':
            inf_fac = np.ones(n)
        self.inf_fac = inf_fac.tolist()
        self.Q_exh = 0.0
        self.P_mech_vent = np.zeros(n)
        m_v = np.zeros(n)
        T_v = wx['T_out'].copy()
        W_v = wx['W_out'].copy()
        self.doas_heat_W = np.zeros(n)
        self.doas_cool_W = np.zeros(n)
        if not self.oa_through_system and self.V_oa_design > 0 and sysn != 'none':
            if res:
                if vent['mech_vent']:
                    self.Q_exh = self.V_oa_design
                    self.P_mech_vent = np.full(n, 0.74 * self.V_oa_design * 1000.0)
            elif sysn in ('fan_coil', 'radiant', 'ideal', 'baseboard_resistance', 'hydronic_baseboard',
                          'wall_furnace'):
                occ_on = (hv > 0.5).astype(float)
                m_v = 1.2 * self.V_oa_design * occ_on
                if sysn != 'ideal':
                    T_ret = np.full(n, s['setpoints']['heat'] + 1.0)
                    T_rec = wx['T_out'] + self.erv_eff * (T_ret - wx['T_out'])
                    T_v = np.clip(T_rec, 18.0, 21.0)
                    W_v = np.minimum(wx['W_out'], ps.w_sat(12.0))
                    self.doas_heat_W = m_v * CP_AIR * np.maximum(T_v - T_rec, 0)
                    self.doas_cool_W = m_v * (CP_AIR * np.maximum(T_rec - T_v, 0) +
                                              np.maximum(wx['W_out'] - W_v, 0) * ps.H_FG_ROOM)
                    self.P_mech_vent = 0.0015 * 1000 * self.V_oa_design * occ_on * 0.6   # DOAS fan 0.9 W/(L/s)
        self.m_v, self.T_v, self.W_v = m_v.tolist(), T_v.tolist(), W_v.tolist()
        # setpoints
        self._setpoints(s, sc, hv)
        # state
        init = dict(init or {})
        T0 = init.get('T_air', 0.5 * (self.hsp[0] + self.csp[0]) if self.hsp[0] > -50 else 20.0)
        if self.multinode:
            zone.theta_m = np.full(zone.n, float(init.get('T_mass', T0)))
        else:
            zone.theta_m = init.get('T_mass', T0)
        self.T_air = T0
        self.W_air = init.get('W_air', float(ps.w_from_rh(T0, 45.0, wx['P'][0])))
        self.V = geo['volume']
        self.M_moist = 1.2 * self.V * s['moisture_capacitance']
        # lists for the loop
        self.T_out = wx['T_out'].tolist()
        self.W_out = wx['W_out'].tolist()
        self.T_wb = wx['T_wb'].tolist()
        self.T_ground_deep = wx['T_ground_deep'].tolist()
        self._th_em = th_em.tolist()
        self._phi_int = phi_int.tolist()
        self._phi_sol = self.phi_sol.tolist()
        self._G_lat = self.G_lat.tolist()
        self.hvac_on = hv.tolist()
        self._T_wall = bnd['T_wall'].tolist()
        self._T_roof = bnd['T_roof'].tolist()
        self._T_gnd = bnd['T_ground'].tolist()
        # system
        self.design = design_loads(self)
        self.system = build_system(self, self.design)
        # PV, battery, prices
        self._pv(s, wx)
        self.bat = s.get('battery')
        self.soc = float(init.get('soc', self.bat['soc0'] if self.bat else 0.0))
        self.prices = _price_array(prices, self.times, wx['cal']) if prices is not None else None
        self.controller = controller
        self.record_trend = record_trend
        self.sensor_noise = sensor_noise
        self.warmup_steps = int(round(warmup_days * 86400 / dt_s)) if warmup_days else 0
        self.warmup_steps = min(self.warmup_steps, n)
        self.setup_s = time.time() - t0
        self.init_state = init
        self._init_records()          # (re-initialised after the warm-up)

    # ------------------------------------------------------------------ helpers built once
    def _dhw(self, s, sc, wx):
        d = s['dhw']
        n = self.n
        f = sc['dhw'] / max(sc['dhw'].mean(), 1e-9)
        Lps = d['L_per_day'] / 86400.0 * f                      # L/s
        Tm = wx['T_mains']
        Q_use = Lps * 0.998 * ps.CP_W * np.maximum(d['T_set'] - Tm, 0)        # W
        T_amb = 21.0
        Q_sb = np.full(n, d['UA_W_K'] * (d['T_set'] - T_amb))
        Q = Q_use + Q_sb
        fuel = d['fuel']
        eff = d['efficiency']
        self.dhw_elec = np.zeros(n)
        self.dhw_gas = np.zeros(n)
        self.dhw_dist = np.zeros(n)
        if fuel == 'gas':
            self.dhw_gas = Q / eff
        elif fuel == 'heat_pump':
            cop = eff * (1 + 0.02 * (T_amb - 19.7))
            self.dhw_elec = Q / cop
        elif fuel == 'district':
            self.dhw_dist = Q / eff
        else:
            self.dhw_elec = Q / eff
        inside = d.get('in_conditioned_space', True)
        to_zone = Q_sb.copy() if inside else np.zeros(n)
        if fuel == 'heat_pump' and inside:
            to_zone = to_zone - (Q - self.dhw_elec)
        self.dhw_to_zone = to_zone
        self.dhw_use_W = Q_use
        self.dhw_standby_W = Q_sb

    def _setpoints(self, s, sc, hv):
        sp = s['setpoints']
        n = self.n
        heat, cool = sp['heat'], sp['cool']
        sbh = sp['setback_heat'] if sp['setback_heat'] is not None else heat
        sbc = sp['setback_cool'] if sp['setback_cool'] is not None else cool
        if self.res:
            mode = sp.get('schedule', 'constant')
            if mode == 'night_setback':
                night = sc['asleep'] > 0.5
                h = np.where(night, sbh, heat)
                c = np.where(night, sbc, cool)
            elif mode == 'away_setback':
                aw = sc['away'] > 0.5
                h = np.where(aw, sbh, heat)
                c = np.where(aw, sbc, cool)
            elif mode == 'setback':
                m = (sc['asleep'] > 0.5) | (sc['away'] > 0.5)
                h = np.where(m, sbh, heat)
                c = np.where(m, sbc, cool)
            else:
                h = np.full(n, heat)
                c = np.full(n, cool)
        else:
            occ = hv > 0.5
            h = np.where(occ, heat, sbh)
            c = np.where(occ, cool, sbc)
        rh = sp.get('recovery_h', 0.0) or 0.0
        if rh > 0:
            r = int(round(rh * 3600 / self.dt))
            hh, cc = h.copy(), c.copy()
            for k in range(n - 2, -1, -1):
                if h[k] < h[k + 1]:
                    for j in range(1, r + 1):
                        if k - j + 1 < 0:
                            break
                        hh[k - j + 1] = max(hh[k - j + 1], h[k] + (h[k + 1] - h[k]) * (1 - j / (r + 1)))
                if c[k] > c[k + 1]:
                    for j in range(1, r + 1):
                        if k - j + 1 < 0:
                            break
                        cc[k - j + 1] = min(cc[k - j + 1], c[k] - (c[k] - c[k + 1]) * (1 - j / (r + 1)))
            h, c = hh, cc
        oh = self.fault_arrays.get('sp_heat')
        oc = self.fault_arrays.get('sp_cool')
        if oh is not None:
            h = np.where(np.isnan(oh), h, oh)
        if oc is not None:
            c = np.where(np.isnan(oc), c, oc)
        c = np.maximum(c, h + 0.5)
        self.hsp_sched = h
        self.csp_sched = c
        self.hsp = h.tolist()
        self.csp = c.tolist()

    def _pv(self, s, wx):
        pv = s.get('pv')
        n = self.n
        if not pv:
            self.pv_W = np.zeros(n)
            return
        sfc = wx['pv'][0]
        iam = np.nan_to_num(pvlib.iam.physical(np.clip(sfc['aoi'], 0, 90)))
        poa = sfc['beam'] * iam + 0.95 * (sfc['sky'] + sfc['ground'])
        Tc = pvlib.temperature.faiman(poa, wx['T_out'], wx['wind'])
        pdc0 = pv['kW'] * 1000.0
        pdc = pvlib.pvsystem.pvwatts_dc(poa, Tc, pdc0, pv['gamma'])
        pdc = np.clip(np.nan_to_num(pdc), 0, None) * (1 - pv['losses'])
        pac = np.minimum(pdc * 0.96, pdc0 / pv['dc_ac'])
        self.pv_W = pac
        self.pv_poa = poa

    # ------------------------------------------------------------------ the loop
    def run(self):
        t0 = time.time()
        if self.warmup_steps > 0:
            ctrl = self.controller
            self.controller = None
            soc0 = self.soc
            self._loop(0, self.warmup_steps, record=False)
            self.controller = ctrl
            self.soc = soc0
            self._reset_accumulators()
        self._init_records()
        self._loop(0, self.n, record=True)
        wall = time.time() - t0
        return self._outputs(wall)

    # ------------------------------------------------------------------ step-wise use (ThermoPilot / MPC / RL)
    def reset(self):
        """Warm up, clear the ledgers and return the first observation. Then call step(action) n times."""
        self._t0 = time.time()
        if self.warmup_steps > 0:
            ctrl = self.controller
            self.controller = None
            soc0 = self.soc
            self._loop(0, self.warmup_steps, record=False)
            self.controller = ctrl
            self.soc = soc0
            self._reset_accumulators()
        self._init_records()
        self.k = 0
        return self._obs(0)

    def step(self, action=None):
        """Advance one time step with an action dict (keys as for controller return values). Returns
        (obs for the next step or None at the end, info) where info has the step's electricity kW, gas kW, zone
        temperature and price."""
        k = self.k
        ctrl = self.controller
        self.controller = (lambda obs: action) if action else None
        self._loop(k, k + 1, record=True)
        self.controller = ctrl
        r = self.system.rec
        be = self._base_elec_list()[k] + self.pv_W[k]
        el = (be + r['E_heat_elec'][k] + r['E_cool_elec'][k] + r['E_fans'][k] + r['E_pumps'][k] + r['E_rej'][k]
              + self.r_bat[k] - self.pv_W[k]) / 1000.0
        info = dict(k=k, electricity_kW=el, gas_kW=(r['E_heat_gas'][k] + self.dhw_gas[k]) / 1000.0,
                    T_zone=self.r_T[k], q_hvac_kW=self.r_q[k] / 1000.0,
                    price=self.prices[k] if self.prices is not None else None)
        self.k = k + 1
        return (self._obs(self.k) if self.k < self.n else None), info

    def finish(self):
        """Outputs after step-wise use (same dict as simulate())."""
        return self._outputs(time.time() - getattr(self, '_t0', time.time()))

    def get_state(self):
        """Dynamic state (zone, battery, system internals) for MPC rollouts: set_state() restores it. Ledger
        accumulators and records are not part of the state."""
        sysd = {k: v for k, v in self.system.__dict__.items() if isinstance(v, (int, float, str, bool))}
        plants = {}
        for nm in ('cool_plant', 'heat_plant'):
            pl = getattr(self.system, nm, None)
            if pl is not None and hasattr(pl, '__dict__'):
                plants[nm] = {k: v for k, v in pl.__dict__.items() if isinstance(v, (int, float))}
        import copy as _c
        return dict(k=getattr(self, 'k', 0), theta_m=_c.copy(self.zone.theta_m), T_air=self.T_air, W_air=self.W_air,
                    soc=self.soc, system=sysd, plants=plants)

    def set_state(self, st):
        self.k = st['k']
        self.zone.theta_m = st['theta_m']
        self.T_air, self.W_air, self.soc = st['T_air'], st['W_air'], st['soc']
        for k, v in st['system'].items():
            setattr(self.system, k, v)
        for nm, d in st['plants'].items():
            pl = getattr(self.system, nm)
            for k, v in d.items():
                setattr(pl, k, v)

    def _reset_accumulators(self):
        sysm = self.system
        for k in sysm.led:
            sysm.led[k] = 0.0
        n = self.n
        sysm.rec = {k: [0.0] * n for k in REC}
        if hasattr(sysm, 'unmet_steps'):
            sysm.unmet_steps = 0

    def _init_records(self):
        n = self.n
        self.r_T = [0.0] * n
        self.r_Ts = [0.0] * n
        self.r_Tm = [0.0] * n
        self.r_W = [0.0] * n
        self.r_q = [0.0] * n
        self.r_mode = [0] * n
        self.r_hsp = [0.0] * n
        self.r_csp = [0.0] * n
        self.r_bat = [0.0] * n
        self.r_soc = [0.0] * n
        self.r_inf = [0.0] * n
        self.r_elec = [0.0] * n
        self.r_cond = [0.0] * n
        self.L = {k: 0.0 for k in ('win', 'win_rad', 'wall', 'roof', 'ground', 'inf', 'vent', 'int', 'sol', 'hvac', 'storage',
                                   'm_inf', 'm_vent', 'm_sup', 'm_lat', 'm_storage', 'm_cond',
                                   'bat_in', 'bat_out', 'bat_loss', 'bat_store')}
        self.soc_start = self.soc

    def _loop(self, k0, k1, record=True):
        z = self.zone
        sysm = self.system
        solve = z.solve
        F = self.F
        dt = self.dt
        Tout, Wout, rho = self.T_out, self.W_out, self.rho
        th_em_a, phi_int_a, phi_sol_a, G_lat_a = self._th_em, self._phi_int, self._phi_sol, self._G_lat
        CwU2, inf_fac, ela, Cs = self.CwU2, self.inf_fac, self.ela, self.Cs
        m_v_a, T_v_a, W_v_a = self.m_v, self.T_v, self.W_v
        hsp_a, csp_a = self.hsp, self.csp
        Qexh2 = self.Q_exh ** 2
        nv_ok, nv_area = self.nv_ok, self.nv_area
        Cs_nv = self.Cs if self.Cs > 0 else 0.000145
        M = self.M_moist
        inj = sysm.inj
        has_tbias = F.has('bias_T_zone')
        ctrl = self.controller
        L = self.L if record else None
        Hw, Hem = z.H_tr_w, z.H_tr_em
        ww, wr, wg = z.w_wall, z.w_roof, z.w_gnd
        Twall, Troof, Tgnd = self._T_wall, self._T_roof, self._T_gnd
        Cm_dt = z.C_m / dt
        f_wr = 0.0 if self.multinode else z.H_tr_w / (9.1 * z.A_t)   # ISO 13790 radiant share lost via windows
        hs_fn, cs_fn = sysm.heat_stages, sysm.cool_stages
        mn = self.multinode
        bat = self.bat
        base_elec = self._base_elec_list() if (bat or ctrl) else None
        for k in range(k0, k1):
            To = Tout[k]
            Ta = self.T_air
            Qi = ela * math.sqrt(Cs * abs(Ta - To) + CwU2[k]) * 1e-3 * inf_fac[k]
            if Qexh2 > 0:
                Qi = math.sqrt(Qi * Qi + Qexh2)

            m_inf = rho[k] * Qi
            H_inf = m_inf * CP_AIR
            m_v = m_v_a[k]
            H_v = m_v * CP_AIR
            H_ve = H_inf + H_v + 1e-6
            th_sup = (H_inf * To + H_v * T_v_a[k]) / H_ve
            th_em = th_em_a[k]
            pi, psol = phi_int_a[k], phi_sol_a[k]
            hsp, csp = hsp_a[k], csp_a[k]
            act = None
            if ctrl is not None:
                act = ctrl(self._obs(k))
                if act:
                    hsp = act.get('heat_sp', hsp)
                    csp = act.get('cool_sp', csp)
                    if act.get('hvac_off'):
                        hsp, csp = -100.0, 100.0
                    if 'hvac_on' in act:
                        self.hvac_on[k] = 1.0 if act['hvac_on'] else 0.0
            if has_tbias:
                bb = F.get('bias_T_zone', k)
                if bb:
                    hsp -= bb
                    csp -= bb
            tm0 = z.theta_m
            if mn:
                z.k = k
            r0 = solve(tm0, To, th_em, H_ve, th_sup, pi, psol, 0.0, 0.0, 0.0)
            if inj == 'air':
                r1 = solve(tm0, To, th_em, H_ve, th_sup, pi, psol, 1000.0, 0.0, 0.0)
            elif inj == 'mass':
                r1 = solve(tm0, To, th_em, H_ve, th_sup, pi, psol, 0.0, 0.0, 1000.0)
            else:
                r1 = solve(tm0, To, th_em, H_ve, th_sup, pi, psol, 0.0, 1000.0, 0.0)
            a = r0[3]
            b = (r1[3] - a) * 1e-3
            if inj != 'air':
                ra = solve(tm0, To, th_em, H_ve, th_sup, pi, psol, 1000.0, 0.0, 0.0)
                b_air = (ra[3] - a) * 1e-3
            else:
                ra, b_air = r1, b
            # natural ventilation: an outdoor-air conductance at the air node, opened only as far as needed to
            # hold the mid-deadband temperature (free cooling before the HVAC is asked)
            g_nv = 0.0
            if nv_ok is not None and nv_ok[k]:
                tgt = 0.5 * (hsp + csp)
                if a > tgt and To < a - 1.0:
                    g_max = rho[k] * nv_area * math.sqrt(Cs_nv * (a - To) + CwU2[k]) * 1e-3 * CP_AIR
                    if g_max > 0:
                        fnv = (a - tgt) / (b_air * g_max * (tgt - To)) if tgt > To else 1.0
                        g_nv = g_max * min(max(fnv, 0.0), 1.0)
                        den_nv = 1.0 + b_air * g_nv
                        a = (a + b_air * g_nv * To) / den_nv
                        b = b / den_nv if inj == 'air' else b - b_air * g_nv * b / den_nv
                        b_air = b_air / den_nv
            base = sysm.plan(k)
            cap = act.get('max_capacity') if act else None
            if cap is not None and cap < 1.0:
                q, T, mode, stage, frac = thermostat_solve(a, b, base, hsp, csp, _scaled(hs_fn, base, cap),
                                                           _scaled(cs_fn, base, cap))
                if mode != 0 and stage == 0:
                    frac *= cap
            else:
                q, T, mode, stage, frac = thermostat_solve(a, b, base, hsp, csp, hs_fn, cs_fn)
            q_nv = g_nv * (To - T) if g_nv > 0 else 0.0
            f = q * 1e-3
            fa = q_nv * 1e-3
            tm_t = r0[0] + f * (r1[0] - r0[0]) + fa * (ra[0] - r0[0])
            tm = r0[1] + f * (r1[1] - r0[1]) + fa * (ra[1] - r0[1])
            ts = r0[2] + f * (r1[2] - r0[2]) + fa * (ra[2] - r0[2])
            m_nv = g_nv / CP_AIR
            m_sup, W_sup = sysm.finish(k, q, T, mode, stage, frac)
            # moisture balance (implicit), with condensation if supersaturated
            Wo = self.W_air
            m_inf += m_nv
            den = M + dt * (m_inf + m_v + m_sup)
            Wn = (M * Wo + dt * (m_inf * Wout[k] + m_v * W_v_a[k] + m_sup * W_sup + G_lat_a[k])) / den
            Wsat = ps.w_sat_s(T)
            cond = 0.0
            if Wn > Wsat:
                cond = (Wn - Wsat) * den / dt
                Wn = Wsat
            z.theta_m = tm_t
            self.T_air = T
            self.W_air = Wn
            if bat:
                P_el = base_elec[k] + _hvac_elec(sysm.rec, k)
                self._battery(k, P_el, act)
            if L is not None and mn:
                fl = z.ledger_flows(tm0, tm_t, k, To)
                L['win'] += fl['win']
                L['wall'] += fl['wall']
                L['roof'] += fl['roof']
                L['ground'] += fl['ground']
                L['storage'] += fl['storage']
                L['inf'] += H_inf * (To - T) + q_nv
                L['vent'] += H_v * (T_v_a[k] - T)
                L['int'] += pi
                L['sol'] += psol
                L['hvac'] += q
                L['m_inf'] += m_inf * (Wout[k] - Wn)
                L['m_vent'] += m_v * (W_v_a[k] - Wn)
                L['m_sup'] += m_sup * (W_sup - Wn)
                L['m_lat'] += G_lat_a[k]
                L['m_storage'] += M * (Wn - Wo) / dt
                L['m_cond'] += cond
                self.r_T[k] = T
                self.r_Ts[k] = ts
                self.r_Tm[k] = float(tm_t[1])
                self.r_W[k] = Wn
                self.r_q[k] = q
                self.r_mode[k] = mode
                self.r_hsp[k] = hsp
                self.r_csp[k] = csp
                self.r_inf[k] = Qi + m_nv / rho[k]
                self.r_cond[k] = cond
            elif L is not None:
                L['win'] += Hw * (To - ts)
                dm = Hem * (th_em - tm)
                L['wall'] += Hem * ww * (Twall[k] - tm)
                L['roof'] += Hem * wr * (Troof[k] - tm)
                L['ground'] += Hem * wg * (Tgnd[k] - tm)
                L['inf'] += H_inf * (To - T) + q_nv
                L['vent'] += H_v * (T_v_a[k] - T)
                L['int'] += pi
                L['sol'] += psol
                L['win_rad'] += f_wr * (0.5 * pi + psol)
                L['hvac'] += q
                L['storage'] += Cm_dt * (tm_t - tm0)
                L['m_inf'] += m_inf * (Wout[k] - Wn)
                L['m_vent'] += m_v * (W_v_a[k] - Wn)
                L['m_sup'] += m_sup * (W_sup - Wn)
                L['m_lat'] += G_lat_a[k]
                L['m_storage'] += M * (Wn - Wo) / dt
                L['m_cond'] += cond
                del dm
                self.r_T[k] = T
                self.r_Ts[k] = ts
                self.r_Tm[k] = tm_t
                self.r_W[k] = Wn
                self.r_q[k] = q
                self.r_mode[k] = mode
                self.r_hsp[k] = hsp
                self.r_csp[k] = csp
                self.r_inf[k] = Qi + m_nv / rho[k]
                self.r_cond[k] = cond

    def _base_elec_list(self):
        if not hasattr(self, '_be'):
            be = (self.P_light + self.P_plug + self.P_ext + self.dhw_elec + self.P_mech_vent +
                  self._doas_elec() + (self.cook if self.spec['internal_gains']['cooking_fuel'] == 'electric'
                                       else 0.0))
            self._be = (be - self.pv_W).tolist()
        return self._be

    def _doas_elec(self):
        return self.doas_cool_W / 3.0 if self.doas_cool_W.any() else 0.0

    def _battery(self, k, P_el, act):
        """P_el: building electricity minus PV [W]. Battery + = charging."""
        b = self.bat
        cap_J = b['kWh'] * 3.6e6
        Pmax = b['kW'] * 1000.0
        cmd = act.get('battery_kW') if act else None
        if cmd is not None:
            P = cmd * 1000.0
        elif b['mode'] == 'self_consumption':
            P = -P_el
        elif b['mode'] == 'tou' and self.prices is not None:
            pr = self.prices[k]
            P = Pmax if pr <= self._p_lo else (-Pmax if pr >= self._p_hi else 0.0)
        else:
            P = 0.0
        P = max(-Pmax, min(Pmax, P))
        dt = self.dt
        e = self.soc * cap_J
        if P > 0:
            room = (1.0 - self.soc) * cap_J
            P = min(P, room / (dt * b['eff_charge']))
            de = P * b['eff_charge'] * dt
            loss = P * (1 - b['eff_charge']) * dt
            self.L['bat_in'] += P * dt
        else:
            avail = (self.soc - b['soc_min']) * cap_J
            Pd = min(-P, max(avail, 0.0) * b['eff_discharge'] / dt)
            P = -Pd
            de = -Pd / b['eff_discharge'] * dt
            loss = Pd * (1 / b['eff_discharge'] - 1) * dt
            self.L['bat_out'] += Pd * dt
        self.L['bat_loss'] += loss
        self.L['bat_store'] += de
        self.soc = (e + de) / cap_J
        self.r_bat[k] = P
        self.r_soc[k] = self.soc

    def _obs(self, k):
        return dict(k=k, t=self.times[k], T_zone=self.T_air, T_mass=self.zone.theta_m,
                    W_zone=self.W_air, T_out=self.T_out[k], RH_out=float(self.wx['RH'][k]),
                    GHI=float(self.wx['ghi'][k]), occupied=self.hvac_sched[k] > 0.5,
                    hour=float(self.wx['cal']['hour'][k]), daytype=int(self.wx['cal']['daytype'][k]),
                    heat_sp=self.hsp[k], cool_sp=self.csp[k], price=self.prices[k] if self.prices is not None
                    else None, soc=self.soc, pv_kW=float(self.pv_W[k]) / 1000.0, dt_s=self.dt, sim=self)

    def forecast(self, k, hours, key='T_out'):
        """Perfect forecast of an input for the next `hours` (for MPC-style controllers)."""
        m = int(round(hours * 3600 / self.dt))
        src = {'T_out': self.wx['T_out'], 'GHI': self.wx['ghi'], 'price': self.prices,
               'pv_kW': self.pv_W / 1000.0}[key]
        return np.asarray(src[k:k + m])

    # ------------------------------------------------------------------ outputs
    def _outputs(self, wall):
        from .outputs import make_outputs
        return make_outputs(self, wall)


def _scaled(fn, base, s):
    def g():
        st = fn()
        out = []
        for x in st:
            if x is None:
                out.append(None)
                continue
            g_, Ts, q0 = x
            gb, Tsb, q0b = base
            gn = gb + s * (g_ - gb)
            gTs = gb * Tsb + s * (g_ * Ts - gb * Tsb)
            out.append((gn, gTs / gn if gn > 0 else 0.0, q0b + s * (q0 - q0b)))
        return out[:1]
    return g


def _hvac_elec(rec, k):
    return (rec['E_heat_elec'][k] + rec['E_cool_elec'][k] + rec['E_fans'][k] + rec['E_pumps'][k] +
            rec['E_rej'][k])


def _price_array(prices, times, cal):
    """prices: number ($/kWh), pandas Series/DataFrame (column 'elec', or agent B's tariff_prices() frame with
    'energy_$per_kWh') on any time grid (tz-aware or UTC), or a TOU tariff dict
    {'base': $/kWh, 'periods': [{'months': [..], 'hours': (h0, h1), 'weekdays_only': True, 'price': $/kWh}],
     'demand_charge': $/kW-month, 'gas': $/kWh}."""
    n = len(times)
    if isinstance(prices, (int, float)):
        return [float(prices)] * n
    if isinstance(prices, dict):
        p = np.full(n, float(prices.get('base', 0.15)))
        for per in prices.get('periods', []):
            m = np.ones(n, bool)
            if per.get('months'):
                m &= np.isin(cal['month'], per['months'])
            h0, h1 = per.get('hours', (0, 24))
            m &= (cal['hour'] >= h0) & (cal['hour'] < h1)
            if per.get('weekdays_only'):
                m &= cal['daytype'] == 0
            p[m] = per['price']
        return p.tolist()
    if isinstance(prices, pd.DataFrame):
        col = 'elec' if 'elec' in prices else ('energy_$per_kWh' if 'energy_$per_kWh' in prices else
                                              prices.select_dtypes('number').columns[0])
        s = prices[col]
    else:
        s = prices
    s = s.copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize('UTC')
    else:
        s.index = s.index.tz_convert('UTC')
    s = s.sort_index()
    v = s.reindex(s.index.union(times)).ffill().bfill().reindex(times).values
    return np.asarray(v, float).tolist()


def simulate(spec, weather, start=None, end=None, dt='1h', controller=None, faults=None, prices=None, init=None,
             rng=None, warmup_days=7, sensor_noise=None, record_trend=True, options=None):
    """Simulate one building. See README for the full contract. Returns a dict (SimResult)."""
    sim = Simulator(spec, weather, start, end, dt, controller=controller, faults=faults, prices=prices, init=init,
                    rng=rng, warmup_days=warmup_days, sensor_noise=sensor_noise, record_trend=record_trend,
                    options=options)
    if sim.prices is not None and sim.bat and sim.bat.get('mode') == 'tou':
        pr = np.asarray(sim.prices)
        sim._p_lo, sim._p_hi = np.percentile(pr, 30), np.percentile(pr, 70)
    return sim.run()
