"""RoadSim physics v2, work package B2: engine warm-up, tyres and axle in the cold, cabin HVAC, accessories.

Everything here runs only when one of the B2 options in options.py is switched on (`make()` returns None otherwise),
so physics v1 is untouched. sim.py calls, once per second:

    aux(...)   before the powertrain plan: HVAC and accessory power (mechanical on the belt or electrical), whether a
               hybrid needs its engine for heat, and the engine's cold-friction / enrichment state for this step;
    crr(...)   the rolling-resistance coefficient for this step (tyre temperature, cold axle drag);
    post(...)  after the plan: coolant, oil, cabin, PTC loop, tyre and axle temperatures.

Every tunable number is in PARAMS (with its source); options['thermal_params'] overrides any of them. The calibration
pass fits them on dev data only. Units: W, J, K or C, kg, m, s.

Main sources
- Engine warm-up as lumped coolant/block and oil capacities: Shayler, Christian & Ma 1993 (SAE 931153, PROMETS);
  friction vs oil viscosity: Patton, Nitschke & Heywood 1989 (SAE 890836); multigrade oil viscosity-temperature in Vogel
  form (ASTM D341 practice) through ~65 cSt at 40 C and ~10.5 cSt at 100 C (SAE 5W-30).
- Tyre rolling resistance vs temperature: ISO 28580 (0.008 /K for passenger tyres); tyre heat transfer from flat-plate
  forced convection; warm-up ~10-20 min (Ejsmont et al. 2017, Int. J. Automotive Technology 18).
- Cabin and A/C: two-node cabin (air + interior mass) as in NREL CoolCalc-type lumped models (Rugh et al.); moist-air
  enthalpy with the Magnus saturation curve; compressor COP as a share of the Carnot COP between evaporator and
  condenser (automotive R134a/R1234yf systems: COP ~1.5-2.5 at 35 C ambient).
- SC03 conditions: 35 C, 100 grains water/lb dry air (~40 % RH), 850 W/m2 solar (40 CFR 86.161).
"""
import math
from .constants import G, T0K

CP_AIR = 1006.0          # J/kg/K
H_FG = 2.45e6            # J/kg, latent heat of water at ~20 C
R_GAS = 8.314

PARAMS = {
    # ---- engine, two-node (engine_thermal='v2'); per litre of displacement where marked
    'oil_C_J_K_per_L': 5.0e3,      # oil (~4-5 L x 0.86 kg/L x 2.1 kJ/kg/K) + crankshaft, sump and rods it wets [PROMETS]
    'oil_C0_J_K': 6.0e3,           # (1.5 L engine: 13.5 kJ/K; 2.5 L: 18.5 kJ/K)
    'oil_heat_frac': 0.07,         # share of (fuel - brake) power that heats the oil directly (friction, piston cooling)
    'oil_UA_W_K_per_L': 8.0,       # block/coolant <-> oil (galleries, liners, oil cooler)
    'oil_hA_W_K_per_L': 1.5,       # sump to ambient, still air
    'oil_hA_speed': 0.25,          # ... plus per m/s of vehicle speed
    'oil_T_over': 12.0,            # K: warm oil settles this far above the coolant thermostat temperature
    'oil_fric_scale': 0.33,        # x calib cold_fric with engine_thermal='v2' (the v1 fit lumped every cold loss on the
                                   # fast coolant node; see docs/v2_b2.md for the value that keeps the 24 C cold start)
    'oil_visc_exp': 0.5,           # cold friction ~ (nu/nu_hot)^exp - 1 (v1 0.5: ring/liner film friction ~ sqrt(viscosity))
    'vogel_theta1': 800.0,         # K, engine oil nu = a exp(theta1 / (T + theta2)) (SAE 5W-30: 65 cSt @40 C, 10.5 @100 C)
    'vogel_theta2': 95.0,          # K
    'atf_theta1': 706.0,           # ATF (~30 cSt @40 C, ~6 @100 C), same theta2
    'gear_theta1': 854.0,          # axle gear oil 75W-90 (~105 cSt @40 C, ~15 @100 C), same theta2
    'enrich_T_ref': 25.0,          # C: after-start enrichment counts K of coolant below this at the start
    'enrich_tau_s': 40.0,          # s: after-start enrichment / catalyst-heating decay
    'warmup_T_ref': 20.0,          # C: warm-up enrichment counts K of coolant below this
    'lowload_kW_per_L': 4.0,       # brake power per litre where the eng_lowload_W_per_L fuel adder has faded to zero
                                   # (idle and light city load; HWFET cruise is ~5-8 kW/L)
    # ---- axle / wheel-bearing cold drag (axle_cold)
    'axle_a0_Nm': 1.2,             # warm no-load drag torque at the wheels per driven axle (car), Nm
    'axle_a1_Nm_s': 0.025,         # ... plus per rad/s of wheel speed
    'axle_visc_exp': 0.3,          # churning drag ~ viscosity^0.3
    'axle_T_ref': 80.0,            # C: the warm reference (losses at this temperature are already in driveline_eff)
    'axle_C_J_K': 1.2e4,           # car axle oil + housing (scaled by (mass/1600)^0.8)
    'axle_hA_W_K': 2.0,            # still air, plus axle_hA_speed per m/s
    'axle_hA_speed': 0.6,
    # ---- tyres (tire_thermal)
    'tire_C_J_K': 7.0e4,           # car: 4 tyres x ~9 kg rubber/steel x ~1.4 kJ/kg/K + inner air (scaled by (mass/1600)^0.85)
    'tire_area_m2': 2.0,           # car: wetted outer area of 4 tyres (scaled by (mass/1600)^0.6)
    'tire_h0': 8.0,                # W/m2/K standing
    'tire_h1': 4.5,                # W/m2/K per (m/s)^0.8 (forced convection)
    'tire_T_ref_amb': 20.0,        # C: ambient at which the spec Crr holds (road; coast-down reference)
    'tire_T_ref_amb_dyno': 23.9,   # C: chassis-dyno test-cell temperature at which the dyno is set
    'tire_v_ref': 22.0,            # m/s: speed whose steady tyre temperature is the Crr reference (coast-down mean)
    'tire_f_clip': [0.7, 1.8],     # limits on the rolling-resistance temperature factor
    'road_solar_K': 0.015,         # K of road-surface warming per W/m2 of sun (asphalt)
    # ---- cabin and HVAC (hvac_model='v2')
    'cabin_air_frac': 0.17,        # share of the spec cabin capacity C_J_K in the fast air node (air + light trim)
    'cabin_mass_frac': 0.67,       # share in the interior-mass node (seats, dashboard, panels)
    'cabin_h_int_W_K': 150.0,      # air <-> interior surfaces, car: ~15 m2 x ~10 W/m2/K (convection + radiation), x UA/120
    'solar_to_interior': 0.8,      # share of transmitted sun absorbed by the interior mass
    'leak_kg_s': 0.012,            # infiltration, car (scaled by UA/120), plus leak_speed per m/s
    'leak_speed': 0.03,
    'air_kg_s_max': {'car': 0.13, 'suv': 0.15, 'pickup': 0.15, 'van': 0.20, 'bus': 2.2, 'truck_md': 0.15, 'truck_hd': 0.17},
    'blower_min': 0.25,            # auto-climate minimum blower level
    'blower_span_K': 15.0,         # cabin error (K) that drives the blower from minimum to full
    'cabin_UA_body_frac': 0.6,     # share of the spec UA_W_K that is conduction through glass and body; the rest of the v1
                                   # lumped UA is ventilation, now explicit (fresh-air share of the blower flow + leakage)
    'blower_exp': 2.5,             # blower electrical power ~ level^exp, full = 2 x spec blower_W
    'ctrl_tau_s': 120.0,           # climate controller time constant for the air node
    'T_sup_max': 55.0,             # C hottest supply air
    'cold_engine_blower_C': 35.0,  # engine-heated cars hold the blower at minimum until the coolant reaches this
    'heater_core_eff': 0.8,        # heater-core effectiveness
    'evap_T_set': 4.0,             # C evaporator air-out set point (belt systems dehumidify and reheat)
    'evap_T_min': 2.0,
    'evap_cap_factor': 1.3,        # evaporator capacity = factor x spec cool_max_W (sensible + latent)
    'comp_rpm_ref': 1800.0,        # engine rpm at which a belt compressor reaches full capacity
    'comp_cap_exp': 0.8,
    'refrig_dT': 6.0,              # K refrigerant evaporating below the evaporator air-out temperature
    'cond_dT0': 12.0,              # K condensing above ambient at speed ...
    'cond_dT1': 13.0,              # ... plus this at standstill, fading with exp(-v / 8 m/s)
    'cop_eta_belt': 0.33,          # COP / Carnot COP, belt compressor incl. mechanical losses (COP ~1.5 idling, ~1.9 at speed, 35 C)
    'cop_eta_el': 0.36,            # electric compressor incl. motor and inverter (no clutch cycling, speed-controlled)
    'cop_clip': [0.8, 5.0],
    'belt_eff': 0.95,
    'cond_fan_factor': 2.0,        # condenser/radiator fan full power = factor x spec blower_W
    'cond_fan_v_off': 20.0,        # m/s: ram air replaces the fan by this speed (floor cond_fan_min)
    'cond_fan_min': 0.25,
    'ptc_max_W': {'car': 5000, 'suv': 6000, 'pickup': 6000, 'van': 7000, 'bus': 24000, 'truck_md': 6000, 'truck_hd': 8000},
    'ptc_eff': 0.97,
    'ptc_type': 'coolant',         # 'coolant' (heater loop, e.g. 2011-12 Leaf 5 kW) | 'air' (direct air PTC)
    'ptc_T_sw': 85.0,              # C PTC switch temperature (power falls to zero over ptc_dT_sw below it)
    'ptc_dT_sw': 20.0,
    'ptc_loop_C_J_K': 1.6e4,       # coolant loop (~3 L + heater core + heater), car
    'ptc_loop_hA_W_K': 4.0,
    'ptc_loop_T_max': 65.0,
    'hp_cap_lo': 0.35,             # heat-pump capacity share of heat_max_W at -15 C, rising ...
    'hp_cap_slope': 0.025,         # ... per K to 1.0
    'occupant_moist_kg_s': 1.4e-5, # ~50 g/h per person
    'seat_W': 75.0,                # per heated seat (driver only by default)
    'n_seats': 1,
    'steer_W': 50.0,
    'seat_dT': 2.0,                # K lower air set point with seat + wheel heating at equal comfort
    'defrost_T_comp_C': 3.0,       # compressor runs for dehumidification above this ambient in defrost mode
    'defrost_blower_min': 0.6,
    # ---- buses (bus_aux)
    'bus_el_W': 2500.0,            # 24 V loads: lights, signs, fare system, electronics, fans
    'alt_eff_hd': 0.55,            # heavy-duty alternator efficiency
    'air_load_W': 8000.0,          # air compressor shaft power when loaded at comp_rpm_ref (engine-driven)
    'air_unload_W': 600.0,         # unloaded drag (engine-driven, spinning)
    'air_el_W': 6500.0,            # electric air compressor input when running
    'air_tank_J': 6.0e5,           # air store, in compressor-work units; cut-in at half
    'air_per_stop_J': 6.0e4,       # door cycles + kneeling per stop
    'air_per_brake_J': 1.2e4,      # per friction-brake application
    'air_leak_W': 150.0,
    'steer_W_base': 800.0,         # engine-driven hydraulic power steering, idle flow
    'steer_W_base_el': 250.0,      # electro-hydraulic, on demand
    'steer_W_manoeuvre': 1200.0,   # extra below 4 m/s while moving (pulling in/out of stops)
    'steer_eff_el': 0.85,
    'bus_edrive_cooling_W': 1500.0,  # hybrid/electric buses: pumps and fans cooling the traction inverters, motors, battery
    'door_n': 2,
    'door_area_m2': 2.4,
    'door_height_m': 2.0,
    'door_cd': 0.6,
    'door_min_ms': 0.15,           # m/s wind/turbulence exchange through an open door at zero temperature difference
    'door_delay_s': 2.0,           # doors open after this long at a stop
}


def _engine_active(o):
    return (o['engine_thermal'] != 'v1' or o['oil_viscosity'] != 'arrhenius' or o['cold_enrich'] != 0.0
            or o['warmup_enrich'] != 0.0 or o['eng_dfmep_bar'] != 0.0 or o['eng_eta_scale'] != 1.0
            or o['eng_lowload_W_per_L'] != 0.0
            or bool(o['eng_family_adj']) or o['fead_drag_Nm'] != 0.0 or o.get('eng_transient_fuel', 0.0) != 0.0
            or o.get('eng_fullload_enrich', 0.0) != 0.0)


def active(o):
    return (_engine_active(o) or o['axle_cold'] or o['tire_thermal'] or o['hvac_model'] != 'v1'
            or o['phev_electric_heat'] or o['bus_aux'])


def make(veh, opt, trip, trip_inputs, st, it, dt):
    return Thermal(veh, opt, trip, trip_inputs, st, it, dt) if active(opt) else None


# ------------------------------------------------------------------ helpers
def p_ws(T):
    """saturation vapour pressure over water (Pa), Magnus form."""
    return 610.94 * math.exp(17.625 * T / (T + 243.04))


def w_sat(T, p=101325.0):
    pw = p_ws(T)
    return 0.622 * pw / max(p - pw, 1.0)


def visc_ratio(T, T_ref, law, theta1=800.0, theta2=95.0):
    """kinematic viscosity nu(T) / nu(T_ref)."""
    T = max(T, -40.0)
    if law == 'vogel':
        return math.exp(theta1 * (1.0 / (T + theta2) - 1.0 / (T_ref + theta2)))
    return math.exp(3776.0 * (1.0 / (T + T0K) - 1.0 / (T_ref + T0K)))


class Thermal:
    def __init__(self, veh, opt, trip, trip_inputs, st, it, dt):
        P = dict(PARAMS)
        P.update(opt.get('thermal_params') or {})
        self.P = P; self.o = opt; self.veh = veh; self.dt = dt
        self.arch = veh.arch; self.conv = veh.arch == 'conv'
        self.hybrid = veh.arch not in ('conv', 'bev')
        self.cls = veh.cls
        e = veh.engine; self.e = e
        self.cab = veh.cabin
        self.T_amb = trip['T_amb'].tolist(); self.solar = trip['solar'].tolist()
        N = trip['N']
        rh = trip_inputs.get('rh') if isinstance(trip_inputs, dict) else None
        if rh is None:
            self.rh = [float(opt['ambient_rh'])] * N
        else:
            import numpy as np
            r = np.asarray(rh, float)
            self.rh = (np.full(N, float(r)) if r.ndim == 0 else np.clip(np.nan_to_num(r, nan=opt['ambient_rh']), 0, 1)).tolist()
        Ta0 = self.T_amb[0]
        self.m = veh.mass; mr = veh.mass / 1600.0
        # ---------------- engine
        self.eng_on_mod = e is not None and _engine_active(opt)
        self.eng_v2 = e is not None and opt['engine_thermal'] == 'v2'
        if e is not None:
            # engine adjustment: global keys, replaced per map family by eng_family_adj[map] = {key: value}
            adj = dict(eng_eta_scale=opt['eng_eta_scale'], eng_lowload_W_per_L=opt['eng_lowload_W_per_L'],
                       eng_dfmep_bar=opt['eng_dfmep_bar'])
            adj.update((opt['eng_family_adj'] or {}).get(e.map_id) or {})
            self.dT_adj = (adj['eng_dfmep_bar'] * 1e5 * e.disp * 1e-3 / (4.0 * math.pi)
                           + (opt['fead_drag_Nm'] if self.conv else 0.0))
            self.f_eta = 1.0 / adj['eng_eta_scale']
            self.ll_W = adj['eng_lowload_W_per_L'] * e.disp                 # extra fuel power at zero brake power ...
            self.ll_P = max(P['lowload_kW_per_L'] * 1e3 * e.disp, 1.0)     # ... fading linearly to zero at this brake power
            self.law = opt['oil_viscosity']
            self.th1 = P['vogel_theta1']; self.th2 = P['vogel_theta2']; self.pexp = P['oil_visc_exp']
            self.g20 = self._g(20.0)
            self.fric_scale = P['oil_fric_scale'] if self.eng_v2 else 1.0
            self.T_oil = it.get('oil', st['T_eng'])
            self.C_oil = P['oil_C0_J_K'] + P['oil_C_J_K_per_L'] * e.disp
            self.UA_oil = P['oil_UA_W_K_per_L'] * e.disp
            self.t_run = 0.0; self.T_start = st['T_eng']; self.was_on = bool(st.get('eng_on', False)) or self.conv
            # transient fuel: gasoline engines in light-duty vehicles (the D3 evidence); 0 for diesel and heavy duty
            self.k_tr = (float(opt.get('eng_transient_fuel', 0.0) or 0.0) / max(e.T_peak, 1.0)
                         if (veh.fuel == 'gasoline' and not veh.hd) else 0.0)   # per (Nm/s)
            self.T_e_prev = None
            e._x2 = (0.0, 1.0, 0.0, 1.0) if self.eng_on_mod else None
            k_en = float(opt.get('eng_fullload_enrich', 0.0) or 0.0)
            e._enr = ((k_en, float(opt.get('eng_fullload_frac', 0.85))) if (k_en and veh.fuel == 'gasoline' and not veh.hd)
                      else None)
        self.w_e = e.w_idle if e is not None else 0.0          # previous step's engine speed (belt-driven loads)
        # ---------------- driveline oil (transmission) viscosity law
        self.f_dl_vogel = opt['oil_viscosity'] == 'vogel'
        # ---------------- tyres
        self.tyre = bool(opt['tire_thermal'])
        self.dyno = not veh.tire_temp_effects
        if self.tyre:
            import numpy as np
            self.kt = float(opt['tire_temp_coef']); self.fdyn = float(opt['tire_dyno_frac'])
            self.C_ty = P['tire_C_J_K'] * mr ** 0.85; self.A_ty = P['tire_area_m2'] * mr ** 0.6
            f_s = np.array([1.0, 0.95, 1.12, 1.6, 2.2])[trip['surface']]
            if self.dyno:
                self.base = (veh.Crr * veh.tire_p_ratio ** -0.4 * f_s).tolist()
                T_ref_amb = P['tire_T_ref_amb_dyno']
            else:
                self.base = (veh.Crr * f_s).tolist()
                T_ref_amb = P['tire_T_ref_amb']
            self.p0 = 240.0 if not veh.hd else 760.0
            v = P['tire_v_ref']
            self.T_ty_ref = T_ref_amb + self._tyre_rise(v, veh.Crr)
            self.pg_ref = self._pg(self.T_ty_ref)
            self.road = [Ta + (0.0 if self.dyno else P['road_solar_K'] * s) for Ta, s in zip(self.T_amb, self.solar)]
            self.T_ty = it.get('tire_C', self.road[0] + float(it.get('tire_warm', 0.0)) * self._tyre_rise(v, veh.Crr))
            self.F_ty = 0.0
        # ---------------- axle
        self.axle = bool(opt['axle_cold'])
        if self.axle:
            sc = (mr ** 0.7) * (2.0 if veh.hd and veh.cls != 'bus' else 1.0)
            self.ax_a0 = P['axle_a0_Nm'] * sc; self.ax_a1 = P['axle_a1_Nm_s'] * sc
            self.ax_C = P['axle_C_J_K'] * mr ** 0.8
            self.T_ax = it.get('axle', it.get('driveline', it.get('engine', Ta0)))
            self.F_ax = 0.0
        # ---------------- cabin / HVAC
        self.hvac2 = opt['hvac_model'] == 'v2'
        cab = self.cab
        if self.hvac2:
            C = cab.C
            self.C_a = P['cabin_air_frac'] * C; self.C_m = P['cabin_mass_frac'] * C
            s = cab.UA / 120.0
            self.h_m = P['cabin_h_int_W_K'] * s
            self.leak0 = P['leak_kg_s'] * s
            self.UA_b = P['cabin_UA_body_frac'] * cab.UA
            self.m_air = 3.5 * (C / 9.0e4)                      # kg of cabin air (humidity balance)
            self.ma_max = P['air_kg_s_max'].get(self.cls, 0.15)
            self.blow_max = 2.0 * cab.blower_W
            self.fan_max = P['cond_fan_factor'] * cab.blower_W
            self.T_int = it.get('cabin_mass', st['T_cab'])
            self.w_cab = self.rh[0] * w_sat(Ta0)
            ptc = P['ptc_max_W']
            self.ptc_max = cab.spec_ptc_W if getattr(cab, 'spec_ptc_W', None) else ptc.get(self.cls, 6000) if isinstance(ptc, dict) else ptc
            self.ptc_type = cab.spec_ptc_type or P['ptc_type']
            self.C_lp = P['ptc_loop_C_J_K'] * (C / 9.0e4) ** 0.5
            self.T_lp = it.get('ptc_loop', st['T_cab'])
            self.Q_lp = 0.0; self.P_ptc = 0.0
        self.src = cab.source
        no_ptc = getattr(cab, 'spec_ptc_W', None) == 0          # spec says: no electric heater (2013 Ford Energi)
        if no_ptc:
            self.ptc_max = 0.0
        if opt['phev_electric_heat'] and self.arch in ('phev_series', 'phev_powersplit') and self.src == 'engine' and not no_ptc:
            self.src = 'engine+ptc'
        if opt.get('phev_engine_heat') and self.arch in ('phev_series', 'phev_powersplit') and self.src in ('ptc', 'heat_pump'):
            # plug-ins with an electric heater also have a heater core on the engine loop: a warm engine heats the cabin
            # (v1 / B2: a 'ptc' plug-in heated electrically even in charge sustaining, the engine recharging the battery)
            self.src = 'engine+ptc' if self.src == 'ptc' else 'engine+hp'
        # step results handed from aux() to post()
        self.m_a = 0.0; self.T_sup = 0.0; self.w_sup = 0.0; self.Q_core_eng = 0.0; self.m_door = 0.0
        self.Q_eng_heat = 0.0; self._Q_del_v1 = 0.0; self.leak = 0.0
        # ---------------- bus accessories
        self.bus = bool(opt['bus_aux']) and veh.cls == 'bus'
        self.t_stop = 0.0; self.moving_prev = True; self.brk_prev = False
        self.tank = P['air_tank_J']; self.loading = False
        self.events_J = 0.0

    # ======================================================== engine
    def _g(self, T):
        r = visc_ratio(T, self.e.T_hot, self.law, self.th1, self.th2)
        if self.pexp == 0.5:
            return math.sqrt(r) - 1.0
        return r ** self.pexp - 1.0

    def _engine_state(self, st):
        """cold friction (extra torque at the fuel-map lookup) and fuel factor for this step -> engine._x2."""
        e = self.e; P = self.P; o = self.o
        T_f = self.T_oil if self.eng_v2 else st['T_eng']
        dT = self.dT_adj
        if T_f < e.T_hot:
            dT += e.c_fric * self.fric_scale * e.disp * self._g(T_f) / self.g20
        f = self.f_eta
        if o['cold_enrich'] or o['warmup_enrich']:
            a = o['cold_enrich'] * max(0.0, P['enrich_T_ref'] - self.T_start) * math.exp(-self.t_run / P['enrich_tau_s'])
            b = o['warmup_enrich'] * max(0.0, P['warmup_T_ref'] - st['T_eng'])
            f *= 1.0 + a + b
        if self.k_tr and self.T_e_prev is not None:
            e._x2 = (dT, f, self.ll_W, self.ll_P, self.T_e_prev, self.k_tr / self.dt)
        else:
            e._x2 = (dT, f, self.ll_W, self.ll_P)

    def f_dl(self, T_dl):
        """transmission spin-loss factor vs ATF temperature (Vogel), relative to 80 C (v1: Arrhenius)."""
        return visc_ratio(T_dl, 80.0, 'vogel', self.P['atf_theta1'], self.P['vogel_theta2']) ** 0.3

    # ======================================================== rolling resistance
    def _tyre_rise(self, v, Crr):
        q = Crr * self.m * G * v
        hA = (self.P['tire_h0'] + self.P['tire_h1'] * v ** 0.8) * self.P['tire_area_m2'] * (self.m / 1600.0) ** 0.6
        return q / hA

    def _pg(self, T):
        p0 = self.p0
        return ((p0 + 101.3) * (T + T0K) / (20.0 + T0K) - 101.3) / p0

    def crr(self, k, st, crr_v1, v0, ck):
        if self.tyre:
            lo, hi = self.P['tire_f_clip']
            f = min(max(math.exp(-self.kt * (self.T_ty - self.T_ty_ref)), lo), hi)
            if self.dyno:
                c = self.base[k] * (1.0 + self.fdyn * (f - 1.0))
            else:
                fp = (max(self._pg(self.T_ty) * self.veh.tire_p_ratio, 0.5) / self.pg_ref) ** -0.4
                c = self.base[k] * f * fp
            self.F_ty = c * self.m * G * ck
        else:
            c = crr_v1
            self.F_ty = c * self.m * G * ck
        if self.axle:
            w = v0 / self.veh.r
            fx = visc_ratio(self.T_ax, self.P['axle_T_ref'], 'vogel', self.P['gear_theta1'], self.P['vogel_theta2']) ** self.P['axle_visc_exp']
            self.F_ax = max(fx - 1.0, 0.0) * (self.ax_a0 + self.ax_a1 * w) / self.veh.r
            c += self.F_ax / (self.m * G * max(ck, 0.5))
        return c

    # ======================================================== HVAC and accessories (before the plan)
    def aux(self, k, st, v0, stopped, v1vals):
        """-> (P_aux_mech, P_aux_el, need_heat, P_hvac_tot). v1vals = the v1 block's values, used where B2 is off."""
        Qd1, Q_del1, Q_eng1, P_el1, P_mech1, need1, P_tms, base = v1vals
        self._Q_del_v1 = Q_del1
        e = self.e
        if self.eng_on_mod:
            self._engine_state(st)
        Ta = self.T_amb[k]
        self.Q_eng_heat = Q_eng1
        if self.hvac2:
            P_mech, P_12v, P_hv, need_heat = self._hvac(k, st, v0, stopped, Ta)
        else:
            P_mech, P_12v, P_hv, need_heat = P_mech1, 0.0, P_el1, need1
            if self.src == 'engine+ptc' and Qd1 > 0 and st['T_eng'] < 50.0:
                # PHEV: electric heater instead of the cold engine (v1 cabin demand), no engine start for heat
                P_hv = Qd1 / 0.95 + self.cab.blower_W; self.Q_eng_heat = 0.0; need_heat = False
                self._Q_del_v1 = Qd1
        # ---- bus accessories replace the lumped base load
        P_bus_mech = 0.0; P_bus_el = 0.0
        if self.bus:
            base, P_bus_mech, P_bus_el = self._bus(st, v0, stopped)
        if self.conv:
            alt = self.P['alt_eff_hd'] if self.bus else 0.65
            if self.hvac2:
                P_aux_mech = (base + P_12v + P_hv) / alt + P_mech + P_bus_mech
            else:
                P_aux_mech = (base + P_hv) / alt + P_mech + P_bus_mech
            P_aux_el = 0.0
        else:
            from .powertrain import ETA_DCDC
            P_aux_mech = 0.0
            if self.hvac2:
                P_aux_el = (base + P_12v) / ETA_DCDC + P_hv + P_tms + P_bus_el
            else:
                bl = self.cab.blower_W * (Qd1 != 0.0)
                P_aux_el = (base + bl) / ETA_DCDC + (P_hv - bl) + P_tms + P_bus_el
        return P_aux_mech, P_aux_el, need_heat, P_mech + P_12v + P_hv + P_tms

    def _bus(self, st, v0, stopped):
        P = self.P
        engine_driven = self.conv
        # air: stops and brake applications draw on the tank; the governor loads the compressor below half
        if stopped and self.moving_prev:
            self.tank -= P['air_per_stop_J']
        self.tank -= P['air_leak_W'] * self.dt
        if self.tank < 0.5 * P['air_tank_J']:
            self.loading = True
        if engine_driven:
            rs = min(max(self.w_e * 30.0 / math.pi / P['comp_rpm_ref'], 0.0), 1.3)
            P_air = (P['air_load_W'] * rs if self.loading else P['air_unload_W'] * rs)
            fill = P['air_load_W'] * rs if self.loading else 0.0
        else:
            P_air = P['air_el_W'] if self.loading else 0.0
            fill = P['air_load_W'] if self.loading else 0.0
        self.tank += fill * self.dt
        if self.tank >= P['air_tank_J']:
            self.tank = P['air_tank_J']; self.loading = False
        man = P['steer_W_manoeuvre'] if (not stopped and v0 < 4.0) else 0.0
        if engine_driven:
            mech = P_air + (P['steer_W_base'] + man if self.w_e > 0 else 0.0)
            return P['bus_el_W'], mech, 0.0
        el = P_air + (P['steer_W_base_el'] + man) / P['steer_eff_el'] + P['bus_edrive_cooling_W']
        return P['bus_el_W'], 0.0, el

    def _hvac(self, k, st, v0, stopped, Ta):
        """two-node cabin with explicit air flows -> (P_mech belt, P_12V, P_HV, need_heat). Sets the supply air state."""
        cab = self.cab; P = self.P; o = self.o
        self.Q_core_eng = 0.0; self.Q_lp = 0.0; self.P_ptc = 0.0; self.Q_eng_heat = 0.0
        sol = self.solar[k] * cab.A_solar
        w_amb = self.rh[k] * w_sat(Ta)
        T = st['T_cab']
        # bus doors: stack-driven exchange through open doors at stops (one-third Cd A sqrt(g H dT / T) per door)
        self.m_door = 0.0
        if self.cls == 'bus':
            self.t_stop = self.t_stop + self.dt if stopped else 0.0
            if self.bus and stopped and self.t_stop > P['door_delay_s']:
                dTd = abs(T - Ta)
                vel = P['door_cd'] / 3.0 * math.sqrt(G * P['door_height_m'] * dTd / (0.5 * (T + Ta) + T0K)) + P['door_min_ms']
                self.m_door = P['door_n'] * 1.2 * P['door_area_m2'] * vel
        leak = self.leak0 * (1.0 + P['leak_speed'] * v0) + self.m_door
        if not cab.enabled:
            self.m_a = 0.0; self.T_sup = T; self.w_sup = self.w_cab; self.leak = leak
            return 0.0, 0.0, 0.0, False
        seat = o['hvac_seat_heat'] and Ta < cab.heat_on_below
        Tset = o['hvac_T_set'] - (P['seat_dT'] if seat else 0.0)
        err = T - Tset
        heat_ok = Ta < cab.heat_on_below or T < Tset - 4.0
        cool_ok = Ta > cab.cool_on_above or T > Tset + 6.0
        defrost = o['hvac_defrost']
        lvl = min(max(P['blower_min'] + abs(err) / P['blower_span_K'], P['blower_min']), 1.0)
        eng_src = self.src in ('engine', 'engine+ptc')
        if heat_ok and err < 0 and self.src == 'engine' and st['T_eng'] < P['cold_engine_blower_C'] and not defrost:
            lvl = P['blower_min']
        if defrost:
            lvl = max(lvl, P['defrost_blower_min'])
        m_a = self.ma_max * lvl
        # heat gains of the air node (without the HVAC supply)
        g_amb = self.UA_b + leak * CP_AIR
        q_gain = (g_amb * (Ta - T) + self.h_m * (self.T_int - T) + cab.Q_people + (1.0 - P['solar_to_interior']) * sol)
        T_req = T - (q_gain + self.C_a * err / P['ctrl_tau_s']) / (m_a * CP_AIR)
        r_h = 0.0 if defrost else o['hvac_recirc_heat']
        r_c = 1.0 if err > 3.0 else o['hvac_recirc_cool']
        heating = heat_ok and T_req > r_h * T + (1.0 - r_h) * Ta + 0.1
        cooling = (not heating) and cool_ok and T_req < r_c * T + (1.0 - r_c) * Ta - 0.1
        r = r_h if heating else (r_c if (cooling and not defrost) else 0.0)
        T_mix = r * T + (1.0 - r) * Ta
        w_mix = r * self.w_cab + (1.0 - r) * w_amb
        P_mech = 0.0; P_12v = self.blow_max * lvl ** P['blower_exp']; P_hv = 0.0
        need_heat = False
        T_sup = T_mix; w_sup = w_mix
        # ---- evaporator: cooling, or dehumidifying in defrost mode
        dehum = defrost and Ta >= P['defrost_T_comp_C']
        if cooling or dehum:
            belt = self.conv
            T_ev = P['evap_T_set'] if (belt or dehum) else max(T_req, P['evap_T_set'])
            T_ev = max(min(T_ev, T_mix), P['evap_T_min'])
            lat = H_FG * max(0.0, w_mix - w_sat(T_ev))
            Q_ev = m_a * (CP_AIR * (T_mix - T_ev) + lat)
            cap = P['evap_cap_factor'] * cab.Q_cool_max
            if belt:
                e = self.e
                cap *= min(max(self.w_e * 30.0 / math.pi / P['comp_rpm_ref'], 0.0), 1.0) ** P['comp_cap_exp']
            if Q_ev > cap:
                f = cap / Q_ev if Q_ev > 0 else 0.0
                T_ev = T_mix - f * (T_mix - T_ev); lat *= f; Q_ev = cap
            w_sup = w_mix - lat / H_FG
            T_sup = T_ev if dehum else max(T_ev, T_req)    # belt systems reheat to the requested supply temperature
            if Q_ev > 0.0:
                Te = T_ev - P['refrig_dT'] + T0K
                Tc = Ta + P['cond_dT0'] + P['cond_dT1'] * math.exp(-v0 / 8.0) + T0K
                lo, hi = P['cop_clip']
                cop = min(max((P['cop_eta_belt'] if belt else P['cop_eta_el']) * Te / max(Tc - Te, 1.0), lo), hi)
                P_comp = Q_ev / cop
                if belt:
                    P_mech = P_comp / P['belt_eff']
                else:
                    P_hv += P_comp
                P_12v += self.fan_max * max(P['cond_fan_min'], 1.0 - v0 / P['cond_fan_v_off'])
            if belt and T_sup > T_ev:
                self.Q_core_eng = m_a * CP_AIR * (T_sup - T_ev)     # blend-door reheat from the coolant
        # ---- heater
        if heating or (dehum and T_req > T_sup):
            T_tgt = min(max(T_req, T_sup), P['T_sup_max'])
            Q_req = m_a * CP_AIR * (T_tgt - T_sup)
            Q_need = max(-(q_gain + self.C_a * err / P['ctrl_tau_s']), Q_req)
            Q_d, P_e, need_heat = self._heat(Q_req, T_sup, m_a, st, Ta, Q_need)
            P_hv += P_e
            T_sup = T_sup + Q_d / (m_a * CP_AIR)
            if seat:
                P_12v += P['n_seats'] * P['seat_W'] + P['steer_W']
        self.m_a = m_a; self.T_sup = T_sup; self.w_sup = w_sup; self.leak = leak
        self.Q_eng_heat = self.Q_core_eng
        return P_mech, P_12v, P_hv, need_heat

    def _heat(self, Q_req, T_in, m_a, st, Ta, Q_need=0.0):
        """heat Q_req into the air stream entering at T_in -> (Q delivered, electrical power, engine needed for heat)."""
        cab = self.cab; P = self.P
        src = self.src
        Q_req = min(Q_req, cab.Q_heat_max)
        Q = 0.0; P_e = 0.0; need = False
        if src in ('engine', 'engine+ptc'):
            Q_av = P['heater_core_eff'] * m_a * CP_AIR * max(0.0, st['T_eng'] - T_in)
            if src == 'engine+ptc' and st['T_eng'] < 50.0:
                Q_av = 0.0                       # PHEV: cold engine -> heat electrically, don't start for heat
            Q = min(Q_req, Q_av)
            self.Q_core_eng += Q
            # engine wanted for heat (B1's strategies decide): the cabin needs real heat, the coolant is too cold to give
            # it, and (PHEV with an electric heater) the heater can't cover it. Same thresholds as v1.
            short = min(Q_need, cab.Q_heat_max) - (self.ptc_max * P['ptc_eff'] if src == 'engine+ptc' else 0.0)
            need = self.hybrid and short > 0.3 * cab.Q_heat_max and st['T_eng'] < 55.0
            rest = Q_req - Q
            if src == 'engine+ptc' and rest > 0.0:
                Qp, P_e = self._ptc(rest, T_in, m_a, Ta)
                Q += Qp
        elif src == 'engine+hp':
            Q = min(Q_req, P['heater_core_eff'] * m_a * CP_AIR * max(0.0, st['T_eng'] - T_in)) if st['T_eng'] >= 50.0 else 0.0
            self.Q_core_eng += Q
            rest = Q_req - Q
            if rest > 0.0:
                cap = cab.Q_heat_max * min(max(P['hp_cap_lo'] + P['hp_cap_slope'] * (Ta + 15.0), P['hp_cap_lo']), 1.0)
                Qh = min(rest, cap)
                P_e = Qh / cab.hp_cop(Ta)
                Q += Qh
                if rest > Qh:
                    Qp, Pp = self._ptc(rest - Qh, T_in, m_a, Ta)
                    Q += Qp; P_e += Pp
        elif src == 'heat_pump':
            cap = cab.Q_heat_max * min(max(P['hp_cap_lo'] + P['hp_cap_slope'] * (Ta + 15.0), P['hp_cap_lo']), 1.0)
            Q = min(Q_req, cap)
            P_e = Q / cab.hp_cop(Ta)
            if Q_req > Q:
                Qp, Pp = self._ptc(Q_req - Q, T_in, m_a, Ta)
                Q += Qp; P_e += Pp
        else:
            Q, P_e = self._ptc(Q_req, T_in, m_a, Ta)
        return Q, P_e, need

    def _ptc(self, Q_req, T_in, m_a, Ta):
        P = self.P
        if self.ptc_type == 'air':
            cap = self.ptc_max * min(max((P['ptc_T_sw'] - T_in) / (2.0 * P['ptc_dT_sw']), 0.3), 1.0)
            P_e = min(Q_req / P['ptc_eff'], cap)
            self.P_ptc += P_e
            return P_e * P['ptc_eff'], P_e
        # coolant loop: the heater core delivers from the loop; the PTC holds the loop at its target temperature
        Q = min(Q_req, P['heater_core_eff'] * m_a * CP_AIR * max(0.0, self.T_lp - T_in))
        T_tgt = min(T_in + Q_req / max(P['heater_core_eff'] * m_a * CP_AIR, 1.0) + 2.0, P['ptc_loop_T_max'])
        cap = self.ptc_max * min(max((P['ptc_T_sw'] - self.T_lp) / P['ptc_dT_sw'], 0.0), 1.0)
        P_e = min(max(self.C_lp * (T_tgt - self.T_lp) / 30.0 + Q + P['ptc_loop_hA_W_K'] * (self.T_lp - Ta), 0.0), cap)
        self.Q_lp += Q; self.P_ptc += P_e
        return Q, P_e

    # ======================================================== state updates (after the plan)
    def post(self, k, st, plan, u, running, F_brk, dt):
        P = self.P; e = self.e
        Ta = self.T_amb[k]
        # ---- engine coolant (and oil)
        if e is not None:
            Q_h = self.Q_eng_heat
            if self.eng_v2:
                q = max(plan['P_fuel'] - plan['P_eng'], 0.0) if running else 0.0
                Tc = st['T_eng']; To = self.T_oil
                q_co = self.UA_oil * (Tc - To)
                hAo = (P['oil_hA_W_K_per_L'] + P['oil_hA_speed'] * u) * e.disp
                fo = P['oil_heat_frac']
                Tc_n = Tc + dt * ((e.f_coolant - fo) * q - q_co - e.hA * (Tc - Ta) - Q_h) / e.C_th
                To_n = To + dt * (fo * q + q_co - hAo * (To - Ta)) / self.C_oil
                st['T_eng'] = min(Tc_n, e.T_hot + 2.0)
                self.T_oil = min(To_n, e.T_hot + P['oil_T_over'])
            else:
                st['T_eng'] = e.thermal_step(st['T_eng'], plan['P_fuel'], plan['P_eng'], Ta, Q_h, dt, running)
            if running:
                if not self.was_on:
                    self.t_run = 0.0; self.T_start = st['T_eng']
                self.t_run += dt
            self.was_on = running
            self.w_e = plan.get('w_e', 0.0) if running else 0.0
            self.T_e_prev = float(plan.get('T_e', 0.0)) if running else None
        # ---- cabin
        sol = self.solar[k] * self.cab.A_solar
        if self.hvac2:
            T = st['T_cab']; Ti = self.T_int
            g_amb = self.UA_b + self.leak * CP_AIR
            gs = self.m_a * CP_AIR
            q = self.cab.Q_people + (1.0 - P['solar_to_interior']) * sol
            Tn = (self.C_a * T + dt * (gs * self.T_sup + g_amb * Ta + self.h_m * Ti + q)) / (self.C_a + dt * (gs + g_amb + self.h_m))
            self.T_int = (self.C_m * Ti + dt * (self.h_m * Tn + P['solar_to_interior'] * sol)) / (self.C_m + dt * self.h_m)
            st['T_cab'] = Tn
            w_amb = self.rh[k] * w_sat(Ta)
            mw = self.m_a + self.leak
            self.w_cab = (self.m_air * self.w_cab + dt * (self.m_a * self.w_sup + self.leak * w_amb + P['occupant_moist_kg_s'])) / (self.m_air + dt * mw)
            if self.ptc_type != 'air':
                self.T_lp += dt * (self.P_ptc * P['ptc_eff'] - self.Q_lp - P['ptc_loop_hA_W_K'] * (self.T_lp - Ta)) / self.C_lp
        else:
            # v1 cabin (the v1 block's delivered heat, or the PHEV electric heater's)
            st['T_cab'] = self.cab.step(st['T_cab'], Ta, self.solar[k], self._Q_del_v1, dt)
        # ---- bus brakes (air)
        if self.bus:
            brk = F_brk > 1.0
            if brk and not self.brk_prev:
                self.tank -= P['air_per_brake_J']
            self.brk_prev = brk
            self.moving_prev = u > 0.05
        # ---- tyres
        if self.tyre:
            hA = (P['tire_h0'] + P['tire_h1'] * max(u, 0.0) ** 0.8) * self.A_ty
            q = self.F_ty * u
            self.T_ty += dt * (q - hA * (self.T_ty - self.road[k])) / self.C_ty
        # ---- axle oil: its own churning (warm part + cold excess)
        if self.axle:
            w = u / self.veh.r
            q = (self.ax_a0 + self.ax_a1 * w) * w + self.F_ax * u
            self.T_ax += dt * (q - (P['axle_hA_W_K'] + P['axle_hA_speed'] * u) * (self.T_ax - Ta)) / self.ax_C

    def final_state(self):
        fs = {}
        if self.e is not None and self.eng_v2:
            fs['oil'] = self.T_oil
        if self.tyre:
            fs['tire_C'] = self.T_ty
        if self.axle:
            fs['axle'] = self.T_ax
        if self.hvac2:
            fs['cabin_mass'] = self.T_int
            if self.ptc_type != 'air':
                fs['ptc_loop'] = self.T_lp
        return fs

    def done(self):
        if self.e is not None:
            self.e._x2 = None
