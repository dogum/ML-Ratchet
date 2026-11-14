"""Forward-facing 1 Hz vehicle simulation with a PI driver, road loads with real weather, and an energy ledger.

Step k covers [t_k, t_k+1]. The driver asks for the force that would reach the next target speed (feed-forward from the
road-load model + PI on the lagged speed error). The powertrain delivers what it can; the vehicle speed then follows from
    m_eff (v_k+1 - v_k)/dt = F_wheel - F_brake - F_aero(u) - F_roll(u) - F_grade,   u = (v_k + v_k+1)/2
so every work term is force x mean speed x dt, and the ledger closes exactly up to solver tolerance.

Output arrays have the trip length N. Interval quantities for step k are stored at index k+1 ("the second ending at
t_k+1", like an OBD logger's rate over the last second); index 0 holds zeros. States (speed, SOC, temperatures) are
instantaneous values at t_k.
"""
import math, time, hashlib, json
from itertools import chain as _chain_it
import numpy as np
from .constants import G, R_AIR, T0K, FUELS, MPH, MILE, GAL, KWH, EPA_KWH_PER_GAL
from .vehicle import Vehicle
from .powertrain import ARCH_CLASSES, ETA_ALT, ETA_DCDC
from .options import resolve as resolve_options, DEFAULT_PHYSICS
from .strategies import make_strategy
from . import thermal as _thermal, aging as _aging

SURFACE_RR = [1.0, 0.95, 1.12, 1.6, 2.2]     # dry asphalt, concrete, wet, gravel, snow (Wong 2008 table 1.1; Ejsmont 2017)
_VEH_CACHE = {}
# with sub-steps, these outputs are states (value at the end of each trip second); the rest are interval means
_STATE_KEYS = frozenset(['v', 'soc', 'T_eng', 'T_batt', 'T_cabin', 'v_target', 'gear'])


def get_vehicle(spec):
    if isinstance(spec, Vehicle):
        return spec
    key = hashlib.md5(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()
    v = _VEH_CACHE.get(key)
    if v is None:
        v = Vehicle(spec)
        if len(_VEH_CACHE) > 256:
            _VEH_CACHE.clear()
        _VEH_CACHE[key] = v
    return v


def isa_density(elev_m, T_C=None):
    p = 101325.0 * (1 - 2.25577e-5 * np.asarray(elev_m, float)) ** 5.25588
    T = (15.0 - 0.0065 * np.asarray(elev_m, float)) if T_C is None else np.asarray(T_C, float)
    return p / (R_AIR * (T + T0K))


def air_density(T_C, p_Pa=None, elev_m=0.0):
    if p_Pa is None:
        p_Pa = 101325.0 * (1 - 2.25577e-5 * np.asarray(elev_m, float)) ** 5.25588
    return np.asarray(p_Pa, float) / (R_AIR * (np.asarray(T_C, float) + T0K))


def normalize_trip(trip):
    """trip dict (contract arrays) -> dict of float arrays with every field present."""
    v = np.asarray(trip['v'], float)
    N = len(v)

    def arr(k, default):
        x = trip.get(k)
        if x is None:
            return np.full(N, default, float)
        x = np.asarray(x, float)
        if x.ndim == 0:
            return np.full(N, float(x))
        x = x.copy()
        bad = ~np.isfinite(x)
        if bad.all():
            return np.full(N, default, float)
        if bad.any():
            idx = np.arange(N)
            x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        return x

    out = dict(v=np.maximum(np.nan_to_num(v), 0.0))
    out['t'] = arr('t', 0.0) if trip.get('t') is not None else np.arange(N, dtype=float)
    out['grade'] = np.clip(arr('grade', 0.0), -0.3, 0.3)
    out['elev'] = arr('elev', 0.0)
    out['T_amb'] = arr('T_amb', 20.0)
    out['wind_speed'] = np.maximum(arr('wind_speed', 0.0), 0.0)
    out['wind_dir'] = arr('wind_dir', 0.0)
    out['heading'] = arr('heading', 0.0)
    rho = trip.get('rho_air')
    if rho is None or not np.isfinite(np.asarray(rho, float)).any():
        out['rho_air'] = air_density(out['T_amb'], elev_m=out['elev'])
    else:
        out['rho_air'] = arr('rho_air', 1.2)
    out['surface'] = np.clip(np.nan_to_num(arr('surface', 0.0)).astype(int), 0, 4)
    out['solar'] = np.maximum(arr('solar', 0.0), 0.0)
    out['N'] = N
    d = np.sum(0.5 * (out['v'][1:] + out['v'][:-1]) * np.diff(out['t'])) if N > 1 else 0.0
    out['distance_m'] = float(d)
    return out


def _f(x):
    return float(x)


_DEFAULT_PHYSICS = object()      # simulate(..., options omitted) -> DEFAULT_PHYSICS; options=None -> physics v1


def simulate(vehicle_spec, trip_inputs, strategy=None, dt=1.0, init_soc=None, init_temps=None, rng=None,
             driver=None, record=True, check=True, ledger_tol=0.005, options=_DEFAULT_PHYSICS):
    """Simulate one trip. See README.md for the full description of inputs and outputs.

    vehicle_spec : dict (contract spec), path/id (see load_vehicle), or a compiled Vehicle
    trip_inputs  : dict with at least 'v' (m/s, 1 Hz); optional grade, elev, T_amb, wind_speed, wind_dir, heading,
                   rho_air, surface, solar [W/m2]
    strategy     : name or {'strategy': name, 'params': {...}} (see list_strategies); default from the spec / architecture
    init_soc     : battery SOC at start (default: HEV mid-window, PHEV/BEV 0.9)
    init_temps   : {'engine': C, 'cabin': C, 'battery': C, 'tire_warm': 0..1} (default: soaked at ambient, cold tires)
    rng          : numpy Generator or seed -> small random driver gain variation (None = deterministic)
    driver       : {'Kp': 1/s, 'Ki': 1/s^2, 'a_max': m/s2, 'a_min': m/s2}
    options      : physics options (roadsim/options.py): a preset name, a dict, or None. Omitted = DEFAULT_PHYSICS
                   ('roadsim-physics-v2' since 13 Nov 2026); None or 'roadsim-physics-v1' = physics v1 exactly; a dict
                   without a 'preset' key applies its values on top of v1
    """
    t_wall = time.perf_counter()
    veh = get_vehicle(vehicle_spec)
    trip = normalize_trip(trip_inputs)
    N = trip['N']
    arch = veh.arch
    strat = make_strategy(arch, strategy if strategy is not None else (veh.control.get('strategy') and
                                                                        {'strategy': veh.control.get('strategy'), 'params': veh.control.get('params', {})}))
    if options is _DEFAULT_PHYSICS:
        options = DEFAULT_PHYSICS
    opt = resolve_options(options, veh.spec)
    strat.opt = opt
    for _m in veh.motors.values():          # B2: light-load motor options, battery state of health (v1 = no change)
        _m.set_options(opt)
    if veh.battery is not None:
        _aging.apply(veh.battery, opt)
    if veh.engine is not None:
        veh.engine._x2 = None
        _thermal.engine_map(veh, opt)        # v3.1: measured engine-era load table (none: the engine's own map)
    PT = ARCH_CLASSES[arch]
    pt = PT(veh, shift=strat.shift) if arch == 'conv' else PT(veh)
    pt.opt = opt
    if arch == 'conv' and getattr(strat, 'idle_stop', False):
        veh.engine.idle_stop = True
    elif arch == 'conv':
        veh.engine.idle_stop = bool(veh.spec['engine'].get('idle_stop', False))
    e = veh.engine; bat = veh.battery; cab = veh.cabin
    hybrid = strat.is_hybrid
    elec = bat is not None
    regen_mode = strat.regen_blend or veh.regen.get('blend', 'series')
    pt.regen_share = float((opt.get('regen_share') or {}).get(arch, 1.0))     # v2 brake-blending calibration (1 = v1)
    if rng is not None and not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # ---------------- driver
    drv = dict(Kp=0.5, Ki=0.05, a_max=4.5 if not veh.hd else 1.5, a_min=-7.5 if not veh.hd else -4.0)
    drv.update(driver or {})
    if rng is not None:
        drv['Kp'] *= float(rng.uniform(0.7, 1.3)); drv['Ki'] *= float(rng.uniform(0.7, 1.3))

    # ---------------- precomputed per-step environment
    v_tgt = trip['v'].tolist()
    T_amb = trip['T_amb'].tolist()
    rho = trip['rho_air'].tolist()
    rel = np.radians(trip['wind_dir'] - trip['heading'])
    w_head = (trip['wind_speed'] * np.cos(rel)).tolist()          # wind FROM ahead -> headwind (+)
    w_cross = np.abs(trip['wind_speed'] * np.sin(rel)).tolist()
    gr = trip['grade']
    gr_mid = np.concatenate([0.5 * (gr[1:] + gr[:-1]), gr[-1:]])
    th = np.arctan(gr_mid)
    sin_t = np.sin(th).tolist(); cos_t = np.cos(th).tolist()
    # rolling resistance multiplier: ISO 28580 ambient temperature correction, cold-tyre pressure, surface
    Tc = np.clip(trip['T_amb'], -30, 45)
    Kt = veh.tire_K_t
    f_T = (1 + Kt * (20 - 25)) / np.maximum(1 + Kt * (Tc - 25), 0.4)
    p0 = 240.0 if not veh.hd else 760.0
    p_g = ((p0 + 101.3) * (Tc + T0K) / (20 + T0K) - 101.3) / p0          # tyres set at 20 C
    f_p = np.maximum(p_g * veh.tire_p_ratio, 0.5) ** -0.4
    if not veh.tire_temp_effects:      # chassis-dyno emulation: road load fixed by the target coefficients
        f_T = np.ones(N); f_p = np.full(N, veh.tire_p_ratio ** -0.4)
    f_s = np.array(SURFACE_RR)[trip['surface']]
    crr_k = (veh.Crr * f_T * f_p * f_s).tolist()
    solar = trip['solar'].tolist()

    m = veh.mass; m_eff = veh.m_eff; CdA = veh.CdA; Crr_v = veh.Crr_v; yaw_k = veh.yaw_k
    mg = m * G; par = veh.parasitic_N
    r = veh.r

    # ---------------- dynamics options (B3: sub-steps, tyre, driver, wind; all off = v1)
    nsub = max(int(opt['substeps'] or 1), 1)
    dt_trip = dt
    if nsub > 1:
        dt = dt_trip / nsub            # from here on dt is the integration step
    M = (N - 1) * nsub                 # integration steps
    if nsub > 1:
        vt_s = np.interp(np.arange(M + 1) / nsub, np.arange(N), trip['v']).tolist()
    else:
        vt_s = v_tgt
    w_head_s, w_cross_s = w_head, w_cross
    f_draft = None
    if opt['wind_profile'] != 'none' or opt['wind_gusts'] or opt['drafting'] or nsub > 1:
        from . import wind as _wind
        from .tyre import CHASSIS as _CH
        hgt = float((veh.spec.get('chassis') or {}).get('height_m') or _CH[veh.cls]['height_m'])
        if opt['wind_profile'] != 'none' or opt['wind_gusts']:
            w_head_s, w_cross_s = _wind.road_wind(trip, trip_inputs, veh, opt, nsub, dt_trip, vt_s, hgt)
        elif nsub > 1:
            w_head_s = np.repeat(w_head, nsub)[:M].tolist(); w_cross_s = np.repeat(w_cross, nsub)[:M].tolist()
        f_draft = _wind.draft_factor(trip_inputs, veh, opt, nsub, vt_s, N)
    tyre_on = bool(opt['tyre_model'])
    if tyre_on:
        from . import tyre as _tyre
        tyr = _tyre.get(veh)
        surf6 = _tyre.surface_codes(trip_inputs, N)
        mu_k = (np.array(_tyre.MU_PEAK)[surf6] * float(opt['tyre_mu_scale'])).tolist()
        ayg_k = (_tyre.lateral_accel(trip, dt_trip) / G).tolist()
        Fcn_k = [tyr.corner_drag(a * G) for a in ayg_k]
        f_s6 = np.array(_tyre.SURFACE_RR6)[surf6] * tyr.crr_ratio(opt['tyre_ref_eu_class'])
        crr_k = (veh.Crr * f_T * f_p * f_s6).tolist()
        regen_frac = float(opt['tyre_regen_mu_frac'])
    human = opt['driver_model'] == 'human'
    if opt['driver_model'] not in ('pi', 'human'):
        raise ValueError(f"driver_model must be 'pi' or 'human' (got {opt['driver_model']!r})")

    # ---------------- states
    it = init_temps or {}
    T0 = T_amb[0]
    st = dict(T_eng=it.get('engine', T0), T_cab=it.get('cabin', T0), T_bat=it.get('battery', T0), v_rc=0.0,
              T_dl=it.get('driveline', it.get('engine', T0)),
              tire=it.get('tire_warm', 0.0), soc=1.0, eng_on=False, t_on=0.0, t_off=99.0, gear=0)
    if elec:
        if init_soc is None:
            init_soc = 0.9 if arch in ('bev', 'phev_series', 'phev_powersplit') else 0.5 * (bat.soc_lo + bat.soc_hi)
        st['soc'] = float(init_soc)
        st['v_rc'] = it.get('v_rc', 0.0)
    pt.reset(st)
    if 'gear' in it:
        st['gear'] = it['gear']
    strat.setup(veh, pt, trip, st)
    if hybrid and it.get('eng_on') is not None:
        st['eng_on'] = bool(it['eng_on'])
    th = _thermal.make(veh, opt, trip, trip_inputs, st, it, dt)     # B2 thermal/HVAC/accessories (None = v1)
    aux_base_o = None if (opt.get('aux_base_W') is None or veh.hd) else float(opt['aux_base_W'])   # v2: light-duty 12 V base load
    _onr = float((opt.get('onroad_aux_W') or {}).get(arch, 0.0))
    if _onr and veh.tire_temp_effects and not veh.hd:      # v2: on-road extra load by architecture (never on a dyno spec)
        aux_base_o = (veh.aux_base if aux_base_o is None else aux_base_o) + _onr

    if human:
        from .driver import HumanDriver
        hdrv = HumanDriver(vt_s, dt, opt['driver_aggressiveness'], opt['driver_reaction_s'], rng,
                           (drv['a_max'], drv['a_min']))

    # ---------------- outputs
    keys = ['v', 'a', 'F_aero', 'F_roll', 'F_grade', 'F_inertia', 'F_wheel', 'F_brake', 'P_wheel', 'gear', 'eng_rpm',
            'eng_torque', 'eng_power', 'fuel_rate', 'eng_on', 'T_eng', 'mot_rpm', 'mot_torque', 'mot_power', 'mot_loss',
            'gen_rpm', 'gen_torque', 'gen_power', 'batt_current', 'batt_voltage', 'soc', 'T_batt', 'aux_power',
            'hvac_power', 'P_brake', 'P_regen', 'T_cabin', 'trans_loss', 'batt_loss', 'v_target', 'rho_air', 'headwind']
    rows = [] if record else None      # one tuple per integration step in `keys` order (columns built after the loop)
    # ledger accumulators (J; the dict L after the loop): fuel, eng_loss, mot_loss, trans_loss, batt_loss, aero, rolling,
    # grade, brake, aux, batt_chem, regen, tyre_slip
    L_fuel = L_eng = L_mot = L_trans = L_bl = L_aero = L_roll = L_grade = L_brake = L_aux = L_chem = L_regen = L_slip = 0.0
    v0 = v_tgt[0]
    v_start = v0
    I_err = 0.0
    fuel_lhv = FUELS[veh.fuel]['lhv'] if e is not None else 1.0
    worst_step = 0.0
    dist = 0.0
    miss_sq = 0.0; miss_max = 0.0; miss_n2 = 0
    sat_prev = False
    tire_tau = veh.tire_warm_tau; tire_amp = veh.tire_warm_amp
    has_box = veh.ttype in ('AT', 'DCT', 'AMT', 'CVT')
    dl_C = 25e3 * (1.0 if not veh.hd else 4.0); dl_hA = 6.0 * (1.0 if not veh.hd else 3.0)
    dl_hc = 25.0 if veh.ttype in ('AT', 'CVT') else 5.0

    row0 = dict.fromkeys(keys, 0.0)    # index 0 of the outputs: initial states
    row0['v'] = v0; row0['soc'] = st['soc']; row0['T_eng'] = st['T_eng']; row0['T_cabin'] = st['T_cab']
    row0['T_batt'] = st['T_bat']; row0['v_target'] = v_tgt[0]; row0['rho_air'] = rho[0]
    if elec:
        row0['batt_voltage'] = bat.ocv(st['soc'])

    # loop invariants (speed only: the same values the loop used to compute every step)
    conv = arch == 'conv'
    Kp = drv['Kp']; Ki = drv['Ki']; a_min = drv['a_min']; a_max = drv['a_max']
    hvac_v1 = th is None or not th.hvac2       # the two-node cabin (thermal.py) does not use the v1 HVAC block
    tms = elec and not conv
    timers = hybrid or conv
    dl_vogel = th is not None and has_box and th.f_dl_vogel
    MPH2 = 2 * MPH
    rec_gear = arch != 'bev'
    if elec:
        ex = math.exp(-dt / bat.tau)
        bat_ocv = bat.ocv; bat_r0 = bat.r0; bat_limits = bat.limits
    pt_plan = pt.plan if conv else None
    if tyre_on:
        tyr_step = tyr.step

    for i in range(M):                 # integration steps; k = trip second (i = k at 1 Hz)
        k = i if nsub == 1 else i // nsub
        Ta = T_amb[k]; rk = rho[k]; wh = w_head_s[i]; sk = sin_t[k]; ck = cos_t[k]
        crr = crr_k[k] * (1.0 + tire_amp * (1.0 - st['tire']))
        if th is not None:
            crr = th.crr(k, st, crr, v0, ck)
        # yaw angle from the cross wind (deg) raises CdA (wind-averaged drag; SAE J1252)
        wc = w_cross_s[i]
        if wc > 0.0:
            psi = math.degrees(math.atan2(wc, max(v0 + wh, 0.5)))
            cda = CdA * (1.0 + yaw_k * min(psi, 30.0) ** 2)
        else:
            cda = CdA
        if f_draft is not None:
            cda *= f_draft[i]
        ka = 0.5 * rk * cda
        Fg = mg * sk
        Frr0 = mg * ck * crr
        Fcn = Fcn_k[k] if tyre_on else 0.0      # cornering (lateral-slip) drag, N

        # ---------------- driver: feed-forward + PI on the lagged speed error (or the human driver model)
        err = vt_s[i] - v0
        # (min / max written as conditionals with the builtins' exact semantics: max(a, b) = b if b > a else a)
        if not sat_prev:
            x_ = I_err + err * dt
            I_err = -5.0 if -5.0 > x_ else x_
            if 5.0 < I_err:
                I_err = 5.0
        if nsub == 1:
            a_cmd = (vt_s[i + 1] - v0) / dt + Kp * err + Ki * I_err
        else:   # same driver in continuous form: target slope + the error closed over one trip second + PI
            a_cmd = (vt_s[i + 1] - vt_s[i]) / dt + err / dt_trip + Kp * err + Ki * I_err
        if vt_s[i + 1] <= 0.0 and err <= 0.0:
            a_cmd = (0.0 - v0) / dt
        if human:
            a_cmd = hdrv.accel(i, v0)
        if a_min > a_cmd:
            a_cmd = a_min
        if a_max < a_cmd:
            a_cmd = a_max
        v1d = v0 + a_cmd * dt
        if v1d < 0.0:
            v1d = 0.0
        stopped = v0 < 1e-3 and v1d < 1e-3

        # ---------------- aux & HVAC
        Q_del = 0.0; Q_eng_heat = 0.0; P_hvac_el = 0.0; P_hvac_mech = 0.0; need_heat = False
        if hvac_v1:
            Qd = cab.demand(st['T_cab'], Ta, solar[k])
            if Qd > 0.0:
                if cab.source == 'engine':
                    Q_av = max(0.0, cab.hx_UA * (st['T_eng'] - st['T_cab']))
                    Q_del = min(Qd, Q_av); Q_eng_heat = Q_del
                    need_heat = hybrid and Qd > 0.3 * cab.Q_heat_max and st['T_eng'] < 55.0
                elif cab.source == 'heat_pump':
                    cop = cab.hp_cop(Ta); Q_del = Qd; P_hvac_el = Qd / cop
                else:
                    Q_del = Qd; P_hvac_el = Qd / 0.95
                P_hvac_el += cab.blower_W
            elif Qd < 0.0:
                Q_del = Qd
                P_comp = -Qd / cab.ac_cop(Ta)
                if conv:
                    P_hvac_mech = P_comp / 0.95
                else:
                    P_hvac_el = P_comp
                P_hvac_el += cab.blower_W
        else:
            Qd = 0.0                   # unused: thermal.aux's two-node cabin replaces this block
        P_tms = 0.0; Q_tms = 0.0
        if tms:
            if bat.heater_W > 0 and st['T_bat'] < bat.T_heat_on:
                Q_tms = bat.heater_W; P_tms = bat.heater_W / 0.95
            elif bat.cooler_W > 0 and st['T_bat'] > bat.T_cool_on:
                Q_tms = -bat.cooler_W; P_tms = bat.cooler_W / 2.5
        base = veh.aux_base if aux_base_o is None else aux_base_o
        if th is not None:
            P_aux_mech, P_aux_el, need_heat, P_hvac_tot = th.aux(k, st, v0, stopped, (Qd, Q_del, Q_eng_heat, P_hvac_el, P_hvac_mech,
                                                                                    need_heat, P_tms, base))
        else:
            if conv:
                P_aux_mech = (base + (P_hvac_el)) / ETA_ALT + P_hvac_mech
                P_aux_el = 0.0
            else:
                P_aux_mech = 0.0
                P_aux_el = (base + cab.blower_W * (Qd != 0.0)) / ETA_DCDC + (P_hvac_el - cab.blower_W * (Qd != 0.0)) + P_tms
            P_hvac_tot = P_hvac_el + P_hvac_mech + P_tms

        if elec:
            p_dis, p_chg = bat_limits(st['soc'], st['T_bat'], st['v_rc'])
        else:
            p_dis = p_chg = 0.0
        # driveline (transmission/axle) oil: spin losses scale with viscosity^0.3 relative to 80 C
        if dl_vogel:
            f_dl = th.f_dl(st['T_dl'])
        else:
            f_dl = math.exp(0.3 * 3776.0 * (1.0 / (max(st['T_dl'], -40.0) + T0K) - 1.0 / (80.0 + T0K))) if has_box else 1.0
        env = dict(P_aux_mech=P_aux_mech, P_aux_el=P_aux_el, p_dis=p_dis, p_chg=p_chg, stopped=stopped,
                   need_heat=need_heat, k=k, f_dl=f_dl)

        # ---------------- requested force at the desired mid-step speed
        u = 0.5 * (v0 + v1d)
        if stopped:
            F_req = 0.0
        else:
            rel_v = u + wh
            F_req = m_eff * (v1d - v0) / dt + ka * rel_v * abs(rel_v) + Frr0 * (1.0 + Crr_v * u) + par + Fg + Fcn

        # ---------------- powertrain plan (iterate if traction-limited so flows match the final speed)
        plan = None; F_brk = 0.0
        converged = False
        u_pt = u; t_lim = False
        if nsub > 1:
            env['start_spread'] = True     # engine-start fuel spread over the first second of running (see _finalize)
        for it_ in range(8):
            env['a'] = 2.0 * (u - v0) / dt
            # each pass asks for the driver's force at the current mean-speed estimate; a limited powertrain returns
            # its maximum at that speed, and the loop converges on the speed that this force produces
            F_ask = F_req; u_pt = u
            if tyre_on and not stopped:   # friction limits at this pass's load transfer; wheels turn at u (1 + slip)
                F_ask, u_pt = tyr_step(F_req, u, env['a'], sk, ck, mu_k[k], ayg_k[k], regen_frac, env, elec)
                t_lim = env['tyre_lim']
            if conv:
                plan = pt_plan(F_ask, u_pt, st, env)       # = _plan for 'conv'
            else:
                plan = _plan(pt, strat, veh, F_ask, u_pt, st, env, regen_mode, hybrid, elec, k, dt, trip, dist)
            Fd = plan['F']
            if stopped:
                F_brk = 0.0; u_new = 0.0; v1 = 0.0; converged = True
                break
            if F_ask < 0.0:
                F_brk = Fd - F_ask               # friction brakes supply the rest of the braking force
                if 0.0 > F_brk:
                    F_brk = 0.0
            else:
                F_brk = 0.0
            F_ext = Fd - F_brk
            if not plan['limited'] and abs(F_ext - F_req) <= 1e-6 * (abs(F_req) + 1.0):
                u_new = u; v1 = v1d; converged = True
                break
            # solve m_eff (2u - 2v0)/dt = F_ext - ka (u+wh)|u+wh| - Frr0 (1 + Crr_v u) - par - Fg  for u
            u_new = _solve_u(v0, F_ext, m_eff, dt, ka, wh, Frr0, Crr_v, par + Fg + Fcn)
            if 2.0 * u_new - v0 < 0.0:
                u_new = 0.5 * v0
            if abs(u_new - u) < 1e-7:
                v1 = 2.0 * u_new - v0; converged = True
                break
            u = u_new if it_ < 3 else 0.5 * (u + u_new)      # damped after a few passes (gear changes make F(u) jumpy)
        if not converged:
            # exact fallback: take the lower speed estimate and ask for exactly the force that produces it
            us = max(min(u, u_new), 0.5 * v0)
            rv = us + wh
            Fs = m_eff * 2.0 * (us - v0) / dt + ka * rv * abs(rv) + Frr0 * (1.0 + Crr_v * us) + par + Fg + Fcn
            env['a'] = 2.0 * (us - v0) / dt
            u_pt = us
            if tyre_on:
                _, u_pt = tyr.step(Fs, us, env['a'], sk, ck, mu_k[k], ayg_k[k], regen_frac, env, elec)
            plan = _plan(pt, strat, veh, Fs, u_pt, st, env, regen_mode, hybrid, elec, k, dt, trip, dist)
            F_brk = max(plan['F'] - Fs, 0.0) if Fs < 0 else 0.0
            u_new = us; v1 = 2.0 * us - v0
        u = u_new
        # aero/rolling/grade work at the final mean speed; brakes take any residual (stop within the step)
        pF = plan['F']
        if stopped or u <= 0.0:
            Fa = 0.0; Fr = 0.0; Fgr = 0.0; F_brk = 0.0; v1 = 0.0
            F_net = 0.0; Fcn = 0.0
        else:
            rel_v = u + wh
            Fa = ka * rel_v * abs(rel_v); Fr = Frr0 * (1.0 + Crr_v * u) + par; Fgr = Fg
            F_net = m_eff * (v1 - v0) / dt
            F_brk = pF - Fa - Fr - Fgr - Fcn - F_net       # exact closure of the momentum balance
            if F_brk < 0.0 and F_brk > -1e-6 * (abs(pF) + 1.0):
                F_brk = 0.0
        # ---------------- battery
        P_b = plan['P_batt']
        if elec:
            soc = st['soc']; Tb = st['T_bat']; v_rc = st['v_rc']
            Voc = bat_ocv(soc)
            R0 = bat_r0(soc, Tb)
            Vt = Voc - v_rc                # Battery.current, inlined (same arithmetic)
            disc = Vt * Vt - 4.0 * R0 * P_b
            if disc < 0.0:
                disc = 0.0
            I = (Vt - math.sqrt(disc)) / (2.0 * R0)
            V = Voc - v_rc - R0 * I
            P_chem = Voc * I
            P_bl = P_chem - P_b
            st['v_rc'] = v_rc * ex + bat.R1 * I * (1.0 - ex)
            st['soc'] = soc - I * dt / bat.Q
            st['T_bat'] = Tb + dt * ((0.0 if 0.0 > P_bl else P_bl) + Q_tms - bat.hA * (Tb - Ta)) / bat.C_th
        else:
            I = 0.0; V = 0.0; P_chem = 0.0; P_bl = 0.0
        # ---------------- engine thermal, cabin, tyres
        running = plan.get('eng_on', False)
        if th is not None:
            th.post(k, st, plan, u, running, F_brk, dt)
        else:
            if e is not None:
                st['T_eng'] = e.thermal_step(st['T_eng'], plan['P_fuel'], plan['P_eng'], Ta, Q_eng_heat, dt, running)
            st['T_cab'] = cab.step(st['T_cab'], Ta, solar[k], Q_del, dt)
        Ptr = plan['P_trans']
        if has_box:   # driveline oil heated by its own losses and (ATF cooler) by the engine coolant
            T_dl = st['T_dl']
            q_dl = 0.5 * (0.0 if 0.0 > Ptr else Ptr) + (dl_hc * (st['T_eng'] - T_dl) if e is not None else 0.0)
            x_ = T_dl + dt * (q_dl - dl_hA * (T_dl - Ta)) / dl_C
            st['T_dl'] = 95.0 if 95.0 < x_ else x_
        if u > 0.5:
            st['tire'] = st['tire'] + (1.0 - st['tire']) * dt / tire_tau
        if timers:
            if running:
                st['t_on'] = st['t_on'] + dt if st['eng_on'] else dt
                st['t_off'] = 0.0
            else:
                st['t_off'] = st['t_off'] + dt if not st['eng_on'] else dt
                st['t_on'] = 0.0
            st['eng_on'] = running
            st['t_runon'] = st.get('t_runon', 0.0) + dt if plan.get('runon') else 0.0
        st['gear'] = plan.get('gear', 0)
        if hybrid:
            strat.step_end(st, plan, dict(dt=dt))

        # ---------------- ledger
        Pf = plan['P_fuel']; Pe = plan['P_eng']; Pm = plan['P_mot']; Pa = plan['P_aux']
        L_fuel += Pf * dt
        L_eng += (Pf - Pe) * dt
        L_mot += Pm * dt
        L_trans += Ptr * dt
        L_bl += P_bl * dt
        L_chem += P_chem * dt
        L_aux += Pa * dt
        L_aero += Fa * u * dt
        L_roll += Fr * u * dt
        L_grade += Fgr * u * dt
        if tyre_on:
            # tyre slip: powertrain force x (wheel-rim speed - vehicle speed), the friction brakes' share of the slip,
            # and cornering drag; the brakes keep the rest of their work
            sl_b = F_brk * u * dt * abs(env.get('tyre_kappa', 0.0)) if F_req < 0.0 else 0.0
            L_brake += F_brk * u * dt - sl_b
            L_slip += pF * (u_pt - u) * dt + Fcn * u * dt + sl_b
        else:
            L_brake += F_brk * u * dt
        P_reg = -P_b + Pa if (F_req < 0.0 and elec) else 0.0
        if P_reg > 0:
            L_regen += P_reg * dt
        # per-step closure (diagnostic)
        lhs = Pf + P_chem
        rhs = (Pf - Pe) + Pm + Ptr + P_bl + Pa + (Fa + Fr + Fgr + F_brk) * u + 0.5 * m_eff * (v1 * v1 - v0 * v0) / dt
        if tyre_on:
            rhs += Fcn * u + pF * (u_pt - u)
        x_ = abs(lhs - rhs)
        if x_ > worst_step:
            worst_step = x_
        dist += u * dt
        if record:
            g_ = plan.get
            rows.append((v1, (v1 - v0) / dt, Fa, Fr, Fgr, m_eff * (v1 - v0) / dt, pF, F_brk, pF * u,
                         g_('gear', 0) + 1 if rec_gear else 1, g_('w_e', 0.0) * 30 / math.pi, g_('T_e', 0.0), Pe,
                         Pf / fuel_lhv * 1e3 if e is not None else 0.0, 1.0 if running else 0.0, st['T_eng'],
                         g_('w_m', 0.0) * 30 / math.pi, g_('T_m', 0.0), g_('P_m_el', 0.0), Pm,
                         g_('w_g', 0.0) * 30 / math.pi, g_('T_g', 0.0), g_('P_g_el', 0.0),
                         I, V, st['soc'], st['T_bat'], Pa, P_hvac_tot, F_brk * u, 0.0 if 0.0 > P_reg else P_reg, st['T_cab'], Ptr,
                         P_bl, vt_s[i + 1], rk, wh))
        if nsub == 1 or i % nsub == nsub - 1:      # trace-miss statistics at the trip seconds
            e1 = v1 - v_tgt[k + 1]
            ae1 = abs(e1)
            miss_sq += e1 * e1
            if ae1 > miss_max:
                miss_max = ae1
            if ae1 > MPH2: miss_n2 += 1
        sat_prev = plan['limited'] or t_lim
        v0 = v1

    L = dict(fuel=L_fuel, eng_loss=L_eng, mot_loss=L_mot, trans_loss=L_trans, batt_loss=L_bl, aero=L_aero, rolling=L_roll,
             grade=L_grade, brake=L_brake, aux=L_aux, batt_chem=L_chem, regen=L_regen, tyre_slip=L_slip)
    if record:                         # output columns: index 0 = initial states, step i at index i + 1
        A_ = np.fromiter(_chain_it.from_iterable(rows), float, count=M * len(keys)).reshape(M, len(keys))
        O = {}
        for c_, k_ in enumerate(keys):
            col = np.empty(M + 1)
            col[0] = row0[k_]
            col[1:] = A_[:, c_]
            O[k_] = col
        del rows, A_

    # ---------------- ledger summary (kJ)
    dKE = 0.5 * m_eff * (v0 * v0 - v_start * v_start)
    led = {k_: v_ / 1e3 for k_, v_ in L.items()}
    led['kinetic_change'] = dKE / 1e3
    led['batt_stored_change'] = -L['batt_chem'] / 1e3
    sources = L['fuel'] + L['batt_chem']
    sinks = (L['eng_loss'] + L['mot_loss'] + L['trans_loss'] + L['batt_loss'] + L['aux'] + L['aero'] + L['rolling']
             + L['grade'] + L['brake'] + dKE)
    if tyre_on:
        sinks += L['tyre_slip']
    resid = sources - sinks
    scale = L['fuel'] + abs(L['batt_chem']) + abs(dKE) + abs(L['grade']) + L['brake'] + 1e-9
    scale = max(scale, L['aero'] + L['rolling'] + L['aux'] + L['brake'] + 1e-9)
    led['residual'] = resid / 1e3
    led['closure_rel'] = abs(resid) / scale
    led['worst_step_W'] = worst_step
    led['ok'] = bool(led['closure_rel'] < ledger_tol)
    led['batt_chem_out'] = led.pop('batt_chem')
    if check and not led['ok']:
        import warnings
        warnings.warn('energy ledger does not close: %.3g %%' % (100 * led['closure_rel']))

    fuel_g = L['fuel'] / fuel_lhv * 1e3 if e is not None else 0.0
    dist_m = dist
    summ = dict(distance_m=dist_m, duration_s=float(trip['t'][-1] - trip['t'][0]), fuel_g=fuel_g,
                fuel_L=fuel_g / FUELS[veh.fuel]['density'] if e is not None else 0.0,
                batt_kWh_out=L['batt_chem'] / KWH, soc_start=float(init_soc) if elec else None,
                soc_end=st['soc'] if elec else None,
                trace_rms_mps=math.sqrt(miss_sq / max(N - 1, 1)), trace_max_mps=miss_max,
                trace_frac_over_2mph=miss_n2 / max(N - 1, 1))
    if dist_m > 0:
        summ['L_per_100km'] = summ['fuel_L'] / (dist_m / 1e5)
        summ['mpg'] = (dist_m / MILE) / (summ['fuel_L'] / GAL) if summ['fuel_L'] > 0 else float('inf')
        summ['kWh_per_100km_dc'] = L['batt_chem'] / KWH / (dist_m / 1e5)
    summ['wall_s'] = time.perf_counter() - t_wall
    summ['x_realtime'] = summ['duration_s'] / max(summ['wall_s'], 1e-9)
    hf = None
    if record and nsub > 1:   # sub-step arrays -> per-second outputs: interval means, states at the second's end
        hf = {k_: np.asarray(v_, float) for k_, v_ in O.items()}
        O = {}
        for k_, x in hf.items():
            y = x[1:].reshape(N - 1, nsub)
            O[k_] = np.concatenate([x[:1], y[:, -1] if k_ in _STATE_KEYS else y.mean(axis=1)])
    out = {k_: np.asarray(v_, np.float32) for k_, v_ in O.items()} if record else {}
    if record:
        out['t'] = np.asarray(trip['t'], np.float32)
        if hf is not None and opt['substep_record']:
            out['hf'] = {k_: x.astype(np.float32) for k_, x in hf.items()}
            out['hf']['t'] = (trip['t'][0] + np.arange(M + 1) * dt).astype(np.float32)
    out['ledger'] = led
    out['summary'] = summ
    out['final_state'] = dict(engine=st['T_eng'], cabin=st['T_cab'], battery=st['T_bat'], tire_warm=st['tire'], driveline=st['T_dl'],
                              soc=st['soc'], v_rc=st['v_rc'], eng_on=st['eng_on'], gear=st['gear'])
    if th is not None:
        out['final_state'].update(th.final_state())
        th.done()
    out['strategy'] = strat.name
    out['vehicle_id'] = veh.id
    return out


def _solve_u(v0, F_ext, m_eff, dt, ka, wh, Frr0, Crr_v, Fc):
    """mean speed u solving 2 m_eff (u - v0)/dt = F_ext - ka (u+wh)|u+wh| - Frr0 (1+Crr_v u) - Fc (Newton, safeguarded)."""
    c = 2.0 * m_eff / dt
    u = max(v0, 0.0)
    for _ in range(30):
        rv = u + wh
        f = c * (u - v0) - F_ext + ka * rv * abs(rv) + Frr0 * (1.0 + Crr_v * u) + Fc
        df = c + 2.0 * ka * abs(rv) + Frr0 * Crr_v
        du = f / df
        u -= du
        if abs(du) < 1e-10:
            break
    return u


def _plan(pt, strat, veh, F, u, st, env, regen_mode, hybrid, elec, k, dt, trip, dist):
    """turn a requested wheel force into a plan for the architecture (traction or braking)."""
    arch = veh.arch
    if arch == 'conv':
        return pt.plan(F, u, st, env)
    r = veh.r
    if F < 0.0 and not env['stopped']:
        # ---- braking: regenerative share by blending rule, engine unfuelled
        F_cap = _regen_cap(pt, veh, u, st, env)
        Fb = -F
        if regen_mode == 'parallel':
            share = veh.regen.get('parallel_share', 0.6)
            F_reg = min(share * Fb, F_cap)
        elif regen_mode == 'one_pedal':
            F_reg = min(Fb, max(F_cap, 0.0))
        else:
            F_reg = min(getattr(pt, 'regen_share', 1.0) * Fb, F_cap)
        if arch == 'bev':
            plan = pt.plan(-F_reg, u, st, env)
        else:
            plan = _runon(pt, veh, strat, -F_reg, u, st, env)
            if plan is None:
                plan = _hybrid_fixed(pt, veh, -F_reg, u, st, env, ctrl0=True)
        plan['limited'] = False
        return plan
    if arch == 'bev':
        return pt.plan(F, u, st, env)
    # ---- hybrid traction
    w_w = u / r
    T_w = F * r
    ctx = dict(u=u, moving=not env['stopped'], need_heat=env['need_heat'], dt=dt, k=k, dist=dist,
               P_aux_el=env['P_aux_el'], p_dis=env['p_dis'], p_chg=env['p_chg'], P_dem_mech=F * u)
    ctx['P_dem_el'] = F * u / 0.85 + env['P_aux_el'] if F > 0 else env['P_aux_el']
    if arch == 'hev_parallel_p2':
        T_o = pt.to_box_out(T_w); w_o = w_w * pt.FD
        g = pt.conv.choose_gear(max(T_o, 0.0), w_o, st, max(F * u, 0.0)) if not env['stopped'] else 0
        env = dict(env, gear=g)
        gr_ = veh.gears[g]; w_in = w_o * gr_
        T_in = pt.conv.box_in(T_o, w_in, gr_)
        w_e = max(w_in, veh.engine.w_min_load)
        ctx['T_in_frac'] = T_in / max(veh.engine.t_max(w_e), 1.0)
    if env['stopped']:
        ctrl = np.array([0.0])
        if hybrid and strat.forced_on(st, ctx) and arch in ('hev_series', 'phev_series', 'hev_powersplit', 'phev_powersplit'):
            # engine runs at a stop (warm-up / cabin heat): largest of a few charging powers the battery can accept
            P0 = max(0.06 * veh.engine.P_max, 3000.0)
            ctrl = np.array([P0, 0.6 * P0, 0.35 * P0, 1200.0, 600.0])
        C = pt.evaluate(0.0, 0.0, ctrl, st, env)
        j = int(np.argmax(C['ok'])) if C['ok'].any() else -1
        if j < 0:
            C = pt.evaluate(0.0, 0.0, np.array([0.0]), st, env); j = 0
        plan = _finalize(pt, veh, C, j, env, st, strat)
        plan['F'] = 0.0
        plan['limited'] = False
        return plan
    ctrl = strat.candidates(st, ctx)
    C = pt.evaluate(T_w, w_w, ctrl, st, env)
    j = strat.select(C, ctrl, st, ctx)
    if j < 0:
        # rule's choice infeasible -> nearest feasible on the full grid, else traction-limited
        grid = strat.grid(ctx)
        C = pt.evaluate(T_w, w_w, grid, st, env)
        if C['ok'].any():
            tgt = float(np.max(ctrl)) if len(ctrl) else 0.0
            j = strat.pick_closest(C, tgt, grid, st)
        else:
            scales = np.linspace(1.0, 0.0, 41)[1:]
            TT = (T_w * scales)[:, None] * np.ones((1, len(grid)))
            CC = pt.evaluate(TT, w_w, np.broadcast_to(grid, TT.shape), st, env)
            okrow = CC['ok'].any(axis=1)
            if okrow.any():
                i = int(np.argmax(okrow))
                hi = 1.0 if i == 0 else scales[i - 1]
                fine = np.linspace(hi, scales[i], 26)          # refine between the last infeasible and first feasible scale
                TT = (T_w * fine)[:, None] * np.ones((1, len(grid)))
                CC = pt.evaluate(TT, w_w, np.broadcast_to(grid, TT.shape), st, env)
                okf = CC['ok'].any(axis=1)
                sc = fine[int(np.argmax(okf))] if okf.any() else scales[i]
                C = pt.evaluate(T_w * sc, w_w, grid, st, env)
                J = C['P_fuel'] + 2.5 * C['P_batt']
                j = int(np.argmin(np.where(C['ok'], J, np.inf)))
            else:
                C = pt.evaluate(0.0, w_w, grid[:1], st, env); j = 0
                C['ok'][:] = True
        plan = _finalize(pt, veh, C, j, env, st, strat)
        plan['limited'] = plan['F'] < F - 1e-6 * abs(F) - 1e-9
        return plan
    plan = _finalize(pt, veh, C, j, env, st, strat)
    plan['limited'] = False
    return plan


def _regen_cap(pt, veh, u, st, env):
    """regenerative braking force cap (N). v1: motor, battery acceptance and the fixed regen.max_decel_g. With the tyre
    model the fixed deceleration is replaced by the driven axle's friction with a stability margin (tyre.py)."""
    cap_t = env.get('tyre_regen_cap')
    if cap_t is None:
        return pt.regen_force_cap(u, st, env['P_aux_el'], st['T_bat'], st['soc'], st['v_rc'])
    rg = veh.regen
    had = 'max_decel_g' in rg; old = rg.get('max_decel_g')
    rg['max_decel_g'] = 1e6          # motor and battery limits only (the powertrain's own code, unchanged)
    try:
        cap = pt.regen_force_cap(u, st, env['P_aux_el'], st['T_bat'], st['soc'], st['v_rc'])
    finally:
        if had:
            rg['max_decel_g'] = old
        else:
            del rg['max_decel_g']
    return min(cap, cap_t)


def _runon(pt, veh, strat, F, u, st, env):
    """option hybrid_runon_s (v2.1): an engine that is running when braking begins keeps running for this many seconds
    at the strategy's lowest engine-on point (charging the battery on top of regen), as the D3 dev hybrids do (median
    run-on 1-3 s at 0.25-0.8 g/s). None if off, not applicable or infeasible (then the engine stops, as in v1)."""
    o = getattr(strat, 'o', None) or {}
    T = o.get('hybrid_runon_s')
    if not T or not st.get('eng_on') or st.get('t_runon', 0.0) >= T or veh.engine is None:
        return None
    w_w = u / veh.r
    if veh.arch == 'hev_parallel_p2':
        env = dict(env, gear=pt._gear_for_regen(w_w * pt.FD))
        ctrl = np.array([0.15, 0.08])
    else:
        P0 = max(0.08 * veh.engine.P_max, 3000.0)
        ctrl = np.array([P0, 0.5 * P0, 1200.0])
    C = pt.evaluate(F * veh.r, w_w, ctrl, st, env)
    ok = np.flatnonzero(C['ok'] & C['on'])
    if not len(ok):
        return None
    plan = _finalize(pt, veh, C, int(ok[0]), env, st, strat)
    plan['runon'] = True
    return plan


def _hybrid_fixed(pt, veh, F, u, st, env, ctrl0=True):
    """hybrid in braking: engine unfuelled (ctrl 0), motor regenerates the requested (negative) force."""
    w_w = u / veh.r
    if veh.arch == 'hev_parallel_p2':
        w_o = w_w * pt.FD
        env = dict(env, gear=pt._gear_for_regen(w_o))
    C = pt.evaluate(F * veh.r, w_w, np.array([0.0]), st, env)
    if not C['ok'][0]:
        # regen request exceeded a limit (rare: cap is approximate) -> back off until feasible
        for s in np.linspace(0.95, 0.0, 20):
            C = pt.evaluate(F * veh.r * s, w_w, np.array([0.0]), st, env)
            if C['ok'][0]:
                break
    return _finalize(pt, veh, C, 0, env, st, None)


def _finalize(pt, veh, C, j, env, st, strat):
    # (x.item(j) = float(x[j]) for the 1-D float arrays of C, without the numpy scalar in between)
    on = bool(C['on'][j])
    Pf = C['P_fuel'].item(j)
    if on and not st.get('eng_on', False) and strat is not None and hasattr(strat, 'start_W'):
        Pf += strat.start_W            # start-up fuel (cranking, enrichment) booked as engine loss
    elif on and env.get('start_spread') and st.get('t_on', 99.0) < 1.0 - 1e-9 and strat is not None and hasattr(strat, 'start_W'):
        Pf += strat.start_W            # sub-steps: the start cost (start_W over one second) lasts the whole first second
    plan = dict(F=C['T_w'].item(j) / veh.r, P_fuel=Pf, P_eng=C['P_e'].item(j) if on or veh.arch in ('hev_powersplit', 'phev_powersplit') else 0.0,
                w_e=C['w_e'].item(j), T_e=C['T_e'].item(j), eng_on=on, gear=int(C.get('gear', 0) or 0),
                P_trans=C['P_trans'].item(j), P_mot=C['P_mot'].item(j), P_batt=C['P_batt'].item(j),
                P_aux=env['P_aux_el'])
    for a in ('T_m', 'w_m', 'P_m_el', 'T_g', 'w_g', 'P_g_el'):
        if a in C:
            plan[a] = C[a].item(j)
    if 'T2' in C:
        plan['T_m'] = C['T2'].item(j); plan['w_m'] = C['w2'].item(j)
        plan['T_g'] = C['T1'].item(j); plan['w_g'] = C['w_s'].item(j)
    return plan
