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
import numpy as np
from .constants import G, R_AIR, T0K, FUELS, MPH, MILE, GAL, KWH, EPA_KWH_PER_GAL
from .vehicle import Vehicle
from .powertrain import ARCH_CLASSES, ETA_ALT, ETA_DCDC
from .strategies import make_strategy

SURFACE_RR = [1.0, 0.95, 1.12, 1.6, 2.2]     # dry asphalt, concrete, wet, gravel, snow (Wong 2008 table 1.1; Ejsmont 2017)
_VEH_CACHE = {}


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


def simulate(vehicle_spec, trip_inputs, strategy=None, dt=1.0, init_soc=None, init_temps=None, rng=None,
             driver=None, record=True, check=True, ledger_tol=0.005):
    """Simulate one trip. See README.md for the full description of inputs and outputs.

    vehicle_spec : dict (contract spec), path/id (see load_vehicle), or a compiled Vehicle
    trip_inputs  : dict with at least 'v' (m/s, 1 Hz); optional grade, elev, T_amb, wind_speed, wind_dir, heading,
                   rho_air, surface, solar [W/m2]
    strategy     : name or {'strategy': name, 'params': {...}} (see list_strategies); default from the spec / architecture
    init_soc     : battery SOC at start (default: HEV mid-window, PHEV/BEV 0.9)
    init_temps   : {'engine': C, 'cabin': C, 'battery': C, 'tire_warm': 0..1} (default: soaked at ambient, cold tires)
    rng          : numpy Generator or seed -> small random driver gain variation (None = deterministic)
    driver       : {'Kp': 1/s, 'Ki': 1/s^2, 'a_max': m/s2, 'a_min': m/s2}
    """
    t_wall = time.perf_counter()
    veh = get_vehicle(vehicle_spec)
    trip = normalize_trip(trip_inputs)
    N = trip['N']
    arch = veh.arch
    strat = make_strategy(arch, strategy if strategy is not None else (veh.control.get('strategy') and
                                                                        {'strategy': veh.control.get('strategy'), 'params': veh.control.get('params', {})}))
    PT = ARCH_CLASSES[arch]
    pt = PT(veh, shift=strat.shift) if arch == 'conv' else PT(veh)
    if arch == 'conv' and getattr(strat, 'idle_stop', False):
        veh.engine.idle_stop = True
    elif arch == 'conv':
        veh.engine.idle_stop = bool(veh.spec['engine'].get('idle_stop', False))
    e = veh.engine; bat = veh.battery; cab = veh.cabin
    hybrid = strat.is_hybrid
    elec = bat is not None
    regen_mode = strat.regen_blend or veh.regen.get('blend', 'series')
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

    # ---------------- outputs
    keys = ['v', 'a', 'F_aero', 'F_roll', 'F_grade', 'F_inertia', 'F_wheel', 'F_brake', 'P_wheel', 'gear', 'eng_rpm',
            'eng_torque', 'eng_power', 'fuel_rate', 'eng_on', 'T_eng', 'mot_rpm', 'mot_torque', 'mot_power', 'mot_loss',
            'gen_rpm', 'gen_torque', 'gen_power', 'batt_current', 'batt_voltage', 'soc', 'T_batt', 'aux_power',
            'hvac_power', 'P_brake', 'P_regen', 'T_cabin', 'trans_loss', 'batt_loss', 'v_target', 'rho_air', 'headwind']
    O = {k: [0.0] * N for k in keys} if record else None
    L = dict(fuel=0.0, eng_loss=0.0, mot_loss=0.0, trans_loss=0.0, batt_loss=0.0, aero=0.0, rolling=0.0, grade=0.0,
             brake=0.0, aux=0.0, batt_chem=0.0, regen=0.0)
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

    def batt_state(v_rc_, soc_, Tb):
        if not elec:
            return 0.0, 0.0
        return bat.limits(soc_, Tb, v_rc_)

    if record:
        O['v'][0] = v0; O['soc'][0] = st['soc']; O['T_eng'][0] = st['T_eng']; O['T_cabin'][0] = st['T_cab']
        O['T_batt'][0] = st['T_bat']; O['v_target'][0] = v_tgt[0]; O['rho_air'][0] = rho[0]
        if elec:
            O['batt_voltage'][0] = bat.ocv(st['soc'])

    for k in range(N - 1):
        Ta = T_amb[k]; rk = rho[k]; wh = w_head[k]; sk = sin_t[k]; ck = cos_t[k]
        crr = crr_k[k] * (1.0 + tire_amp * (1.0 - st['tire']))
        # yaw angle from the cross wind (deg) raises CdA (wind-averaged drag; SAE J1252)
        if w_cross[k] > 0.0:
            psi = math.degrees(math.atan2(w_cross[k], max(v0 + wh, 0.5)))
            cda = CdA * (1.0 + yaw_k * min(psi, 30.0) ** 2)
        else:
            cda = CdA
        ka = 0.5 * rk * cda
        Fg = mg * sk
        Frr0 = mg * ck * crr

        # ---------------- driver: feed-forward + PI on the lagged speed error
        err = v_tgt[k] - v0
        if not sat_prev:
            I_err = min(max(I_err + err * dt, -5.0), 5.0)
        a_cmd = (v_tgt[k + 1] - v0) / dt + drv['Kp'] * err + drv['Ki'] * I_err
        if v_tgt[k + 1] <= 0.0 and err <= 0.0:
            a_cmd = (0.0 - v0) / dt
        a_cmd = min(max(a_cmd, drv['a_min']), drv['a_max'])
        v1d = v0 + a_cmd * dt
        if v1d < 0.0:
            v1d = 0.0
        stopped = v0 < 1e-3 and v1d < 1e-3

        # ---------------- aux & HVAC
        Qd = cab.demand(st['T_cab'], Ta, solar[k])
        Q_del = 0.0; Q_eng_heat = 0.0; P_hvac_el = 0.0; P_hvac_mech = 0.0; need_heat = False
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
            if arch == 'conv':
                P_hvac_mech = P_comp / 0.95
            else:
                P_hvac_el = P_comp
            P_hvac_el += cab.blower_W
        P_tms = 0.0; Q_tms = 0.0
        if elec and arch != 'conv':
            if bat.heater_W > 0 and st['T_bat'] < bat.T_heat_on:
                Q_tms = bat.heater_W; P_tms = bat.heater_W / 0.95
            elif bat.cooler_W > 0 and st['T_bat'] > bat.T_cool_on:
                Q_tms = -bat.cooler_W; P_tms = bat.cooler_W / 2.5
        base = veh.aux_base
        if arch == 'conv':
            P_aux_mech = (base + (P_hvac_el)) / ETA_ALT + P_hvac_mech
            P_aux_el = 0.0
        else:
            P_aux_mech = 0.0
            P_aux_el = (base + cab.blower_W * (Qd != 0.0)) / ETA_DCDC + (P_hvac_el - cab.blower_W * (Qd != 0.0)) + P_tms
        P_hvac_tot = P_hvac_el + P_hvac_mech + P_tms

        p_dis, p_chg = batt_state(st['v_rc'], st['soc'], st['T_bat'])
        # driveline (transmission/axle) oil: spin losses scale with viscosity^0.3 relative to 80 C
        f_dl = math.exp(0.3 * 3776.0 * (1.0 / (max(st['T_dl'], -40.0) + T0K) - 1.0 / (80.0 + T0K))) if has_box else 1.0
        env = dict(P_aux_mech=P_aux_mech, P_aux_el=P_aux_el, p_dis=p_dis, p_chg=p_chg, stopped=stopped,
                   need_heat=need_heat, k=k, f_dl=f_dl)

        # ---------------- requested force at the desired mid-step speed
        u = 0.5 * (v0 + v1d)
        if stopped:
            F_req = 0.0
        else:
            rel_v = u + wh
            F_req = m_eff * (v1d - v0) / dt + ka * rel_v * abs(rel_v) + Frr0 * (1.0 + Crr_v * u) + par + Fg

        # ---------------- powertrain plan (iterate if traction-limited so flows match the final speed)
        plan = None; F_brk = 0.0
        converged = False
        u_hist = []
        for it_ in range(8):
            env['a'] = 2.0 * (u - v0) / dt
            # each pass asks for the driver's force at the current mean-speed estimate; a limited powertrain returns
            # its maximum at that speed, and the loop converges on the speed that this force produces
            plan = _plan(pt, strat, veh, F_req, u, st, env, regen_mode, hybrid, elec, k, dt, trip, dist)
            Fd = plan['F']
            if stopped:
                F_brk = 0.0; u_new = 0.0; v1 = 0.0; converged = True
                break
            if F_req < 0.0:
                F_brk = max(Fd - F_req, 0.0)     # friction brakes supply the rest of the braking force
            else:
                F_brk = 0.0
            F_ext = Fd - F_brk
            if not plan['limited'] and abs(F_ext - F_req) <= 1e-6 * (abs(F_req) + 1.0):
                u_new = u; v1 = v1d; converged = True
                break
            # solve m_eff (2u - 2v0)/dt = F_ext - ka (u+wh)|u+wh| - Frr0 (1 + Crr_v u) - par - Fg  for u
            u_new = _solve_u(v0, F_ext, m_eff, dt, ka, wh, Frr0, Crr_v, par + Fg)
            if 2.0 * u_new - v0 < 0.0:
                u_new = 0.5 * v0
            if abs(u_new - u) < 1e-7:
                v1 = 2.0 * u_new - v0; converged = True
                break
            u_hist.append(u)
            u = u_new if it_ < 3 else 0.5 * (u + u_new)      # damped after a few passes (gear changes make F(u) jumpy)
        if not converged:
            # exact fallback: take the lower speed estimate and ask for exactly the force that produces it
            us = max(min(u, u_new), 0.5 * v0)
            rv = us + wh
            Fs = m_eff * 2.0 * (us - v0) / dt + ka * rv * abs(rv) + Frr0 * (1.0 + Crr_v * us) + par + Fg
            env['a'] = 2.0 * (us - v0) / dt
            plan = _plan(pt, strat, veh, Fs, us, st, env, regen_mode, hybrid, elec, k, dt, trip, dist)
            F_brk = max(plan['F'] - Fs, 0.0) if Fs < 0 else 0.0
            u_new = us; v1 = 2.0 * us - v0
        u = u_new
        u = u_new
        # aero/rolling/grade work at the final mean speed; brakes take any residual (stop within the step)
        if stopped or u <= 0.0:
            Fa = 0.0; Fr = 0.0; Fgr = 0.0; F_brk = 0.0; v1 = 0.0
            F_net = 0.0
        else:
            rel_v = u + wh
            Fa = ka * rel_v * abs(rel_v); Fr = Frr0 * (1.0 + Crr_v * u) + par; Fgr = Fg
            F_net = m_eff * (v1 - v0) / dt
            F_brk = plan['F'] - Fa - Fr - Fgr - F_net       # exact closure of the momentum balance
            if F_brk < 0.0 and F_brk > -1e-6 * (abs(plan['F']) + 1.0):
                F_brk = 0.0
        # ---------------- battery
        P_b = plan['P_batt']
        if elec:
            soc = st['soc']; Tb = st['T_bat']
            I = bat.current(P_b, soc, Tb, st['v_rc'])
            Voc = bat.ocv(soc)
            V = Voc - st['v_rc'] - bat.r0(soc, Tb) * I
            P_chem = Voc * I
            P_bl = P_chem - P_b
            ex = math.exp(-dt / bat.tau)
            st['v_rc'] = st['v_rc'] * ex + bat.R1 * I * (1.0 - ex)
            st['soc'] = soc - I * dt / bat.Q
            st['T_bat'] = Tb + dt * (max(P_bl, 0.0) + Q_tms - bat.hA * (Tb - Ta)) / bat.C_th
        else:
            I = 0.0; V = 0.0; P_chem = 0.0; P_bl = 0.0
        # ---------------- engine thermal, cabin, tyres
        running = plan.get('eng_on', False)
        if e is not None:
            st['T_eng'] = e.thermal_step(st['T_eng'], plan['P_fuel'], plan['P_eng'], Ta, Q_eng_heat, dt, running)
        st['T_cab'] = cab.step(st['T_cab'], Ta, solar[k], Q_del, dt)
        if has_box:   # driveline oil heated by its own losses and (ATF cooler) by the engine coolant
            q_dl = 0.5 * max(plan['P_trans'], 0.0) + (dl_hc * (st['T_eng'] - st['T_dl']) if e is not None else 0.0)
            st['T_dl'] = min(st['T_dl'] + dt * (q_dl - dl_hA * (st['T_dl'] - Ta)) / dl_C, 95.0)
        if u > 0.5:
            st['tire'] = st['tire'] + (1.0 - st['tire']) * dt / tire_tau
        if hybrid or arch == 'conv':
            if running:
                st['t_on'] = st['t_on'] + dt if st['eng_on'] else dt
                st['t_off'] = 0.0
            else:
                st['t_off'] = st['t_off'] + dt if not st['eng_on'] else dt
                st['t_on'] = 0.0
            st['eng_on'] = running
        st['gear'] = plan.get('gear', 0)
        if hybrid:
            strat.step_end(st, plan, dict(dt=dt))

        # ---------------- ledger
        Pf = plan['P_fuel']
        L['fuel'] += Pf * dt
        L['eng_loss'] += (Pf - plan['P_eng']) * dt
        L['mot_loss'] += plan['P_mot'] * dt
        L['trans_loss'] += plan['P_trans'] * dt
        L['batt_loss'] += P_bl * dt
        L['batt_chem'] += P_chem * dt
        L['aux'] += plan['P_aux'] * dt
        L['aero'] += Fa * u * dt
        L['rolling'] += Fr * u * dt
        L['grade'] += Fgr * u * dt
        L['brake'] += F_brk * u * dt
        P_reg = -plan['P_batt'] + plan['P_aux'] if (F_req < 0.0 and elec) else 0.0
        if P_reg > 0:
            L['regen'] += P_reg * dt
        # per-step closure (diagnostic)
        lhs = Pf + P_chem
        rhs = (Pf - plan['P_eng']) + plan['P_mot'] + plan['P_trans'] + P_bl + plan['P_aux'] + (Fa + Fr + Fgr + F_brk) * u + 0.5 * m_eff * (v1 * v1 - v0 * v0) / dt
        worst_step = max(worst_step, abs(lhs - rhs))
        dist += u * dt
        if record:
            j = k + 1
            O['v'][j] = v1; O['a'][j] = (v1 - v0) / dt; O['F_aero'][j] = Fa; O['F_roll'][j] = Fr; O['F_grade'][j] = Fgr
            O['F_inertia'][j] = m_eff * (v1 - v0) / dt; O['F_wheel'][j] = plan['F']; O['F_brake'][j] = F_brk
            O['P_wheel'][j] = plan['F'] * u; O['gear'][j] = plan.get('gear', 0) + 1 if arch != 'bev' else 1
            we = plan.get('w_e', 0.0)
            O['eng_rpm'][j] = we * 30 / math.pi; O['eng_torque'][j] = plan.get('T_e', 0.0); O['eng_power'][j] = plan['P_eng']
            O['fuel_rate'][j] = Pf / fuel_lhv * 1e3 if e is not None else 0.0
            O['eng_on'][j] = 1.0 if running else 0.0; O['T_eng'][j] = st['T_eng']
            O['mot_rpm'][j] = plan.get('w_m', 0.0) * 30 / math.pi; O['mot_torque'][j] = plan.get('T_m', 0.0)
            O['mot_power'][j] = plan.get('P_m_el', 0.0); O['mot_loss'][j] = plan['P_mot']
            O['gen_rpm'][j] = plan.get('w_g', 0.0) * 30 / math.pi; O['gen_torque'][j] = plan.get('T_g', 0.0)
            O['gen_power'][j] = plan.get('P_g_el', 0.0)
            O['batt_current'][j] = I; O['batt_voltage'][j] = V; O['soc'][j] = st['soc']; O['T_batt'][j] = st['T_bat']
            O['aux_power'][j] = plan['P_aux']; O['hvac_power'][j] = P_hvac_tot; O['P_brake'][j] = F_brk * u
            O['P_regen'][j] = max(P_reg, 0.0); O['T_cabin'][j] = st['T_cab']; O['trans_loss'][j] = plan['P_trans']
            O['batt_loss'][j] = P_bl; O['v_target'][j] = v_tgt[j]; O['rho_air'][j] = rk; O['headwind'][j] = wh
        e1 = v1 - v_tgt[k + 1]
        miss_sq += e1 * e1; miss_max = max(miss_max, abs(e1))
        if abs(e1) > 2 * MPH: miss_n2 += 1
        sat_prev = plan['limited']
        v0 = v1

    # ---------------- ledger summary (kJ)
    dKE = 0.5 * m_eff * (v0 * v0 - v_start * v_start)
    led = {k_: v_ / 1e3 for k_, v_ in L.items()}
    led['kinetic_change'] = dKE / 1e3
    led['batt_stored_change'] = -L['batt_chem'] / 1e3
    sources = L['fuel'] + L['batt_chem']
    sinks = (L['eng_loss'] + L['mot_loss'] + L['trans_loss'] + L['batt_loss'] + L['aux'] + L['aero'] + L['rolling']
             + L['grade'] + L['brake'] + dKE)
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
    out = {k_: np.asarray(v_, np.float32) for k_, v_ in O.items()} if record else {}
    if record:
        out['t'] = np.asarray(trip['t'], np.float32)
    out['ledger'] = led
    out['summary'] = summ
    out['final_state'] = dict(engine=st['T_eng'], cabin=st['T_cab'], battery=st['T_bat'], tire_warm=st['tire'], driveline=st['T_dl'],
                              soc=st['soc'], v_rc=st['v_rc'], eng_on=st['eng_on'], gear=st['gear'])
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
        F_cap = pt.regen_force_cap(u, st, env['P_aux_el'], st['T_bat'], st['soc'], st['v_rc'])
        Fb = -F
        if regen_mode == 'parallel':
            share = veh.regen.get('parallel_share', 0.6)
            F_reg = min(share * Fb, F_cap)
        elif regen_mode == 'one_pedal':
            F_reg = min(Fb, max(F_cap, 0.0))
        else:
            F_reg = min(Fb, F_cap)
        if arch == 'bev':
            plan = pt.plan(-F_reg, u, st, env)
        else:
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
    on = bool(C['on'][j])
    Pf = float(C['P_fuel'][j])
    if on and not st.get('eng_on', False) and strat is not None and hasattr(strat, 'start_W'):
        Pf += strat.start_W            # start-up fuel (cranking, enrichment) booked as engine loss
    plan = dict(F=float(C['T_w'][j]) / veh.r, P_fuel=Pf, P_eng=float(C['P_e'][j]) if on or veh.arch in ('hev_powersplit', 'phev_powersplit') else 0.0,
                w_e=float(C['w_e'][j]), T_e=float(C['T_e'][j]), eng_on=on, gear=int(C.get('gear', 0) or 0),
                P_trans=float(C['P_trans'][j]), P_mot=float(C['P_mot'][j]), P_batt=float(C['P_batt'][j]),
                P_aux=env['P_aux_el'])
    for a, b in (('T_m', 'T_m'), ('w_m', 'w_m'), ('P_m_el', 'P_m_el'), ('T_g', 'T_g'), ('w_g', 'w_g'), ('P_g_el', 'P_g_el')):
        if b in C:
            plan[a] = float(C[b][j])
    if 'T2' in C:
        plan['T_m'] = float(C['T2'][j]); plan['w_m'] = float(C['w2'][j])
        plan['T_g'] = float(C['T1'][j]); plan['w_g'] = float(C['w_s'][j])
    return plan
