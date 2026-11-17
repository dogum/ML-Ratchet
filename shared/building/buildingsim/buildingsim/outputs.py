"""Turn a finished Simulator into the result dict: time series, sensor trend log, summary, ledger."""
import numpy as np
import pandas as pd

from . import psychro as ps
from .comfort import pmv_ppd, degree_hours
from .ledger import build_ledger

MET = {'office': 1.1, 'education': 1.2, 'lodging': 1.0, 'retail': 1.4, 'healthcare': 1.2, 'assembly': 1.1,
       'warehouse': 2.0, 'residential_single': 1.0, 'residential_multi': 1.0, 'food': 1.6, 'lab': 1.3,
       'other': 1.1}


def end_use_arrays(sim):
    """W per step by end use and fuel."""
    r = sim.system.rec
    a = {k: np.asarray(v) for k, v in r.items()}
    ig = sim.spec['internal_gains']
    cook_gas = sim.cook if ig['cooking_fuel'] == 'gas' else np.zeros(sim.n)
    cook_el = sim.cook if ig['cooking_fuel'] != 'gas' else np.zeros(sim.n)
    doas_el = sim.doas_cool_W / 3.0
    doas_gas = sim.doas_heat_W / 0.8
    eu = {
        'heating_electricity': a['E_heat_elec'],
        'heating_gas': a['E_heat_gas'] + doas_gas,
        'heating_district': a['E_heat_dist'],
        'cooling_electricity': a['E_cool_elec'] + doas_el,
        'cooling_district': a['E_cool_dist'],
        'fans_electricity': a['E_fans'] + sim.P_mech_vent,
        'pumps_electricity': a['E_pumps'],
        'heat_rejection_electricity': a['E_rej'],
        'interior_lighting_electricity': sim.P_light,
        'exterior_lighting_electricity': sim.P_ext,
        'interior_equipment_electricity': sim.P_plug + cook_el,
        'water_systems_electricity': sim.dhw_elec,
        'water_systems_gas': sim.dhw_gas,
        'water_systems_district': sim.dhw_dist,
        'cooking_gas': cook_gas,
        'pv_electricity': -sim.pv_W,
        'battery_electricity': np.asarray(sim.r_bat),
    }
    cp = getattr(sim, 'campus', None)
    if cp is not None:      # physics 1.1: loop base while a district loop runs (campus.py)
        if cp['heat']:
            eu['plant_heating_district'] = cp['base_heat_W']
        if cp['cool']:
            eu['plant_cooling_district'] = cp['base_cool_W']
    return eu


def meters_from_end_uses(eu, sim):
    dist_heat = 'hot_water' if sim.spec['hvac'].get('district_heat') == 'hot_water' else 'steam'
    m = {'electricity': np.zeros(sim.n), 'gas': np.zeros(sim.n), dist_heat: np.zeros(sim.n),
         'chilled_water': np.zeros(sim.n)}
    for k, v in eu.items():
        if k.endswith('_electricity'):
            m['electricity'] = m['electricity'] + v
        elif k.endswith('_gas'):
            m['gas'] = m['gas'] + v
        elif k in ('cooling_district', 'plant_cooling_district'):
            m['chilled_water'] = m['chilled_water'] + v
        elif k.endswith('_district'):
            m[dist_heat] = m[dist_heat] + v
    return m


def make_outputs(sim, wall):
    n, dt = sim.n, sim.dt
    dth = dt / 3600.0
    idx = sim.times
    if getattr(sim, 'hp_home', False) and sim.physics != '1.0' and not getattr(sim, '_stored', False):
        from .campus import at_least, emitter_store
        if at_least(sim.physics, '1.3'):     # 1.3: heat-pump electricity through the emitter store
            sim.system.rec['E_heat_elec'] = emitter_store(sim.system.rec['E_heat_elec'], dt,
                                                          float(sim._campus_P['tau_e'])).tolist()
            sim._stored = True
    r = {k: np.asarray(v) for k, v in sim.system.rec.items()}
    T = np.asarray(sim.r_T)
    Ts = np.asarray(sim.r_Ts)
    W = np.asarray(sim.r_W)
    P = sim.wx['P']
    RH = np.clip(ps.rh_from_w(T, W, P), 0, 100)
    occ_sched = sim.sched['occ'] > 0.05
    eu = end_use_arrays(sim)
    met = meters_from_end_uses(eu, sim)
    # comfort (vectorised PMV): clothing by season from a 7-day running mean outdoor temperature
    Trm = pd.Series(sim.wx['T_out']).rolling(int(7 * 86400 / dt), min_periods=1).mean().values
    clo = np.clip(1.0 - 0.5 * (Trm - 5.0) / 20.0, 0.5, 1.0)
    pmv, ppd = pmv_ppd(T, Ts, 0.1, RH, MET.get(sim.spec['primary_use'], 1.1), clo)
    hsp, csp = np.asarray(sim.r_hsp), np.asarray(sim.r_csp)
    ts = pd.DataFrame(dict(T_zone=T, T_surface=Ts, T_mass=np.asarray(sim.r_Tm), T_op=0.3 * T + 0.7 * Ts,
                           RH_zone=RH, W_zone=W, heat_sp=hsp, cool_sp=csp, occupied=sim.hvac_sched,
                           occupancy=sim.sched['occ'], hvac_mode=np.asarray(sim.r_mode),
                           Q_hvac_sens_kW=np.asarray(sim.r_q) / 1000.0, Q_heat_kW=r['Q_heat'] / 1000.0,
                           Q_cool_sens_kW=r['Q_cool_sens'] / 1000.0, Q_cool_lat_kW=r['Q_cool_lat'] / 1000.0,
                           plr_heat=r['plr_heat'], plr_cool=r['plr_cool'], cop_heat=r['cop_heat'],
                           cop_cool=r['cop_cool'], infiltration_m3s=np.asarray(sim.r_inf), PMV=pmv, PPD=ppd,
                           T_out=sim.wx['T_out'], RH_out=sim.wx['RH'], GHI=sim.wx['ghi'],
                           wind_speed=sim.wx['wind']), index=idx)
    for k, v in eu.items():
        ts[k + '_kW'] = v / 1000.0
    for k, v in met.items():
        ts['meter_' + k + '_kW'] = v / 1000.0
    ts['pv_kW'] = sim.pv_W / 1000.0
    ts['battery_kW'] = np.asarray(sim.r_bat) / 1000.0
    ts['soc'] = np.asarray(sim.r_soc) if sim.bat else 0.0
    if sim.prices is not None:
        ts['price'] = sim.prices
    ts['fault'] = sim.fault_labels
    # sensor trend log (what an AHU/RTU trend would record); biased sensors report the biased value
    F = sim.fault_arrays

    def bias(name):
        b = F.get('bias_' + name)
        return np.nan_to_num(b) if b is not None else 0.0
    trend = pd.DataFrame(dict(
        OAT=sim.wx['T_out'] + bias('T_out'), OA_RH=sim.wx['RH'],
        RAT=r['T_ret'] + bias('T_ret'), MAT=r['T_mix'] + bias('T_mix'), SAT=r['T_sup'] + bias('T_sup'),
        ZAT=T + bias('T_zone'), ZAT_heat_sp=hsp + bias('T_zone'), ZAT_cool_sp=csp + bias('T_zone'), ZONE_RH=RH,
        OA_damper_pct=100 * r['oa_frac'], cool_cmd_pct=100 * np.clip(r['cool_cmd'], 0, 1),
        heat_cmd_pct=100 * np.clip(r['heat_cmd'], 0, 1), reheat_cmd_pct=100 * np.clip(r['reheat_cmd'], 0, 1),
        fan_speed_pct=100 * np.clip(r['fan_frac'], 0, 1), fan_status=(r['fan_frac'] > 0.01).astype(int),
        supply_flow_m3s=r['m_sup'] / 1.2, fan_kW=r['E_fans'] / 1000.0,
        hvac_kW=(r['E_heat_elec'] + r['E_cool_elec'] + r['E_fans'] + r['E_pumps'] + r['E_rej']) / 1000.0,
        building_kW=met['electricity'] / 1000.0 + sim.pv_W / 1000.0 - np.asarray(sim.r_bat) / 1000.0,
        gas_kW=met['gas'] / 1000.0, occupied=sim.hvac_sched.astype(int)), index=idx)
    if sim.sensor_noise:
        rng = sim.rng
        for c, sd in sim.sensor_noise.items():
            if c in trend:
                trend[c] = trend[c] + rng.normal(0, sd, n)
    trend['fault'] = sim.fault_labels
    led = build_ledger(sim, eu, met)
    # summary
    kwh = {k: float(v.sum() * dth / 1000.0) for k, v in eu.items()}
    mkwh = {k: float(v.sum() * dth / 1000.0) for k, v in met.items()}
    A = sim.geo['A_f']
    years = n * dt / (365 * 86400)
    site = sum(v for k, v in mkwh.items())
    unmet_h = float(((T < hsp - 0.3) & occ_sched).sum() * dth)
    unmet_c = float(((T > csp + 0.3) & occ_sched).sum() * dth)
    dh = degree_hours(0.3 * T + 0.7 * Ts, 20.0, 26.0, dth, occ_sched)
    z = sim.zone
    H_ve_mean = float(np.mean(np.asarray(sim.r_inf) * 1.2 * 1020.0)) + float(np.mean(sim.m_v)) * 1020.0
    summary = dict(
        kWh_by_end_use=kwh, kWh_by_meter=mkwh, site_kWh=site, EUI_kWh_m2_yr=site / A / years if years > 0 else None,
        peak_electricity_kW=float(met['electricity'].max() / 1000.0),
        peak_hvac_electric_kW=float(trend['hvac_kW'].max()),
        unmet_heating_h=unmet_h, unmet_cooling_h=unmet_c, comfort_degree_hours_below_20=dh[0],
        comfort_degree_hours_above_26=dh[1], mean_PPD_occupied=float(np.mean(ppd[occ_sched])) if occ_sched.any()
        else None, UA_envelope_W_K=z.UA, UA_with_air_W_K=z.UA + H_ve_mean, C_mass_J_K=z.C_m,
        tau_h=z.time_constant_h(max(H_ve_mean, 1.0)), floor_area_m2=A, sizing=_jsonable(sim.sizing),
        wall_s=wall, setup_s=sim.setup_s, x_realtime=(n * dt) / max(wall, 1e-9), steps=n, dt_s=dt)
    if sim.prices is not None:
        pr = np.asarray(sim.prices)
        e_kwh = met['electricity'] / 1000.0 * dth
        energy_cost = float((np.where(e_kwh > 0, e_kwh, e_kwh * 0.0 if not sim.spec.get('net_metering')
                                      else e_kwh) * pr).sum())
        summary['electricity_cost'] = energy_cost
    final = dict(T_air=sim.T_air, T_mass=sim.zone.theta_m, W_air=sim.W_air, soc=sim.soc)
    return dict(ts=ts, trend=trend, ledger=led, summary=summary, spec=sim.spec, final_state=final,
                sim=sim, physics=sim.physics)


def _jsonable(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out[k] = _jsonable(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out
