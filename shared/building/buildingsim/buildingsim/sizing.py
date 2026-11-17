"""Design loads and equipment sizing (ASHRAE-style peak-load method, no dynamics), then system construction.

Heating design: envelope UA x (T_heat_sp - T_99.6 %) + infiltration at design + outdoor air through the system,
x sizing factor (1.25), no gains or sun. Cooling design: UA x (T_0.4 % - T_cool_sp) + 5 K sol-air on opaque
surfaces + 70 % of the year's peak transmitted window solar (mass damping) + 90 % of peak internal gains +
outdoor-air sensible and latent at the 0.4 % dry bulb and the 99.6 % wet bulb, x sizing factor (1.15).
Design temperatures come from the weather given to the run. Rated DX capacity = design coil load / CAPFT at
design conditions; airflow 400 cfm/ton for DX, 11 K supply dT for VAV.
"""
import math

import numpy as np

from . import psychro as ps
from . import spec as sp
from .plant import ChillerPlant, BoilerPlant, District
from .systems import (UnitarySystem, VAVSystem, ZonalSystem, IdealSystem, NoSystem, DXCoil, HPHeating)
from .zone import CP_AIR

M3S_PER_W_DX = 0.0537e-3      # 400 cfm per ton
RHO = 1.2


def design_loads(sim):
    s, z, wx = sim.spec, sim.zone, sim.wx
    cl = wx['climate']
    hsp, csp = s['setpoints']['heat'], s['setpoints']['cool']
    Thd, Tcd = cl['T_heat_design'], cl['T_cool_design']
    Twb_d = cl['T_wb_cool_design']
    P = float(np.median(wx['P']))
    UA_above = z.H_tr_w + sim.bnd['H_wall'] + sim.bnd['H_roof']
    Hg = sim.bnd['H_ground']
    Tg_min = float(np.min(sim.bnd['T_ground']))
    Tg_max = float(np.max(sim.bnd['T_ground']))
    V = sim.geo['volume']
    q_inf_d = sim.ela * math.sqrt(sim.Cs * max(hsp - Thd, 0) + sim.Cw0 * 16.0) / 1000.0     # m3/s
    q_inf_c = sim.ela * math.sqrt(sim.Cs * max(Tcd - csp, 0) + sim.Cw0 * 9.0) / 1000.0
    oa = sim.V_oa_design      # m3/s at design occupancy
    oa_sys = oa if sim.oa_through_system else 0.0
    oa_dir = oa if not sim.oa_through_system else 0.0
    Qh = (UA_above * (hsp - Thd) + Hg * max(hsp - Tg_min, 0) + RHO * CP_AIR * (q_inf_d + oa_dir * (1 - sim.erv_eff))
          * (hsp - Thd))
    Qh_oa = RHO * CP_AIR * oa_sys * (hsp - Thd)
    A = sim.geo['A_f']
    ig = s['internal_gains']
    people_s = ig['people_W_m2'] * A * (1 - ig['people_latent_frac'])
    people_l = ig['people_W_m2'] * A * ig['people_latent_frac']
    internal = people_s + (ig['lighting'] + ig['plug'] + ig['process']) * A + ig['cooking'] * A * \
        ig['cooking_to_space'] * 0.67
    sol_peak = float(np.percentile(sim.phi_sol, 99.9)) if len(sim.phi_sol) else 0.0
    W_d = ps.w_sat_s(Twb_d, P)
    W_d = float(ps.w_from_pw(ps.pw_from_w(W_d), P))
    h_out = ps.enthalpy_s(Tcd, min(W_d, ps.w_sat_s(Tcd, P)))
    W_in = ps.w_from_rh(csp, 50.0, P)
    h_in = ps.enthalpy_s(csp, float(W_in))
    Qc_s = (UA_above * (Tcd - csp) + sim.bnd['H_wall'] * 5.0 + sim.bnd['H_roof'] * 10.0 + 0.7 * sol_peak +
            0.9 * internal + RHO * CP_AIR * (q_inf_c + oa_dir * (1 - sim.erv_eff)) * (Tcd - csp)
            - Hg * max(csp - Tg_max, 0))
    Qc_s = max(Qc_s, 0.2 * internal + 100.0)
    Qc_lat_zone = 0.9 * people_l + RHO * (q_inf_c + oa_dir) * max(min(W_d, 0.03) - float(W_in), 0) * ps.H_FG_ROOM
    Qc_oa_tot = RHO * oa_sys * max(h_out - h_in, 0)
    return dict(Q_heat_zone=max(Qh, 500.0), Q_heat_oa=Qh_oa, Q_cool_sens=Qc_s, Q_cool_lat=Qc_lat_zone,
                Q_cool_oa=Qc_oa_tot, T_heat_design=Thd, T_cool_design=Tcd, T_wb_design=Twb_d,
                internal_peak=internal, solar_peak=sol_peak, V_oa=oa)


def build_system(sim, D):
    """Construct the HVAC system object from spec + design loads. Records capacities in sim.sizing."""
    s = sim.spec
    h = s['hvac']
    sysname = s['system']
    use = s['primary_use']
    res = use.startswith('residential')
    fh, fc = h['sizing_factor_heat'], h['sizing_factor_cool']
    P = float(np.median(sim.wx['P']))
    Qh = h['heating_capacity_W'] or (D['Q_heat_zone'] + D['Q_heat_oa']) * fh
    Qc_coil = (D['Q_cool_sens'] + D['Q_cool_lat'] + D['Q_cool_oa']) * fc
    sz = dict(design=D, system=sysname)
    sim.sizing = sz
    if sysname == 'none':
        return NoSystem(sim)
    if sysname == 'ideal':
        return IdealSystem(sim)
    cooling = bool(h.get('cooling', True))
    speed = h.get('speed', 'single')
    if sysname in ('ptac', 'packaged_rtu', 'split_ac_furnace', 'heat_pump_air', 'heat_pump_ground',
                   'baseboard_resistance', 'hydronic_baseboard', 'wall_furnace'):
        from .curves import dx_cool_capft
        capft_d = dx_cool_capft(19.4, D['T_cool_design'])
        Qr = h['cooling_capacity_W'] or max(Qc_coil / max(capft_d, 0.6), 1500.0)
        if not cooling:
            Qr = 0.0
        # airflow: 400 cfm/ton of cooling, or a 30 K rise for heating-only units, whichever is larger
        m_c = RHO * M3S_PER_W_DX * Qr
        m_h = Qh / (CP_AIR * 30.0)
        m = max(m_c, m_h, 0.05)
        zonal_heat = sysname in ('baseboard_resistance', 'hydronic_baseboard', 'wall_furnace')
        if zonal_heat:
            m = max(m_c, 0.03)
        V = m / RHO
        if res:
            wcfm = h.get('fan_W_per_cfm') or (0.3 if s['year_built'] >= 2015 else 0.5)
            if h.get('room_ac') or (zonal_heat and not h.get('central_ac')):
                wcfm = min(wcfm, 0.15)       # room / window units: small fans, no ducts
            P_fan = wcfm * V * 2118.88
        elif sysname == 'ptac':
            P_fan = (h['fan_W_per_m3s'] or 700.0) * V
        else:
            P_fan = (h['fan_W_per_m3s'] or 1360.0) * V
        if res or sysname in ('split_ac_furnace',):
            eer = sp.seer_to_eer(h['SEER'], speed)
        else:
            eer = h['EER'] if sysname != 'ptac' else min(h['EER'], 10.0)
        # rated COP excluding the supply fan (EnergyPlus convention): total input = Q / EER (Btu/Wh)
        P_tot = Qr * 3.412 / eer if Qr > 0 else 0.0
        P_fan_rated = P_fan * (m_c / m if m > 0 else 1.0)
        cop_gross = Qr / max(P_tot - min(P_fan_rated, 0.4 * P_tot), 1.0) if Qr > 0 else 3.0
        m_rated = max(m_c, 1e-3)
        coil = DXCoil(Qr, cop_gross, m_rated if Qr > 0 else 1.0, SHR=0.78 if res else 0.75, speed=speed, P=P) \
            if Qr > 0 else None
        cfg = dict(m_dot=m, P_fan=P_fan, cool=dict(kind='dx', coil=coil) if coil else None,
                   fan_mode='cycling' if (res or sysname == 'ptac') else 'continuous_occupied',
                   oa_frac=min(1.0, sim.V_oa_design / V) if sim.oa_through_system else 0.0,
                   economizer=bool(s['ventilation']['economizer']) and sysname == 'packaged_rtu' and Qr > 0,
                   econ_high_limit=s['ventilation']['econ_high_limit'])
        hs = h.get('heat_source')
        if (sysname == 'split_ac_furnace' and hs not in ('elec', 'electric')) or \
                (sysname in ('packaged_rtu', 'ptac') and hs in ((None, 'gas', 'furnace') if sysname == 'packaged_rtu'
                                                               else ('gas', 'furnace'))):
            cfg['heat_kind'] = 'furnace'
            cfg['heat'] = dict(Q=Qh, eff=h['AFUE'] if res else max(h['AFUE'], 0.78))
        elif sysname in ('packaged_rtu', 'ptac') and hs in ('boiler', 'district'):
            cfg['heat_kind'] = hs
            cfg['heat'] = dict(Q=Qh, eff=h['boiler_eff'] * 0.97)
        elif sysname == 'heat_pump_air' or (sysname in ('packaged_rtu', 'ptac') and hs == 'hp'):
            hspf = h['HSPF']
            cop47 = sp.hspf_to_cop47(hspf, speed)
            Qhr = max(Qr, Qh / 2.0) if Qr > 0 else Qh / 1.5     # HP rated heating ~ rated cooling
            coil_h = HPHeating(Qhr, cop47 * 1.08, speed=speed, min_T=h['hp_min_T'], P=P)
            cfg['heat_kind'] = 'hp'
            cfg['heat'] = dict(Q=Qhr, coil=coil_h)
            cfg['backup_W'] = h['backup_capacity_W'] if h['backup_capacity_W'] is not None else Qh
            sz['hp_rated_heat_W'] = Qhr
        elif sysname == 'heat_pump_ground':
            cop_h = h.get('COP_heat_gshp', 3.8 if s['year_built'] < 2010 else 4.2)
            Qhr = max(Qr, Qh / 1.25)
            cfg['heat_kind'] = 'gshp'
            cfg['heat'] = dict(Q=Qhr, COP=cop_h)
            cfg['backup_W'] = h['backup_capacity_W'] if h['backup_capacity_W'] is not None else 0.5 * Qh
            if coil is not None:
                eer_g = h.get('EER_gshp', 16.0)
                coil.COPr = (eer_g / 3.412) * 1.15
                cfg['cool'] = dict(kind='gshp', coil=coil)
            UA = Qhr / 8.0
            cfg['gshp'] = dict(UA=UA, C=UA * 86400 * 15.0, T_ground=float(np.mean(sim.wx['T_ground_deep'])))
            sz['gshp_loop_UA'] = UA
        elif zonal_heat:
            cfg['heat_kind'] = 'baseboard'
            if sysname == 'baseboard_resistance' or hs in ('elec', 'electric'):
                cfg['heat'] = dict(Q=Qh, fuel='elec')
            elif sysname == 'wall_furnace':
                cfg['heat'] = dict(Q=Qh, fuel='gas', eff=h['AFUE'] if h.get('AFUE') else 0.68)
            elif hs == 'district':
                cfg['heat'] = dict(Q=Qh, fuel='district', eff=1.0)
            else:
                cfg['heat'] = dict(Q=Qh, fuel='boiler', eff=h['boiler_eff'] * 0.95)
            cfg['fan_mode'] = 'cycling'
            cfg['oa_frac'] = 0.0
        else:   # ptac (electric resistance) or rtu electric
            cfg['heat_kind'] = 'elec'
            cfg['heat'] = dict(Q=Qh)
        sz.update(cooling_rated_W=Qr, heating_W=cfg['heat']['Q'], supply_m3s=V, fan_W=P_fan, EER=eer,
                  COP_cool_gross=cop_gross, backup_W=cfg.get('backup_W', 0.0))
        return UnitarySystem(sim, cfg)
    if sysname in ('vav_chiller_boiler', 'district_steam_chw'):
        dT = max(s['setpoints']['cool'] - h['sat_cool'], 6.0)
        m_max = 1.1 * D['Q_cool_sens'] * fc / (CP_AIR * dT)
        m_oa = RHO * sim.V_oa_design
        m_max = max(m_max, m_oa * 1.2, 0.05)
        P_fan = (h['fan_W_per_m3s'] or 1660.0) * m_max / RHO
        Qcc = Qc_coil + m_max * CP_AIR * 1.0 * 0.0
        Qcc = max(Qcc, (D['Q_cool_sens'] + D['Q_cool_lat']) * fc + D['Q_cool_oa'] * fc)
        T_sa = h['sat_cool']
        Q_ph = m_oa * CP_AIR * max(T_sa - D['T_heat_design'], 0)
        min_frac = h['vav_min_frac']
        Q_rh = max(min_frac * m_max * CP_AIR * (35.0 - T_sa), D['Q_heat_zone'] * fh)
        if sysname == 'district_steam_chw':
            cool_plant = District(Qcc, h['district_eff'], 'chw') if cooling else None
            heat_plant = District(Q_ph + Q_rh, h['district_eff'], 'steam')
        else:
            cs = h.get('cooling_source', 'chiller')
            if not cooling:
                cool_plant = None
            elif cs == 'dx':
                cool_plant = DXCoil(Qcc, h['EER'] / 3.412 * 1.2, RHO * M3S_PER_W_DX * Qcc, SHR=0.75,
                                    speed='two', P=P)
            else:
                cond = h['chiller_condenser']
                cop = h['chiller_COP'] if cond == 'water' else min(h['chiller_COP'], 3.1)
                cool_plant = ChillerPlant(Qcc, cop, cond, h['chw_supply_C'])
            hs = h.get('heat_source')
            if hs == 'district':
                heat_plant = District(Q_ph + Q_rh, h['district_eff'], 'steam')
            elif hs in ('elec', 'electric'):
                heat_plant = BoilerPlant(Q_ph + Q_rh, 0.98, 'noncondensing', 'electric')
            else:
                heat_plant = BoilerPlant((Q_ph + Q_rh) * 1.0, h['boiler_eff'], h['boiler_type'], 'gas')
        cfg = dict(m_max=m_max, min_frac=min_frac, P_fan=P_fan, m_oa=m_oa,
                   economizer=bool(s['ventilation']['economizer']), econ_high_limit=s['ventilation']['econ_high_limit'],
                   sat=T_sa, sat_reset=h.get('sat_reset', True), cool_plant=cool_plant, heat_plant=heat_plant,
                   Q_cc_design=Qcc, Q_ph_design=max(Q_ph, 1.0), Q_rh_design=max(Q_rh, 1.0),
                   T_chw=h['chw_supply_C'], reheat_elec=h.get('reheat', 'hw') == 'elec',
                   static_reset=h['static_reset'], Q_heat_zone=D['Q_heat_zone'] * fh)
        sz.update(supply_max_m3s=m_max / RHO, oa_m3s=m_oa / RHO, fan_W=P_fan, Q_cooling_coil_W=Qcc,
                  Q_preheat_W=Q_ph, Q_reheat_W=Q_rh, plant=type(cool_plant).__name__ if cool_plant else None)
        return VAVSystem(sim, cfg)
    if sysname in ('fan_coil', 'radiant'):
        Qcs = D['Q_cool_sens'] * fc if cooling else 0.0
        hs = h.get('heat_source')
        if hs == 'hp':
            heat_src = ('hp', h.get('COP_hp_water', 3.5))
        elif hs in ('elec', 'electric'):
            heat_src = 'elec'
        elif hs == 'district':
            heat_src = District(Qh, h['district_eff'], 'steam')
        else:
            heat_src = BoilerPlant(Qh, h['boiler_eff'], h['boiler_type'], 'gas')
        cool_plant = None
        if cooling:
            cond = h['chiller_condenser']
            cop = h['chiller_COP'] if cond == 'water' else min(h['chiller_COP'], 3.1)
            cool_plant = ChillerPlant(Qcs * 1.3, cop, cond, h['chw_supply_C'])
        m_fc = max(Qcs, Qh * 0.5) / (CP_AIR * 10.0)
        P_fan = 0.3 * m_fc / RHO * 2118.88 if sysname == 'fan_coil' else 0.0
        cfg = dict(inj='air' if sysname == 'fan_coil' else 'mass', Q_heat=Qh, Q_cool=Qcs, P_fan=P_fan, m_fc=m_fc,
                   heat_src=heat_src, cool_plant=cool_plant, shr=0.75)
        sz.update(heating_W=Qh, cooling_sens_W=Qcs, fan_W=P_fan)
        return ZonalSystem(sim, cfg)
    raise ValueError(f'unknown system {sysname}; known: {list(sp.SYSTEMS)}')
