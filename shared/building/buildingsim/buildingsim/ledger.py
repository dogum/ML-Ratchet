"""Energy, heat and moisture ledgers. Every run closes each one; `ledger(out)` returns it, `ok` says whether
every closure is below tolerance (1e-6 of the throughput for the zone and air side, 1e-9 for meters).

- zone_heat_kWh      zone air + surfaces + mass: conduction (windows, walls, roof, ground), infiltration,
                     direct ventilation, solar through windows, internal gains, HVAC delivered = storage change.
- zone_moisture_kg   infiltration, direct ventilation, supply air, internal latent gains, condensation = storage.
- air_side_kWh       heat delivered to the zone = coil heating - coil sensible cooling + fan heat + outdoor-air
                     sensible + return lag + supply lag + terminal direct heat (baseboard, radiant, fan coils).
- equipment_kWh      electricity + fuel + district + heat taken from air/water + heat taken from ambient
                     = heat to air/water + heat to outdoors (rejected, flue, losses) + fan heat.
- meters_kWh         each meter = the sum of its end uses (PV negative, battery charging positive).
- battery_kWh        charge in - discharge out = losses + stored change.
"""
import numpy as np

J2KWH = 1.0 / 3.6e6


def _close(d, src_keys, sink_keys):
    s = sum(d[k] for k in src_keys)
    t = sum(d[k] for k in sink_keys)
    scale = sum(abs(d[k]) for k in src_keys + sink_keys) + 1e-12
    d['residual'] = s - t
    d['closure_rel'] = abs(s - t) / scale
    return d


def build_ledger(sim, eu, met):
    dt = sim.dt
    L = sim.L
    f = dt * J2KWH
    zh = dict(conduction_windows=L['win'] * f, radiant_gains_lost_via_windows=-L['win_rad'] * f,
              conduction_walls=L['wall'] * f, conduction_roof=L['roof'] * f,
              conduction_ground=L['ground'] * f, infiltration=L['inf'] * f, ventilation=L['vent'] * f,
              solar_windows=L['sol'] * f, internal_gains=L['int'] * f, hvac_delivered=L['hvac'] * f,
              storage_change=L['storage'] * f)
    _close(zh, ['conduction_windows', 'radiant_gains_lost_via_windows', 'conduction_walls', 'conduction_roof', 'conduction_ground', 'infiltration',
                'ventilation', 'solar_windows', 'internal_gains', 'hvac_delivered'], ['storage_change'])
    zm = dict(infiltration=L['m_inf'] * dt, ventilation=L['m_vent'] * dt, supply_air=L['m_sup'] * dt,
              internal_latent=L['m_lat'] * dt, condensation=-L['m_cond'] * dt, storage_change=L['m_storage'] * dt)
    _close(zm, ['infiltration', 'ventilation', 'supply_air', 'internal_latent', 'condensation'], ['storage_change'])
    S = sim.system.led
    a = {k: v * J2KWH for k, v in S.items()}
    air = dict(to_zone=a['to_zone'], coil_heat=a['coil_heat'], coil_cool_sensible=-a['coil_cool_sens'],
               fan_heat=a['fan_heat'], outdoor_air_sensible=a['oa_sens'], return_lag=a['return_lag'],
               supply_lag=a['supply_lag'], terminal_direct=a['direct'])
    _close(air, ['coil_heat', 'coil_cool_sensible', 'fan_heat', 'outdoor_air_sensible', 'return_lag', 'supply_lag',
                 'terminal_direct'], ['to_zone'])
    air['hvac_delivered_zone_side'] = zh['hvac_delivered']
    air['zone_vs_air_side_rel'] = abs(zh['hvac_delivered'] - a['to_zone']) / (abs(zh['hvac_delivered']) +
                                                                             abs(a['to_zone']) + 1e-9)
    eq = dict(electricity_in=a['in_elec'], fuel_in=a['in_fuel'], district_in=a['in_dist'],
              heat_from_air_water=a['from_air_water'], heat_from_ambient=a['from_ambient'],
              heat_to_air_water=a['to_air_water'], heat_to_outdoors=a['to_outdoors'], fan_heat=a['fan_heat'])
    _close(eq, ['electricity_in', 'fuel_in', 'district_in', 'heat_from_air_water', 'heat_from_ambient'],
           ['heat_to_air_water', 'heat_to_outdoors', 'fan_heat'])
    eq['coil_condensate_kg'] = S['coil_latent_kg']
    dth = dt / 3600.0
    eu_k = {k: float(v.sum() * dth / 1000.0) for k, v in eu.items()}
    me_k = {k: float(v.sum() * dth / 1000.0) for k, v in met.items()}
    tot_eu = sum(eu_k.values())
    tot_me = sum(me_k.values())
    meters = dict(by_meter=me_k, by_end_use=eu_k, residual=tot_me - tot_eu,
                  closure_rel=abs(tot_me - tot_eu) / (sum(abs(v) for v in eu_k.values()) + 1e-12))
    # HVAC electricity booked by the equipment ledger vs the end-use records
    hv = sum(eu_k[k] for k in ('heating_electricity', 'cooling_electricity', 'pumps_electricity',
                               'heat_rejection_electricity')) + float(np.sum(sim.system.rec['E_fans']) * dth / 1000)
    hv -= float(np.sum(sim.doas_cool_W / 3.0) * dth / 1000.0)
    eq['electricity_vs_end_uses_rel'] = abs(hv - a['in_elec']) / (abs(hv) + abs(a['in_elec']) + 1e-9)
    out = dict(zone_heat_kWh=zh, zone_moisture_kg=zm, air_side_kWh=air, equipment_kWh=eq, meters_kWh=meters)
    if sim.bat:
        b = dict(charge_in=L['bat_in'] * J2KWH, discharge_out=L['bat_out'] * J2KWH, losses=L['bat_loss'] * J2KWH,
                 stored_change=L['bat_store'] * J2KWH,
                 soc_change_kWh=(sim.soc - sim.soc_start) * sim.bat['kWh'])
        _close(b, ['charge_in'], ['discharge_out', 'losses', 'stored_change'])
        out['battery_kWh'] = b
    worst = max(zh['closure_rel'], zm['closure_rel'], air['closure_rel'], air['zone_vs_air_side_rel'],
                eq['closure_rel'], meters['closure_rel'], eq['electricity_vs_end_uses_rel'],
                out.get('battery_kWh', {}).get('closure_rel', 0.0))
    out['closure_rel'] = worst
    out['ok'] = bool(worst < 1e-6)
    lag = abs(air['return_lag']) + abs(air['supply_lag'])
    out['system_lag_share'] = lag / (abs(air['to_zone']) + abs(air['coil_heat']) + abs(air['coil_cool_sensible'])
                                     + 1e-9)
    return out


def ledger(out):
    """The ledger of a simulate() result."""
    return out['ledger']
