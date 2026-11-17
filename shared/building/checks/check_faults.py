"""Fault injection: every fault type on the system it applies to, two weeks at 15 min (healthy week, then the
faulted week), summer and winter. Records the signature on the trend log (what a FaultFinder model would see)
and on energy, checks the expected direction of each signature, and checks the ledgers close with faults on.
Weather: Chicago TMYx (5A)."""
import numpy as np
import pandas as pd

from _common import write
from buildingsim import simulate, read_epw, epw_path

RTU = dict(primary_use='office', floor_area_m2=2500, n_floors=2, year_built=2005, climate_zone='5A',
           system='packaged_rtu')
VAV = dict(primary_use='office', floor_area_m2=12000, n_floors=4, year_built=2005, climate_zone='5A',
           system='vav_chiller_boiler', hvac=dict(chiller_condenser='water'))
HP = dict(primary_use='residential_single', floor_area_m2=180, year_built=2010, climate_zone='5A',
          system='heat_pump_air')

# (name, spec, fault dict, season, expected signature: list of (channel, sign) on faulted-minus-healthy means
#  over occupied hours; channel names of out['trend'] or 'hvac_kWh', 'heat_kWh', 'cool_kWh')
CASES = [
    ('rtu_damper_stuck_closed_summer', RTU, dict(type='oa_damper_stuck', position=0.0), 'spring', [('OA_damper_pct', -1), ('cool_kWh', +1)]),
    ('rtu_damper_stuck_open_winter', RTU, dict(type='oa_damper_stuck', position=1.0), 'winter', [('OA_damper_pct', +1), ('MAT', -1), ('heat_kWh', +1)]),
    ('rtu_damper_leak_winter', RTU, dict(type='oa_damper_leak', leak=0.5), 'winter', [('OA_damper_pct', +1), ('heat_kWh', +1)]),
    ('rtu_economizer_disabled', RTU, dict(type='economizer_disabled'), 'spring', [('OA_damper_pct', -1), ('cool_kWh', +1)]),
    ('rtu_econ_high_limit_low', RTU, dict(type='economizer_high_limit', high_limit=10.0), 'spring', [('OA_damper_pct', -1), ('cool_kWh', +1)]),
    ('rtu_undercharge_30pct', RTU, dict(type='refrigerant_charge', charge=-0.3), 'summer', [('cool_kWh', +1)]),
    ('rtu_condenser_fouling', RTU, dict(type='condenser_fouling', severity=0.6), 'summer', [('cool_kWh', +1)]),
    ('rtu_low_airflow', RTU, dict(type='low_airflow', severity=0.4), 'summer', [('supply_flow_m3s', -1), ('SAT', -1)]),
    ('rtu_compressor_stuck_on', RTU, dict(type='cooling_valve_stuck', position=0.6), 'spring', [('cool_kWh', +1), ('SAT', -1)]),
    ('rtu_zone_sensor_bias_+2K_winter', RTU, dict(type='sensor_bias', sensor='T_zone', bias=2.0), 'winter', [('T_zone_true', -1), ('heat_kWh', -1)]),
    ('rtu_oat_sensor_bias_-6K', RTU, dict(type='sensor_bias', sensor='T_out', bias=-6.0), 'summer', [('OA_damper_pct', +1)]),
    ('rtu_schedule_override_hvac_on', RTU, dict(type='schedule_override', mode='hvac_always_on'), 'winter', [('hvac_kWh', +1)]),
    ('rtu_setpoint_override', RTU, dict(type='setpoint_override', heat=23.5, cool=24.0), 'winter', [('ZAT', +1), ('heat_kWh', +1)]),
    ('vav_cooling_valve_leak_winter', VAV, dict(type='cooling_valve_leak', leak=0.3), 'winter', [('cool_kWh', +1), ('heat_kWh', +1)]),
    ('vav_heating_valve_leak_summer', VAV, dict(type='heating_valve_leak', leak=0.3), 'summer', [('heat_kWh', +1), ('cool_kWh', +1)]),
    ('vav_cooling_valve_stuck', VAV, dict(type='cooling_valve_stuck', position=0.0), 'summer', [('SAT', +1), ('ZAT', +1)]),
    ('vav_reheat_valve_leak', VAV, dict(type='reheat_valve_leak', leak=0.4), 'summer', [('heat_kWh', +1), ('cool_kWh', +1)]),
    ('vav_low_airflow', VAV, dict(type='low_airflow', severity=0.4), 'summer', [('T_zone_true', +1)]),
    ('vav_sat_offset_-3K', VAV, dict(type='sat_setpoint_offset', offset=-3.0), 'spring', [('SAT', -1), ('cool_kWh', +1)]),
    ('vav_box_stuck_full', VAV, dict(type='vav_box_stuck', position=1.0), 'winter', [('fan_kW', +1), ('hvac_kWh', +1)]),
    ('vav_coil_fouling', VAV, dict(type='coil_fouling', severity=0.5), 'summer', [('fan_kW', +1)]),
    ('vav_chiller_fouling', VAV, dict(type='chiller_fouling', severity=1.0), 'summer', [('cool_kWh', +1)]),
    ('vav_boiler_fouling', VAV, dict(type='boiler_fouling', severity=0.2), 'winter', [('heat_kWh', +1)]),
    ('vav_sat_sensor_bias_+2K', VAV, dict(type='sensor_bias', sensor='T_sup', bias=2.0), 'summer', [('SAT_true', -1)]),
    ('vav_damper_stuck_open_winter', VAV, dict(type='oa_damper_stuck', position=1.0), 'winter', [('OA_damper_pct', +1), ('heat_kWh', +1)]),
    ('hp_undercharge_winter', HP, dict(type='refrigerant_charge', charge=-0.3), 'winter', [('heat_kWh', +1)]),
]
SEASON = {'winter': ('2019-01-07', '2019-01-21'), 'spring': ('2019-04-22', '2019-05-06'),
          'summer': ('2019-07-08', '2019-07-22')}


def metrics(out, m):
    ts, tr = out['ts'], out['trend']
    occ = tr['occupied'].values.astype(bool) & m
    dth = 0.25
    r = {c: float(tr[c].values[occ].mean()) for c in ('OA_damper_pct', 'MAT', 'SAT', 'ZAT', 'supply_flow_m3s', 'fan_kW')}
    r['T_zone_true'] = float(ts['T_zone'].values[occ].mean())
    r['SAT_true'] = float(np.asarray(out['sim'].system.rec['T_sup'])[occ].mean())
    r['hvac_kWh'] = float(tr['hvac_kW'].values[m].sum() * dth + ts['meter_gas_kW'].values[m].sum() * dth -
                          ts['water_systems_gas_kW'].values[m].sum() * dth)
    r['heat_kWh'] = float((ts['heating_electricity_kW'] + ts['heating_gas_kW'] + ts['heating_district_kW']).values[m].sum() * dth)
    r['cool_kWh'] = float((ts['cooling_electricity_kW'] + ts['cooling_district_kW']).values[m].sum() * dth)
    return r


def main():
    w = read_epw(epw_path('Chicago'))
    rows = {}
    n_ok = 0
    for name, spec, fault, season, expect in CASES:
        a, b = SEASON[season]
        mid = pd.Timestamp(a) + pd.Timedelta(days=7)
        f = dict(fault, start=str(mid))
        healthy = simulate(spec, w, a, b, dt='15min')
        faulty = simulate(spec, w, a, b, dt='15min', faults=[f])
        idx = faulty['ts'].index
        m2 = np.asarray(idx >= mid.tz_localize('UTC'))
        mh, mf = metrics(healthy, m2), metrics(faulty, m2)
        sig = {k: round(mf[k] - mh[k], 4) for k in mh}
        checks = {f'{c} {"up" if s > 0 else "down"}': bool(np.sign(sig[c]) == s and abs(sig[c]) > 1e-6) for c, s in expect}
        ok = all(checks.values())
        n_ok += ok
        lab = faulty['ts']['fault']
        rows[name] = dict(fault=fault, season=season, signature_faulted_minus_healthy=sig, expected=checks, ok=ok,
                          label_steps=int((lab != '').sum()), label_example=str(lab[lab != ''].iloc[0]) if (lab != '').any() else '',
                          ledger_closure=faulty['ledger']['closure_rel'], system_lag_share=faulty['ledger']['system_lag_share'])
        print(name, 'OK' if ok else 'CHECK', checks, {k: sig[k] for k in ('OA_damper_pct', 'SAT', 'SAT_true', 'T_zone_true', 'heat_kWh', 'cool_kWh')},
              'clos %.0e' % faulty['ledger']['closure_rel'], flush=True)
    write('faults.json', dict(what='fault signatures (faulted week minus the same week healthy), Chicago TMYx, 15 min',
                              n_cases=len(CASES), n_expected_direction_ok=n_ok, cases=rows,
                              max_ledger_closure=max(r['ledger_closure'] for r in rows.values())))


if __name__ == '__main__':
    main()
