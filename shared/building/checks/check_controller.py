"""Controller hook, step-wise API, tariffs, PV and battery.
1. TOU pre-cooling of a heat-pump house in Phoenix (July week): default thermostat vs a rule pre-cooler.
2. The same policy through the step-wise API gives identical results.
3. get_state / set_state: a rollout from a saved state is reproduced exactly.
4. Demand limiting (max_capacity) lowers the peak.
5. PV + battery in self-consumption: battery ledger closes, grid import falls.
"""
import time

import numpy as np

from _common import write
from buildingsim import simulate, read_epw, epw_path
from buildingsim.simulate import Simulator

HOUSE = dict(primary_use='residential_single', floor_area_m2=200, year_built=2005, climate_zone='2B',
             system='heat_pump_air', setpoints=dict(heat=20.0, cool=24.4, setback_heat=None, setback_cool=None))
TOU = dict(base=0.12, periods=[dict(months=list(range(1, 13)), hours=(16, 21), weekdays_only=True, price=0.45)])
A, B = '2019-07-08 07:00', '2019-07-15 07:00'


def precool(obs):
    h = obs['hour']
    if obs['daytype'] == 0 and 12 <= h < 16:
        return dict(cool_sp=22.5)
    if obs['daytype'] == 0 and 16 <= h < 21:
        return dict(cool_sp=27.0)
    return None


def summarise(out):
    ts = out['ts']
    pr = np.asarray(ts['price'])
    el = ts['meter_electricity_kW'].values
    peak = pr > 0.3
    hot = np.clip(ts['T_op'].values - 26.0, 0, None)
    return dict(kWh=float(el.sum() * 0.25), cost=float((el * pr).sum() * 0.25), peak_kWh=float(el[peak].sum() * 0.25),
                max_peak_kW=float(el[peak].max()), degree_h_above_26_op=float(hot.sum() * 0.25),
                ledger_closure=out['ledger']['closure_rel'])


def main():
    w = read_epw(epw_path('Phoenix'))
    base = simulate(HOUSE, w, A, B, dt='15min', prices=TOU)
    t0 = time.time()
    pc = simulate(HOUSE, w, A, B, dt='15min', prices=TOU, controller=precool)
    t_ctrl = time.time() - t0
    t0 = time.time()
    simulate(HOUSE, w, A, B, dt='15min', prices=TOU)
    t_none = time.time() - t0
    r1 = dict(default=summarise(base), precool=summarise(pc))
    print(r1)
    # 2. step-wise API
    sim = Simulator(HOUSE, w, A, B, dt='15min', prices=TOU)
    obs = sim.reset()
    while obs is not None:
        obs, info = sim.step(precool(obs))
    sw = sim.finish()
    eq = abs(summarise(sw)['kWh'] - r1['precool']['kWh'])
    # 3. state save / restore
    sim = Simulator(HOUSE, w, A, B, dt='15min', prices=TOU)
    obs = sim.reset()
    for _ in range(300):
        obs, info = sim.step(None)
    st = sim.get_state()
    a1 = [sim.step(dict(cool_sp=23.0))[1]['electricity_kW'] for _ in range(40)]
    sim.set_state(st)
    a2 = [sim.step(dict(cool_sp=23.0))[1]['electricity_kW'] for _ in range(40)]
    rollback_err = float(np.max(np.abs(np.array(a1) - np.array(a2))))
    # 4. demand limiting during peak
    lim = simulate(HOUSE, w, A, B, dt='15min', prices=TOU,
                   controller=lambda o: dict(max_capacity=0.3) if (o['daytype'] == 0 and 16 <= o['hour'] < 21) else None)
    r4 = summarise(lim)
    # 5. PV + battery
    pvb = dict(HOUSE, pv=dict(kW=6.0, tilt=20, azimuth=180), battery=dict(kWh=13.5, kW=5.0, mode='self_consumption'))
    pv_only = simulate(dict(HOUSE, pv=dict(kW=6.0, tilt=20, azimuth=180)), w, A, B, dt='15min')
    both = simulate(pvb, w, A, B, dt='15min')
    def imp(o):
        e = o['ts']['meter_electricity_kW'].values
        return float(np.clip(e, 0, None).sum() * 0.25), float(-np.clip(e, None, 0).sum() * 0.25)
    bl = both['ledger']['battery_kWh']
    r5 = dict(pv_kWh=float(pv_only['ts']['pv_kW'].sum() * 0.25), import_pv_only=imp(pv_only)[0],
              export_pv_only=imp(pv_only)[1], import_pv_battery=imp(both)[0], export_pv_battery=imp(both)[1],
              battery_ledger=bl, ledger_closure=both['ledger']['closure_rel'])
    print(r5)
    write('controller.json', dict(
        what='controller hook, step-wise API, TOU tariff, demand limiting, PV + battery (Phoenix TMYx, July week, '
             'heat-pump house, 15 min)', tou_precooling=r1,
        precool_effect=dict(peak_kWh_change=r1['precool']['peak_kWh'] - r1['default']['peak_kWh'],
                            cost_change=r1['precool']['cost'] - r1['default']['cost'],
                            kWh_change=r1['precool']['kWh'] - r1['default']['kWh']),
        stepwise_vs_callable_kWh_diff=eq, state_rollback_max_diff_kW=rollback_err,
        demand_limit_30pct_peak=dict(max_peak_kW=r4['max_peak_kW'], default_max_peak_kW=r1['default']['max_peak_kW'],
                                      degree_h_above_26=r4['degree_h_above_26_op']),
        pv_battery=r5, controller_overhead=dict(with_controller_s=t_ctrl, without_s=t_none)))


if __name__ == '__main__':
    main()
