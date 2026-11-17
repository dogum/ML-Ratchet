"""Speed: single thread (OMP_NUM_THREADS=1), shared 2-core machine. Building-years at 15 min and 1 h."""
import os
import time

import numpy as np

from _common import write
from buildingsim import simulate, read_epw, epw_path, generate_buildings
from buildingsim import weather as wxm

CASES = {'house_split_ac_furnace': dict(primary_use='residential_single', system='split_ac_furnace'),
         'house_heat_pump': dict(primary_use='residential_single', system='heat_pump_air'),
         'office_rtu': dict(primary_use='office', floor_area_m2=2500, system='packaged_rtu'),
         'office_vav_chiller_tower': dict(primary_use='office', floor_area_m2=40000, n_floors=10,
                                          system='vav_chiller_boiler', hvac=dict(chiller_condenser='water')),
         'hotel_ptac': dict(primary_use='lodging', system='ptac'),
         'school_fan_coil': dict(primary_use='education', system='fan_coil')}


def main():
    w = read_epw(epw_path('Chicago'))
    year = 365 * 86400.0
    res = {}
    for name, spec in CASES.items():
        for dt in ('15min', '1h'):
            wxm._CACHE.clear()
            t0 = time.time()
            out = simulate(spec, w, '2019-01-01', '2020-01-01', dt=dt)
            cold = time.time() - t0
            t0 = time.time()
            out = simulate(spec, w, '2019-01-01', '2020-01-01', dt=dt)
            warm = time.time() - t0
            res[f'{name}_{dt}'] = dict(total_s_first=cold, total_s_cached_weather=warm, loop_s=out['summary']['wall_s'],
                                      x_realtime_total=year / warm, x_realtime_loop=year / out['summary']['wall_s'])
            print(name, dt, res[f'{name}_{dt}'], flush=True)
    specs = generate_buildings(40, seed=5)
    ws = {}
    t0 = time.time()
    for s in specs:
        c = s['weather_hint']
        if c not in ws:
            ws[c] = read_epw(epw_path(c))
        simulate(s, ws[c], '2019-01-01', '2020-01-01', dt='15min')
    gen = (time.time() - t0) / len(specs)
    res['generated_mixed_15min_mean'] = dict(s_per_building_year=gen, x_realtime=year / gen)
    print(res['generated_mixed_15min_mean'])
    write('speed.json', dict(what='wall time per simulated building-year, single thread, shared 2-core Xeon',
                             target='>= 1,000x real time for a single-zone building-year at 15 min',
                             cpu=os.popen('grep -m1 "model name" /proc/cpuinfo').read().strip(), results=res))


if __name__ == '__main__':
    main()
