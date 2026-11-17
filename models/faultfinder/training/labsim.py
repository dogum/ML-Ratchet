"""Page lab: one simulated VAV air handler (office, Chicago TMYx) on three days (winter, spring, summer), healthy and
with six fault types at six severities each. BMS-style logging (commands, not positions) as in bank v3, and one fixed
real-noise draw per day stored separately (delta), so the page can scale the noise with a slider.
Usage: python3 labsim.py  -> web/lab_raw.npz"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RI, sim_raw, to_day  # noqa: E402
from simbank import log_commands, weather  # noqa: E402
from noise import Noise  # noqa: E402
import buildingsim as bs  # noqa: E402

SPEC = dict(primary_use='office', floor_area_m2=6000, n_floors=3, year_built=2000, climate_zone='5A',
            system='vav_chiller_boiler', hvac=dict(sat_cool=12.8, sat_reset=True, vav_min_frac=0.35))
DAYS = {'winter': '2019-01-23', 'spring': '2019-04-24', 'summer': '2019-07-17'}
FAULTS = {   # name -> (BuildingSim fault builder, six slider values, label, unit)
    'oa_damper_stuck': (lambda v: dict(type='oa_damper_stuck', position=v), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        'Outdoor-air damper stuck at', 'open fraction'),
    'cooling_valve_stuck': (lambda v: dict(type='cooling_valve_stuck', position=v), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            'Cooling valve stuck at', 'open fraction'),
    'heating_valve_leak': (lambda v: dict(type='heating_valve_leak', leak=v), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                           'Heating valve leaking', 'leak fraction'),
    'sensor_bias_T_sa': (lambda v: dict(type='sensor_bias', sensor='T_sup', bias=v), [-5.0, -3.0, -1.5, 1.5, 3.0, 5.0],
                         'Supply-air sensor reads off by', 'K'),
    'coil_fouling': (lambda v: dict(type='coil_fouling', severity=v), [0.1, 0.2, 0.3, 0.45, 0.6, 0.7],
                     'Cooling coil fouled', 'capacity loss'),
    'low_airflow': (lambda v: dict(type='low_airflow', severity=v), [0.1, 0.2, 0.3, 0.4, 0.5, 0.55],
                    'Supply airflow restricted (dirty filter)', 'flow loss'),
}


class _Keep:   # rng stand-in for log_commands: never keep the actual position (BMS logs the command)
    def random(self):
        return 1.0


def sim_day(day, fault):
    w = weather('Chicago')
    tz = w[1]['tz']
    d0 = pd.Timestamp(day)
    start = (d0 - pd.Timedelta(days=2)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    end = (d0 + pd.Timedelta(days=1)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    f = None
    if fault is not None:
        f = dict(fault, start=str(d0.tz_localize(tz).tz_convert('UTC').tz_localize(None)))   # fault from local midnight
        f = [f]
    out = bs.simulate(json.loads(json.dumps(SPEC)), w, str(start), str(end), dt='15min', faults=f)
    return to_day(sim_raw(out, tz), d0)


def main():
    nz = Noise(f'{ROOT}/data/noise_v1.npz', patterns=False)
    res = {}
    for dn, day in DAYS.items():
        xh = sim_day(day, None)
        rng = np.random.default_rng({'winter': 1, 'spring': 2, 'summer': 3}[dn])
        delta = nz.apply(xh, rng, 'vav_chiller_boiler') - xh
        delta = np.nan_to_num(delta)
        res[f'{dn}__healthy'] = xh
        res[f'{dn}__noise'] = delta
        for fn, (mk, vals, _, _) in FAULTS.items():
            for k, v in enumerate(vals):
                fd = mk(v)
                xf = sim_day(day, fd)
                xf = log_commands(xh, xf, fd, _Keep())
                res[f'{dn}__{fn}__{k}'] = xf
        print(dn, 'done', flush=True)
    os.makedirs(f'{ROOT}/web', exist_ok=True)
    np.savez_compressed(f'{ROOT}/web/lab_raw.npz', **res)
    json.dump({fn: dict(values=v[1], label=v[2], unit=v[3]) for fn, v in FAULTS.items()},
              open(f'{ROOT}/web/lab_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
