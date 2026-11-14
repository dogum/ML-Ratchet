"""Full DemandCast pipeline for one grid and one day, from raw inputs (needs the ML-Ratchet grid data: GridData's
EIA-930, fleet and weather files, and DemandCast's weather bundles built by training/wxprep.py into work/data/wx/;
wxprep.py needs NASA POWER series at every weather point: GridData's data rebuild, `power` stage, not the v1.0 zip).

    python3 pipeline.py ERCO 2024-08-20            # actual weather (upper bound)
    python3 pipeline.py ERCO 2024-08-20 forecast   # day-ahead forecast weather (Open-Meteo previous runs, if cached)

Steps: weather bundle -> building-stock physics with the grid's dev-fitted parameters (2019, 2021) -> wind/solar farm
physics with the real fleet and the dev-calibrated scale -> features (last 168 h of real output vs physics) ->
three MLPs -> hourly demand, wind and solar with 95 % bands."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from demandcast import DemandCast


def forecast(ba, day, weather='actual'):
    if os.path.join(HERE, 'training') not in sys.path:   # training/ (paths via training/_paths.py) only when run
        sys.path.insert(0, os.path.join(HERE, 'training'))
    from common import day_start_index
    import real
    g = real.real_region(ba)
    i0 = day_start_index(ba, day)
    fc = None
    if weather == 'forecast':
        fc = real.Forecasts(ba).fc(i0 - 14, i0 + 24)
        if fc is None: raise SystemExit('no archived forecast for that day')
    s = g.sample(i0, fc)
    m = DemandCast()
    out = dict(demand=m.predict_demand(s['xd'][None], s['pd_total'][None]), demand_actual=s['d_true'],
               ba_day_ahead=g.df[i0:i0 + 24])
    if 'xw' in s:
        out['wind'] = {k: v * s['capw'] for k, v in m.predict_cf('w', s['xw'][None], s['pw'][None]).items() if k != 'sigma'}
        out['wind_actual'] = s['yw'] * s['capw']
    if 'xs' in s:
        out['solar'] = {k: v * s['caps'] for k, v in m.predict_cf('s', s['xs'][None], s['ps'][None], s['night'][None]).items() if k != 'sigma'}
        out['solar_actual'] = s['ys'] * s['caps']
    return out


if __name__ == '__main__':
    ba, day = sys.argv[1], sys.argv[2]
    o = forecast(ba, day, sys.argv[3] if len(sys.argv) > 3 else 'actual')
    print('hour  demand MW [95 % band]        actual   grid DA')
    for h in range(24):
        d = o['demand']
        print(f"{h:4d}  {d['mw'][0, h]:8.0f} [{d['lo95'][0, h]:7.0f}-{d['hi95'][0, h]:7.0f}]  {o['demand_actual'][h]:8.0f} {o['ba_day_ahead'][h]:8.0f}")
