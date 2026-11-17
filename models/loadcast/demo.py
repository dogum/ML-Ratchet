"""LoadCast demo: forecast local 2017 hourly energy for 3 real BDG2 exam building-meters (a fixed draw, not picked by
result) from their metadata only (cold) and from their own 2016 meter (warm), and compare with ASHRAE Guideline 14
CV(RMSE) of the classical baselines BuildingData fitted on 2016.  Runs in a few seconds:  python3 demo.py
"""
import json, os, time
import numpy as np, pandas as pd
import loadcast as lc

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, 'examples', 'bdg2')   # BDG2-derived example data (CC BY-SA 4.0, see LICENSE-DATA.md there)


def main():
    t0 = time.time()
    m = lc.LoadCast()
    info = json.load(open(os.path.join(EX, 'examples.json')))
    print(f"{'building / meter':45s} {'mode':6s} {'CV(RMSE)%':>9s} {'NMBE%':>7s} {'cov90':>6s} | baselines CV: TOWT  change-point  persist-24h*")
    for e in info:
        w = pd.read_csv(os.path.join(EX, e['weather_file']), index_col=0, parse_dates=True)
        y = pd.read_csv(os.path.join(EX, e['meter_file']), index_col=0, parse_dates=True)['kWh']
        loc = y.index.tz_convert(e['tz'])
        hist = y[loc.year == 2016]; real = y[loc.year == 2017]
        frame = m._frame(w, e['tz'])
        for mode, h in (('cold', None), ('warm', hist)):
            fc = m.forecast(w, e['tz'], e['lat'], e['use'], e['area_m2'], e['meter'], e['meters_present'],
                            start='2017-01-01', end='2018-01-01', history=h, year_built=e['year_built'], frame=frame)
            r = real.reindex(fc.index).values
            cv, nmbe, n = lc.g14(r, fc.kW.values)
            ok = np.isfinite(r)
            cov = np.mean((r[ok] >= fc.lo90.values[ok]) & (r[ok] <= fc.hi90.values[ok]))
            b = e['baselines_2017']
            print(f"{e['building'] + ' / ' + e['meter']:45s} {mode:6s} {cv:9.1f} {nmbe:7.1f} {cov:6.2f} | "
                  f"{b['towt']:5.1f}  {b['g14_changepoint']:5.1f}  {b['persistence_24h']:5.1f}")
    print('* persistence uses the actual 2017 reading of the day before (a nowcast, not a forecast from 2016).')
    print(f'done in {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
