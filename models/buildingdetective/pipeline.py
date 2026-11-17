"""BuildingDetective command line.

  python3 pipeline.py hourly  meters.csv   # columns: time (UTC ISO), T_out (degC), GHI (W/m2), and any of
                                           #   elec, fuel, dheat, dcool (kW, hour-beginning); --tz America/New_York --lat 40.1
  python3 pipeline.py monthly bills.csv    # columns: month (1-12), T_mean (degC), HDD18 and CDD18 per day, days,
                                           #   and any of elec, fuel, dheat, dcool as mean kW over the billing month; --lat
  python3 pipeline.py annual --elec 9800 --fuel 14500 --hdd 2900 --cdd 600      (kWh per year, degree days base 18.33 C)
Prints a JSON reading: each quantity as [value, low, high] at +-1 sigma, classes as probabilities.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buildingdetective import Detective  # noqa: E402

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')


def clean(r):
    return {k: v for k, v in r.items() if not k.startswith('_')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['hourly', 'monthly', 'annual'])
    ap.add_argument('csv', nargs='?')
    ap.add_argument('--tz', default='UTC'); ap.add_argument('--lat', type=float, default=40.0)
    for c in ('elec', 'fuel', 'dheat', 'dcool'):
        ap.add_argument('--' + c, type=float)
    ap.add_argument('--hdd', type=float); ap.add_argument('--cdd', type=float)
    a = ap.parse_args()
    det = Detective(W)
    if a.mode == 'annual':
        kwh = {c: getattr(a, c) for c in ('elec', 'fuel', 'dheat', 'dcool') if getattr(a, c)}
        r = det.read_annual(kwh, a.hdd, a.cdd)
    else:
        import pandas as pd
        d = pd.read_csv(a.csv)
        chans = [c for c in ('elec', 'fuel', 'dheat', 'dcool') if c in d]
        if a.mode == 'hourly':
            idx = pd.DatetimeIndex(pd.to_datetime(d['time'], utc=True)).tz_convert(a.tz)
            r = det.read_hourly({c: d[c].to_numpy(float) for c in chans}, d['T_out'].to_numpy(float),
                                d['GHI'].to_numpy(float), idx, a.lat)
        else:
            d = d.set_index('month').reindex(range(1, 13))
            r = det.read_monthly({c: d[c].to_numpy(float) for c in chans}, d['T_mean'].fillna(0).to_numpy(float),
                                 d['HDD18'].fillna(0).to_numpy(float), d['CDD18'].fillna(0).to_numpy(float),
                                 d['days'].fillna(0).to_numpy(float), a.lat)
    print(json.dumps(clean(r), indent=1))


if __name__ == '__main__':
    main()
