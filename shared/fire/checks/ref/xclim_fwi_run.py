"""Run xclim's CFFWIS element functions (isolated venv) on checks/out/fwi_inputs.csv -> out/fwi_xclim.csv.
usage: python xclim_fwi_run.py [out_dir]   (default ../out)"""
import os
import sys
import numpy as np
import pandas as pd
from xclim.compute.fire import _cffwis as c
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out')
x = pd.read_csv(os.path.join(d, 'fwi_inputs.csv'), parse_dates=['date'])
f, p, dc = 85.0, 6.0, 15.0
rows = []
for _, r in x.iterrows():
    f = float(c._fine_fuel_moisture_code(r['T'], r['rain_24h'], r['wind_kmh'], r['RH'], f))
    p = float(c._duff_moisture_code(r['T'], r['rain_24h'], r['RH'], int(r['date'].month), 46.0, p))
    dc = float(c._drought_code(r['T'], r['rain_24h'], int(r['date'].month), 46.0, dc))
    i = float(c.initial_spread_index(np.array([r['wind_kmh']]), np.array([f]))[0])
    b = float(c.build_up_index(np.array([p]), np.array([dc]))[0])
    w = float(c.fire_weather_index(np.array([i]), np.array([b]))[0])
    rows.append((r['date'], f, p, dc, i, b, w))
pd.DataFrame(rows, columns=['date', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']).to_csv(os.path.join(d, 'fwi_xclim.csv'), index=False)
print('ok', len(rows))
