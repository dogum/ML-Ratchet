"""Canadian FWI System: (1) the standard first test row (Van Wagner & Pickett 1985 test data, as in Wang et al.
2015): 13 April, T 17 C, RH 42 %, wind 25 km/h, no rain, start FFMC 85 / DMC 6 / DC 15 -> FFMC 87.69, DMC 8.55,
DC 19.01, ISI 10.85, BUI 8.49, FWI 10.10; (2) a 180-day generated season vs xclim 1.0.0 (an independent
implementation; isolated venv)."""
import os
import subprocess

import numpy as np
import pandas as pd

from _common import write, OUT, CHECKS, FIRE_ROOT
import firesim as fs
from firesim import fwi

ref = dict(FFMC=87.69, DMC=8.55, DC=19.01, ISI=10.85, BUI=8.49, FWI=10.10)
row = pd.DataFrame(dict(T=[17.0], RH=[42.0], wind_kmh=[25.0], rain_24h=[0.0]), index=[pd.Timestamp('1985-04-13')])
m = fwi.fwi_series(row).iloc[0]
test = {k: dict(firesim=float(m[k]), published=v, abs_err=float(abs(m[k] - v))) for k, v in ref.items()}
print(test)
# season
w = fs.generate_weather('2021-04-01', hours=24 * 180, climate='northern_rockies_summer', lat=46.0, lon=-100.0, rng=5)
d = fwi.daily_from_hourly(w, lon=-100.0)
d.index.name = 'date'
# xclim runs in the isolated reference venv (firesim/.refenv under the data root). Without it, the xclim output
# shipped in checks/out (`mlratchet data fire --from-zip`) is reused, but only if this run's inputs equal the shipped ones.
REFPY = os.path.join(FIRE_ROOT, 'firesim', '.refenv', 'Scripts' if os.name == 'nt' else 'bin', 'python')
if os.path.exists(REFPY) or os.path.exists(REFPY + '.exe'):
    d.reset_index().to_csv(f'{OUT}/fwi_inputs.csv', index=False)
    subprocess.run([REFPY, f'{CHECKS}/ref/xclim_fwi_run.py', OUT], check=True)
    xsrc = 'xclim run now in firesim/.refenv'
elif os.path.exists(f'{OUT}/fwi_xclim.csv') and os.path.exists(f'{OUT}/fwi_inputs.csv'):
    shipped = pd.read_csv(f'{OUT}/fwi_inputs.csv', parse_dates=['date']).set_index('date')
    same = shipped.shape == d.shape and all(np.allclose(shipped[c].to_numpy(float), d[c].to_numpy(float), rtol=0, atol=1e-9)
                                            for c in d.columns)
    xsrc = 'shipped checks/out/fwi_xclim.csv (same inputs)' if same else None
else:
    xsrc = None
print('xclim:', xsrc or 'skipped (no firesim/.refenv and no shipped checks/out/fwi_xclim.csv for these inputs)')
cmp = {}
if xsrc:
    x = pd.read_csv(f'{OUT}/fwi_xclim.csv', parse_dates=['date']).set_index('date')
    mine = fwi.fwi_series(d)
    for k in ('FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'):
        e = np.abs(mine[k].to_numpy() - x[k].to_numpy())
        cmp[k] = dict(max_abs_diff=float(e.max()), mean=float(mine[k].mean()), max=float(mine[k].max()))
print(cmp)
write('fwi.json', dict(what='Canadian FWI System vs the standard test row and vs xclim 1.0.0 on a 180-day season',
                       test_row=test, xclim_season=dict(n_days=len(d), rain_days=int((d.rain_24h > 0.5).sum()), elements=cmp)
                       if xsrc else dict(skipped='xclim reference environment absent')))
