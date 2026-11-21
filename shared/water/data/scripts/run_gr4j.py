"""Per-basin calibrated GR4J + degree-day snow for all 671 CAMELS-US basins (resumable, one JSON per basin).

Calibration: 1999-10-01..2008-09-30 (spin-up 1998-10-01..1999-09-30), objective NSE, DE seed 20261118.
Forcing: Caravan ERA5-Land P, T_mean, PET (FAO-56 PM). Obs: Caravan (CAMELS) Q_mm.
Output: proc/baselines/gr4j/<id>.json (params, cal NSE) and, at the end, proc/baselines/sims/gr4j_gauged.parquet
(daily simulated Q_mm 1980-10-01..2023-12-30, one column per basin).
"""
import json, os, sys, time
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
sys.path.insert(0, _CODE)
import waterdata as wd
from waterdata.models import calibrate_gr4j, gr4j_snow

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
OUT = f'{ROOT}/proc/baselines/gr4j'
SIM0, SIM1 = '1980-10-01', '2023-12-30'


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(f'{ROOT}/proc/baselines/sims', exist_ok=True)
    ids = wd.basins('camels')
    for k, b in enumerate(ids):
        p = f'{OUT}/{b}.json'
        if os.path.exists(p):
            continue
        t0 = time.time()
        f = wd.forcing(b, '1998-10-01', '2008-09-30')
        q = wd.streamflow(b, 'caravan').Q_mm.reindex(f.index)
        th, n = calibrate_gr4j(f.P.values, f.T_mean.values, f.PET.values, q.values, spin=365, seed=20261118)
        json.dump({'basin': b, 'params': [float(x) for x in th], 'names': ['x1', 'x2', 'x3', 'x4', 'tt', 'ddf'],
                   'nse_cal': float(n), 'n_obs_cal': int(q.iloc[365:].notna().sum()), 'secs': round(time.time() - t0, 1)},
                  open(p + '.tmp', 'w'))
        os.replace(p + '.tmp', p)
        if k % 20 == 0:
            print(k, b, round(n, 3), round(time.time() - t0, 1), 's', flush=True)
    sims = {}
    for b in ids:
        th = json.load(open(f'{OUT}/{b}.json'))['params']
        f = wd.forcing(b, SIM0, SIM1)
        sims[b] = gr4j_snow(th, f.P.values, f.T_mean.values, f.PET.values).astype(np.float32)
    idx = wd.forcing(ids[0], SIM0, SIM1).index
    pd.DataFrame(sims, index=idx).to_parquet(f'{ROOT}/proc/baselines/sims/gr4j_gauged.parquet', compression='zstd')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
