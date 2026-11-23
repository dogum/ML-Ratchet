"""Real CAMELS-US basins for the water-balance law: inputs from the validation-decade forcing, truth from gauged Q.

Dev (52 basins) is built here. Exam basins are read ONLY by exam_water.py (for_exam=True).
Period: WaterData validation decade 1989-10-01..1999-09-30 (the main exam period); dev basins may use any period.
"""
import os, sys, json
import numpy as np, pandas as pd
import watersim as ws, waterdata as wd
from water_bank import features, soil_proxy

from _paths import ROOT  # the work folder (training/_paths.py)
VAL = ('1989-10-01', '1999-09-30')


def basin_record(bid, truth=True):
    f = wd.forcing(bid, *VAL)
    rec = features(f['P'].values, f['PET'].values, f['T_mean'].values, f.index)
    spec = ws.fill_spec(wd.spec(bid))
    spec = spec[0] if isinstance(spec, tuple) else spec
    rec['soil_proxy'] = soil_proxy(spec)
    rec['basin'] = bid
    if truth:
        q = wd.streamflow(bid, 'best')['Q_mm'].loc[VAL[0]:VAL[1]].dropna()
        P = f['P'].reindex(q.index)
        rec['rr'] = float(q.sum() / P.sum()) if len(q) >= 3000 else np.nan
        rec['n_q_days'] = int(len(q))
    return rec


if __name__ == '__main__':
    ids = wd.basins('camels', split='dev')
    df = pd.DataFrame([basin_record(b) for b in ids])
    os.makedirs(f'{ROOT}/results', exist_ok=True)
    df.to_parquet(f'{ROOT}/results/water_dev.parquet')
    print(df.describe().T[['mean', 'min', 'max']])
