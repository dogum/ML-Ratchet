"""Dumb baseline: median of each quantity per technology over the dev units (dev years 2019, 2021)."""
from common import *
import numpy as np, pandas as pd
from pdcore import truth, REG
from evaluate import real_unit, DEV_YEARS
import realdata
U = realdata.units(); U = U[U.split == 'dev']
rows = []
for uid in U.unit_id:
    rec, t = real_unit('dev', uid)
    m = np.isin(pd.to_datetime(t, unit='s').year, DEV_YEARS)
    tr = truth(*(rec[k][m] for k in ('op', 'P', 'HI', 'meas')))
    rows.append(dict(tech=realdata.truth_tech(U.set_index('unit_id').loc[uid]), **{k: tr[k] for k in REG}))
d = pd.DataFrame(rows).groupby('tech').median()
d.to_csv(f'{DATA}/dev_medians.csv'); print(d.round(3).to_string())
