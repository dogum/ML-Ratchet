"""proc/lfmc/lfmc_all.parquet -> proc/lfmc/lfmc_us.parquet: Globe-LFMC 2.0 samples in the contiguous US, every year and
value, Globe-LFMC's own column names, plus a `date` column (the sampling date). build_gridmet.py,
baselines_moisture.py and FireDetective read it; they apply their own year and value filters.

RECONSTRUCTED on import (21 Nov 2026): the sandbox step that wrote this file was not shipped. CONUS here is
Country == 'USA' without Alaska (the table has no Hawaii rows). Checked against the shipped
proc/baselines/moisture_exam.json: with baselines_moisture.py's filters (2012-2022, 20-400 %) it gives 1,167 sites and
15,361 dev + 138,113 exam samples, and the seeded site split (20261118 + 3) gives 122 dev sites, all as shipped.
"""
import os
import pandas as pd

_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
P = f'{_DATA}/data/proc/lfmc'

d = pd.read_parquet(f'{P}/lfmc_all.parquet')
us = d[(d['Country'] == 'USA') & (d['State/Region'] != 'Alaska')].copy()
us['date'] = pd.to_datetime(us['Sampling date (YYYYMMDD)']).dt.normalize()
us = us.reset_index(drop=True)
us.to_parquet(f'{P}/lfmc_us.parquet')
print('lfmc_us', len(us), 'rows', us['Site name'].nunique(), 'sites',
      us['date'].min().date(), '..', us['date'].max().date(), flush=True)
