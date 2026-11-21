"""The area CAMELS/Caravan used to turn USGS discharge into mm/day, recovered from the data:
implied_area_km2 = median(Q_nwis[m3/s] * 86.4 / Q_caravan[mm/day]) over days with Q_caravan >= 0.05 mm/day (>= 50 days).
It equals the NWIS contributing drainage area where one is published (GAGES-II convention) and the NWIS drainage area
otherwise (median ratio 0.9995; 36 basins differ by > 2%, up to 6x for prairie basins with large non-contributing areas).
Out: proc/attributes/camels_implied_area.csv   (run after build_nwis.py has produced proc/nwis/daily)
"""
import sys
import numpy as np, pandas as pd
import os
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
a = pd.read_parquet(f'{ROOT}/proc/attributes/camels.parquet').set_index('gauge_id')
rows = []
for b in a.index:
    c = pd.read_parquet(f'{ROOT}/proc/caravan/camels/{b}.parquet', columns=['Q_mm']).Q_mm
    n = pd.read_parquet(f'{ROOT}/proc/nwis/daily/{b}.parquet', columns=['Q_cms']).Q_cms
    j = pd.concat([c, n], axis=1, keys=['mm', 'cms']).dropna(); j = j[(j.mm >= 0.05) & (j.cms > 0)]
    r = j.cms * 86.4 / j.mm
    ok = len(r) > 50
    rows.append({'gauge_id': b, 'implied_area_km2': float(r.median()) if ok else np.nan,
                 'implied_spread_q90_q10': float(r.quantile(0.9) / r.quantile(0.1)) if ok else np.nan, 'n_days': len(r)})
d = pd.DataFrame(rows)
d.to_csv(f'{ROOT}/proc/attributes/camels_implied_area.csv', index=False)
print(d.describe())
