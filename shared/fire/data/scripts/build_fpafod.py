"""FPA-FOD 6th ed. (Short 2022, RDS-2013-0009.6; 1992-2020, 2.30 M US wildfires) -> proc/fpafod/fod.parquet
(selected columns, 2001-2020 only, CONUS), then the extracted 0.93 GB gpkg is deleted (zip kept)."""
import os, warnings, numpy as np, pandas as pd, pyogrio
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
warnings.filterwarnings('ignore')
X = f'{_DATA}/data/raw/fpafod/x/Data/FPA_FOD_20221014.gpkg'
COLS = ['FOD_ID', 'FIRE_NAME', 'MTBS_ID', 'ICS_209_PLUS_INCIDENT_JOIN_ID', 'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY',
        'DISCOVERY_TIME', 'NWCG_CAUSE_CLASSIFICATION', 'NWCG_GENERAL_CAUSE', 'CONT_DATE', 'CONT_TIME', 'FIRE_SIZE',
        'FIRE_SIZE_CLASS', 'LATITUDE', 'LONGITUDE', 'OWNER_DESCR', 'STATE', 'NWCG_REPORTING_AGENCY']
parts = []
for y in range(2001, 2021):
    m, t = pyogrio.raw.read_arrow(X, columns=COLS, read_geometry=False, where=f'FIRE_YEAR = {y}')
    d = t.to_pandas()
    d = pd.DataFrame({c: np.array(d[c].tolist(), dtype=object) for c in d.columns})
    parts.append(d); print(y, len(d), flush=True)
d = pd.concat(parts, ignore_index=True)
for c in ['FOD_ID', 'FIRE_YEAR', 'DISCOVERY_DOY']: d[c] = pd.to_numeric(d[c]).astype('int64')
for c in ['FIRE_SIZE', 'LATITUDE', 'LONGITUDE']: d[c] = pd.to_numeric(d[c]).astype('float64')
for c in ['DISCOVERY_DATE', 'CONT_DATE']: d[c] = pd.to_datetime(d[c], errors='coerce')
for c in d.columns:
    if d[c].dtype == object: d[c] = d[c].astype('string')
d = d[~d.STATE.isin(['AK', 'HI', 'PR'])]
d.to_parquet(f'{_DATA}/data/proc/fpafod/fod.parquet', compression='zstd')
print('rows', len(d))
os.remove(X)
