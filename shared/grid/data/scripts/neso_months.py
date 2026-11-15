"""Full months of NESO 1 s GB frequency via the CKAN datastore SQL API, paged by time (6 h per query, <= 21,600 rows).
Stored compactly: proc/frequency/GB/1s/<YYYY-MM>.npz with t0_unix (int64, first second, UTC), df_mHz (int16,
(f - 50 Hz) in mHz on a regular 1 s grid; -32768 = missing). griddata.frequency() returns contract arrays t [s], f [Hz].
Resumable per 6 h chunk (raw/neso/month_chunks/)."""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neso_sql import sql
from gb_events import parse_win
RAW = f'{GRID_ROOT}/data/raw/neso'; OUT = f'{GRID_ROOT}/data/proc/frequency/GB/1s'
os.makedirs(OUT, exist_ok=True); os.makedirs(f'{RAW}/month_chunks', exist_ok=True)
RES = json.load(open(f'{RAW}/freq_resources.json'))
months = sys.argv[1:] or ['2019-08', '2018-07', '2020-07', '2022-01']
for ym in months:
    if os.path.exists(f'{OUT}/{ym}.npz'): continue
    rid = RES[ym]
    # time format of this resource: inspect one row
    one = sql(f'SELECT dtm::text AS dtm FROM "{rid}" LIMIT 1')[0]['dtm']
    sep = 'T' if 'T' in one else ' '
    start = pd.Timestamp(ym + '-01'); end = start + pd.offsets.MonthBegin(1)
    # local-time resources (2018-2019, '+0100' suffix) are queried in their own text time; 6 h chunks overlap DST safely
    chunks = pd.date_range(start - pd.Timedelta(hours=6), end + pd.Timedelta(hours=6), freq='6h')
    parts = []
    for a, b in zip(chunks[:-1], chunks[1:]):
        f = f'{RAW}/month_chunks/{ym}_{a:%Y%m%d%H}.parquet'
        if not os.path.exists(f):
            q = f'''SELECT dtm::text AS dtm, f FROM "{rid}" WHERE dtm >= '{a.strftime(f"%Y-%m-%d{sep}%H:%M:%S")}' AND dtm < '{b.strftime(f"%Y-%m-%d{sep}%H:%M:%S")}' ORDER BY dtm'''
            try: rec = sql(q, timeout=600)
            except Exception as e: print('FAIL', ym, a, e, flush=True); continue
            pd.DataFrame(rec, columns=['dtm', 'f']).to_parquet(f, index=False)
        parts.append(pd.read_parquet(f))
    w = pd.concat(parts, ignore_index=True)
    x = parse_win(w)
    x = x[~x.index.duplicated()]
    t0 = pd.Timestamp(ym + '-01', tz='UTC'); t1 = t0 + pd.offsets.MonthBegin(1)
    grid = pd.date_range(t0, t1, freq='1s', inclusive='left')
    x = x.reindex(grid)
    df = np.round((x.values - 50.0) * 1000)
    arr = np.where(np.isfinite(df), df, -32768).astype(np.int16)
    np.savez_compressed(f'{OUT}/{ym}.npz', t0_unix=np.int64(t0.timestamp()), df_mHz=arr)
    json.dump(dict(region='GB', nominal_hz=50.0, start_utc=str(t0), source='NESO system frequency data (1 s), CKAN datastore API',
                   missing_s=int((arr == -32768).sum()), n_s=int(len(arr))), open(f'{OUT}/{ym}.json', 'w'))
    print('month', ym, 'missing', int((arr == -32768).sum()), 'of', len(arr), flush=True)
    for a in chunks[:-1]:
        f = f'{RAW}/month_chunks/{ym}_{a:%Y%m%d%H}.parquet'
        if os.path.exists(f): os.remove(f)
print('DONE')
