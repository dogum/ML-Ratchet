"""NOAA National Water Model retrospective v3.0 streamflow at the CAMELS-US gauges (resumable).

Source: s3://noaa-nwm-retrospective-3-0-pds (public, anonymous HTTPS), CONUS/netcdf/CHANOBS/<YYYY>/<YYYYMMDDHH00>.CHANOBS_DOMAIN1
(hourly netCDF4 with the 8,643 gauge reaches). The streamflow chunk is the last block of every file at a fixed byte
offset (162458, checked on files from 1980..2022), so one HTTP Range request returns just the zlib-compressed
streamflow vector (~32 KB instead of 195 KB). Crosswalk feature_id -> USGS gage from chrtout.zarr gage_id.

Every hour is fetched (~6 s per month-of-3-hourly, ~18 s per month hourly with 6 threads).
Stored per month: raw/nwm/chanobs_hourly/<YYYY-MM>.parquet (rows: UTC valid time; columns: CAMELS gauges; m3/s).
Daily aggregation (UTC day and local-standard-time day) is done in build_nwm.py.
"""
import io, os, sys, time, zlib, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests, pyarrow as pa, h5py, threading
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
OUT = f'{ROOT}/raw/nwm/chanobs_hourly'
XW = f'{ROOT}/proc/nwm/crosswalk.csv'
B = 'https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com'
OFF = 162458
NFEAT = 8643
PERIODS = [('1989-10-01', '1999-09-30'), ('2015-10-01', '2023-01-31'), ('1999-10-01', '2015-09-30'),
           ('1980-01-01', '1989-09-30')]
tl = threading.local()


def sess():
    if not hasattr(tl, 's'):
        tl.s = requests.Session()
    return tl.s


def crosswalk():
    if os.path.exists(XW):
        return pd.read_csv(XW, dtype={'gauge_id': str})
    os.makedirs(os.path.dirname(XW), exist_ok=True)
    c = pa.Codec('zstd'); n = 2776734
    z = f'{B}/CONUS/zarr/chrtout.zarr'
    g = np.frombuffer(c.decompress(requests.get(f'{z}/gage_id/0', timeout=300).content, decompressed_size=n * 15), dtype='S15')
    fid = np.frombuffer(c.decompress(requests.get(f'{z}/feature_id/0', timeout=300).content, decompressed_size=n * 8), dtype='<i8')
    d = {gg.strip().decode(): int(ff) for gg, ff in zip(g, fid) if gg.strip()}
    raw = requests.get(f'{B}/CONUS/netcdf/CHANOBS/1990/199001010000.CHANOBS_DOMAIN1', timeout=300).content
    cf = h5py.File(io.BytesIO(raw), 'r')['feature_id'][:]
    pos = {int(f): i for i, f in enumerate(cf)}
    a = pd.read_csv(f'{ROOT}/raw/caravan/attributes/camels/attributes_other_camels.csv', dtype=str)
    rows = []
    for gid in a.gauge_id:
        u = gid.split('_')[1]
        f = d.get(u)
        rows.append({'gauge_id': u, 'feature_id': f, 'chanobs_index': pos.get(f) if f is not None else None})
    df = pd.DataFrame(rows)
    df.to_csv(XW, index=False)
    return df


def fetch(ts):
    url = f'{B}/CONUS/netcdf/CHANOBS/{ts:%Y}/{ts:%Y%m%d%H}00.CHANOBS_DOMAIN1'
    for i in range(8):
        try:
            r = sess().get(url, headers={'Range': f'bytes={OFF}-'}, timeout=(20, 60))
            if r.status_code == 404:
                return None
            r.raise_for_status()
            v = np.frombuffer(zlib.decompress(r.content), dtype='<f4')
            if v.size != NFEAT:
                raise IOError(f'bad size {v.size}')
            return v
        except Exception as e:
            time.sleep(3 * (i + 1))
    print('FAILED', url, flush=True)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    xw = crosswalk()
    xw = xw.dropna(subset=['chanobs_index'])
    idx = xw.chanobs_index.astype(int).values
    cols = xw.gauge_id.tolist()
    print(len(cols), 'gauges in CHANOBS', flush=True)
    pool = ThreadPoolExecutor(12)
    for a, b in PERIODS:
        for m in pd.period_range(a, b, freq='M'):
            p = f'{OUT}/{m}.parquet'
            if os.path.exists(p):
                continue
            days = pd.date_range(max(pd.Timestamp(a), m.start_time.normalize()), min(pd.Timestamp(b), m.end_time.normalize()), freq='D')
            times = [d + pd.Timedelta(hours=h) for d in days for h in range(0, 24)]
            t0 = time.time()
            vals = list(pool.map(fetch, times))
            arr = np.full((len(times), len(cols)), np.nan, dtype=np.float32)
            nmiss = 0
            for i, v in enumerate(vals):
                if v is None:
                    nmiss += 1
                    continue
                x = v[idx].astype(np.float32)
                x[x < -1] = np.nan
                arr[i] = x
            df = pd.DataFrame(arr, index=pd.DatetimeIndex(times, name='time_utc'), columns=cols)
            df.to_parquet(p + '.tmp', compression='zstd'); os.replace(p + '.tmp', p)
            print(m, len(times), 'files', nmiss, 'missing', f'{time.time() - t0:.0f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
