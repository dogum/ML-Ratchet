"""gridMET daily (Abatzoglou 2013; northwestknowledge.net/metdata/data, public domain) at points.
Each var-year file (~100-300 MB) is downloaded to tmp, read in 61-day chunks, and deleted (never stored).
Points: (A) every FEDS-MTBS CONUS fire centroid, full year of ignition -> proc/weather/gridmet/fires_<year>.parquet
        (B) FPA-FOD fires 2012-2020: discovery day and the day before -> proc/weather/gridmet/fod_<year>.parquet
        (C) Globe-LFMC CONUS sites, 2012-2022, full year -> proc/weather/gridmet/lfmc_<year>.npz
Resumable per year. Units: erc, bi (index), fm100/fm1000 (%), vs (m/s at 10 m), th (deg from), rmin/rmax (%),
tmmx/tmmn (K -> stored in degC), pr (mm), vpd (kPa). Day = local calendar day (gridMET day is 06 UTC-06 UTC? no:
gridMET daily values are for the local day as in PRISM/NLDAS aggregation; see DATA.md)."""
import os, sys, time, io, numpy as np, pandas as pd, h5py, requests
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
P = f'{_DATA}/data/proc'
OUT = f'{P}/weather/gridmet'; os.makedirs(OUT, exist_ok=True)
TMP = f'{_DATA}/data/tmp/gridmet'; os.makedirs(TMP, exist_ok=True)
VARS = ['erc', 'bi', 'fm100', 'fm1000', 'vs', 'th', 'rmin', 'rmax', 'tmmx', 'tmmn', 'pr', 'vpd']
LFMC_VARS = ['erc', 'fm100', 'fm1000', 'pr', 'tmmx', 'rmin', 'vs', 'vpd']
URL = 'https://www.northwestknowledge.net/metdata/data/{v}_{y}.nc'
S = requests.Session()


def download(v, y):
    fn = f'{TMP}/{v}_{y}.nc'
    for i in range(5):
        try:
            with S.get(URL.format(v=v, y=y), stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                with open(fn, 'wb') as f:
                    for ch in r.iter_content(1 << 20): f.write(ch)
            return fn
        except Exception as e:
            print('retry', v, y, e, flush=True); time.sleep(10 * (i + 1))
    raise RuntimeError(f'download failed {v} {y}')


def idx(h, lon, lat):
    la = h['lat'][:]; lo = h['lon'][:]
    i = np.clip(np.round((lat - la[0]) / (la[1] - la[0])).astype(int), 0, len(la) - 1)
    j = np.clip(np.round((lon - lo[0]) / (lo[1] - lo[0])).astype(int), 0, len(lo) - 1)
    return i, j


def read_points(fn, lon, lat):
    """Return [ndays, npts] float32 (scaled) for all days of the file."""
    with h5py.File(fn, 'r') as h:
        name = [k for k in h.keys() if k not in ('lat', 'lon', 'day', 'crs')][0]
        d = h[name]; sf = float(d.attrs.get('scale_factor', [1.0])[0]); of = float(d.attrs.get('add_offset', [0.0])[0])
        fill = d.attrs.get('_FillValue', [None])[0]
        i, j = idx(h, lon, lat)
        nt = d.shape[0]; out = np.empty((nt, len(lon)), np.float32)
        for t0 in range(0, nt, 61):
            blk = d[t0:t0 + 61]
            v = blk[:, i, j].astype(np.float32)
            if fill is not None: v[blk[:, i, j] == fill] = np.nan
            out[t0:t0 + 61] = v * sf + of
        days = pd.Timestamp('1900-01-01') + pd.to_timedelta(h['day'][:], unit='D')
    return out, days


def main():
    from pyproj import Transformer
    tr = Transformer.from_crs(5070, 4326, always_xy=True)
    f = pd.read_parquet(f'{P}/fires.parquet'); f = f[f.region != 'nonCONUS']
    f['lon'], f['lat'] = tr.transform(f.cx.values, f.cy.values)
    fod = pd.read_parquet(f'{P}/fpafod/fod.parquet', columns=['FOD_ID', 'FIRE_YEAR', 'DISCOVERY_DOY', 'LATITUDE', 'LONGITUDE'])
    lf = pd.read_parquet(f'{P}/lfmc/lfmc_us.parquet')
    sites = lf.groupby('Site name')[['Latitude (WGS84, EPSG:4326)', 'Longitude (WGS84, EPSG:4326)']].first()
    years = [int(a) for a in sys.argv[1:]] or list(range(2012, 2025))
    for y in years:
        fa = f[f.year == y]; fo = fod[fod.FIRE_YEAR == y] if y <= 2020 else fod.iloc[:0]
        do_l = y <= 2022
        outs = dict(A=f'{OUT}/fires_{y}.parquet', B=f'{OUT}/fod_{y}.parquet', C=f'{OUT}/lfmc_{y}.npz')
        need = [not os.path.exists(outs['A']), len(fo) and not os.path.exists(outs['B']), do_l and not os.path.exists(outs['C'])]
        if not any(need): continue
        A, B, C = {}, {}, {}
        for v in VARS:
            t = time.time(); fn = download(v, y)
            arr, days = read_points(fn, np.r_[fa.lon.values, fo.LONGITUDE.values, sites.iloc[:, 1].values],
                                    np.r_[fa.lat.values, fo.LATITUDE.values, sites.iloc[:, 0].values])
            os.remove(fn)
            if v in ('tmmx', 'tmmn'): arr = arr - 273.15
            na, nb = len(fa), len(fo)
            A[v] = arr[:, :na]
            if nb:
                doy = np.clip(fo.DISCOVERY_DOY.values - 1, 0, len(days) - 1)
                B[v] = arr[doy, na + np.arange(nb)]; B[v + '_m1'] = arr[np.maximum(doy - 1, 0), na + np.arange(nb)]
            if do_l and v in LFMC_VARS: C[v] = arr[:, na + nb:].astype(np.float16)
            print(y, v, f'{time.time() - t:.0f}s', flush=True)
        rows = pd.DataFrame({'Event_ID': np.repeat(fa.Event_ID.values, len(days)), 'date': np.tile(days.values, len(fa))})
        for v in VARS: rows[v] = A[v].T.ravel()
        rows.to_parquet(outs['A'], compression='zstd')
        if len(fo):
            b = pd.DataFrame({'FOD_ID': fo.FOD_ID.values, **B}); b.to_parquet(outs['B'], compression='zstd')
        if do_l:
            np.savez_compressed(outs['C'], days=days.values.astype('datetime64[D]').astype(np.int64),
                                sites=np.array(sites.index, dtype=str), **C)
        print('year done', y, flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
