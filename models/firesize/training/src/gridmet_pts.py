"""gridMET daily (Abatzoglou 2013; northwestknowledge.net/metdata/data HTTP files, public domain) at the training and
twin ignition points, for the days [discovery - 24 d, discovery + 5 d] (30 days). Same nearest-cell rule and units as
firedata's build_gridmet (temperatures degC). Each var-year file is downloaded to tmp, read, and deleted.
-> data/wx_points.npz: wx[n_ign, 12 vars, 30 days] float32, ign ids, var names. Resumable per year (data/wx_parts).
`gridmet_pts.py <kind>` (27 Nov 2026): only the ignitions of that kind (e.g. trainsb from prior_sb.py), into
data/wx_parts_<kind>/ and data/wx_points_<kind>.npz; data/wx_points.npz is not touched."""
import os, sys, time
from common import *
import h5py, requests

URL = 'https://www.northwestknowledge.net/metdata/data/{v}_{y}.nc'
TMP = f'{ROOT}/tmp/gridmet'
KIND = sys.argv[1] if len(sys.argv) > 1 else None
PART = f'{DATA}/wx_parts' + (f'_{KIND}' if KIND else '')
os.makedirs(TMP, exist_ok=True)
os.makedirs(PART, exist_ok=True)
S = requests.Session()
ND, OFF = 30, 24


def download(v, y):
    fn = f'{TMP}/{v}_{y}.nc'
    for i in range(5):
        try:
            with S.get(URL.format(v=v, y=y), stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                with open(fn, 'wb') as f:
                    for ch in r.iter_content(1 << 20):
                        f.write(ch)
            return fn
        except Exception as e:
            print('retry', v, y, e, flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(f'download failed {v} {y}')


def read_points(fn, lon, lat):
    with h5py.File(fn, 'r') as h:
        name = [k for k in h.keys() if k not in ('lat', 'lon', 'day', 'crs')][0]
        d = h[name]
        sf = float(d.attrs.get('scale_factor', [1.0])[0]); of = float(d.attrs.get('add_offset', [0.0])[0])
        fill = d.attrs.get('_FillValue', [None])[0]
        la = h['lat'][:]; lo = h['lon'][:]
        i = np.clip(np.round((lat - la[0]) / (la[1] - la[0])).astype(int), 0, len(la) - 1)
        j = np.clip(np.round((lon - lo[0]) / (lo[1] - lo[0])).astype(int), 0, len(lo) - 1)
        nt = d.shape[0]
        out = np.empty((nt, len(lon)), np.float32)
        for t0 in range(0, nt, 61):
            blk = d[t0:t0 + 61]
            v = blk[:, i, j].astype(np.float32)
            if fill is not None:
                v[blk[:, i, j] == fill] = np.nan
            out[t0:t0 + 61] = v * sf + of
        days = (pd.Timestamp('1900-01-01') + pd.to_timedelta(h['day'][:], unit='D')).normalize()
    return out, days


def main():
    from pyproj import Transformer
    I = pd.read_parquet(f'{DATA}/ignitions.parquet')
    if KIND:
        I = I[I.kind == KIND].reset_index(drop=True)
    tr = Transformer.from_crs(5070, 4326, always_xy=True)
    lon, lat = tr.transform(I.x.values + 45, I.y.values + 45)
    d0 = pd.to_datetime(I.date).values.astype('datetime64[D]')
    first = d0 - np.timedelta64(OFF, 'D')
    years = sorted(set(pd.DatetimeIndex(first).year) | set(pd.DatetimeIndex(first + np.timedelta64(ND - 1, 'D')).year))
    for y in years:
        pf = f'{PART}/{y}.npz'
        if os.path.exists(pf):
            continue
        ys, ye = np.datetime64(f'{y}-01-01'), np.datetime64(f'{y}-12-31')
        sel = np.nonzero((first <= ye) & (first + np.timedelta64(ND - 1, 'D') >= ys))[0]
        if not len(sel):
            continue
        G = np.full((len(sel), len(WX), ND), np.nan, np.float32)
        for vi, v in enumerate(WX):
            t = time.time()
            fn = download(v, y)
            arr, days = read_points(fn, lon[sel], lat[sel])
            os.remove(fn)
            if v in ('tmmx', 'tmmn'):
                arr = arr - 273.15
            dd = days.values.astype('datetime64[D]')
            for k, s in enumerate(sel):
                rel = (dd - first[s]).astype(int)
                ok = (rel >= 0) & (rel < ND)
                G[k, vi, rel[ok]] = arr[ok, k]
            print(y, v, len(sel), f'{time.time() - t:.0f}s', flush=True)
        np.savez(pf, sel=sel, G=G)
    W = np.full((len(I), len(WX), ND), np.nan, np.float32)
    for y in years:
        z = np.load(f'{PART}/{y}.npz')
        cur = W[z['sel']]
        W[z['sel']] = np.where(np.isnan(cur), z['G'], cur)
    np.savez_compressed(f'{DATA}/wx_points' + (f'_{KIND}' if KIND else '') + '.npz', wx=W, ign=I.ign.values, vars=np.array(WX), off=OFF)
    print('DONE missing frac', float(np.isnan(W).mean()), flush=True)


if __name__ == '__main__':
    main()
