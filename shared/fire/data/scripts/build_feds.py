"""FEDS-MTBS (Zenodo 20187963, CC BY 4.0) -> compact tables.

Outputs (proc/feds/):
  fires.parquet       one row per MTBS event (fireslist attributes + FEDS summary), no geometry
  final_perims.parquet  MTBS final perimeter, WKB in EPSG:5070
  perims/<year>.parquet 12-hourly FEDS perimeters (t, n_pixels, n_newpixels, farea, fperim, duration, pixden,
                        sumFRP, flinelen, geometry WKB EPSG:5070, simplified 15 m)
  firelines/<year>.parquet active fire lines (WKB EPSG:5070)
  firepix/<year>.parquet  VIIRS S-NPP 375 m pixels used by FEDS (lat, lon, FRP, conf, DN flag, DT, DS, t, Event_ID)
Resumable per year. Reads gpkg members from the zip one at a time (extracted to tmp, deleted after).
"""
import os, sys, zipfile, tempfile, warnings, io
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import pyogrio, shapely
from pyproj import Transformer
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
warnings.filterwarnings('ignore')
RAW = f'{_DATA}/data/raw/feds_mtbs'
OUT = f'{_DATA}/data/proc/feds'
TMP = f'{_DATA}/data/tmp/feds'
for d in (OUT, f'{OUT}/perims', f'{OUT}/firelines', f'{OUT}/firepix', TMP): os.makedirs(d, exist_ok=True)
TR = Transformer.from_crs(4326, 5070, always_xy=True)
TR83 = Transformer.from_crs(4269, 5070, always_xy=True)


def to5070(geoms, tr=TR, tol=15.0):
    out = []
    for g in geoms:
        if g is None: out.append(None); continue
        g2 = shapely.transform(g, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
        out.append(shapely.to_wkb(shapely.simplify(g2, tol) if tol else g2))
    return out


def numpy_frame(d):
    """Copy into plain numpy columns: pandas-3 str columns from pyogrio keep GDAL's Arrow batch buffers alive (~4 MB
    per fire), which leaked >1 GB per year."""
    o = {}
    for c in d.columns:
        if c == 't': o[c] = pd.to_datetime(d[c]).dt.tz_localize(None).values.astype('datetime64[ns]')
        else: o[c] = pd.to_numeric(pd.Series(np.array(d[c].tolist(), dtype=object)), errors='coerce').to_numpy(np.float64)
    return pd.DataFrame(o)


def fireslist():
    if os.path.exists(f'{OUT}/fires_raw.parquet'): return
    p = f'{RAW}/fireslist_FEDS25MTBS_2012-2024.geojson'
    meta, tbl = pyogrio.raw.read_arrow(p)
    df = tbl.to_pandas()
    geom = shapely.from_wkb(df.pop(meta['geometry_name'] or 'wkb_geometry'))
    df['Ig_Date'] = pd.to_datetime(df['Ig_Date']).dt.tz_localize(None)
    g5 = to5070(geom, TR83, tol=15.0)
    pq.write_table(pa.table({'Event_ID': df.Event_ID.values, 'wkb': g5}), f'{OUT}/final_perims.parquet', compression='zstd')
    gg = shapely.from_wkb(g5)
    df['mtbs_area_km2'] = shapely.area(gg) / 1e6
    b = shapely.bounds(gg); df['xmin'], df['ymin'], df['xmax'], df['ymax'] = b.T
    c = shapely.centroid(gg); df['cx'], df['cy'] = shapely.get_x(c), shapely.get_y(c)
    df.to_parquet(f'{OUT}/fires_raw.parquet')
    print('fireslist', len(df))


def year(yr):
    out = f'{OUT}/perims/{yr}.parquet'
    if os.path.exists(out): return
    z = zipfile.ZipFile(f'{RAW}/FEDS25MTBS.zip')
    names = sorted(i.filename for i in z.infolist() if i.filename.startswith(f'{yr}/') and i.filename.endswith('.gpkg'))
    P, L = [], []
    for k, n in enumerate(names):
        ev = n.split('/')[1][:-5]
        fn = f'{TMP}/cur.gpkg'
        open(fn, 'wb').write(z.read(n))
        try:
            m, t = pyogrio.raw.read_arrow(fn, layer='perimeter')
            d = t.to_pandas(); g = shapely.from_wkb(np.array(d.pop(m['geometry_name']).tolist(), dtype=object))
            d = numpy_frame(d); d['wkb'] = to5070(g); d['Event_ID'] = ev; P.append(d); del t
            m, t = pyogrio.raw.read_arrow(fn, layer='fireline')
            d = t.to_pandas(); g = shapely.from_wkb(np.array(d.pop(m['geometry_name']).tolist(), dtype=object))
            d = numpy_frame(d); d['wkb'] = to5070(g, tol=0); d['Event_ID'] = ev; L.append(d); del t
        except Exception as e:
            print('FAIL', n, repr(e), flush=True)
        os.remove(fn)
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        if k % 200 == 0 or rss > 900: print(yr, k, len(names), ev, f'maxrss {rss:.0f} MB', flush=True)
        if rss > 1100: raise MemoryError(f'RSS {rss} MB at {ev}')
    P = pd.concat(P, ignore_index=True); L = pd.concat(L, ignore_index=True)
    for c in ['n_pixels', 'n_newpixels', 'farea', 'fperim', 'duration', 'pixden', 'sumFRP', 'flinelen']:
        P[c] = pd.to_numeric(P[c], errors='coerce')
    L.to_parquet(f'{OUT}/firelines/{yr}.parquet', compression='zstd')
    # firepix
    fp = pd.read_csv(io.TextIOWrapper(z.open(f'firepix/Firepix_{yr}.csv')), index_col=0)
    fp['t'] = pd.to_datetime(fp['t'])
    fp = fp.astype({'Lat': 'float32', 'Lon': 'float32', 'FRP': 'float32', 'DT': 'float32', 'DS': 'float32'})
    fp.to_parquet(f'{OUT}/firepix/{yr}.parquet', compression='zstd')
    P.to_parquet(out, compression='zstd')
    print('year', yr, len(names), len(P), len(fp), flush=True)


if __name__ == '__main__':
    fireslist()
    yrs = [int(a) for a in sys.argv[1:]] or [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2012, 2013, 2014, 2015, 2016]
    for y in yrs: year(y)
    print('DONE')
