"""firedata: real wildfire data for the Wildfire project (Phase 1, agent B). See shared/fire/data/DATA.md.

    import sys; sys.path.insert(0, 'shared/fire')     # the code; the data live under $ML_RATCHET_FIRE (default shared/fire)
    import firedata as fd
    s   = fd.splits()                         # frozen splits + rules (data/proc/splits.json)
    f   = fd.fires(split='dev')               # fire table (MTBS attributes, region, fuel group, windows, FEDS stats)
    pr  = fd.perimeters('CA3966012280920200817')      # FEDS 12-hourly perimeters (shapely, EPSG:5070)
    g   = fd.grid('CA3966012280920200817')    # 90 m EPSG:5070 grid of the fire's window (contract meta)
    prog= fd.progression('CA3966012280920200817')     # PM-to-PM daily burned masks on that grid
    L   = fd.landscape('CA3966012280920200817')       # contract landscape stack (dev + replay subset)
    W   = fd.weather('CA3966012280920200817')         # hourly HRRR (series at centroid + coarse grid) + daily gridMET
    ig  = fd.ignition('CA3966012280920200817')
    fd.guard.check_landscape(bbox); fd.guard.check_weather(bbox, t0, t1)   # raise ExamLeakError
"""
import os, json, functools
import numpy as np, pandas as pd, shapely, pyarrow.parquet as pq
from . import grid as _grid, guard, metrics  # noqa: F401
from .guard import ExamLeakError  # noqa: F401

ROOT = os.path.join(os.environ.get('ML_RATCHET_FIRE') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data').replace(os.sep, '/')
P = f'{ROOT}/proc'
EXAM_FLAG_MSG = ('{} is an EXAM fire. Exam data are for final scoring only: pass for_exam_scoring=True in scoring code. '
                 'Never use it to train, calibrate, tune or choose anything.')


@functools.lru_cache(None)
def splits():
    return json.load(open(f'{P}/splits.json'))


@functools.lru_cache(None)
def _fires():
    return pd.read_parquet(f'{P}/fires.parquet')


def fires(split=None, period=None):
    """Fire table. split in {'dev','exam','aux_rx','excluded_nonCONUS'}; period in {'early','main','recent'}."""
    f = _fires()
    if split: f = f[f.split == split]
    if period: f = f[f.exam_period == period]
    return f.copy()


def _row(fire_id):
    f = _fires(); r = f[f.Event_ID == fire_id]
    if not len(r): raise KeyError(fire_id)
    return r.iloc[0]


def _check(fire_id, for_exam_scoring):
    if _row(fire_id).split == 'exam' and not for_exam_scoring:
        raise ExamLeakError(EXAM_FLAG_MSG.format(fire_id))


def _t_utc(t, lon):
    """FEDS 't' is a nominal local half-day label: 00:00 = S-NPP night overpass (~01:30 local solar time),
    12:00 = day overpass (~13:30 local solar). Approximate UTC = label + 1.5 h - lon/15 h."""
    return t + pd.to_timedelta(1.5 - np.asarray(lon) / 15.0, unit='h')


@functools.lru_cache(64)
def _perims_year(year):
    return pd.read_parquet(f'{P}/feds/perims/{year}.parquet')


def perimeters(fire_id, for_exam_scoring=False):
    """FEDS-MTBS 12-hourly cumulative perimeters. Columns: t (nominal local label), t_utc (approx.), half ('AM' night
    overpass / 'PM' day overpass), farea_km2, n_pixels, n_newpixels, flinelen_km, sumFRP, geometry (EPSG:5070)."""
    _check(fire_id, for_exam_scoring)
    r = _row(fire_id)
    d = _perims_year(int(r.year)); d = d[d.Event_ID == fire_id].sort_values('t').copy()
    lon = _lonlat(r)[0]
    d['t_utc'] = _t_utc(d.t, lon); d['half'] = np.where(d.t.dt.hour == 12, 'PM', 'AM')
    d['geometry'] = shapely.from_wkb(d.wkb.values)
    d['farea'] = d['farea'] / 1e6  # FEDS farea is m2; fperim and flinelen are km
    return d.rename(columns={'farea': 'farea_km2', 'fperim': 'fperim_km', 'flinelen': 'flinelen_km'}).drop(columns=['wkb']).reset_index(drop=True)


def _lonlat(r):
    from pyproj import Transformer
    return Transformer.from_crs(5070, 4326, always_xy=True).transform(r.cx, r.cy)


@functools.lru_cache(None)
def _final():
    t = pq.read_table(f'{P}/feds/final_perims.parquet')
    return dict(zip(t.column('Event_ID').to_pylist(), t.column('wkb').to_pylist()))


def final_perimeter(fire_id, for_exam_scoring=False):
    _check(fire_id, for_exam_scoring)
    return shapely.from_wkb(_final()[fire_id])


def firepix(fire_id, for_exam_scoring=False):
    """VIIRS S-NPP 375 m detections FEDS used for this fire (lat, lon, FRP MW, Confidence, DNFlag, DT/DS km, t)."""
    _check(fire_id, for_exam_scoring)
    r = _row(fire_id)
    d = pd.read_parquet(f'{P}/feds/firepix/{int(r.year)}.parquet', filters=[('Event_ID', '=', fire_id)])
    return d.reset_index(drop=True)


def grid(fire_id, res=90.0):
    """Contract grid meta for the fire window (EPSG:5070; 90 m default; aligned with LANDFIRE 30 m cells)."""
    r = _row(fire_id)
    return _grid.grid_from_window(r.wx0, r.wy0, r.wx1, r.wy1, res)


def progression(fire_id, res=90.0, half='PM', for_exam_scoring=False):
    """Daily (PM-to-PM) cumulative burned masks on the fire grid.
    Returns dict(grid, t (labels), t_utc, burned [n, ny, nx] bool, area_km2 (FEDS), arrival_idx (int16, -1 never))."""
    pr = perimeters(fire_id, for_exam_scoring)
    if half: pr = pr[pr.half == half]
    g = grid(fire_id, res)
    B = np.stack([_grid.burn([geom], g) for geom in pr.geometry]) if len(pr) else np.zeros((0,) + g['shape'], bool)
    B = np.maximum.accumulate(B, axis=0) if len(B) else B
    arr = np.full(g['shape'], -1, np.int16)
    for i in range(len(B) - 1, -1, -1): arr[B[i]] = i
    return dict(grid=g, t=pr.t.values, t_utc=pr.t_utc.values, burned=B, area_km2=pr.farea_km2.values, arrival_idx=arr)


def ignition(fire_id, for_exam_scoring=False):
    """MTBS ignition date (local calendar date, no time), MTBS burn-bound point, and the first FEDS perimeter and its
    approximate UTC time (the earliest observed state: use it as the initial perimeter)."""
    _check(fire_id, for_exam_scoring)
    r = _row(fire_id)
    pr = perimeters(fire_id, for_exam_scoring)
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(float(r.BurnBndLon), float(r.BurnBndLat))
    return dict(date_local=r.Ig_Date.date(), point_xy_5070=(x, y), point_note='MTBS BurnBndLat/Lon (burn-bound centre, '
                'not the true origin)', first_perimeter=pr.geometry.iloc[0] if len(pr) else None,
                first_t_utc=pr.t_utc.iloc[0] if len(pr) else None, crs='EPSG:5070')


def landscape(fire_id, for_exam_scoring=False):
    """Contract landscape stack (90 m, EPSG:5070): fuel, cc, ch, cbh, cbd, elev, slope, aspect + meta."""
    _check(fire_id, for_exam_scoring)
    fn = f'{P}/landscapes/{fire_id}.npz'
    if not os.path.exists(fn): raise FileNotFoundError(f'no landscape package for {fire_id} (dev + replay subset only)')
    z = np.load(fn, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    L = {k: z[k] for k in z.files if k != 'meta'}
    for k in ('cc', 'ch', 'cbh', 'cbd', 'elev', 'slope', 'aspect'):
        L[k] = L[k].astype(np.float32)
    L['aspect'] = np.where(L['aspect'] >= 0, L['aspect'] % 360.0, -1.0).astype(np.float32)  # float16 can round 359.9 up
    return dict(layers=L, **meta)


def weather(fire_id, for_exam_scoring=False):
    """Hourly HRRR analyses (UTC) for the fire window: 'series' DataFrame at the window centre (contract columns) and
    'grid' arrays [nt, ny, nx] on HRRR 3 km cells; plus 'daily' gridMET (local day) at the fire centroid."""
    _check(fire_id, for_exam_scoring)
    out = {}
    fn = f'{P}/weather/hrrr/{fire_id}.npz'
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        t = z['time'].astype('datetime64[h]') if z['time'].dtype.kind == 'M' else z['time'].astype(np.int64).astype('datetime64[h]')
        keys = meta['vars']
        grids = {k: z[k].astype(np.float32) for k in keys}
        iy, ix = meta['centre_index']
        out['series'] = pd.DataFrame({k: grids[k][:, iy, ix] for k in keys}, index=pd.DatetimeIndex(t, name='time_utc'))
        out['grid'] = dict(time=t, lat=z['lat'], lon=z['lon'], **grids)
        out['hrrr_meta'] = meta
    fn = f'{P}/weather/gridmet/fires_{int(_row(fire_id).year)}.parquet'
    if os.path.exists(fn):
        d = pd.read_parquet(fn, filters=[('Event_ID', '=', fire_id)])
        out['daily'] = d.drop(columns='Event_ID').set_index('date')
    return out


def gofer(name_year=None):
    """GOFER-Combined hourly perimeters (28 CA fires). name_year like 'Dixie|2021'. Exam ones need for_exam_scoring
    in scoring code; this loader returns all and leaves the check to fd.guard.is_exam / splits()['gofer']."""
    d = pd.read_parquet(f'{P}/gofer/fireProg.parquet')
    if name_year:
        n, y = name_year.split('|'); d = d[(d.fname == n) & (d.fyear == int(y))]
    d = d.copy(); d['geometry'] = shapely.from_wkb(d.wkb.values)
    return d.drop(columns='wkb')


def wsts(fire_id, for_exam_scoring=False):
    """WildfireSpreadTS fire (e.g. 'fire_21890058'): daily VIIRS AF (hhmm UTC of detection, -1 none), static layers,
    daily gridMET/GFS weather sampled every 8 px. Native grid: UTM 375 m (meta)."""
    s = splits()['wsts'].get(fire_id)
    if s is None: raise KeyError(fire_id)
    if s['split'] != 'dev' and not for_exam_scoring: raise ExamLeakError(EXAM_FLAG_MSG.format(fire_id))
    meta = json.load(open(f'{P}/wsts/meta/{fire_id}.json'))
    fn = f'{P}/wsts/fires/{fire_id}.npz'
    data = dict(np.load(fn)) if os.path.exists(fn) else None
    return dict(meta=meta, data=data)
