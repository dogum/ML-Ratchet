"""Adapters from agent B's real-data package (fire/firedata) to FireSim inputs.

* landscape_from_firedata(L): fd.landscape() dict -> Landscape (rasterio/affine transform -> GDAL 6-tuple; EPSG:5070).
* grid_convergence_deg(x, y): clockwise angle from grid north (EPSG:5070 Albers) to true north at a point. HRRR wind
  directions are relative to true north; LANDFIRE aspect and the simulation grid are relative to grid north, so winds
  are rotated by this angle (grid = true + gamma; about +15 deg in the far west of CONUS, -12 deg in Maine).
* weather_series(W, x, y): the hourly HRRR series at the HRRR cell nearest to (x, y) (EPSG:5070), in the contract
  (wind_dir rotated to grid north; NaN precipitation set to 0 and flagged; cloud % -> fraction).
* weather_zones(W, ls, coarsen=2, bbox=None): gridded weather as {'series': [...], 'zones': raster}: each landscape cell
  takes the nearest block of coarsen x coarsen HRRR cells.
* random_landscape(shape, res_m, rng): a random window from B's real dev landscapes that passes
  firedata.guard.check_landscape (no exam footprint); used by generate_landscape(kind='real').
"""
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

from .landscape import Landscape

FIRE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # shared/fire: the firedata code (its data: $ML_RATCHET_FIRE)


def _fd():
    if FIRE_ROOT not in sys.path:
        sys.path.insert(0, FIRE_ROOT)
    import firedata as fd
    return fd


def _gdal_transform(t):
    """Accept GDAL (x0, dx, 0, y0, 0, dy) or affine (a, b, c, d, e, f) = (dx, 0, x0, 0, dy, y0)."""
    t = tuple(float(v) for v in t)
    if abs(t[1]) < 1e-9 and abs(t[3]) < 1e-9 and t[0] > 0 and t[4] < 0:      # affine order
        return (t[2], t[0], 0.0, t[5], 0.0, t[4])
    return t


def landscape_from_firedata(L):
    lay = L['layers']
    meta = dict(crs=L.get('crs', 'EPSG:5070'), res_m=float(L['res_m']), transform=_gdal_transform(L['transform']),
                nodata=-9999, fire_id=L.get('fire_id'), landfire_version=L.get('landfire_version'),
                landfire_postfire=L.get('landfire_postfire'), source=L.get('source'))
    layers = {k: np.asarray(lay[k]) for k in ('fuel', 'cc', 'ch', 'cbh', 'cbd', 'elev', 'slope', 'aspect') if k in lay}
    layers['fuel'] = np.where(layers['fuel'] <= 0, 99, layers['fuel']).astype(np.int64)
    if 'aspect' in layers:
        layers['aspect'] = np.where(layers['aspect'] < 0, 0.0, layers['aspect'])
    ls = Landscape(meta=meta, **layers)
    return ls


def lonlat_of(x, y):
    from pyproj import Transformer
    return Transformer.from_crs(5070, 4326, always_xy=True).transform(x, y)


def grid_convergence_deg(x, y):
    """Clockwise angle from grid north to true north (degrees): a true azimuth A is A + gamma in the grid
    (west of -96 deg, true north leans clockwise of grid north in CONUS Albers, so gamma > 0 there)."""
    from pyproj import Transformer
    tr = Transformer.from_crs(5070, 4326, always_xy=True)
    inv = Transformer.from_crs(4326, 5070, always_xy=True)
    lon, lat = tr.transform(x, y)
    x2, y2 = inv.transform(lon, lat + 0.01)
    # direction of true north in grid coordinates, clockwise from grid north
    return math.degrees(math.atan2(x2 - x, y2 - y))


def _hrrr_xy(W):
    from pyproj import Transformer
    g = W['grid']
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    hx, hy = tr.transform(np.asarray(g['lon'], float), np.asarray(g['lat'], float))
    return np.asarray(hx), np.asarray(hy)


def _contract(df, gamma):
    w = pd.DataFrame(index=pd.DatetimeIndex(df.index).tz_localize('UTC') if df.index.tz is None else df.index)
    w['T'] = df['T'].to_numpy(float)
    w['RH'] = np.clip(df['RH'].to_numpy(float), 1.0, 100.0)
    w['wind_speed'] = np.maximum(df['wind_speed'].to_numpy(float), 0.0)
    w['wind_dir'] = (df['wind_dir'].to_numpy(float) + gamma) % 360.0
    p = df['precip'].to_numpy(float) if 'precip' in df else np.zeros(len(df))
    w['precip_missing'] = ~np.isfinite(p)
    w['precip'] = np.where(np.isfinite(p), np.maximum(p, 0.0), 0.0)
    if 'srad' in df:
        w['srad'] = np.maximum(df['srad'].to_numpy(float), 0.0)
    if 'cloud' in df:
        c = df['cloud'].to_numpy(float)
        w['cloud'] = np.clip(c / 100.0 if np.nanmax(c) > 1.5 else c, 0, 1)
    if 'gust' in df:
        w['gust'] = df['gust'].to_numpy(float)
    # fill short gaps
    for c in ('T', 'RH', 'wind_speed', 'srad'):
        if c in w and w[c].isna().any():
            w[c] = w[c].interpolate(limit=6).bfill().ffill()
    if w['wind_dir'].isna().any():
        w['wind_dir'] = w['wind_dir'].ffill().bfill()
    return w


def weather_series(W, x, y):
    """Contract weather at the HRRR cell nearest to (x, y) in EPSG:5070."""
    g = W['grid']
    hx, hy = _hrrr_xy(W)
    d = (hx - x) ** 2 + (hy - y) ** 2
    iy, ix = np.unravel_index(np.argmin(d), d.shape)
    t = pd.DatetimeIndex(np.asarray(g['time']).astype('datetime64[ns]'))
    df = pd.DataFrame({k: np.asarray(g[k])[:, iy, ix] for k in ('T', 'RH', 'wind_speed', 'wind_dir', 'precip', 'srad',
                                                                 'cloud', 'gust') if k in g}, index=t)
    w = _contract(df, grid_convergence_deg(x, y))
    w.attrs.update(hrrr_cell=(int(iy), int(ix)), lonlat=lonlat_of(x, y))
    return w


def weather_zones(W, ls, coarsen=2, bbox=None):
    """Gridded weather: blocks of coarsen x coarsen HRRR cells; each landscape cell takes the nearest block centre."""
    g = W['grid']
    hx, hy = _hrrr_xy(W)
    ny, nx = hx.shape
    t = pd.DatetimeIndex(np.asarray(g['time']).astype('datetime64[ns]'))
    gt = ls.meta['transform']
    yy, xx = np.mgrid[0:ls.shape[0], 0:ls.shape[1]]
    cx = gt[0] + (xx + 0.5) * gt[1]
    cy = gt[3] + (yy + 0.5) * gt[5]
    blocks = []
    for by in range(0, ny, coarsen):
        for bx in range(0, nx, coarsen):
            sl = (slice(by, by + coarsen), slice(bx, bx + coarsen))
            bxm, bym = float(hx[sl].mean()), float(hy[sl].mean())
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                pad = 3000.0 * coarsen
                if not (x0 - pad <= bxm <= x1 + pad and y0 - pad <= bym <= y1 + pad):
                    continue
            blocks.append((sl, bxm, bym))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.array([[b[1], b[2]] for b in blocks]))
    _, zone = tree.query(np.column_stack([cx.ravel(), cy.ravel()]))
    zone = zone.reshape(ls.shape)
    used = np.unique(zone)
    remap = -np.ones(len(blocks), np.int64)
    remap[used] = np.arange(len(used))
    series = []
    gam = grid_convergence_deg(float(cx.mean()), float(cy.mean()))
    for u in used:
        sl = blocks[u][0]
        df = {}
        for k in ('T', 'RH', 'wind_speed', 'precip', 'srad', 'cloud'):
            if k in g:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    df[k] = np.nanmean(np.asarray(g[k])[:, sl[0], sl[1]].reshape(len(t), -1), axis=1)
        # wind direction: vector mean
        ws = np.asarray(g['wind_speed'])[:, sl[0], sl[1]].reshape(len(t), -1)
        wd = np.radians(np.asarray(g['wind_dir'])[:, sl[0], sl[1]].reshape(len(t), -1))
        ue, vn = np.nanmean(ws * np.sin(wd), 1), np.nanmean(ws * np.cos(wd), 1)
        df['wind_dir'] = np.degrees(np.arctan2(ue, vn)) % 360.0
        series.append(_contract(pd.DataFrame(df, index=t), gam))
    return dict(series=series, zones=remap[zone])


def random_landscape(shape=(256, 256), res_m=90.0, rng=None, fire_ids=None, max_tries=50, blank_exam=True):
    """Random window of a real (dev) LANDFIRE landscape that touches no exam footprint (firedata.guard).
    res_m must be the package resolution (90 m). If no clean window is found after max_tries, the best window is
    used with exam cells blanked to non-burnable (blank_exam=True) or an error is raised."""
    fd = _fd()
    rng = np.random.default_rng(rng)
    if fire_ids is None:
        d = os.path.join(fd.P, 'landscapes')
        dev = set(fd.fires('dev').Event_ID)
        fire_ids = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npz') and f[:-4] in dev)
    best = None
    for _ in range(max_tries):
        fid = fire_ids[int(rng.integers(len(fire_ids)))]
        L = fd.landscape(fid)
        if abs(float(L['res_m']) - res_m) > 1e-6:
            raise ValueError(f"real landscapes are at {L['res_m']} m")
        ny, nx = L['layers']['fuel'].shape
        if ny < shape[0] or nx < shape[1]:
            continue
        r0 = int(rng.integers(0, ny - shape[0] + 1))
        c0 = int(rng.integers(0, nx - shape[1] + 1))
        ls = landscape_from_firedata(L).subset(r0, r0 + shape[0], c0, c0 + shape[1])
        t = ls.meta['transform']
        bbox = (t[0], t[3] + shape[0] * t[5], t[0] + shape[1] * t[1], t[3])
        burnable = float(((ls['fuel'] < 91) | (ls['fuel'] > 99)).mean())
        if burnable < 0.5:
            continue
        try:
            fd.guard.check_landscape(bbox)
            ls.meta.update(source_fire=fid, window_rowcol=(r0, c0), bbox=bbox, exam_blanked=0, generator='real')
            return ls
        except fd.guard.ExamLeakError:
            if best is None:
                best = (ls, fid, r0, c0, bbox)
    if best is None or not blank_exam:
        raise RuntimeError('no exam-free real window found')
    ls, fid, r0, c0, bbox = best
    g = dict(crs='EPSG:5070', res_m=res_m, shape=ls.shape, bounds=bbox,
             transform=(ls.meta['transform'][1], 0.0, ls.meta['transform'][0], 0.0, ls.meta['transform'][5],
                        ls.meta['transform'][3]))
    m = np.asarray(fd.guard.exam_mask(g), bool)
    ls['fuel'][m] = 99
    for k in ('cc', 'ch', 'cbh', 'cbd'):
        ls[k][m] = 0.0
    ls.meta.update(source_fire=fid, window_rowcol=(r0, c0), bbox=bbox, exam_blanked=int(m.sum()), generator='real')
    return ls


def random_weather(start=None, hours=72, rng=None, fire_ids=None, spinup_h=72, max_tries=50):
    """Real hourly HRRR weather (contract format) from a random dev-fire package, a random HRRR cell and a random
    start inside the package period; passes firedata.guard.check_weather (not exam fire-day weather).
    The returned index is shifted to `start` when given (keeps the real diurnal phase: shifts are whole days);
    attrs hold the source fire, cell and real times. `hours` + `spinup_h` hours are returned (spin-up first)."""
    fd = _fd()
    rng = np.random.default_rng(rng)
    if fire_ids is None:
        d = os.path.join(fd.P, 'weather', 'hrrr')
        dev = set(fd.fires('dev').Event_ID)
        fire_ids = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npz') and f[:-4] in dev)
    need = hours + spinup_h
    for _ in range(max_tries):
        fid = fire_ids[int(rng.integers(len(fire_ids)))]
        W = fd.weather(fid)
        g = W['grid']
        nt = len(g['time'])
        if nt < need + 1:
            continue
        k0 = int(rng.integers(0, nt - need))
        iy = int(rng.integers(0, g['lat'].shape[0]))
        ix = int(rng.integers(0, g['lat'].shape[1]))
        hx, hy = _hrrr_xy(W)
        x, y = float(hx[iy, ix]), float(hy[iy, ix])
        t = pd.DatetimeIndex(np.asarray(g['time']).astype('datetime64[ns]'))[k0:k0 + need]
        try:
            fd.guard.check_weather((x - 1500, y - 1500, x + 1500, y + 1500), t[0], t[-1])
        except fd.guard.ExamLeakError:
            continue
        df = pd.DataFrame({k: np.asarray(g[k])[k0:k0 + need, iy, ix] for k in
                           ('T', 'RH', 'wind_speed', 'wind_dir', 'precip', 'srad', 'cloud', 'gust') if k in g}, index=t)
        if df[['T', 'RH', 'wind_speed', 'wind_dir']].isna().mean().max() > 0.05:
            continue
        w = _contract(df, grid_convergence_deg(x, y))
        real_t0 = w.index[0]
        if start is not None:
            s = pd.Timestamp(start)
            s = s.tz_localize('UTC') if s.tz is None else s
            shift = (s - pd.Timedelta(hours=spinup_h)).normalize() - real_t0.normalize()
            w.index = w.index + shift
        w.attrs.update(source_fire=fid, hrrr_cell=(iy, ix), real_start=str(real_t0), lonlat=lonlat_of(x, y),
                       generator='real', spinup_h=spinup_h)
        return w
    raise RuntimeError('no exam-free real weather window found')


def fill_precip_from_daily(w, daily, lon):
    """Where HRRR precipitation is missing (precip_missing), spread the gridMET daily total (local calendar day,
    column 'pr', mm) evenly over that day's 24 hours. Returns a new DataFrame; bool column precip_from_daily marks filled hours."""
    w = w.copy()
    if 'precip_missing' not in w or not w['precip_missing'].any() or daily is None or 'pr' not in daily:
        w['precip_from_daily'] = False
        return w
    local_day = (w.index + pd.to_timedelta(lon / 15.0, unit='h')).tz_convert(None).normalize()
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize() if getattr(d.index, 'tz', None) else \
        pd.DatetimeIndex(d.index).normalize()
    pr = d['pr'].reindex(local_day).to_numpy(float) / 24.0
    miss = w['precip_missing'].to_numpy(bool)
    w.loc[miss, 'precip'] = np.nan_to_num(pr[miss], nan=0.0)
    w['precip_from_daily'] = miss          # True: gridMET daily total / 24
    return w
