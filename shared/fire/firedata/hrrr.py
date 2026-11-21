"""Hourly HRRR analysis subsets from the public HRRR-Zarr archive (s3://hrrrzarr, University of Utah MesoWest,
AWS Open Data; 2016-08-25 onward). Reads only the 150x150 chunks that cover a box (~23 KB each); never bulk.

Variables returned (contract units):
  T (degC, 2 m), RH (%, 2 m), wind_speed (m/s, 10 m, earth-relative), wind_dir (deg FROM, true north),
  gust (m/s, surface), precip (mm/h, from analysis PRATE x 3600: an instantaneous rate, not a 1 h total),
  srad (W/m2 downward shortwave, DSWRF), cloud (% total cloud cover).
Gotchas: HRRR U/V are grid-relative on the Lambert grid; they are rotated here (cone = sin 38.5 deg).
Values are float16 in the archive until 2023 (T resolution ~0.25 K near 300 K), float32 in 2024.
"""
import numpy as np, requests, time
from concurrent.futures import ThreadPoolExecutor
from pyproj import Transformer

BASE = 'https://hrrrzarr.s3.amazonaws.com/sfc'
HRRR_PROJ = '+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +R=6371229 +units=m +no_defs'
X0, Y0, DX, NX, NY, CH = -2697520.14252193, -1587306.15255666, 3000.0, 1799, 1059, 150
VARS = {'TMP': '2m_above_ground/TMP', 'RH': '2m_above_ground/RH', 'UGRD': '10m_above_ground/UGRD',
        'VGRD': '10m_above_ground/VGRD', 'GUST': 'surface/GUST', 'PRATE': 'surface/PRATE',
        'DSWRF': 'surface/DSWRF', 'TCDC': 'entire_atmosphere/TCDC'}
FIRST = np.datetime64('2016-08-25T00')
_S = requests.Session()
_S.mount('https://', requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16))
_T5070 = Transformer.from_crs(5070, HRRR_PROJ, always_xy=True)
_TLL = Transformer.from_crs(HRRR_PROJ, 4326, always_xy=True)


def _blosc():
    import numcodecs
    return numcodecs.Blosc()


def box_cells(xmin, ymin, xmax, ymax, src_crs=5070, pad=1):
    """HRRR (row, col) index ranges covering a box given in EPSG:5070 metres (rows south->north)."""
    tr = _T5070 if src_crs == 5070 else Transformer.from_crs(src_crs, HRRR_PROJ, always_xy=True)
    xs = np.linspace(xmin, xmax, 9); ys = np.linspace(ymin, ymax, 9)
    gx, gy = np.meshgrid(xs, ys); hx, hy = tr.transform(gx.ravel(), gy.ravel())
    c0 = int(np.floor((hx.min() - X0) / DX + 0.5)) - pad; c1 = int(np.floor((hx.max() - X0) / DX + 0.5)) + pad
    r0 = int(np.floor((hy.min() - Y0) / DX + 0.5)) - pad; r1 = int(np.floor((hy.max() - Y0) / DX + 0.5)) + pad
    return max(r0, 0), min(r1, NY - 1), max(c0, 0), min(c1, NX - 1)


def _fetch_chunk(url, retries=5):
    for i in range(retries):
        try:
            r = _S.get(url, timeout=(20, 60))
            if r.status_code == 200: return r.content
            if r.status_code in (403, 404): return None
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None


def extract(xmin, ymin, xmax, ymax, t0, t1, threads=8, src_crs=5070):
    """Hourly HRRR analyses for every hour in [t0, t1] (UTC, numpy datetime64 or str) over the box.
    Returns dict: time (UTC datetime64[h]), lat, lon (2-D cell centres), and arrays [nt, ny, nx] float32 in contract
    units. Missing hours are NaN."""
    t0 = np.datetime64(t0, 'h'); t1 = np.datetime64(t1, 'h')
    times = np.arange(max(t0, FIRST), t1 + 1, np.timedelta64(1, 'h'))
    r0, r1, c0, c1 = box_cells(xmin, ymin, xmax, ymax, src_crs)
    ny, nx = r1 - r0 + 1, c1 - c0 + 1
    chunks = [(a, b) for a in range(r0 // CH, r1 // CH + 1) for b in range(c0 // CH, c1 // CH + 1)]
    raw = {k: np.full((len(times), ny, nx), np.nan, np.float32) for k in VARS}
    blosc = _blosc()
    jobs = []
    for it, t in enumerate(times):
        s = str(t).replace('-', '').replace('T', '')  # YYYYMMDDHH
        d, hh = s[:8], s[8:10]
        for k, path in VARS.items():
            for (a, b) in chunks:
                jobs.append((it, k, a, b, f'{BASE}/{d}/{d}_{hh}z_anl.zarr/{path}/{path}/{a}.{b}'))
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(lambda j: _fetch_chunk(j[4]), jobs))
    for (it, k, a, b, _), content in zip(jobs, res):
        if content is None: continue
        raw_b = blosc.decode(content)
        # archive dtype changed from float16 to float32 in 2024 (fill -10000 -> -9999): infer from the chunk size
        dt = '<f4' if len(raw_b) == CH * CH * 4 else '<f2'
        arr = np.frombuffer(raw_b, dt).astype(np.float32)
        hh = min(CH, NY - a * CH); ww = min(CH, NX - b * CH)
        if arr.size != CH * CH: continue
        arr = arr.reshape(CH, CH)[:hh, :ww]
        arr[arr <= -9999] = np.nan
        ra0, ca0 = a * CH, b * CH
        rs, re = max(r0, ra0), min(r1, ra0 + hh - 1); cs, ce = max(c0, ca0), min(c1, ca0 + ww - 1)
        if rs > re or cs > ce: continue
        raw[k][it, rs - r0:re - r0 + 1, cs - c0:ce - c0 + 1] = arr[rs - ra0:re - ra0 + 1, cs - ca0:ce - ca0 + 1]
    cx = X0 + DX * np.arange(c0, c1 + 1); cy = Y0 + DX * np.arange(r0, r1 + 1)
    gx, gy = np.meshgrid(cx, cy); lon, lat = _TLL.transform(gx, gy)
    ang = np.deg2rad(np.sin(np.deg2rad(38.5)) * (lon - (-97.5)))
    u = np.cos(ang) * raw['UGRD'] + np.sin(ang) * raw['VGRD']
    v = -np.sin(ang) * raw['UGRD'] + np.cos(ang) * raw['VGRD']
    out = dict(time=times, lat=lat.astype(np.float32), lon=lon.astype(np.float32), rows=(r0, r1), cols=(c0, c1),
               T=raw['TMP'] - 273.15, RH=np.clip(raw['RH'], 0, 100), wind_speed=np.hypot(u, v),
               wind_dir=(np.degrees(np.arctan2(-u, -v)) % 360.0), gust=raw['GUST'],
               precip=np.clip(raw['PRATE'], 0, None) * 3600.0, srad=raw['DSWRF'], cloud=raw['TCDC'])
    return out


_dt_cache = {}


def _fcst_dtype(ct, path):
    s = str(ct).replace('-', '').replace('T', ''); d, hh = s[:8], s[8:10]
    key = (d, hh)
    if key not in _dt_cache:
        import json
        r = _S.get(f'{BASE}/{d}/{d}_{hh}z_fcst.zarr/{path}/.zarray', timeout=(20, 60))
        _dt_cache[key] = json.loads(r.text)['dtype'] if r.status_code == 200 else '<f2'
    return _dt_cache[key]


def precip(xmin, ymin, xmax, ymax, t0, t1, threads=8, src_crs=5070, cycles=(0, 12), lead=12):
    """Hourly precipitation (mm per hour ending at the time stamp) from HRRR forecasts: the analysis PRATE chunks are
    absent in hrrrzarr, so we use surface/APCP_1hr_acc_fcst from the 00z and 12z cycles, leads f01..f12 (each hour
    is taken from the most recent of those cycles). Returns dict(time, precip [nt, ny, nx])."""
    t0 = np.datetime64(t0, 'h'); t1 = np.datetime64(t1, 'h')
    times = np.arange(max(t0, FIRST), t1 + 1, np.timedelta64(1, 'h'))
    r0, r1, c0, c1 = box_cells(xmin, ymin, xmax, ymax, src_crs)
    ny, nx = r1 - r0 + 1, c1 - c0 + 1
    chunks = [(a, b) for a in range(r0 // CH, r1 // CH + 1) for b in range(c0 // CH, c1 // CH + 1)]
    out = np.full((len(times), ny, nx), np.nan, np.float32)
    step = 24 // len(cycles)
    cyc_t = np.unique([t - np.timedelta64((int(str(t)[11:13]) - 1) % step + 1, 'h') for t in times])
    path = 'surface/APCP_1hr_acc_fcst/surface/APCP_1hr_acc_fcst'
    jobs = []
    for ct in cyc_t:
        s = str(ct).replace('-', '').replace('T', ''); d, hh = s[:8], s[8:10]
        for (a, b) in chunks:
            jobs.append((ct, a, b, f'{BASE}/{d}/{d}_{hh}z_fcst.zarr/{path}/0.{a}.{b}'))
    blosc = _blosc()
    tix = {t: i for i, t in enumerate(times)}
    with ThreadPoolExecutor(threads) as ex:
        for (ct, a, b, _), content in zip(jobs, ex.map(lambda j: _fetch_chunk(j[3]), jobs)):
            if content is None: continue
            raw_b = blosc.decode(content)
            arr = np.frombuffer(raw_b, _fcst_dtype(ct, path)).astype(np.float32)
            nl = arr.size // (CH * CH)
            arr = arr.reshape(nl, CH, CH)
            hh = min(CH, NY - a * CH); ww = min(CH, NX - b * CH)
            ra0, ca0 = a * CH, b * CH
            rs, re = max(r0, ra0), min(r1, ra0 + hh - 1); cs, ce = max(c0, ca0), min(c1, ca0 + ww - 1)
            if rs > re or cs > ce: continue
            for L in range(1, min(lead, nl) + 1):
                t = ct + np.timedelta64(L, 'h')
                if t not in tix or L > step: continue
                v = arr[L - 1, rs - ra0:re - ra0 + 1, cs - ca0:ce - ca0 + 1].copy(); v[v <= -9999] = np.nan
                out[tix[t], rs - r0:re - r0 + 1, cs - c0:ce - c0 + 1] = np.clip(v, 0, None)
    return dict(time=times, precip=out)
