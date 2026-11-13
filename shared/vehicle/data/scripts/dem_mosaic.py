"""Build a 2 m elevation mosaic (UTM 17N) over the VED geofence from USGS 3DEP 1 m lidar DEM tiles
(Washtenaw County 2016, bare earth). Downloads one tile at a time, block-averages 2x2, crops, deletes the tile.
Output: proc/dem_annarbor_2m.npz with elev_cm (uint16, 0 = nodata), x0, y0 (top-left corner, metres), res, crs.
Resumable via proc/dem_tmp/<tile>.npy pieces.
"""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, subprocess, numpy as np, rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds

ROOT = _VEH + '/data'
TILES = ['x26y468', 'x26y469', 'x27y468', 'x27y469']
URL = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/MI_31Co_Washtenaw_2016/TIFF/USGS_one_meter_{t}_MI_31Co_Washtenaw_2016.tif'
LON = (-83.812, -83.666); LAT = (42.214, 42.332)
RES = 2.0
TMP = f'{ROOT}/proc/dem_tmp'; os.makedirs(TMP, exist_ok=True)

xs, ys = transform('EPSG:4326', 'EPSG:26917', [LON[0], LON[1], LON[0], LON[1]], [LAT[0], LAT[0], LAT[1], LAT[1]])
X0, X1 = np.floor(min(xs) / 10) * 10, np.ceil(max(xs) / 10) * 10
Y0, Y1 = np.floor(min(ys) / 10) * 10, np.ceil(max(ys) / 10) * 10
NX, NY = int((X1 - X0) / RES), int((Y1 - Y0) / RES)
print('grid', X0, X1, Y0, Y1, NX, NY, flush=True)

for t in TILES:
    piece = f'{TMP}/{t}.npz'
    if os.path.exists(piece): continue
    tif = f'{ROOT}/raw/elev/{t}.tif'
    if not os.path.exists(tif + '.ok'):
        subprocess.run(['curl', '-sS', '-L', '--retry', '5', '-C', '-', '-o', tif, URL.format(t=t)], check=True)
        open(tif + '.ok', 'w').close()
    with rasterio.open(tif) as src:
        print(t, src.crs, src.res, src.bounds, src.nodata, flush=True)
        assert abs(src.res[0] - 1) < 1e-6
        b = src.bounds
        lx0, lx1 = max(b.left, X0), min(b.right, X1)
        ly0, ly1 = max(b.bottom, Y0), min(b.top, Y1)
        lx0 = X0 + np.ceil((lx0 - X0) / RES) * RES; lx1 = X0 + np.floor((lx1 - X0) / RES) * RES
        ly0 = Y0 + np.ceil((ly0 - Y0) / RES) * RES; ly1 = Y0 + np.floor((ly1 - Y0) / RES) * RES
        w = from_bounds(lx0, ly0, lx1, ly1, src.transform).round_offsets().round_lengths()
        a = src.read(1, window=w).astype(np.float32)
        nd = src.nodata
        if nd is not None: a[a == nd] = np.nan
        a[a < -100] = np.nan
        h, wd = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
        a = np.nanmean(a[:h, :wd].reshape(h // 2, 2, wd // 2, 2), axis=(1, 3))
        np.savez(piece, a=a, x0=lx0, y1=ly1)
    os.remove(tif); os.remove(tif + '.ok')

M = np.zeros((NY, NX), np.uint16)
for t in TILES:
    p = np.load(f'{TMP}/{t}.npz'); a = p['a']
    c0 = int(round((float(p['x0']) - X0) / RES)); r0 = int(round((Y1 - float(p['y1'])) / RES))
    sub = M[r0:r0 + a.shape[0], c0:c0 + a.shape[1]]
    a = a[:sub.shape[0], :sub.shape[1]]
    v = np.where(np.isfinite(a), np.clip(np.round(a * 100), 1, 65535), 0).astype(np.uint16)
    sub[v > 0] = v[v > 0]
np.savez_compressed(f'{ROOT}/proc/dem_annarbor_2m.npz', elev_cm=M, x0=X0, y1=Y1, res=RES, crs='EPSG:26917',
                    source='USGS 3DEP 1 m DEM, MI_31Co_Washtenaw_2016 (bare earth), 2x2 block mean')
print('nodata frac', (M == 0).mean())
