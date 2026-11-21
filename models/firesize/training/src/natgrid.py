"""National coarse LANDFIRE grids at 250 m (nearest-neighbour samples of the 30 m rasters) for per-ignition landscape
summaries. LF2016 Remap FBFM40, CC (%), CH (m x 10 -> m), LF2020 SlpD (deg). EPSG:5070, origin = LANDFIRE extent
(-2362425, 3177435). Fetched in 4000 x 4000 tiles from lfps.usgs.gov exportImage (public). Resumable.
Stored uint8 (fuel: FBFM40 code, 0 = nodata; cc 255 = nodata; ch m; slope deg) -> data/nat/<layer>.npz"""
import os, time, io
from common import *
import requests, rasterio

BASE = 'https://lfps.usgs.gov/arcgis/rest/services'
SVC = dict(fuel='Landfire_LF2016/LF2016_FBFM40_CONUS', cc='Landfire_LF2016/LF2016_CC_CONUS',
           ch='Landfire_LF2016/LF2016_CH_CONUS', slope='Landfire_Topo/LF2020_SlpD_CONUS')
X0, Y1 = -2362425.0, 3177435.0
R = 250.0
NX, NY = 18505, 11674
T = 4000
OUT = f'{DATA}/nat'
os.makedirs(f'{OUT}/tiles', exist_ok=True)
S = requests.Session()


def fetch(layer, x0, y1, nx, ny):
    p = dict(bbox=f'{x0},{y1 - ny * R},{x0 + nx * R},{y1}', bboxSR=5070, imageSR=5070, size=f'{nx},{ny}',
             format='tiff', interpolation='RSP_NearestNeighbor', f='image')
    for i in range(6):
        try:
            r = S.get(f'{BASE}/{SVC[layer]}/ImageServer/exportImage', params=p, timeout=(30, 600))
            if r.status_code == 200 and r.content[:2] in (b'II', b'MM'):
                with rasterio.MemoryFile(r.content) as mf, mf.open() as s:
                    return s.read(1)
            err = r.text[:200]
        except Exception as e:
            err = repr(e)
        print('retry', layer, err, flush=True)
        time.sleep(10 * (i + 1))
    raise RuntimeError(err)


def to_u8(layer, a):
    a = a.astype(np.int32)
    if layer == 'fuel':
        return np.where((a >= 1) & (a <= 255), a, 0).astype(np.uint8)
    if layer == 'cc':
        return np.where((a >= 0) & (a <= 100), a, 255).astype(np.uint8)
    if layer == 'ch':
        return np.where(a > 0, np.clip(np.round(a / 10.0), 0, 254), 0).astype(np.uint8)
    return np.where((a >= 0) & (a <= 90), a, 0).astype(np.uint8)


def main():
    for layer in ['fuel', 'cc', 'slope', 'ch']:
        if os.path.exists(f'{OUT}/{layer}.npz'):
            continue
        for ty in range(0, NY, T):
            for tx in range(0, NX, T):
                fn = f'{OUT}/tiles/{layer}_{ty}_{tx}.npy'
                if os.path.exists(fn):
                    continue
                nx, ny = min(T, NX - tx), min(T, NY - ty)
                t = time.time()
                a = fetch(layer, X0 + tx * R, Y1 - ty * R, nx, ny)
                assert a.shape == (ny, nx), a.shape
                np.save(fn, to_u8(layer, a))
                print(layer, ty, tx, f'{time.time() - t:.1f}s', flush=True)
        full = np.zeros((NY, NX), np.uint8)
        for ty in range(0, NY, T):
            for tx in range(0, NX, T):
                fn = f'{OUT}/tiles/{layer}_{ty}_{tx}.npy'
                a = np.load(fn)
                full[ty:ty + a.shape[0], tx:tx + a.shape[1]] = a
        np.savez_compressed(f'{OUT}/{layer}.npz', a=full)
        del full
        for ty in range(0, NY, T):
            for tx in range(0, NX, T):
                os.remove(f'{OUT}/tiles/{layer}_{ty}_{tx}.npy')
        print('layer done', layer, os.path.getsize(f'{OUT}/{layer}.npz') / 1e6, 'MB', flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
