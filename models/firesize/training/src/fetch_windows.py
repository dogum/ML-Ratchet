"""90 m LANDFIRE simulation windows (201 x 201 cells, 18.1 km) at the window centres of data/windows.parquet.
Nearest-neighbour 90 m samples of the 30 m rasters (one 30 m cell per 90 m cell; documented simplification vs B's 3x3
mode). LF2016 Remap FBFM40 / CC / CH / CBH / CBD and LF2020 Elev. Twin windows that touch an exam footprint have those
cells blanked (all layers 0, fuel 99 = non-burnable) right after download; they are never stored.
-> data/win/<win>.npz  (fuel u8, cc u8, ch/cbh u16 (m x10), cbd u16 (kg/m3 x100), elev i16, blank bool). Resumable."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from common import *
import requests, rasterio
import firedata as fd
import importlib
fgrid = importlib.import_module('firedata.grid')

BASE = 'https://lfps.usgs.gov/arcgis/rest/services'
SVC = dict(fuel='Landfire_LF2016/LF2016_FBFM40_CONUS', cc='Landfire_LF2016/LF2016_CC_CONUS',
           ch='Landfire_LF2016/LF2016_CH_CONUS', cbh='Landfire_LF2016/LF2016_CBH_CONUS',
           cbd='Landfire_LF2016/LF2016_CBD_CONUS', elev='Landfire_Topo/LF2020_Elev_CONUS')
OUT = f'{DATA}/win'
os.makedirs(OUT, exist_ok=True)
S = requests.Session()


def fetch(layer, x0, y1, n):
    p = dict(bbox=f'{x0},{y1 - n * RES},{x0 + n * RES},{y1}', bboxSR=5070, imageSR=5070, size=f'{n},{n}',
             format='tiff', interpolation='RSP_NearestNeighbor', f='image')
    err = ''
    for i in range(6):
        try:
            r = S.get(f'{BASE}/{SVC[layer]}/ImageServer/exportImage', params=p, timeout=(30, 300))
            if r.status_code == 200 and r.content[:2] in (b'II', b'MM'):
                with rasterio.MemoryFile(r.content) as mf, mf.open() as s:
                    a = s.read(1)
                    assert a.shape == (n, n)
                    return a
            err = r.text[:200]
        except Exception as e:
            err = repr(e)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f'{layer} {err}')


def one(row):
    fn = f'{OUT}/{row.win}.npz'
    if os.path.exists(fn):
        return 0
    x0, y1 = row.cx - 100 * RES, row.cy + 101 * RES
    L = {k: fetch(k, x0, y1, WIN).astype(np.int32) for k in SVC}
    blank = np.zeros((WIN, WIN), bool)
    if row.blank:
        g = fgrid.grid_from_window(x0, y1 - WIN * RES, x0 + WIN * RES, y1)
        blank = fd.guard.exam_mask(g)
        for k in L:
            L[k][blank] = 0
        L['fuel'][blank] = 99
    nod = L['fuel'] < 0
    fuel = np.where(nod, 0, L['fuel'])
    np.savez_compressed(fn + '.tmp.npz', fuel=fuel.astype(np.uint8), cc=np.clip(L['cc'], 0, 100).astype(np.uint8),
                        ch=np.clip(L['ch'], 0, 65000).astype(np.uint16), cbh=np.clip(L['cbh'], 0, 65000).astype(np.uint16),
                        cbd=np.clip(L['cbd'], 0, 65000).astype(np.uint16),
                        elev=np.clip(L['elev'], -500, 32000).astype(np.int16), blank=blank,
                        x0=x0, y1=y1, res=RES)
    os.replace(fn + '.tmp.npz', fn)
    return 1


def safe(r):
    try:
        return one(r)
    except Exception as e:
        print('FAIL', r.win, repr(e)[:200], flush=True)
        return -1


def main():
    W = pd.read_parquet(f'{DATA}/windows.parquet')
    # twins first (needed for calibration), then train windows in id order
    W = pd.concat([W[W.kind == 'twin'], W[W.kind == 'train']])
    if len(sys.argv) > 2 and sys.argv[1] == 'frozen':
        # `fetch_windows.py frozen N`: only the frozen-set windows of the first N train ignitions (the 22 Nov 2026
        # physics p2 re-score needs the frozen set only: 147 windows for N = 8500)
        I = pd.read_parquet(f'{DATA}/ignitions.parquet')
        use = set(I[I.kind == 'train'].head(int(sys.argv[2])).win)
        W = W[(W.kind == 'train') & (W.frozen == 1) & W.win.isin(use)]
    todo = [r for r in W.itertuples() if not os.path.exists(f'{OUT}/{r.win}.npz')]
    print('todo', len(todo), flush=True)
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(3) as ex:
        for k, r in enumerate(ex.map(lambda r: (safe(r), r.win), todo)):
            n += 1
            if n % 50 == 0:
                print(n, r[1], f'{(time.time() - t0) / n:.2f}s/win', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
