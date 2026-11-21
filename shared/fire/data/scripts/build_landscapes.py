"""Contract landscape stacks (90 m, EPSG:5070) from LANDFIRE ImageServers for a list of fires.
Usage: python3 -I build_landscapes.py dev|replay|<Event_ID> ...     -> proc/landscapes/<Event_ID>.npz
Layers: fuel (FBFM40, 3x3 mode), cc %, ch m, cbh m, cbd kg/m3 (3x3 means), elev m (mean), slope deg (mean of LF2020 SlpD),
aspect deg from N (circular mean of LF2020 Asp; -1 flat). Fuel/canopy version = lf.lf_version(fire year) (pre-fire
except 2012-2016 fires). Resumable."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lf
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
P = f'{_DATA}/data/proc'
OUT = f'{P}/landscapes'; os.makedirs(OUT, exist_ok=True)
TILE = 2700  # 30 m pixels per request side (81 km)
SCALE = {'CC': 1.0, 'CH': 0.1, 'CBH': 0.1, 'CBD': 0.01}


def mode3(a):
    ny, nx = a.shape[0] // 3, a.shape[1] // 3
    b = a[:ny * 3, :nx * 3].reshape(ny, 3, nx, 3).transpose(0, 2, 1, 3).reshape(ny, nx, 9)
    cnt = np.zeros(b.shape, np.int8)
    for k in range(9): cnt += (b == b[..., k:k + 1])
    return np.take_along_axis(b, cnt.argmax(-1)[..., None], -1)[..., 0]


def mean3(a, valid):
    ny, nx = a.shape[0] // 3, a.shape[1] // 3
    r = lambda x: x[:ny * 3, :nx * 3].reshape(ny, 3, nx, 3).transpose(0, 2, 1, 3).reshape(ny, nx, 9)
    v = r(valid); s = (r(a) * v).sum(-1); n = v.sum(-1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def build(fire):
    fn = f'{OUT}/{fire.Event_ID}.npz'
    if os.path.exists(fn): return 'skip'
    ver, post = lf.lf_version(int(fire.year))
    x0, y0, x1, y1 = fire.wx0, fire.wy0, fire.wx1, fire.wy1
    NX, NY = int(round((x1 - x0) / 30)), int(round((y1 - y0) / 30))
    out = {k: np.full((NY // 3, NX // 3), np.nan, np.float32) for k in ['fuel', 'cc', 'ch', 'cbh', 'cbd', 'elev', 'slope', 'aspect']}
    out['fuel'] = np.full((NY // 3, NX // 3), -9999, np.int16)
    for ty in range(0, NY, TILE):
        for tx in range(0, NX, TILE):
            ny, nx = min(TILE, NY - ty), min(TILE, NX - tx)
            gx0, gy1 = x0 + tx * 30, y1 - ty * 30
            sl = (slice(ty // 3, (ty + ny) // 3), slice(tx // 3, (tx + nx) // 3))
            a = lf.export('FBFM40', ver, gx0, gy1, nx, ny)
            m = mode3(a); out['fuel'][sl] = np.where(m > 0, m, -9999)
            for lay in ['CC', 'CH', 'CBH', 'CBD']:
                a = lf.export(lay, ver, gx0, gy1, nx, ny).astype(np.float32)
                out[lay.lower()][sl] = mean3(a * SCALE[lay], a >= 0)
            a = lf.export('Elev', None, gx0, gy1, nx, ny).astype(np.float32); out['elev'][sl] = mean3(a, a > -1000)
            a = lf.export('SlpD', None, gx0, gy1, nx, ny).astype(np.float32); out['slope'][sl] = mean3(a, a >= 0)
            a = lf.export('Asp', None, gx0, gy1, nx, ny).astype(np.float32)
            ok = a >= 0
            s = mean3(np.sin(np.deg2rad(a)), ok); c = mean3(np.cos(np.deg2rad(a)), ok)
            asp = np.degrees(np.arctan2(s, c)) % 360
            out['aspect'][sl] = np.where(np.hypot(s, c) < 1e-6, -1, asp)
    out['aspect'][out['slope'] < 0.5] = -1.0
    meta = dict(fire_id=fire.Event_ID, crs='EPSG:5070', res_m=90.0, transform=[90.0, 0.0, x0, 0.0, -90.0, y1],
                shape=list(out['fuel'].shape), nodata={'fuel': -9999, 'float': 'NaN'},
                landfire_version=f'LF{ver}', landfire_postfire=bool(post), topo='LF2020 (Elev, SlpD, Asp)',
                encoding={'fuel': 'FBFM40 (Scott & Burgan 40; 91-99 non-burnable), 3x3 mode of 30 m',
                          'cc': '% (mean)', 'ch': 'm (LF value/10, mean)', 'cbh': 'm (LF value/10, mean)',
                          'cbd': 'kg/m3 (LF value/100, mean)', 'elev': 'm', 'slope': 'degrees', 'aspect': 'deg from N, -1 flat'},
                source='LANDFIRE via lfps.usgs.gov ImageServer exportImage (nearest, native 30 m), aggregated 3x3')
    np.savez_compressed(fn, meta=json.dumps(meta), fuel=out['fuel'],
                        **{k: out[k].astype(np.float16) for k in ['cc', 'ch', 'cbh', 'cbd', 'slope', 'aspect']},
                        elev=out['elev'].astype(np.float32))
    return 'ok'


if __name__ == '__main__':
    f = pd.read_parquet(f'{P}/fires.parquet')
    ids = []
    for a in sys.argv[1:]:
        if a == 'dev': ids += list(f.Event_ID[f.split == 'dev'])
        elif a == 'replay': ids += json.load(open(f'{P}/baselines/replay_subset.json'))['fires']
        else: ids.append(a)
    for e in ids:
        r = f[f.Event_ID == e].iloc[0]
        t = time.time()
        try: s = build(r)
        except Exception as ex: s = f'FAIL {ex!r}'
        print(e, s, f'{time.time() - t:.0f}s', flush=True)
    print('DONE')
