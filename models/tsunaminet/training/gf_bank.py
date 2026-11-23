"""Green's-function bank: for each DART site, one receiver-started linear SWE run on the ETOPO 2022 Pacific grid
(8' and dt = 20 s for the released bank; 4' and dt = 10 s, as NOAA's MOST propagation database, for the check), projected every 60 s onto every Pacific SIFT unit source
(Okada, 1 m slip, rake 90). Output per site: G[unit, minute] in mm per metre of slip (float16), t = 0..1440 min.
Resumable: one file per site. Usage: python gf_bank.py sites_todo.json"""
import os, sys, json, time, numpy as np
from _paths import DATA, BANK
import swe
from okada import okada_uz
GRID = os.path.join(DATA, 'etopo_pacific_4m.npz')
STRIDE = int(os.environ.get('GF_STRIDE', '2'))     # 1 = 4' (dt 10 s), 2 = 8' (dt 20 s, the released bank)
OUT = os.path.join(BANK, 'gf4' if STRIDE == 1 else 'gf%d' % (4 * STRIDE)); os.makedirs(OUT, exist_ok=True)
DT = 10.0 * STRIDE; EVERY = 6 // STRIDE; NT = int(86400 / DT)

def unit_csr(g, U, thresh=1e-3):
    fn = os.path.join(BANK, 'unit_csr_%dm.npz' % (4 * STRIDE))
    if os.path.exists(fn):
        z = np.load(fn); return z['indptr'], z['idx'], z['w']
    indptr = [0]; idx = []; w = []
    for u in U:
        j0 = np.searchsorted(g.lat, u['lat'] - 3); j1 = np.searchsorted(g.lat, u['lat'] + 3)
        lo = (u['lon'] - 3.5 - g.lon[0]); i0 = max(0, int(lo / np.degrees(g.dlon))); i1 = min(g.nx, i0 + int(7 / np.degrees(g.dlon)) + 1)
        X, Y = np.meshgrid(g.lon[i0:i1], g.lat[j0:j1])
        uz = okada_uz(X, Y, u); uz[g.h[j0:j1, i0:i1] <= 0] = 0.0
        m = np.abs(uz) > thresh * max(np.abs(uz).max(), 1e-9)
        jj, ii = np.nonzero(m)
        idx.append(((jj + j0) * g.nx + (ii + i0)).astype(np.int64)); w.append((uz[m] * g.area[jj + j0]).astype(np.float64))
        indptr.append(indptr[-1] + int(m.sum()))
    indptr = np.array(indptr, np.int64); idx = np.concatenate(idx); w = np.concatenate(w)
    np.savez(fn, indptr=indptr, idx=idx, w=w); return indptr, idx, w

def main(todo):
    g = swe.Grid(GRID, stride=STRIDE)
    assert g.stable_dt() > DT, g.stable_dt()
    U = json.load(open(os.path.join(DATA, 'unit_sources_pacific.json')))
    indptr, idx, w = unit_csr(g, U)
    damp = g.sponge()
    for s in todo:
        fn = os.path.join(OUT, s['site'] + '.npz')
        if os.path.exists(fn): continue
        t0 = time.time()
        jr, ir = g.index(s['lon'], s['lat'])
        if g.h[jr, ir] <= 0:   # nearest wet cell
            J, I = np.nonzero(g.h[max(0, jr - 3):jr + 4, max(0, ir - 3):ir + 4] > 0); k = np.argmin((J - 3) ** 2 + (I - 3) ** 2); jr, ir = jr - 3 + J[k], ir - 3 + I[k]
        eta = np.zeros((g.ny, g.nx), np.float32); M = np.zeros_like(eta); N = np.zeros_like(eta)
        eta[jr, ir] = 1.0
        Gm = np.zeros((len(U), NT // EVERY + 1), np.float32); buf = np.zeros(len(U))
        swe.project(eta.ravel(), indptr, idx, w, buf); Gm[:, 0] = buf
        for n in range(1, NT + 1):
            swe.step(eta, M, N, g.h, g.cos_c, g.cos_n, g.dlon, g.dlat, DT, damp)
            if n % EVERY == 0:
                swe.project(eta.ravel(), indptr, idx, w, buf); Gm[:, n // EVERY] = buf
        Gm *= 1000.0 / g.area[jr]   # mm per metre of slip
        np.savez(fn + '.tmp.npz', G=Gm.astype(np.float16), cell=np.array([jr, ir]), depth=float(g.h[jr, ir]),
                 lat=float(g.lat[jr]), lon=float(g.lon[ir]))
        os.replace(fn + '.tmp.npz', fn)
        print(s['site'], 'done %.0fs' % (time.time() - t0), 'max mm %.2f' % np.abs(Gm).max(), flush=True)

if __name__ == '__main__':
    main(json.load(open(sys.argv[1])))
