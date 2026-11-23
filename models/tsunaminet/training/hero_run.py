"""Forward 8' run of the 2011 Tohoku tsunami from the USGS finite-fault slip on SIFT unit sources (for the page):
records the arrival time (first |eta| > 1 cm... or 2 mm far field) and the 24 h maximum |eta| on the grid, coarsened to 0.5 deg."""
import os, json, numpy as np
from _paths import DATA, BANK
import swe, tsim, realdata
from okada import okada_uz
g = swe.Grid(os.path.join(DATA, 'etopo_pacific_4m.npz'), stride=2)
geo = tsim.Geometry()
ev = next(e for e in json.load(open(os.path.join(DATA, 'events.json'))) if e['id'] == 5413)
units, slips = realdata.ffm_units(ev, geo)
X, Y = np.meshgrid(g.lon, g.lat); eta0 = np.zeros_like(X)
for u, s in zip(units, slips):
    U = geo.U[u]; m = (np.abs(Y - U['lat']) < 3) & (np.abs(X - U['lon']) < 4)
    eta0[m] += s * okada_uz(X[m], Y[m], U)
eta0[g.h <= 0] = 0
eta = eta0.astype(np.float32); M = np.zeros_like(eta); N = np.zeros_like(eta); damp = g.sponge()
arr = np.full(eta.shape, np.inf, np.float32); amax = np.zeros_like(eta); dt = 20.0
for n in range(1, int(86400 / dt) + 1):
    swe.step(eta, M, N, g.h, g.cos_c, g.cos_n, g.dlon, g.dlat, dt, damp)
    swe.track_max(eta, amax)
    if n % 3 == 0:
        hit = (np.abs(eta) > 0.002) & ~np.isfinite(arr); arr[hit] = n * dt / 60.0
c = 4   # 8' -> 32' blocks
ny, nx = (g.ny // c) * c, (g.nx // c) * c
A = amax[:ny, :nx].reshape(ny // c, c, nx // c, c).max((1, 3))
T = arr[:ny, :nx].reshape(ny // c, c, nx // c, c).min((1, 3))
H = g.h[:ny, :nx].reshape(ny // c, c, nx // c, c).mean((1, 3))
np.savez_compressed(os.path.join(BANK, 'hero_tohoku.npz'), amax=A, arr=T, h=H, lat=g.lat[:ny].reshape(-1, c).mean(1), lon=g.lon[:nx].reshape(-1, c).mean(1), eta0max=float(eta0.max()), eta0min=float(eta0.min()))
print('done', A.shape, float(eta0.max()), float(eta0.min()))
