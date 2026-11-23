"""Verify the SWE solver (swe.py) against analytic cases and its own reciprocity:
 1. 1-D flat basin: a Gaussian hump splits into two halves moving at sqrt(g h) (d'Alembert).
 2. Green's law: a long wave crossing a gentle slope from 4000 m to 250 m grows as h^(-1/4).
 3. 2-D flat basin: radial spreading of a Gaussian hump vs the exact Hankel-transform solution
    eta(r,t) = int_0^inf k A(k) J0(k r) cos(c k t) dk, A(k) = a s^2 exp(-k^2 s^2 / 2).
 4. Reciprocity on the real 8' Pacific grid: forward run from a SIFT unit source, read at a DART cell, vs one
    receiver-started run projected onto the same source."""
import os, json, numpy as np
from scipy.special import j0
from _paths import DATA
import swe
from okada import okada_uz
OUT = {}

class Synth:
    def __init__(self, h, lat, lon):
        self.h = h.astype(np.float32); self.lat = lat; self.lon = lon
        self.dlon = np.radians(lon[1] - lon[0]); self.dlat = np.radians(lat[1] - lat[0])
        self.cos_c = np.cos(np.radians(lat)); self.cos_n = np.cos(np.radians(lat + 0.5 * np.degrees(self.dlat)))
        self.ny, self.nx = self.h.shape
    run = None

def run(g, eta0, dt, nsteps, rec=None, every=1, damp=None):
    eta = eta0.astype(np.float32).copy(); M = np.zeros_like(eta); N = np.zeros_like(eta)
    damp = np.ones_like(eta) if damp is None else damp
    out = []
    for n in range(1, nsteps + 1):
        swe.step(eta, M, N, g.h, g.cos_c, g.cos_n, g.dlon, g.dlat, dt, damp)
        if rec is not None and n % every == 0: out.append(rec(eta))
    return eta, out

# ---- 1. 1-D flat basin (equatorial strip, 1 row of water between walls)
nx = 4000; dx_deg = 1 / 60.0; lon = np.arange(nx) * dx_deg; lat = np.array([-dx_deg, 0.0, dx_deg])
h = np.zeros((3, nx)); h[1] = 4000.0
g = Synth(h, lat, lon)
x = np.radians(lon) * swe.R; x0 = x[nx // 2]; sig = 20e3
eta0 = np.zeros((3, nx)); eta0[1] = np.exp(-0.5 * ((x - x0) / sig) ** 2)
c = np.sqrt(9.81 * 4000); dt = 0.5 * (x[1] - x[0]) / c; T = 2400.0; ns = int(T / dt)
eta, _ = run(g, eta0, dt, ns)
exact = 0.5 * np.exp(-0.5 * ((x - x0 - c * ns * dt) / sig) ** 2) + 0.5 * np.exp(-0.5 * ((x - x0 + c * ns * dt) / sig) ** 2)
pk = np.argmax(eta[1, nx // 2:]) + nx // 2
OUT['flat_1d'] = dict(grid_km=float((x[1] - x[0]) / 1e3), t_s=ns * dt, peak_num=float(eta[1].max()), peak_exact=0.5,
                      speed_num=float((x[pk] - x0) / (ns * dt)), speed_exact=float(c),
                      rel_rms=float(np.sqrt(np.mean((eta[1] - exact[1 if exact.ndim > 1 else slice(None)]) ** 2)) / 0.5) if exact.ndim == 1 else None)
OUT['flat_1d']['rel_rms'] = float(np.sqrt(np.mean((eta[1] - exact) ** 2)) / 0.5)

# ---- 2. Green's law on a gentle slope
nx = 2600; lon = np.arange(nx) * dx_deg; x = np.radians(lon) * swe.R
hh = np.full(nx, 4000.0); a_, b_ = 700, 1900
hh[a_:b_] = 4000.0 * (250 / 4000.0) ** ((np.arange(a_, b_) - a_) / (b_ - a_))   # exponential profile, gentle
hh[b_:] = 250.0
h = np.zeros((3, nx)); h[1] = hh; g = Synth(h, lat, lon)
sig = 30e3; x0 = x[350]; eta0 = np.zeros((3, nx)); eta0[1] = np.exp(-0.5 * ((x - x0) / sig) ** 2)
# keep only the right-going half: initial flux M = c*eta on east faces
dt = 0.5 * (x[1] - x[0]) / np.sqrt(9.81 * 4000)
eta = eta0.astype(np.float32).copy(); M = np.zeros_like(eta); N = np.zeros_like(eta); damp = np.ones_like(eta)
damp[:, -300:] = (1 - 0.02 * (np.arange(300) / 300.0) ** 2)[None]; damp[:, :60] = 0.98
cface = np.sqrt(9.81 * 4000)
M[1, :-1] = (cface * 0.5 * (eta0[1, :-1] + eta0[1, 1:])).astype(np.float32)
amax = np.zeros(nx)
tt = 0.0
while tt < 45000:
    swe.step(eta, M, N, g.h, g.cos_c, g.cos_n, g.dlon, g.dlat, dt, damp); tt += dt
    amax = np.maximum(amax, eta[1])
ip = [a_ - 200, (a_ + b_) // 2, b_ - 50]
num = [amax[i] / amax[a_ - 200] for i in ip]; gl = [(hh[a_ - 200] / hh[i]) ** 0.25 for i in ip]
OUT['greens_law'] = dict(depths=[float(hh[i]) for i in ip], amp_ratio_num=[float(v) for v in num], amp_ratio_greens=[float(v) for v in gl],
                         rel_err_at_250m=float(num[-1] / gl[-1] - 1))

# ---- 3. 2-D radial spreading in a flat basin (equatorial box, 2' cells)
n = 600; d = 2 / 60.0; lat = (np.arange(n) - n / 2) * d; lon = 180 + (np.arange(n) - n / 2) * d
h = np.full((n, n), 4000.0); g = Synth(h, lat, lon)
X, Y = np.meshgrid(np.radians(lon - 180) * swe.R, np.radians(lat) * swe.R)
sig = 25e3; a = 1.0; eta0 = a * np.exp(-0.5 * (X ** 2 + Y ** 2) / sig ** 2)
c = np.sqrt(9.81 * 4000); dt = 0.4 * (swe.R * np.radians(d)) / c
T = 3000.0; ns = int(T / dt); tt = ns * dt
eta, _ = run(g, eta0, dt, ns)
r = np.linspace(0, 1000e3, 400); k = np.linspace(0, 12 / sig, 6000)
A = a * sig ** 2 * np.exp(-0.5 * (k * sig) ** 2)
ex = np.trapezoid(k[None] * A[None] * j0(k[None] * r[:, None]) * np.cos(c * k[None] * tt), k, axis=1)
jc = n // 2; num = np.interp(r, Y[jc:, jc], eta[jc:, jc])   # along north
num_e = np.interp(r, X[jc, jc:], eta[jc, jc:])                # along east
OUT['flat_2d'] = dict(t_s=tt, grid_km=float(swe.R * np.radians(d) / 1e3), peak_exact=float(ex.max()), peak_num_north=float(num.max()), peak_num_east=float(num_e.max()),
                      r_peak_exact_km=float(r[np.argmax(ex)] / 1e3), r_peak_num_km=float(r[np.argmax(num)] / 1e3),
                      rel_rms=float(np.sqrt(np.mean((num - ex) ** 2)) / ex.max()))
print(json.dumps(OUT, indent=1), flush=True)
json.dump(OUT, open(os.path.join(DATA, 'checks', 'swe_check.json'), 'w'), indent=1)

if os.environ.get('SKIP_RECIP'):
    raise SystemExit
# ---- 4. reciprocity on the real Pacific grid (8')
g = swe.Grid(os.path.join(DATA, 'etopo_pacific_4m.npz'), stride=2)
U = {u['name']: u for u in json.load(open(os.path.join(DATA, 'unit_sources_pacific.json')))}
src = U['acsza10']; rcv = (196.4, 51.1)    # near DART 46402
Xg, Yg = np.meshgrid(g.lon, g.lat)
eta0 = np.where(g.h > 0, okada_uz(Xg, Yg, src), 0.0)
jr, ir = g.index(*rcv)
dt = 20.0; ns = int(6 * 3600 / dt); damp = g.sponge()
_, fwd = run(g, eta0, dt, ns, rec=lambda e: float(e[jr, ir]), every=3, damp=damp)
e1 = np.zeros((g.ny, g.nx)); e1[jr, ir] = 1.0
Aw = (g.area[:, None] * np.ones((1, g.nx))) * eta0 / g.area[jr]
_, rev = run(g, e1, dt, ns, rec=lambda e: float((Aw * e).sum()), every=3, damp=damp)
fwd, rev = np.array(fwd), np.array(rev)
OUT['reciprocity_8min'] = dict(source='acsza10', receiver=rcv, max_abs_fwd_m=float(np.abs(fwd).max()),
                               max_abs_diff_m=float(np.abs(fwd - rev).max()), rel=float(np.abs(fwd - rev).max() / np.abs(fwd).max()))
print(json.dumps(OUT['reciprocity_8min'], indent=1))
json.dump(OUT, open(os.path.join(DATA, 'checks', 'swe_check.json'), 'w'), indent=1)

