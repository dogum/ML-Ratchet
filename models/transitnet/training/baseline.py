"""The classical comparison: a least-squares transit fit (radius ratio, impact parameter, a/R*, a small time shift; limb
darkening fixed at typical TESS values u1 = 0.40, u2 = 0.25, circular orbit) to the same folded, binned transit TransitNet
reads, from six starting points. It returns no error bars and no vetting."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, time, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, _P.SRC); import transitsim as TS, pipeline as PL
X = np.linspace(-2.5, 2.5, PL.NL)

def fit_one(local, scale, P, dur):
    v, m = local[0], local[1] > 0
    if m.sum() < 20: return dict(k=np.nan, b=np.nan, aR=np.nan, cost=np.nan)
    y = 1 + v[m] * scale; t = X[m] * dur
    def model(th):
        k, b, laR, dt = th; aR = 10 ** laR; inc = np.arccos(np.clip(b / aR, 0, 1)); return TS.planet_flux(t, dt, P, k, aR, inc, 0.40, 0.25, supersample=1)
    depth = max(1 - y[np.abs(X[m]) < 0.3].mean(), 1e-5); best = None
    aR0 = max(1.5, P / (np.pi * dur))
    for k0 in (np.sqrt(depth), 1.2 * np.sqrt(depth)):
        for b0 in (0.2, 0.6, 0.9):
            try:
                r = least_squares(lambda th: (model(th) - y) / scale, [k0, b0, np.log10(aR0), 0.0], bounds=([0.002, 0, 0.1, -0.3 * dur], [0.6, 1.5, 3.0, 0.3 * dur]), max_nfev=200)
                if best is None or r.cost < best.cost: best = r
            except Exception:
                pass
    if best is None: return dict(k=np.nan, b=np.nan, aR=np.nan, cost=np.nan)
    return dict(k=float(best.x[0]), b=float(best.x[1]), aR=float(10 ** best.x[2]), cost=float(best.cost))

def run(idx):
    Z = np.load(_P.WORK + '/data/real.npz'); Mt = json.load(open(_P.WORK + '/data/real_meta.json')); out = []; t = time.time()
    for i in idx:
        m = Mt[i]; out.append(fit_one(Z['local'][i], m['info']['scale'], m['info']['P'], m['dur']))
    return out, (time.time() - t) / max(len(idx), 1)
