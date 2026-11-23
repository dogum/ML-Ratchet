"""Turn DART minute series into the network's input (shared by simulations, real dev/exam events and the release).
Per station: a 90-minute window around the predicted arrival ta (ta-20 .. ta+70), 20 channels:
  [obs / scale / 10, mask, 18 basis responses / scale]  where basis = 9 columns x 2 rows of unit sources around the
  epicentre and scale = largest basis response in the window (mm per m of slip). Scalars: ta/180, log10(scale)/2,
  log10(max |obs| / scale + 1e-3)/2. Stations: deployed, sorted by ta, ta <= 290 min, window coverage >= 50 %,
  scale >= 0.05 mm/m; the first K = 3 are the input."""
import numpy as np
from tsim import WIN0, WIN1, NW, NCOL
K = 3; NCH = 2 + 2 * NCOL; NSC = 3
TA_MAX = 290; MIN_COV = 0.5; MIN_SCALE = 0.05

def basis(gf, site, units, ta):
    G = gf.G[site]; B = np.zeros((2 * NCOL, NW), np.float32)
    i0 = ta + WIN0
    for k, u in enumerate(units):
        if u < 0: continue
        seg = G[u, max(i0, 0): i0 + NW].astype(np.float32)
        B[k, max(0, -i0): max(0, -i0) + len(seg)] = seg
    return B

def station_block(gf, site, units, ta, eta, mask):
    """eta (m), mask on the minute grid from the origin (len >= ta+WIN1). Returns (x[NCH, NW], s[NSC], cov, scale)."""
    B = basis(gf, site, units, ta); scale = float(np.abs(B).max())
    i0 = ta + WIN0; o = np.zeros(NW, np.float32); m = np.zeros(NW, np.float32)
    lo, hi = max(i0, 0), min(i0 + NW, len(eta))
    o[lo - i0: hi - i0] = eta[lo:hi]; m[lo - i0: hi - i0] = mask[lo:hi]
    o *= m; cov = float(m.mean())
    if scale <= 0: return None, None, cov, scale
    x = np.concatenate([(o * 1000.0 / scale / 10.0)[None], m[None], B / scale], 0)
    s = np.array([ta / 180.0, np.log10(scale) / 2.0, np.log10(np.abs(o).max() * 1000.0 / scale + 1e-3) / 2.0], np.float32)
    return x, s, cov, scale

def select(cands):
    """cands: list of (ta, cov, scale, payload). Returns first K usable by arrival."""
    ok = [c for c in cands if c[0] is not None and c[0] <= TA_MAX and c[1] >= MIN_COV and c[2] >= MIN_SCALE]
    ok.sort(key=lambda c: c[0])
    return ok[:K]

# ---------------------------------------------------------------- physics anchor (rung r3+)
NEX = 23
def nnls_features(X, M):
    """SIFT-style non-negative least squares on the network's own input (per example): slip (m) on the 18 basis unit
    sources fitted to all observed minutes of the input DARTs. Returns 23 features: Mw_nnls - 8.3, centroid column,
    deep fraction, relative misfit, log10(total slip + 0.1), and the 18 slip fractions."""
    from scipy.optimize import nnls
    X = np.asarray(X, np.float64); out = np.zeros(NEX, np.float32)
    A, b = [], []
    for k in range(X.shape[0]):
        if not M[k]: continue
        m = X[k, 1] > 0
        A.append(X[k, 2:, m] if X[k, 2:, m].shape[-1] == 18 else X[k, 2:][:, m].T); b.append(X[k, 0, m] * 10.0)
    if not A: return out
    A = np.concatenate(A); b = np.concatenate(b)
    s, res = nnls(A, b, maxiter=500)
    tot = s.sum()
    if tot <= 0:
        out[0] = -1.5; out[3] = 1.0; out[4] = -1.0; return out
    mw = (np.log10(4.0e10 * 100e3 * 50e3 * tot) - 9.1) / 1.5
    cols = np.tile(np.arange(9) - 4, 2)
    out[0] = mw - 8.3; out[1] = (cols * s).sum() / tot; out[2] = s[9:].sum() / tot
    out[3] = res / (np.linalg.norm(b) + 1e-9); out[4] = np.log10(tot + 0.1); out[5:] = s / tot
    return out
