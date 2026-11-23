"""AirfoilNet simulator front end: geometry I/O, normalisation and robust polar sweeps with mfoil_fast.

The physics is Fidkowski's mfoil (XFOIL-class: linear-vorticity panels, two-equation integral boundary layer,
e^N transition, lag-entrainment turbulence, strong viscous-inviscid coupling by global Newton). mfoil_fast only
speeds it up. This file adds what a bank needs: reading UIUC coordinates, putting the chord on [0,1], and an
alpha sweep that warm-starts, retries cold and sub-steps, and stops a few failures past stall.
"""
import os, sys, copy, warnings, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(HERE, '..', 'numba_cache'))
import mfoil_fast  # noqa: E402
mf = mfoil_fast.mf
warnings.filterwarnings('ignore')


def _set_coords(M, X):
    """mfoil.set_coords with its X.shape(1) typo fixed (same logic otherwise)."""
    if X.shape[0] > X.shape[1]:
        X = X.transpose()
    A = 0.
    for i in range(X.shape[1] - 1):
        A += (X[0, i + 1] - X[0, i]) * (X[1, i + 1] + X[1, i])
    if A < 0:
        X = np.fliplr(X)
    M.geom.npoint = X.shape[1]
    M.geom.xpoint = X
    M.geom.chord = max(X[0, :]) - min(X[0, :])


mf.set_coords = _set_coords


# ------------------------------------------------------------------ geometry
def read_dat(path):
    """Selig or Lednicer .dat -> (n,2) array, Selig order (TE -> upper -> LE -> lower -> TE)."""
    rows = []
    with open(path, errors='ignore') as f:
        for line in f:
            p = line.replace(',', ' ').split()
            if len(p) < 2:
                rows.append(None); continue
            try:
                rows.append((float(p[0]), float(p[1])))
            except ValueError:
                rows.append(None)
    pts = [r for r in rows if r is not None]
    if not pts:
        raise ValueError('no points')
    a = np.array(pts)
    # Lednicer: first numeric row holds point counts (e.g. 61. 61.)
    if a[0, 0] > 1.5 and a[0, 1] > 1.5:
        nu, nl = int(a[0, 0]), int(a[0, 1])
        up = a[1:1 + nu]; lo = a[1 + nu:1 + nu + nl]
        return np.vstack([up[::-1], lo[1:] if np.allclose(lo[0], up[0]) else lo])
    return a


def normalize(X):
    """Put LE at (0,0) and TE midpoint at (1,0): translate, rotate, scale. Selig order kept."""
    X = np.asarray(X, float)
    # drop exact duplicate consecutive points
    keep = np.r_[True, np.any(np.abs(np.diff(X, axis=0)) > 1e-9, axis=1)]
    X = X[keep]
    te = 0.5 * (X[0] + X[-1])
    d = np.hypot(X[:, 0] - te[0], X[:, 1] - te[1])
    le = X[np.argmax(d)]
    v = te - le
    c = np.hypot(*v); ang = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    Y = (X - le) @ R.T / c
    # Selig order: first point at TE, then upper surface (y>0 near middle of first half)
    i_le = np.argmin(Y[:, 0])
    if np.mean(Y[:i_le + 1, 1]) < np.mean(Y[i_le:, 1]):
        Y = Y[::-1]
    return Y


def surfaces(Y):
    """split a normalised Selig-order airfoil into upper/lower (x ascending)."""
    i = np.argmin(Y[:, 0])
    up = Y[:i + 1][::-1]; lo = Y[i:]
    return up, lo


def sensors(Y, xs):
    """upper and lower y at stations xs (0..1) by monotone interpolation of each surface."""
    up, lo = surfaces(Y)
    def interp(s):
        o = np.argsort(s[:, 0]); s = s[o]
        _, u = np.unique(s[:, 0], return_index=True); s = s[u]
        return np.interp(xs, s[:, 0], s[:, 1])
    return interp(up), interp(lo)


# ------------------------------------------------------------------ solver
def make_model(Y, npanel=120, ncrit=9.0):
    M = mf.mfoil(coords=np.asarray(Y, float).T.copy(), npanel=npanel)
    M.param.verb = 0; M.param.doplot = False; M.param.ncrit = ncrit
    M.param.rtol = 1e-7   # mfoil default 1e-10; Newton is quadratic, forces change < 1e-6 (XFOIL uses ~1e-4)
    return M


def _solve(M, alpha, Re, warm, niglob):
    M.setoper(alpha=alpha, Re=Re)
    M.oper.initbl = not warm
    M.param.niglob = niglob
    try:
        M.solve()
    except Exception:
        return False
    p = M.post
    conv = bool(M.glob.conv) or bool(getattr(M.glob, 'conv_loose', False))
    ok = conv and np.isfinite(p.cl) and np.isfinite(p.cd) and 0 < p.cd < 1.0 and abs(p.cl) < 4
    return ok


def _record(M, want_cp, cp_x):
    p = M.post
    cl, cd, cm = float(p.cl), float(p.cd), float(p.cm)
    loose = (not M.glob.conv) and bool(getattr(M.glob, 'conv_loose', False))
    if loose:
        h = M.glob.hist[-mfoil_fast.LOOSE['window']:]
        cl, cd, cm = (float(v) for v in np.median(h[:, 1:4], axis=0))
    xt = np.asarray(M.vsol.Xt)
    r = dict(cl=cl, cd=cd, cm=cm, cdf=float(p.cdf), loose=loose,
             xtr_u=float(xt[1, 1]) if xt.ndim == 2 and xt[1, 1] > 0 else 1.0,
             xtr_l=float(xt[0, 1]) if xt.ndim == 2 and xt[0, 1] > 0 else 1.0,
             nit=int(len(M.glob.hist)))
    if want_cp:
        X = M.foil.x; cp = np.asarray(p.cp)[:X.shape[1]]
        i = np.argmin(X[0])
        lo = (X[0, :i + 1][::-1], cp[:i + 1][::-1]); up = (X[0, i:], cp[i:])
        r['cp_u'] = np.interp(cp_x, up[0], up[1]).astype(np.float32)
        r['cp_l'] = np.interp(cp_x, lo[0], lo[1]).astype(np.float32)
    return r


def solve_point(Y, alpha, Re, ncrit=9.0, panels=(120, 160, 100), niglob=40):
    """single cold solve with panel-count retries. Returns (record, npanel) or (None, None)."""
    for npan in panels:
        M = make_model(Y, npan, ncrit)
        if _solve(M, alpha, Re, False, niglob):
            return _record(M, False, None), npan
    return None, None


def polar_sweep(Y, Re, alphas_up=None, alphas_dn=None, ncrit=9.0, npanel=120, want_cp=False,
                cp_x=None, max_fail=2, time_limit=120.):
    """Run an alpha sweep. Returns list of dicts (alpha, ok, cl, cd, cm, ...)."""
    if alphas_up is None:
        alphas_up = np.arange(0., 21., 1.)
    if alphas_dn is None:
        alphas_dn = np.arange(-1., -9., -1.)
    t0 = time.time()
    M = make_model(Y, npanel, ncrit)
    out = []
    a0 = float(alphas_up[0])
    ok = _solve(M, a0, Re, False, 30)
    if not ok:  # try a nearby start
        for a_try in (a0 + 2, a0 - 2):
            M = make_model(Y, npanel, ncrit)
            if _solve(M, a_try, Re, False, 30):
                ok = True; a0 = a_try; break
        if not ok:
            return out
    good0 = copy.deepcopy(M)
    out.append(dict(alpha=a0, **_record(M, want_cp, cp_x)))
    for seq in (list(alphas_up[1:]) if a0 == alphas_up[0] else [a for a in alphas_up if a > a0],
                list(alphas_dn)):
        good = copy.deepcopy(good0); a_good = a0; nfail = 0
        for a in seq:
            if time.time() - t0 > time_limit:
                break
            a = float(a)
            M = copy.deepcopy(good)
            ok = _solve(M, a, Re, True, 12)
            if not ok:
                M = copy.deepcopy(good)
                ok = _solve(M, a, Re, False, 30)
            if not ok:  # sub-step
                M = copy.deepcopy(good)
                if _solve(M, 0.5 * (a + a_good), Re, True, 15):
                    ok = _solve(M, a, Re, True, 15)
            if ok:
                out.append(dict(alpha=a, **_record(M, want_cp, cp_x)))
                good = copy.deepcopy(M); a_good = a; nfail = 0
            else:
                nfail += 1
                if nfail >= max_fail:
                    break
    out.sort(key=lambda r: r['alpha'])
    return out


if __name__ == '__main__':
    Y = normalize(read_dat(sys.argv[1]))
    Re = float(sys.argv[2])
    t = time.time()
    res = polar_sweep(Y, Re)
    for r in res:
        print('%6.2f %7.4f %8.5f %8.4f  xtr %.3f %.3f' % (r['alpha'], r['cl'], r['cd'], r['cm'], r['xtr_u'], r['xtr_l']))
    print('n', len(res), 'time', round(time.time() - t, 1))
