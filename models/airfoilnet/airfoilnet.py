"""AirfoilNet: an airfoil's lift, drag and moment polars from its shape and Reynolds number, with error bars.

numpy-only inference (no torch, no mlratchet). Loads airfoilnet_weights.npz next to this file.

    from airfoilnet import AirfoilNet, read_dat
    net = AirfoilNet()
    pol = net.polar(read_dat('examples/sd7037.dat'), Re=2e5)        # alpha grid -6..20 deg
    pol['cl'], pol['cl_sd'], pol['cd'], pol['cm'], pol['clmax'], pol['alpha_stall']

The network is a DeepONet: a branch net reads the shape (thickness and camber at 24 stations) and log Re,
a trunk net reads the angle of attack, and their dot product gives Cl, ln Cd, Cm and a log-sigma for each.
Lift and moment are predicted as corrections to thin-airfoil theory computed from the camber line.
Trained only on mfoil (XFOIL-class) simulations; graded on UIUC wind-tunnel data.
"""
import os, json
import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
N_W = 8
XS = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 26)))[1:-1]        # 24 sensor stations


# ----------------------------------------------------------------- geometry
def read_dat(path):
    """Selig or Lednicer .dat -> (n,2) array in Selig order."""
    pts = []
    for line in open(path, errors='ignore'):
        p = line.replace(',', ' ').split()
        try:
            pts.append((float(p[0]), float(p[1])))
        except (ValueError, IndexError):
            pass
    a = np.array(pts)
    if a[0, 0] > 1.5 and a[0, 1] > 1.5:                          # Lednicer counts
        nu, nl = int(a[0, 0]), int(a[0, 1])
        up = a[1:1 + nu]; lo = a[1 + nu:1 + nu + nl]
        return np.vstack([up[::-1], lo[1:] if np.allclose(lo[0], up[0]) else lo])
    return a


def normalize(X):
    """LE at (0,0), TE midpoint at (1,0); Selig order (TE, upper, LE, lower, TE)."""
    X = np.asarray(X, float)
    keep = np.r_[True, np.any(np.abs(np.diff(X, axis=0)) > 1e-9, axis=1)]
    X = X[keep]
    te = 0.5 * (X[0] + X[-1])
    le = X[np.argmax(np.hypot(X[:, 0] - te[0], X[:, 1] - te[1]))]
    v = te - le; c = np.hypot(*v); ang = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    Y = (X - le) @ R.T / c
    i = np.argmin(Y[:, 0])
    if np.mean(Y[:i + 1, 1]) < np.mean(Y[i:, 1]):
        Y = Y[::-1]
    return Y


def _basis(x, n=N_W - 1):
    x = np.asarray(x, float)
    C = np.sqrt(np.clip(x, 0, None)) * (1 - x)
    return C[:, None] * np.stack([comb(n, i) * x ** i * (1 - x) ** (n - i) for i in range(n + 1)], axis=1)


def _le(x, n=N_W - 1):
    return np.sqrt(np.clip(x, 0, None)) * (1 - x) ** (n + 0.5)


def cst_eval(p, x):
    B = _basis(x); L = _le(x)
    yu = B @ p[:N_W] + p[2 * N_W] * L + x * p[2 * N_W + 1] / 2
    yl = B @ p[N_W:2 * N_W] + p[2 * N_W] * L - x * p[2 * N_W + 1] / 2
    return yu, yl


def cst_fit(Y, reg=1e-6):
    """Kulfan CST (8 weights a side + LE term + TE thickness) least-squares fit; returns (p, rms)."""
    i = np.argmin(Y[:, 0]); up = Y[:i + 1]; lo = Y[i:]
    xu = np.clip(up[:, 0], 0, 1); xl = np.clip(lo[:, 0], 0, 1)
    nu, nl = len(xu), len(xl)
    A = np.zeros((nu + nl, 2 * N_W + 2))
    A[:nu, :N_W] = _basis(xu); A[nu:, N_W:2 * N_W] = _basis(xl)
    A[:nu, 2 * N_W] = _le(xu); A[nu:, 2 * N_W] = _le(xl)
    A[:nu, 2 * N_W + 1] = xu / 2; A[nu:, 2 * N_W + 1] = -xl / 2
    b = np.r_[up[:, 1], lo[:, 1]]
    p = np.linalg.solve(A.T @ A + reg * np.eye(A.shape[1]), A.T @ b)
    if p[-1] < 0:                         # crossed trailing edge: refit with a closed (sharp) TE
        A2 = A[:, :-1]
        p = np.r_[np.linalg.solve(A2.T @ A2 + reg * np.eye(A2.shape[1]), A2.T @ b), 0.0]
    return p, float(np.sqrt(np.mean((A @ p - b) ** 2)))


def thin_airfoil(p):
    """thin-airfoil theory from the CST camber line: zero-lift angle (deg) and Cm about c/4."""
    th = np.linspace(1e-4, np.pi - 1e-4, 801); x = 0.5 * (1 - np.cos(th))
    yu, yl = cst_eval(p, x); c = 0.5 * (yu + yl)
    dc = np.gradient(c, x)
    aL0 = -np.trapezoid(dc * (np.cos(th) - 1), th) / np.pi
    A1 = 2 / np.pi * np.trapezoid(dc * np.cos(th), th); A2 = 2 / np.pi * np.trapezoid(dc * np.cos(2 * th), th)
    return float(np.degrees(aL0)), float(np.pi / 4 * (A2 - A1))


def features(p, Re):
    """branch input: thickness/0.12 and camber/0.04 at 24 stations, (log10 Re - 5)/0.6, tmax/0.12"""
    yu, yl = cst_eval(p, XS)
    t = yu - yl; c = 0.5 * (yu + yl)
    return np.r_[t / 0.12, c / 0.04, (np.log10(Re) - 5.0) / 0.6, t.max() / 0.12]


def tsfc(t):
    return 1 + 0.77 * t


OUT_SCALE = np.array([0.5, 1.0, 0.05])       # network output units: Cl/0.5, ln Cd, Cm/0.05


def trunk_features(A):
    A = np.asarray(A, float)
    return np.stack([A / 10.0, np.sin(np.radians(2 * A)), np.cos(np.radians(A))], 1)


def anchors(p, A):
    """thin-airfoil Cl (thickness-corrected slope) and Cm about c/4, and ln(0.01) for drag;
    the network predicts corrections to these."""
    A = np.asarray(A, float)
    aL0, cm0 = thin_airfoil(p)
    yu, yl = cst_eval(p, XS); tmax = float((yu - yl).max())
    cl = 2 * np.pi * tsfc(tmax) * np.sin(np.radians(A - aL0))
    return np.stack([cl, np.full_like(A, np.log(0.01)), np.full_like(A, cm0)], 1)


def anchors_none(A):
    A = np.asarray(A, float)
    return np.stack([np.zeros_like(A), np.full_like(A, np.log(0.01)), np.zeros_like(A)], 1)


# ----------------------------------------------------------------- network
def _act(x):
    return x / (1 + np.exp(-x))          # SiLU / swish


def _mlp(params, prefix, x):
    i = 0
    while f'{prefix}{i}.w' in params:
        x = x @ params[f'{prefix}{i}.w'].T + params[f'{prefix}{i}.b']
        if f'{prefix}{i + 1}.w' in params:
            x = _act(x)
        i += 1
    return x


class AirfoilNet:
    OUT = ('cl', 'lncd', 'cm')

    def __init__(self, weights=None):
        z = np.load(weights or os.path.join(HERE, 'airfoilnet_weights.npz'), allow_pickle=False)
        self.meta = json.loads(str(z['meta']))
        self.members = []
        for m in range(self.meta['n_members']):
            self.members.append({k.split('/', 1)[1]: z[k].astype(np.float64) for k in z.files
                                 if k.startswith(f'm{m}/')})
        self.widen = np.array(self.meta.get('widen', [1.0, 1.0, 1.0]))
        self.P = self.meta['p_latent']

    # raw network: F (n,50) branch features, A (n,) alpha deg, anc (n,3) -> mean (n,3), sigma (n,3)
    def _forward(self, F, A, anc):
        mus, vars_ = [], []
        tfeat = trunk_features(A)
        for prm in self.members:
            if self.meta['arch'] == 'deeponet':
                b = _mlp(prm, 'branch.', F).reshape(len(F), self.P, 6)
                t = _mlp(prm, 'trunk.', tfeat)
                o = np.einsum('npk,np->nk', b, t) + prm['bias']
            else:
                o = _mlp(prm, 'net.', np.concatenate([F, tfeat], 1))
            mu = o[:, :3] * OUT_SCALE + anc
            ls = np.clip(o[:, 3:], -7, 3) + np.log(OUT_SCALE)
            mus.append(mu); vars_.append(np.exp(2 * ls))
        mus = np.array(mus); vars_ = np.array(vars_)
        return mus.mean(0), np.sqrt(vars_.mean(0) + mus.var(0))     # ensemble mixture

    def anchors(self, p, A):
        return anchors(p, A) if self.meta.get('anchor', True) else anchors_none(A)

    def predict(self, p, Re, alphas, raw_sigma=False):
        A = np.atleast_1d(np.asarray(alphas, float))
        F = np.repeat(features(p, Re)[None, :], len(A), 0)
        mu, sd = self._forward(F, A, self.anchors(p, A))
        if not raw_sigma:
            sd = sd * self.widen[None, :]
        return dict(alpha=A, cl=mu[:, 0], cd=np.exp(mu[:, 1]), cm=mu[:, 2],
                    cl_sd=sd[:, 0], lncd_sd=sd[:, 1], cm_sd=sd[:, 2])

    def polar(self, coords=None, Re=2e5, alphas=None, p=None, raw_sigma=False):
        """coords: (n,2) airfoil coordinates (any chord/orientation, Selig or Lednicer order) or p: CST params"""
        if p is None:
            p, rms = cst_fit(normalize(coords))
        if alphas is None:
            alphas = np.arange(-6, 20.01, 0.25)
        r = self.predict(p, Re, alphas, raw_sigma)
        k = int(np.argmax(r['cl']))
        r.update(clmax=float(r['cl'][k]), alpha_stall=float(r['alpha'][k]), p=p)
        return r
