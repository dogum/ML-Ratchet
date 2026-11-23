"""Airfoil shape parameterisation for AirfoilNet: Kulfan CST fit, reconstruction, sensors, synthetic shapes.

CST (Kulfan 2008): y(x) = sqrt(x)(1-x) * sum_i w_i K_i x^i (1-x)^(n-i) + x*dz_te/2  (+ the leading-edge
modification term of Kulfan 2020, LE * sqrt(x) (1-x)^(n+0.5), on both surfaces), n = 7 (8 weights a side).
The network reads the shape at fixed sensor stations (upper and lower y at 24 cosine-spaced x), the
"function sampled at sensors" input of a DeepONet branch.
"""
import numpy as np
from math import comb

N_W = 8
XS = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 26)))[1:-1]      # 24 sensor stations (exclude 0 and 1)


def _basis(x, n=N_W - 1):
    x = np.asarray(x, float)
    C = np.sqrt(x) * (1 - x)
    B = np.stack([comb(n, i) * x ** i * (1 - x) ** (n - i) for i in range(n + 1)], axis=1)
    return C[:, None] * B


def _le_term(x, n=N_W - 1):
    return np.sqrt(x) * (1 - x) ** (n + 0.5)


def cst_eval(p, x):
    """p = [wu(8), wl(8), le, te] -> yu(x), yl(x)"""
    wu, wl, le, te = p[:N_W], p[N_W:2 * N_W], p[2 * N_W], p[2 * N_W + 1]
    B = _basis(x); L = _le_term(x)
    yu = B @ wu + le * L + x * te / 2
    yl = B @ wl + le * L - x * te / 2
    return yu, yl


def cst_fit(Y, reg=1e-6):
    """least-squares CST fit to a normalised Selig-order airfoil Y (n,2). Returns p (18,), rms error."""
    i = np.argmin(Y[:, 0])
    up = Y[:i + 1]; lo = Y[i:]
    xu = np.clip(up[:, 0], 0, 1); xl = np.clip(lo[:, 0], 0, 1)
    Bu = _basis(xu); Bl = _basis(xl)
    nu, nl = len(xu), len(xl)
    A = np.zeros((nu + nl, 2 * N_W + 2))
    A[:nu, :N_W] = Bu; A[nu:, N_W:2 * N_W] = Bl
    A[:nu, 2 * N_W] = _le_term(xu); A[nu:, 2 * N_W] = _le_term(xl)
    A[:nu, 2 * N_W + 1] = xu / 2; A[nu:, 2 * N_W + 1] = -xl / 2
    b = np.r_[up[:, 1], lo[:, 1]]
    p = np.linalg.solve(A.T @ A + reg * np.eye(A.shape[1]), A.T @ b)
    if p[-1] < 0:                         # crossed trailing edge: refit with a closed (sharp) TE
        A2 = A[:, :-1]
        p = np.r_[np.linalg.solve(A2.T @ A2 + reg * np.eye(A2.shape[1]), A2.T @ b), 0.0]
    return p, float(np.sqrt(np.mean((A @ p - b) ** 2)))


def cst_coords(p, n=100):
    """Selig-order coordinates (2n+1 points, cosine spacing) from CST parameters."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
    yu, yl = cst_eval(p, x)
    up = np.c_[x[::-1], yu[::-1]]
    lo = np.c_[x[1:], yl[1:]]
    return np.vstack([up, lo])


def sensors_from_p(p):
    yu, yl = cst_eval(p, XS)
    return np.r_[yu, yl]


def geom_stats(p):
    x = np.linspace(0.001, 0.999, 400)
    yu, yl = cst_eval(p, x)
    t = yu - yl; c = 0.5 * (yu + yl)
    return dict(tmax=float(t.max()), xt=float(x[np.argmax(t)]), cmax=float(c[np.argmax(np.abs(c))]),
                xc=float(x[np.argmax(np.abs(c))]), tmin=float(t[x < 0.95].min()), te=float(p[-1]))


def valid(p):
    g = geom_stats(p)
    return (0.03 <= g['tmax'] <= 0.25 and g['tmin'] > 0.002 and -0.02 <= g['cmax'] <= 0.10 and
            -0.001 <= g['te'] <= 0.02)
