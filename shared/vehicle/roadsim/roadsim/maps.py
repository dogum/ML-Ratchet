"""Map storage and fast interpolation.

All maps are on uniform rectangular grids so a lookup is two index computations and a bilinear blend. `Grid2.__call__`
is pure-Python (fast for the per-step scalar loop); `Grid2.vec` is the numpy version for candidate arrays.
"""
import os, json
import numpy as np
try:                        # the ufunc np.clip ends in for float arrays with both bounds, without its Python wrappers
    from numpy._core.umath import clip as _clip
except ImportError:         # pragma: no cover
    _clip = np.clip

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_CACHE = {}


def load_maps():
    if 'maps' not in _CACHE:
        z = np.load(os.path.join(DATA, 'maps.npz'))
        _CACHE['maps'] = {k: z[k] for k in z.files}
        _CACHE['meta'] = json.load(open(os.path.join(DATA, 'maps_meta.json')))
    return _CACHE['maps'], _CACHE['meta']


class Grid2:
    """z(x, y) on a uniform grid x in [x0, x1] (nx points) and y in [y0, y1] (ny points); clamped at the edges."""
    __slots__ = ('x0', 'y0', 'ix', 'iy', 'nx', 'ny', 'z', 'Z', 'dz', 'nx1', 'ny1', 'Zf', 'dZf')

    def __init__(self, x0, x1, y0, y1, Z):
        Z = np.asarray(Z, float)
        self.nx, self.ny = Z.shape
        self.x0, self.y0 = float(x0), float(y0)
        self.ix = (self.nx - 1) / float(x1 - x0)
        self.iy = (self.ny - 1) / float(y1 - y0)
        self.Z = Z
        self.z = Z.ravel().tolist()
        # speed: the y-differences z[i, j+1] - z[i, j] once (the same subtraction the lookup did), flat copies for .vec
        dZ = np.zeros_like(Z)
        dZ[:, :-1] = Z[:, 1:] - Z[:, :-1]
        self.Zf = Z.ravel().copy(); self.dZf = dZ.ravel()
        self.dz = self.dZf.tolist()
        self.nx1 = self.nx - 1.000001; self.ny1 = self.ny - 1.000001

    def __call__(self, x, y):
        fx = (x - self.x0) * self.ix
        fy = (y - self.y0) * self.iy
        nx1 = self.nx1
        ny1 = self.ny1
        if fx < 0.0: fx = 0.0
        elif fx > nx1: fx = nx1
        if fy < 0.0: fy = 0.0
        elif fy > ny1: fy = ny1
        i = int(fx); j = int(fy)
        wx = fx - i; wy = fy - j
        z = self.z; dz = self.dz; ny = self.ny; k = i * ny + j
        return (z[k] + dz[k] * wy) * (1.0 - wx) + (z[k + ny] + dz[k + ny] * wy) * wx

    def vec(self, x, y):
        fx = _clip((np.asarray(x, float) - self.x0) * self.ix, 0.0, self.nx1)
        fy = _clip((np.asarray(y, float) - self.y0) * self.iy, 0.0, self.ny1)
        i = fx.astype(np.int64); j = fy.astype(np.int64)
        wx = fx - i; wy = fy - j
        ny = self.ny; k = i * ny + j; k1 = k + ny
        Z = self.Zf; dZ = self.dZf
        return (Z[k] + dZ[k] * wy) * (1.0 - wx) + (Z[k1] + dZ[k1] * wy) * wx


class Interp1:
    """y(x) on a uniform grid (clamped). Pure-python call and numpy .vec."""
    __slots__ = ('x0', 'ix', 'n', 'y', 'Y', 'dy', 'dY', 'n1')

    def __init__(self, x0, x1, Y):
        Y = np.asarray(Y, float)
        self.n = len(Y); self.x0 = float(x0); self.ix = (self.n - 1) / float(x1 - x0)
        self.Y = Y; self.y = Y.tolist()
        dY = np.zeros_like(Y)
        dY[:-1] = Y[1:] - Y[:-1]               # y[i+1] - y[i] once (speed; the same subtraction)
        self.dY = dY; self.dy = dY.tolist()
        self.n1 = self.n - 1.000001

    def __call__(self, x):
        f = (x - self.x0) * self.ix
        if f <= 0.0: return self.y[0]
        if f >= self.n - 1: return self.y[-1]
        i = int(f); w = f - i
        return self.y[i] + self.dy[i] * w

    def vec(self, x):
        f = _clip((np.asarray(x, float) - self.x0) * self.ix, 0.0, self.n1)
        i = f.astype(np.int64); w = f - i
        return self.Y[i] + self.dY[i] * w


def uniform_from_points(xp, yp, x0, x1, n=64):
    xs = np.linspace(x0, x1, n)
    return Interp1(x0, x1, np.interp(xs, xp, yp))
