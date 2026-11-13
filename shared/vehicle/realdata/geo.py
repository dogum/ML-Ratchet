"""Geometry helpers (numpy only): UTM projection, DEM sampling, along-track elevation/grade, heading."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter1d

DATA = _VEH + '/data'

# --- UTM forward projection (GRS80 ~ WGS84; NAD83 vs WGS84 differ by ~1 m here, below GPS error) ---
_A = 6378137.0; _F = 1 / 298.257222101; _K0 = 0.9996
_E2 = _F * (2 - _F); _EP2 = _E2 / (1 - _E2)


def utm_forward(lat, lon, zone=17):
    lat = np.radians(np.asarray(lat, float)); lon = np.asarray(lon, float)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    lam = np.radians(lon) - lon0
    N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2; C = _EP2 * np.cos(lat) ** 2; A = np.cos(lat) * lam
    e4, e6 = _E2 ** 2, _E2 ** 3
    M = _A * ((1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat - (3 * _E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * lat)
              + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * lat) - (35 * e6 / 3072) * np.sin(6 * lat))
    x = _K0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * _EP2) * A ** 5 / 120) + 500000.0
    y = _K0 * (M + N * np.tan(lat) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                      + (61 - 58 * T + T * T + 600 * C - 330 * _EP2) * A ** 6 / 720))
    return x, y


class Dem:
    """2 m bare-earth elevation mosaic over the VED geofence (USGS 3DEP 1 m lidar, block-averaged)."""

    def __init__(self, path=f'{DATA}/proc/dem_annarbor_2m.npz'):
        z = np.load(path)
        self.e = z['elev_cm'].astype(np.float32) / 100.0
        self.e[self.e == 0] = np.nan
        self.x0 = float(z['x0']); self.y1 = float(z['y1']); self.res = float(z['res'])

    def sample_xy(self, x, y):
        # pixel centres at x0 + (j+0.5)*res, y1 - (i+0.5)*res ; bilinear
        fj = (np.asarray(x) - self.x0) / self.res - 0.5
        fi = (self.y1 - np.asarray(y)) / self.res - 0.5
        j0 = np.floor(fj).astype(int); i0 = np.floor(fi).astype(int)
        ny, nx = self.e.shape
        ok = (j0 >= 0) & (i0 >= 0) & (j0 < nx - 1) & (i0 < ny - 1)
        j0c = np.clip(j0, 0, nx - 2); i0c = np.clip(i0, 0, ny - 2)
        wj = fj - j0c; wi = fi - i0c
        e = self.e
        z = ((1 - wi) * (1 - wj) * e[i0c, j0c] + (1 - wi) * wj * e[i0c, j0c + 1]
             + wi * (1 - wj) * e[i0c + 1, j0c] + wi * wj * e[i0c + 1, j0c + 1])
        return np.where(ok, z, np.nan)

    def sample_ll(self, lat, lon):
        return self.sample_xy(*utm_forward(lat, lon))


def elevation_profile(dem, x_of_s, y_of_s, s_max, ds=5.0, close_m=120.0, sigma_m=40.0, gmax=0.15):
    """Elevation and grade along the travelled path.
    x_of_s, y_of_s: callables giving UTM position at distance s (m).
    Steps: sample DEM every ds metres -> fill NaN -> morphological closing over close_m
    (fills narrow dips: bare-earth DEM shows the valley/river under bridges, and GPS points that
    fall into roadside ditches) -> Gaussian smoothing (sigma_m) -> grade = dz/ds, clipped to +-gmax.
    Returns s grid, raw z, cleaned z, grade, and the fraction of samples changed by > 2 m by the closing."""
    n = max(int(np.ceil(s_max / ds)) + 1, 2)
    s = np.linspace(0, max(s_max, ds), n)
    zr = dem.sample_xy(x_of_s(s), y_of_s(s))
    good = np.isfinite(zr)
    if good.sum() < 2:
        return s, zr, np.full(n, np.nan), np.full(n, np.nan), np.nan, 1.0
    zf = np.interp(s, s[good], zr[good])
    w = max(int(round(close_m / ds)), 1)
    zc = minimum_filter1d(maximum_filter1d(zf, w, mode='nearest'), w, mode='nearest')
    zs = gaussian_filter1d(zc, sigma_m / ds, mode='nearest')
    g = np.clip(np.gradient(zs, s), -gmax, gmax)
    filled = float(np.mean(np.abs(zc - zf) > 2.0))
    return s, zr, zs, g, filled, 1 - good.mean()


def heading_from_xy(x, y, moving, half=3):
    """Travel heading [deg from north, clockwise] from 1 Hz UTM positions; central difference over
    +-half seconds; held through stops. Note: UTM grid north differs from true north by ~-1.8 deg here (ignored)."""
    n = len(x)
    i0 = np.clip(np.arange(n) - half, 0, n - 1); i1 = np.clip(np.arange(n) + half, 0, n - 1)
    dx = x[i1] - x[i0]; dy = y[i1] - y[i0]
    h = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    ok = moving & (np.hypot(dx, dy) > 3.0)
    if ok.sum() == 0:
        return np.full(n, np.nan, np.float32)
    idx = np.where(ok, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    first = np.argmax(ok)
    idx[idx < 0] = first
    return h[idx].astype(np.float32)
