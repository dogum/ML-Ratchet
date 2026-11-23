"""Linear shallow-water equations on a spherical lon-lat C-grid (no Coriolis, no friction), forward-backward time
stepping as in MOST/COMCOT's linear propagation. Land (depth < HMIN) is a reflecting wall; a sponge damps the outer
edge of the domain. The discrete operator is symmetric under the cell-area inner product, so a run started from a
unit hump in one receiver cell gives, by reciprocity, the receiver's response to every source cell at once:
    eta_rcv(t) = (1/A_rcv) * sum_j A_j * eta0_j * G_j(t)     (G = the field of the receiver-started run)."""
import os, numpy as np
import _paths  # sets NUMBA_CACHE_DIR / threads
import numba as nb
G = 9.81; R = 6371.0e3; HMIN = 20.0

class Grid:
    def __init__(self, path, stride=1, lon_range=None, lat_range=None):
        d = np.load(path)
        z, lat, lon = d['z'], d['lat'], d['lon']
        if lat_range is not None:
            m = (lat >= lat_range[0]) & (lat <= lat_range[1]); z, lat = z[m], lat[m]
        if lon_range is not None:
            m = (lon >= lon_range[0]) & (lon <= lon_range[1]); z, lon = z[:, m], lon[m]
        if stride > 1:
            ny, nx = (z.shape[0] // stride) * stride, (z.shape[1] // stride) * stride
            z = z[:ny, :nx].reshape(ny // stride, stride, nx // stride, stride).mean((1, 3))
            lat = lat[:ny].reshape(-1, stride).mean(1); lon = lon[:nx].reshape(-1, stride).mean(1)
        self.lat, self.lon = lat.astype(np.float64), lon.astype(np.float64)
        self.h = np.where(-z >= HMIN, -z, 0.0).astype(np.float32)   # water depth (m), 0 = land/wall
        self.dlon = np.radians(lon[1] - lon[0]); self.dlat = np.radians(lat[1] - lat[0])
        self.cos_c = np.cos(np.radians(self.lat))
        self.cos_n = np.cos(np.radians(self.lat + 0.5 * np.degrees(self.dlat)))   # north faces
        self.area = (R * R * self.cos_c * self.dlon * self.dlat)                  # per row
        self.ny, self.nx = self.h.shape

    def stable_dt(self, cfl=0.9):
        c = np.sqrt(G * self.h.max(1))
        dx = R * self.cos_c * self.dlon; dy = R * self.dlat
        return cfl * float(np.min(1.0 / (np.maximum(c, 1e-3) * np.sqrt(1 / dx ** 2 + 1 / dy ** 2))))

    def index(self, lon, lat):
        i = int(np.argmin(np.abs(self.lon - (lon % 360)))); j = int(np.argmin(np.abs(self.lat - lat)))
        return j, i

    def sponge(self, width=25, strength=0.02):
        """Damping factor per cell (1 = none). Applied to eta and fluxes every step near open edges."""
        ny, nx = self.ny, self.nx
        jj = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))[:, None]
        ii = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))[None, :]
        d = np.minimum(jj, ii).astype(float)
        s = np.where(d < width, strength * ((width - d) / width) ** 2, 0.0)
        return (1.0 - s).astype(np.float32)

@nb.njit(cache=True, fastmath=True)
def step(eta, M, N, h, cos_c, cos_n, dlon, dlat, dt, damp):
    ny, nx = eta.shape
    # fluxes (M on east faces i+1/2 for i=0..nx-2; N on north faces j+1/2 for j=0..ny-2)
    for j in range(ny):
        a = G * dt / (R * cos_c[j] * dlon)
        for i in range(nx - 1):
            he = min(h[j, i], h[j, i + 1])
            if he > 0.0:
                M[j, i] = (M[j, i] - a * 0.5 * (h[j, i] + h[j, i + 1]) * (eta[j, i + 1] - eta[j, i])) * damp[j, i]
            else:
                M[j, i] = 0.0
    b = G * dt / (R * dlat)
    for j in range(ny - 1):
        for i in range(nx):
            hn = min(h[j, i], h[j + 1, i])
            if hn > 0.0:
                N[j, i] = (N[j, i] - b * 0.5 * (h[j, i] + h[j + 1, i]) * (eta[j + 1, i] - eta[j, i])) * damp[j, i]
            else:
                N[j, i] = 0.0
    for j in range(ny):
        c1 = dt / (R * cos_c[j] * dlon); c2 = dt / (R * cos_c[j] * dlat)
        cn = cos_n[j]; cs = cos_n[j - 1] if j > 0 else 0.0
        for i in range(nx):
            if h[j, i] > 0.0:
                mw = M[j, i - 1] if i > 0 else 0.0
                ns = N[j - 1, i] if j > 0 else 0.0
                nn = N[j, i] if j < ny - 1 else 0.0
                me = M[j, i] if i < nx - 1 else 0.0
                eta[j, i] = (eta[j, i] - c1 * (me - mw) - c2 * (nn * cn - ns * cs)) * damp[j, i]

@nb.njit(cache=True, fastmath=True)
def project(eta_flat, indptr, idx, w, out):
    for k in range(len(indptr) - 1):
        s = 0.0
        for p in range(indptr[k], indptr[k + 1]):
            s += w[p] * eta_flat[idx[p]]
        out[k] = s

@nb.njit(cache=True, fastmath=True)
def track_max(eta, amax):
    ny, nx = eta.shape
    for j in range(ny):
        for i in range(nx):
            v = abs(eta[j, i])
            if v > amax[j, i]: amax[j, i] = v
