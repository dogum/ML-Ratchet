"""QuakeNet physics: a seismogram at one station from a point source with any moment tensor, built from full-wavefield
Green's functions (ak135f, AxiSEM via Syngine) on a depth x distance grid, convolved with a source time function, rotated
into the station's Z/N/E frame by the back-azimuth.
Moment tensor convention: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp (N m); x = north, y = east, z = down for the Green's-function sum."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, json, os
GF = _P.WORK + '/data/gf/'
DEPTHS = [2, 5, 10, 15, 20, 25, 33, 40, 50, 60, 70, 85, 100, 120, 140, 165, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
DISTS = list(range(25, 96)); _cache = {}

def gf(z, d):
    k = (z, d)
    if k not in _cache:
        _cache[k] = np.load(f'{GF}z{z:03d}_d{d:03d}.npy')
        if len(_cache) > 400: _cache.pop(next(iter(_cache)))
    return _cache[k]

def zrt(G, M, az):
    """G: (10, L) [ZSS ZDS ZDD ZEP RSS RDS RDD REP TSS TDS]; M: (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp); az: source-to-receiver azimuth (rad)"""
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = M; Mxx, Myy, Mzz, Mxy, Mxz, Myz = Mtt, Mpp, Mrr, Mtp, Mrt, Mrp     # signs as Syngine's Green's functions need them (fitted, physics_check.py)
    az = np.pi - az          # Syngine's Green's functions are referenced to this azimuth (fitted against its full synthetics)
    c2, s2, c1, s1 = np.cos(2 * az), np.sin(2 * az), np.cos(az), np.sin(az)
    def vr(SS, DS, DD, EP):
        return (Mxx * (SS / 2 * c2 - DD / 6 + EP / 3) + Myy * (-SS / 2 * c2 - DD / 6 + EP / 3) + Mzz * (DD / 3 + EP / 3) + Mxy * SS * s2 + Mxz * DS * c1 + Myz * DS * s1)
    Z = vr(*G[0:4]); R = vr(*G[4:8]); TSS, TDS = G[8], G[9]
    T = Mxx * TSS / 2 * s2 - Myy * TSS / 2 * s2 - Mxy * TSS * c2 + Mxz * TDS * s1 - Myz * TDS * c1
    return Z, R, T

def zne(Z, R, T, baz):
    """obspy's convention: R = -N cos(baz) - E sin(baz), T = N sin(baz) - E cos(baz)"""
    N = -R * np.cos(baz) + T * np.sin(baz); E = -R * np.sin(baz) - T * np.cos(baz); return np.stack([Z, N, E])

def stf(half_duration, dt=0.5):
    n = max(1, int(round(half_duration / dt))); tri = np.concatenate([np.arange(1, n + 1), np.arange(n - 1, 0, -1)]).astype(float); return tri / tri.sum()

def mw_to_m0(mw): return 10 ** (1.5 * mw + 9.1)          # N m
def half_duration(m0): return 1.05e-8 * (m0 * 1e7) ** (1 / 3)   # s, the GCMT scaling (M0 in dyne cm)

def double_couple(strike, dip, rake, m0):
    """Aki & Richards (Box 4.4) in r, theta, phi: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp"""
    s, d, r = np.radians([strike, dip, rake])
    Mxx = -m0 * (np.sin(d) * np.cos(r) * np.sin(2 * s) + np.sin(2 * d) * np.sin(r) * np.sin(s) ** 2)
    Mxy = m0 * (np.sin(d) * np.cos(r) * np.cos(2 * s) + 0.5 * np.sin(2 * d) * np.sin(r) * np.sin(2 * s))
    Mxz = -m0 * (np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2 * d) * np.sin(r) * np.sin(s))
    Myy = m0 * (np.sin(d) * np.cos(r) * np.sin(2 * s) - np.sin(2 * d) * np.sin(r) * np.cos(s) ** 2)
    Myz = -m0 * (np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2 * d) * np.sin(r) * np.cos(s))
    Mzz = m0 * np.sin(2 * d) * np.sin(r)
    return np.array([Mzz, Mxx, Myy, Mxz, -Myz, -Mxy])     # x north, y east, z down -> r, t, p

def seismogram(z, d, M, az, baz, dt_out=1.0):
    """velocity (m/s), (3, L) Z N E at dt_out, starting 60 s before P"""
    G = gf(z, d); Z, R, T = zrt(G, M, az); m0 = np.sqrt(0.5 * np.sum(np.asarray(M[:3]) ** 2) + np.sum(np.asarray(M[3:]) ** 2))
    k = stf(half_duration(m0)); Z, R, T = (np.convolve(x, k)[:len(x)] for x in (Z, R, T))
    X = zne(Z, R, T, baz); step = int(round(dt_out / 0.5)); return X[:, ::step]
