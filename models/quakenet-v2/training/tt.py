"""Travel times and spherical geometry for QuakeNet v2.
ak135 P and S first-arrival times (ObsPy TauP) on QuakeNet's 27 source depths x 0-100 degrees every 0.25 degrees,
built once into WORK/data/tt_ak135.npz (training/_paths.py H; the release ships the same table as ../tt_ak135.npz); interpolated bilinearly (distance, depth). Geometry on a sphere: positions as unit
vectors, great-circle distance and azimuth, and the point at a given distance and azimuth."""
import os, numpy as np
from _paths import H                     # the work folder (ends with a separator), as every other training script
SHIPPED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tt_ak135.npz')   # release copy
DEPTHS = np.array([2, 5, 10, 15, 20, 25, 33, 40, 50, 60, 70, 85, 100, 120, 140, 165, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700], float)
DIST = np.arange(0, 100.001, 0.25); R_KM = 6371.0; KM_PER_DEG = np.pi * R_KM / 180
_T = None

def build(path=os.path.join(H, 'data', 'tt_ak135.npz')):
    from obspy.taup import TauPyModel
    tp = TauPyModel('ak135'); P = np.full((len(DEPTHS), len(DIST)), np.nan); S = P.copy()
    for i, z in enumerate(DEPTHS):
        for j, d in enumerate(DIST):
            a = tp.get_travel_times(z, max(d, 0.01), ['p', 'P', 'Pdiff']); P[i, j] = min(x.time for x in a) if a else np.nan
            a = tp.get_travel_times(z, max(d, 0.01), ['s', 'S', 'Sdiff']); S[i, j] = min(x.time for x in a) if a else np.nan
    os.makedirs(os.path.dirname(path), exist_ok=True); np.savez(path, depths=DEPTHS, dist=DIST, P=P, S=S); return path

def tables():
    global _T
    if _T is None:
        w = os.path.join(H, 'data', 'tt_ak135.npz')
        z = np.load(w if os.path.exists(w) else SHIPPED); _T = (z['P'], z['S'])
    return _T

def tt(phase, dist, depth):
    """bilinear in (distance deg, depth km); arrays broadcast"""
    T = tables()[0 if phase == 'P' else 1]; d = np.clip(np.asarray(dist, float), 0, 100); z = np.clip(np.asarray(depth, float), 2, 700)
    fj = d / 0.25; j = np.clip(np.floor(fj).astype(int), 0, len(DIST) - 2); wj = fj - j
    i = np.clip(np.searchsorted(DEPTHS, z) - 1, 0, len(DEPTHS) - 2); wi = (z - DEPTHS[i]) / (DEPTHS[i + 1] - DEPTHS[i])
    return (T[i, j] * (1 - wj) + T[i, j + 1] * wj) * (1 - wi) + (T[i + 1, j] * (1 - wj) + T[i + 1, j + 1] * wj) * wi

def slowness(dist, depth, phase='P'):
    """dT/dDelta, s/deg"""
    return (tt(phase, np.asarray(dist) + 0.25, depth) - tt(phase, np.asarray(dist) - 0.25, depth)) / 0.5

def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon); return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

def latlon(u):
    u = u / np.linalg.norm(u, axis=-1, keepdims=True); return np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1))), np.degrees(np.arctan2(u[..., 1], u[..., 0]))

def dist(lat1, lon1, lat2, lon2):
    """great-circle distance, degrees"""
    a, b = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    c = np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(dl); s = np.hypot(np.cos(b) * np.sin(dl), np.cos(a) * np.sin(b) - np.sin(a) * np.cos(b) * np.cos(dl))
    return np.degrees(np.arctan2(s, c))

def azimuth(lat1, lon1, lat2, lon2):
    """azimuth at point 1 toward point 2, degrees clockwise from north"""
    a, b = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    return np.degrees(np.arctan2(np.sin(dl) * np.cos(b), np.cos(a) * np.sin(b) - np.sin(a) * np.cos(b) * np.cos(dl))) % 360

def destination(lat, lon, az, d):
    """point at distance d (deg) along azimuth az (deg) from (lat, lon)"""
    a, l, t, dd = np.radians(lat), np.radians(lon), np.radians(az), np.radians(d)
    b = np.arcsin(np.sin(a) * np.cos(dd) + np.cos(a) * np.sin(dd) * np.cos(t))
    l2 = l + np.arctan2(np.sin(t) * np.sin(dd) * np.cos(a), np.cos(dd) - np.sin(a) * np.sin(b))
    return np.degrees(b), (np.degrees(l2) + 180) % 360 - 180

def angdiff(a, b): return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180

if __name__ == '__main__':
    print(build())
