"""Great-circle geometry on a sphere and a 0.1-degree land mask (Natural Earth 1:50m via world-atlas topojson)."""
import os, json, numpy as np
R_E = 6371.0e3   # m
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def dist(lat1, lon1, lat2, lon2):
    """great-circle distance in m (haversine)"""
    p1, p2 = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - lon1); dp = p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_E * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def bearing(lat1, lon1, lat2, lon2):
    """initial compass bearing (deg, clockwise from north) from point 1 toward point 2"""
    p1, p2 = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - lon1)
    y = np.sin(dl) * np.cos(p2); x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360

def dest(lat, lon, brg, d):
    """point reached from (lat, lon) after d metres on initial bearing brg"""
    p1, l1, b, a = np.radians(lat), np.radians(lon), np.radians(brg), np.asarray(d) / R_E
    p2 = np.arcsin(np.sin(p1) * np.cos(a) + np.cos(p1) * np.sin(a) * np.cos(b))
    l2 = l1 + np.arctan2(np.sin(b) * np.sin(a) * np.cos(p1), np.cos(a) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540) % 360 - 180

def path(lat1, lon1, lat2, lon2, step=25e3):
    D = float(dist(lat1, lon1, lat2, lon2)); n = max(2, int(D / step) + 1); s = np.linspace(0, D, n)
    la, lo = dest(lat1, lon1, bearing(lat1, lon1, lat2, lon2), s); return la, lo, s

_MASK = None
def _decode_topo(fn):
    t = json.load(open(fn)); sc, tr = t['transform']['scale'], t['transform']['translate']; arcs = []
    for a in t['arcs']:
        a = np.cumsum(np.array(a, float), 0); arcs.append(np.stack([a[:, 0] * sc[0] + tr[0], a[:, 1] * sc[1] + tr[1]], 1))
    def ring(ids):
        pts = []
        for i in ids:
            seg = arcs[i] if i >= 0 else arcs[~i][::-1]; pts.append(seg if not pts else seg[1:])
        return np.concatenate(pts)
    polys = []
    for g in t['objects']['land']['geometries']:
        P = g['arcs'] if g['type'] == 'MultiPolygon' else [g['arcs']]
        for p in P: polys.append([ring(r) for r in p])
    return polys

def land_mask():
    """boolean (1800, 3600) grid, True on land; row i = lat -90 + 0.1 (i + 0.5), col j = lon -180 + 0.1 (j + 0.5)"""
    global _MASK
    if _MASK is not None: return _MASK
    cache = os.path.join(ROOT, 'data/geo/landmask_0p1.npy')
    if os.path.exists(cache): _MASK = np.load(cache); return _MASK
    from matplotlib.path import Path
    polys = _decode_topo(os.path.join(ROOT, 'data/geo/land-50m.json')); M = np.zeros((1800, 3600), bool)
    for p in polys:
        outer = p[0]; x0, y0 = outer.min(0); x1, y1 = outer.max(0)
        if x1 - x0 > 300: continue   # Antarctica ring wraps the dateline: handle below
        j0, j1 = int(max(0, (x0 + 180) * 10)), int(min(3600, (x1 + 180) * 10 + 1)); i0, i1 = int(max(0, (y0 + 90) * 10)), int(min(1800, (y1 + 90) * 10 + 1))
        if j1 <= j0 or i1 <= i0: continue
        gx, gy = np.meshgrid(-180 + 0.1 * (np.arange(j0, j1) + 0.5), -90 + 0.1 * (np.arange(i0, i1) + 0.5))
        q = np.stack([gx.ravel(), gy.ravel()], 1); inside = Path(outer).contains_points(q)
        for h in p[1:]: inside &= ~Path(h).contains_points(q)
        M[i0:i1, j0:j1] |= inside.reshape(gx.shape)
    M[:int((90 - 63) * 10), :] |= False
    M[:int((90 - 70) * 10) + 1, :] = True   # everything south of 70S treated as land (Antarctica)
    np.save(cache, M); _MASK = M; return M

def on_land(lat, lon):
    M = land_mask(); i = np.clip(((np.asarray(lat) + 90) * 10).astype(int), 0, 1799); j = np.clip(((np.asarray(lon) + 180) % 360 * 10).astype(int), 0, 3599)
    return M[i, j]

def ocean_path(lat1, lon1, lat2, lon2, skip_end=40e3, skip_start=0.0):
    """True if the great circle from point 1 (storm) to point 2 (buoy) stays over water, ignoring the last skip_end metres"""
    la, lo, s = path(lat1, lon1, lat2, lon2); D = s[-1]; k = (s >= skip_start) & (s <= D - skip_end)
    return not on_land(la[k], lo[k]).any()
