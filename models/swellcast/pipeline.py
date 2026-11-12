"""Read NOAA NDBC spectral wave files and cut the 10-day window SwellCast reads.

Files (https://www.ndbc.noaa.gov/): historical  data/historical/swden/<id>w<year>.txt.gz  (energy density, m^2/Hz)
                                                data/historical/swdir/<id>d<year>.txt.gz  (alpha1, deg, direction waves come FROM)
                                                data/historical/swr1/<id>j<year>.txt.gz   (r1)
             or realtime  data/realtime2/<id>.data_spec, <id>.swdir, <id>.swr1 (last 45 days).
The window is centred so that the swell's 0.06 Hz (16.7 s) energy arrives near the middle (within about +-36 h)."""
import gzip, numpy as np
FGRID = 0.0325 + 0.005 * np.arange(34); DT_H, HALF_H = 3, 120

def _open(p):
    return gzip.open(p, 'rt', errors='ignore') if str(p).endswith('.gz') else open(p, errors='ignore')

def to_grid(fr, V, logint):
    """native NDBC bins -> FGRID; energy is interpolated in log (as in training), moments linearly"""
    out = np.full((len(V), len(FGRID)), np.nan)
    for i, v in enumerate(V):
        ok = np.isfinite(v) & (fr > 0.025)
        if ok.sum() < 5: continue
        x = np.log(np.maximum(v[ok], 1e-4)) if logint else v[ok]; y = np.interp(FGRID, fr[ok], x, left=np.nan, right=np.nan)
        out[i] = np.exp(y) if logint else y
    return out

def parse_raw(path):
    """-> hours since 1970-01-01 (int array), native frequencies (nf,), values (n, nf) with NaN for missing"""
    lines = _open(path).read().splitlines(); out_t, rows = [], []
    if any('(' in l for l in lines[:5]):        # realtime: value (freq) pairs
        freqs = None
        for l in lines:
            if l.startswith('#') or '(' not in l: continue
            p = l.replace('(', ' ').replace(')', ' ').split(); y, mo, d, h, mi = map(int, p[:5]); rest = p[5:]
            if len(rest) % 2: rest = rest[1:]                       # data_spec carries a separation frequency first
            v = np.array(rest[0::2], float); f = np.array(rest[1::2], float); v[v >= 998.9] = np.nan
            if freqs is None: freqs = f
            if len(f) != len(freqs) or np.any(f != freqs): v = np.interp(freqs, f, v)
            rows.append(v); out_t.append(int(np.datetime64(f'{y:04d}-{mo:02d}-{d:02d}T{h:02d}', 'h').astype(np.int64) + round(mi / 60)))
        return np.array(out_t), freqs, np.array(rows)
    hdr = [l for l in lines if l.startswith('#') or l[:2] in ('YY', 'YYYY')]; h = (hdr[0] if hdr else lines[0]).replace('#', '').split()
    nt = 5 if h[4].lower() in ('mm', 'mn') else 4; freqs = np.array([float(x) for x in h[nt:]])
    for l in lines:
        if not l or l.startswith('#') or l[0].isalpha(): continue
        p = l.split()
        if len(p) != nt + len(freqs): continue
        y = int(p[0]); y = y + 1900 if y < 100 else y; mi = int(p[4]) if nt == 5 else 0
        out_t.append(int(np.datetime64(f'{y:04d}-{int(p[1]):02d}-{int(p[2]):02d}T{int(p[3]):02d}', 'h').astype(np.int64) + round(mi / 60)))
        v = np.array(p[nt:], float); v[(v >= 98.9) & (v <= 99.01)] = np.nan; v[v >= 998.9] = np.nan; rows.append(v)
    return np.array(out_t), freqs, np.array(rows)

def read_ndbc(swden, swdir=None, swr1=None):
    """-> dict(h0, E, C, S) on an hourly grid from h0; C, S = r1 cos(alpha1), r1 sin(alpha1) (NaN if no directional files)"""
    t, fr, V = parse_raw(swden); h0, h1 = t.min(), t.max() + 1; n = h1 - h0
    Eg = np.full((n, 34), np.nan, np.float32); Eg[t - h0] = to_grid(fr, V, True); Cg = np.full_like(Eg, np.nan); Sg = np.full_like(Eg, np.nan)
    if swdir and swr1:
        td, frd, D = parse_raw(swdir); tj, frj, R = parse_raw(swr1)
        if np.nanmax(R) > 1.5: R = R / 100.0
        R = np.clip(R, 0, 1); common, a, b = np.intersect1d(td, tj, return_indices=True); k = (common >= h0) & (common < h1)
        ang = np.radians(D[a][k]); r = R[b][k]; Cg[common[k] - h0] = to_grid(frd, r * np.cos(ang), False); Sg[common[k] - h0] = to_grid(frd, r * np.sin(ang), False)
    return dict(h0=int(h0), E=Eg, C=Cg, S=Sg)

def coarse(E, C, S, dt=DT_H):
    n = E.shape[0] // dt; E = E[:n * dt].reshape(n, dt, -1); C = C[:n * dt].reshape(n, dt, -1); S = S[:n * dt].reshape(n, dt, -1)
    okE = np.isfinite(E); ne = okE.sum(1); Em = np.where(ne > 0, np.nansum(np.where(okE, E, 0), 1) / np.maximum(ne, 1), np.nan)
    okD = okE & np.isfinite(C) & np.isfinite(S); w = np.where(okD, E, 0); sw = w.sum(1)
    Cm = np.where(sw > 0, (w * np.where(okD, C, 0)).sum(1) / np.maximum(sw, 1e-12), np.nan); Sm = np.where(sw > 0, (w * np.where(okD, S, 0)).sum(1) / np.maximum(sw, 1e-12), np.nan)
    Cm = np.where(okD.sum(1) > 0, Cm, np.nan); Sm = np.where(okD.sum(1) > 0, Sm, np.nan)
    return Em.astype(np.float32), Cm.astype(np.float32), Sm.astype(np.float32)

def window(rec, centre_hour):
    """rec from read_ndbc; centre_hour = hours since 1970 (UTC). -> ((3, 80, 34) window, grid centre hour used).
    t0 returned by SwellCast is relative to that grid centre."""
    tc = int(np.floor(centre_hour / DT_H) * DT_H); a = tc - HALF_H - rec['h0']; n = 2 * HALF_H
    E = np.full((n, 34), np.nan, np.float32); C = E.copy(); S = E.copy(); lo, hi = max(0, a), min(len(rec['E']), a + n)
    if hi > lo: E[lo - a:hi - a] = rec['E'][lo:hi]; C[lo - a:hi - a] = rec['C'][lo:hi]; S[lo - a:hi - a] = rec['S'][lo:hi]
    return np.stack(coarse(E, C, S)), tc

def hour(s):
    """'2018-08-20T06' -> hours since 1970"""
    return int(np.datetime64(s, 'h').astype(np.int64))

def utc(h):
    return str(np.datetime64(int(round(h)), 'h'))
