"""Real buoy data access on the network's grid: 3-hour means over a 240-hour window, 34 frequencies 0.0325-0.1975 Hz.
  E  (80, 34) energy density m^2/Hz (NaN = missing)
  C, S (80, 34) r1 cos(alpha1), r1 sin(alpha1) (energy-weighted over the 3 hours), alpha1 = direction waves come FROM.
Also builds the per-buoy 3-hourly background store used for injection (data/bg/<buoy>.npz)."""
import os, glob, numpy as np
import swellsim as SS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BUOY = os.path.join(ROOT, 'data/buoy'); BG = os.path.join(ROOT, 'data/bg')
_cache = {}

def _year(b, y):
    k = (b, y)
    if k not in _cache:
        fn = os.path.join(BUOY, f'{b}_{y}.npz')
        if not os.path.exists(fn): _cache[k] = None
        else:
            z = np.load(fn); _cache[k] = (int(z['h0']), z['E'].astype(np.float32), z['C'].astype(np.float32), z['S'].astype(np.float32))
        if len(_cache) > 12: _cache.pop(next(iter(_cache)))
    return _cache[k]

def hourly(b, h_start, n):
    """hourly E, C, S for hours [h_start, h_start + n) (hours since 1970)"""
    E = np.full((n, 34), np.nan, np.float32); C = E.copy(); S = E.copy()
    y0 = int(np.datetime64(int(h_start), 'h').astype('datetime64[Y]').astype(int) + 1970); y1 = int(np.datetime64(int(h_start + n), 'h').astype('datetime64[Y]').astype(int) + 1970)
    for y in range(y0, y1 + 1):
        d = _year(b, y)
        if d is None: continue
        h0, Ey, Cy, Sy = d; a = max(h_start, h0); z = min(h_start + n, h0 + len(Ey))
        if z > a: E[a - h_start:z - h_start] = Ey[a - h0:z - h0]; C[a - h_start:z - h_start] = Cy[a - h0:z - h0]; S[a - h_start:z - h_start] = Sy[a - h0:z - h0]
    return E, C, S

def coarse(E, C, S, dt=SS.DT_H):
    """hourly -> dt-hour means; C, S energy-weighted. NaN where no valid hour."""
    n = E.shape[0] // dt; E = E[:n * dt].reshape(n, dt, -1); C = C[:n * dt].reshape(n, dt, -1); S = S[:n * dt].reshape(n, dt, -1)
    okE = np.isfinite(E); ne = okE.sum(1); Em = np.where(ne > 0, np.nansum(np.where(okE, E, 0), 1) / np.maximum(ne, 1), np.nan)
    okD = okE & np.isfinite(C) & np.isfinite(S); w = np.where(okD, E, 0); sw = w.sum(1)
    Cm = np.where(sw > 0, (w * np.where(okD, C, 0)).sum(1) / np.maximum(sw, 1e-12), np.nan); Sm = np.where(sw > 0, (w * np.where(okD, S, 0)).sum(1) / np.maximum(sw, 1e-12), np.nan)
    Cm = np.where(okD.sum(1) > 0, Cm, np.nan); Sm = np.where(okD.sum(1) > 0, Sm, np.nan)
    return Em.astype(np.float32), Cm.astype(np.float32), Sm.astype(np.float32)

def window(b, tc_h):
    """network window centred on hour tc_h (rounded down to the 3-hour grid): returns (E, C, S) coarse and the grid centre used"""
    tc = int(np.floor(tc_h / SS.DT_H) * SS.DT_H); E, C, S = hourly(b, tc - SS.HALF_H, 2 * SS.HALF_H); return coarse(E, C, S), tc

def build_bg(b):
    os.makedirs(BG, exist_ok=True); fs = sorted(glob.glob(os.path.join(BUOY, f'{b}_*.npz')))
    if not fs: return None
    h0 = np.load(fs[0])['h0'].item(); h0 = int(np.floor(h0 / SS.DT_H) * SS.DT_H)
    last = np.load(fs[-1]); h1 = int(last['h0']) + len(last['E']); n = (h1 - h0) // SS.DT_H * SS.DT_H
    E, C, S = hourly(b, h0, n); Em, Cm, Sm = coarse(E, C, S)
    np.savez(os.path.join(BG, f'{b}.npz'), E=Em.astype(np.float16), C=Cm.astype(np.float16), S=Sm.astype(np.float16), h0=h0)
    return Em.shape

def load_bg(b):
    z = np.load(os.path.join(BG, f'{b}.npz')); return int(z['h0']), z['E'], z['C'], z['S']

if __name__ == '__main__':
    import sys, fetch
    for b in fetch.BUOYS: print(b, build_bg(b), flush=True)
