"""FireDetective shared feature code (numpy/scipy only; vendored into the release).

One fire = a sequence of FEDS-like half-day steps (S-NPP night ~01:30 and day ~13:30 local overpasses):
  masks[k]  cumulative burned mask on a 180 m EPSG:5070 grid (real: rasterised FEDS perimeter; sim: FEDS emulator)
  dets[k]   VIIRS detections of that overpass: x, y (EPSG:5070 m), frp (MW)
  day[k]    True for the day (PM) overpass
  wx[k]     gridMET daily row for the local day that drives the growth reported at step k
plus a static landscape summary around the burned area.
"""
import numpy as np
from scipy import ndimage as ndi

T_STEPS = 12          # 6 days of half-day steps
RES = 180.0           # m
SEQ_NAMES = ['valid', 'is_day', 'logA', 'dlogA', 'dr_km', 'logN', 'logFRP', 'logFRPmean', 'frac_new', 'elong',
             'gx', 'gy', 'tmmx', 'rmin', 'rmax', 'vs', 'u_from', 'v_from', 'fm100', 'fm1000', 'erc', 'bi', 'pr',
             'vpd', 'wind_hidden']
WIND_COLS = [SEQ_NAMES.index(c) for c in ('vs', 'u_from', 'v_from')]
STAT_NAMES = ['f_GR', 'f_GS', 'f_SH', 'f_TU', 'f_TL', 'f_SB', 'f_NB', 'cc', 'ch', 'cbh', 'cbd', 'slope', 'slope90',
              'elev', 'doy_s', 'doy_c', 'n_steps']
GROUPS = {'GR': list(range(101, 110)) + [1, 2, 3], 'GS': list(range(121, 125)), 'SH': list(range(141, 150)) + [4, 5, 6, 7],
          'TU': list(range(161, 166)), 'TL': list(range(181, 190)) + [8, 9, 10], 'SB': list(range(201, 205)) + [11, 12, 13]}


def fuel_group_masks(fuel):
    out = {g: np.isin(fuel, v) for g, v in GROUPS.items()}
    out['NB'] = ~np.any(np.stack(list(out.values())), axis=0)
    return out


# ---------------------------------------------------------------------------------------------------------------
# FEDS emulator: cumulative VIIRS pixel footprints -> closed, hole-filled burned mask (calibrated on dev, see
# calib_feds.py). Applied to simulated detections only; real fires use the real FEDS polygons.
FEDS_PARAMS = dict(close_cells=3, shrink=1.0)   # dev-calibrated, out/calib_feds.json


def disk(r):
    if r <= 0:
        return np.ones((1, 1), bool)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + 0.5


def stamp_pixels(shape, x0, y1, res, xs, ys, ds_km, dt_km, shrink=1.0):
    """Rasterise detection footprints (rectangles ds x dt km, centred at x, y) on a grid with top-left (x0, y1)."""
    m = np.zeros(shape, bool)
    ny, nx = shape
    for x, y, a, b in zip(xs, ys, ds_km, dt_km):
        hx = 500.0 * a * shrink
        hy = 500.0 * b * shrink
        c0 = int(np.floor((x - hx - x0) / res + 0.5)); c1 = int(np.floor((x + hx - x0) / res + 0.5))
        r0 = int(np.floor((y1 - (y + hy)) / res + 0.5)); r1 = int(np.floor((y1 - (y - hy)) / res + 0.5))
        c0, c1 = max(c0, 0), min(max(c1, c0 + 1), nx)
        r0, r1 = max(r0, 0), min(max(r1, r0 + 1), ny)
        if c0 < nx and r0 < ny and c1 > 0 and r1 > 0:
            m[r0:r1, c0:c1] = True
    return m


def feds_emulate(shape, x0, y1, res, steps, params=None):
    """steps: list of dicts with arrays x, y, ds, dt (km). Returns cumulative masks [T, ny, nx]."""
    P = dict(FEDS_PARAMS); P.update(params or {})
    U = np.zeros(shape, bool)
    out = []
    st = disk(int(P['close_cells']))
    prev = np.zeros(shape, bool)
    for s in steps:
        if len(s['x']):
            U |= stamp_pixels(shape, x0, y1, res, s['x'], s['y'], s['ds'], s['dt'], P['shrink'])
        M = ndi.binary_closing(np.pad(U, 8), structure=st)[8:-8, 8:-8] if U.any() else U.copy()
        M = ndi.binary_fill_holes(M) | prev
        out.append(M)
        prev = M
    return np.array(out) if out else np.zeros((0,) + shape, bool)


# ---------------------------------------------------------------------------------------------------------------
def _moments(m, res_km):
    r, c = np.nonzero(m)
    if len(r) < 3:
        return np.nan, np.nan, 1.0
    x = c * res_km; y = -r * res_km
    cx, cy = x.mean(), y.mean()
    C = np.cov(np.vstack([x, y])) + np.eye(2) * (res_km ** 2) / 12
    ev = np.sort(np.linalg.eigvalsh(C))
    return cx, cy, float(np.sqrt(ev[1] / max(ev[0], 1e-9)))


def seq_features(masks, dets, day, wx, res=RES, wind_hidden=False, gamma_deg=0.0):
    """masks: [n, ny, nx] cumulative bool; dets: list of dicts with x, y arrays in grid pixel coords (col, row floats)
    and frp; day: [n] bool; wx: list of dicts (gridMET daily) or None. Returns [T_STEPS, len(SEQ_NAMES)] float32."""
    F = np.zeros((T_STEPS, len(SEQ_NAMES)), np.float32)
    n = min(len(masks), T_STEPS)
    rk = res / 1000.0
    cell_km2 = rk * rk
    prevA = 0.0
    prev = np.zeros(masks.shape[1:], bool)
    pcx = pcy = None
    for k in range(n):
        m = masks[k]
        A = float(m.sum()) * cell_km2
        f = F[k]
        f[0] = 1.0
        f[1] = float(day[k])
        f[2] = np.log10(A + 0.01)
        f[3] = np.log10(A + 0.01) - np.log10(prevA + 0.01) if k > 0 else 0.0
        f[4] = np.sqrt(A / np.pi) - np.sqrt(prevA / np.pi)
        d = dets[k]
        N = len(d['frp'])
        f[5] = np.log10(1 + N)
        f[6] = np.log10(1 + float(np.sum(d['frp'])))
        f[7] = np.log10(1 + float(np.mean(d['frp']))) if N else 0.0
        if N and k > 0:
            rr = np.clip(np.round(d['row']).astype(int), 0, m.shape[0] - 1)
            cc = np.clip(np.round(d['col']).astype(int), 0, m.shape[1] - 1)
            f[8] = float(np.mean(~prev[rr, cc]))
        elif N:
            f[8] = 1.0
        cx, cy, el = _moments(m, rk)
        f[9] = np.log(el) if np.isfinite(el) else 0.0
        new = m & ~prev
        if k > 0 and new.sum() >= 2 and pcx is not None:
            nx_, ny_, _ = _moments(new, rk)
            if np.isfinite(nx_):
                g = np.array([nx_ - pcx, ny_ - pcy])
                f[10], f[11] = g / 10.0
        if np.isfinite(cx):
            pcx, pcy = cx, cy
        w = wx[k] if wx is not None else None
        if w is not None:
            f[12] = w['tmmx'] / 30.0
            f[13] = w['rmin'] / 100.0
            f[14] = w['rmax'] / 100.0
            th = np.radians(w['th'] + gamma_deg)
            f[15] = w['vs'] / 10.0
            f[16] = w['vs'] * np.sin(th) / 10.0
            f[17] = w['vs'] * np.cos(th) / 10.0
            f[18] = w['fm100'] / 20.0
            f[19] = w['fm1000'] / 20.0
            f[20] = w['erc'] / 100.0
            f[21] = w['bi'] / 100.0
            f[22] = np.log1p(max(w['pr'], 0.0))
            f[23] = w['vpd'] / 3.0
        if wind_hidden:
            f[WIND_COLS] = 0.0
            f[24] = 1.0
        prevA = A
        prev = m
    return F


def static_features(L, area_mask, doy, n_steps, res=RES):
    """L: dict of layers on the same grid as area_mask (fuel, cc, ch, cbh, cbd, slope, elev). area_mask: cells
    within 2 km of the window's burned area."""
    S = np.zeros(len(STAT_NAMES), np.float32)
    sel = area_mask & (L['fuel'] > 0)
    if sel.sum() == 0:
        sel = area_mask
    fg = fuel_group_masks(L['fuel'])
    for i, g in enumerate(['GR', 'GS', 'SH', 'TU', 'TL', 'SB', 'NB']):
        S[i] = float(fg[g][sel].mean()) if sel.any() else 0.0
    forest = sel & (fg['TU'] | fg['TL'] | fg['SB']) & (L['cc'] > 0)
    if forest.sum() > 0:
        S[7] = float(L['cc'][forest].mean()) / 100.0
        S[8] = float(L['ch'][forest].mean()) / 30.0
        S[9] = float(L['cbh'][forest].mean()) / 10.0
        S[10] = float(L['cbd'][forest].mean()) * 10.0
    sl = L['slope'][sel] if sel.any() else np.zeros(1)
    S[11] = float(sl.mean()) / 30.0
    S[12] = float(np.percentile(sl, 90)) / 45.0
    S[13] = float(L['elev'][sel].mean()) / 3000.0 if sel.any() else 0.0
    S[14] = np.sin(2 * np.pi * doy / 365.25)
    S[15] = np.cos(2 * np.pi * doy / 365.25)
    S[16] = n_steps / T_STEPS
    return S


def area_mask_from(masks, res=RES, buf_m=2000.0):
    last = masks[min(len(masks), T_STEPS) - 1] if len(masks) else None
    if last is None or not last.any():
        return np.zeros(masks.shape[1:], bool)
    r = int(round(buf_m / res))
    d = ndi.distance_transform_edt(~last)
    return d <= r


def coarsen_layers(L, k=2):
    """90 m contract layers -> k x coarser: fuel = most common burnable code in the block (non-burnable if more than
    half the block is non-burnable), canopy/terrain means; aspect via vector mean."""
    fuel = L['fuel']
    ny, nx = (fuel.shape[0] // k) * k, (fuel.shape[1] // k) * k
    def blocks(a):
        return a[:ny, :nx].reshape(ny // k, k, nx // k, k).transpose(0, 2, 1, 3).reshape(ny // k, nx // k, k * k)
    fb = blocks(fuel)
    nb = (fb >= 91) & (fb <= 99) | (fb <= 0)
    out = {}
    # mode of burnable codes: pick the code at the first burnable position after sorting by frequency
    fsort = np.sort(np.where(nb, 9999, fb), axis=2)
    # simple mode via counts of each of the k*k entries
    cnt = (fsort[:, :, :, None] == fsort[:, :, None, :]).sum(axis=3)
    cnt = np.where(fsort == 9999, -1, cnt)
    mode = np.take_along_axis(fsort, cnt.argmax(axis=2)[:, :, None], axis=2)[:, :, 0]
    nbfrac = nb.mean(axis=2)
    nbcode = np.take_along_axis(fb, np.argmax(nb, axis=2)[:, :, None], axis=2)[:, :, 0]
    out['fuel'] = np.where(nbfrac > 0.5, np.where(nbcode > 0, nbcode, 99), mode).astype(np.int64)
    for key in ('cc', 'ch', 'cbh', 'cbd', 'elev', 'slope'):
        if key in L:
            out[key] = blocks(np.asarray(L[key], np.float32)).mean(axis=2).astype(np.float32)
    if 'aspect' in L:
        a = np.radians(np.asarray(L['aspect'], np.float32))
        out['aspect'] = (np.degrees(np.arctan2(blocks(np.sin(a)).mean(2), blocks(np.cos(a)).mean(2))) % 360).astype(np.float32)
    return out
