"""FireCast input features: one code path for simulated and real fire-days (vendored into the release).

Raw fire-day record (full grid, 90 m):
    now, prev (bool; prev may be None), fl (bool fire line), pix_pm / pix_am (detection tables: row, col, ps, pt km,
    frp MW), land (optional dict: fg uint8 fuel group code, slope deg, aspect deg, cc %), wx (16 floats, see wxfeat),
    sc (fire scalars). The network input is built in two steps:
    1. full-grid fields: masks, pixel rasters, distance fields and the offset vector from the nearest fire-line cell
       (computed on the whole grid so crops and full windows agree);
    2. crop / dihedral augmentation, then the wind-aligned offsets (along / across the d1 spread vector).
Vector convention: (east, north) components in grid coordinates (row 0 = north).
"""
import numpy as np
from scipy import ndimage

RES = 90.0
FG_CODES = {'NB': 0, 'GR': 1, 'GS': 2, 'SH': 3, 'TU': 4, 'TL': 5, 'SB': 5, 'A_grass': 1, 'A_shrub': 3,
            'A_timber': 5, 'A_slash': 5}
SPATIAL = ['now', 'grew', 'fl', 'pix_pm', 'pix_am', 'd_burn', 'd_fl', 'd_pix', 'along', 'across', 'inwin',
           'nonburn', 'GR', 'GS', 'SH', 'TU', 'TL', 'slope', 'up_e', 'up_n', 'cc', 'land_ok']
N_SPATIAL = len(SPATIAL)
N_FIRE = 6


def fuel_group_code(fuel):
    """LANDFIRE FBFM40/13 numbers -> 0 NB, 1 GR, 2 GS, 3 SH, 4 TU, 5 TL/SB."""
    f = np.asarray(fuel)
    g = np.zeros(f.shape, np.uint8)
    g[((f >= 101) & (f <= 109)) | ((f >= 1) & (f <= 3))] = 1
    g[(f >= 121) & (f <= 124)] = 2
    g[((f >= 141) & (f <= 149)) | ((f >= 4) & (f <= 7))] = 3
    g[(f >= 161) & (f <= 165)] = 4
    g[((f >= 181) & (f <= 189)) | ((f >= 201) & (f <= 204)) | ((f >= 8) & (f <= 13))] = 5
    return g


def land_from_landscape(ls):
    """Compact landscape layers (uint8) from a firesim Landscape / dict of contract layers."""
    asp = np.asarray(ls['aspect'], float)
    return dict(fg=fuel_group_code(ls['fuel']), slope=np.clip(np.asarray(ls['slope'], float), 0, 89).astype(np.uint8),
                aspect=(np.where(asp < 0, 0, asp) / 2.0).astype(np.uint8),
                cc=np.clip(np.asarray(ls['cc'], float), 0, 100).astype(np.uint8))


def pixel_raster(det, shape, frp_key='frp'):
    """Max log1p(FRP) over each detection's footprint (ps km along columns, pt km along rows)."""
    out = np.zeros(shape, np.float32)
    if det is None or len(det['row']) == 0:
        return out
    ny, nx = shape
    for r, c, ps, pt, f in zip(det['row'], det['col'], det['ps'], det['pt'], det[frp_key]):
        hx = 0.5 * ps * 1000.0 / RES
        hy = 0.5 * pt * 1000.0 / RES
        r0, r1 = int(np.floor(r - hy + 0.5)), int(np.floor(r + hy + 0.5))
        c0, c1 = int(np.floor(c - hx + 0.5)), int(np.floor(c + hx + 0.5))
        # At least one cell, then drop footprints that miss the grid, then clip (PR #7 review, 21 Nov 2026: clipping
        # first pulled a footprint lying wholly off the north or west edge onto row or column 0).
        r1, c1 = max(r1, r0 + 1), max(c1, c0 + 1)
        if r0 >= ny or c0 >= nx or r1 <= 0 or c1 <= 0:
            continue
        r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, ny), min(c1, nx)
        v = np.log1p(max(float(f), 0.0))
        sub = out[r0:r1, c0:c1]
        np.maximum(sub, v, out=sub)
    return out


def full_fields(now, prev, fl, pix_pm, pix_am, land=None, valid=None):
    """Full-grid fields (dict of float32 arrays). valid: cells that exist (sim landscape interior / real grid)."""
    shape = now.shape
    now = now.astype(bool)
    F = {}
    F['now'] = now.astype(np.float32)
    F['grew'] = (now & ~prev).astype(np.float32) if prev is not None else np.zeros(shape, np.float32)
    fl = fl.astype(bool) & now if fl is not None else np.zeros(shape, bool)
    F['fl'] = fl.astype(np.float32)
    P = pixel_raster(pix_pm, shape)
    A = pixel_raster(pix_am, shape)
    F['pix_pm'] = P / 4.0
    F['pix_am'] = A / 4.0
    d_burn = ndimage.distance_transform_edt(~now, sampling=RES) if now.any() else np.full(shape, 1e5)
    F['d_burn'] = np.clip(d_burn / 1000.0, 0, 10) / 5.0
    src = fl if fl.any() else (now & ~ndimage.binary_erosion(now, border_value=0))
    if src.any():
        d_fl, (ir, ic) = ndimage.distance_transform_edt(~src, sampling=RES, return_indices=True)
        rr, cc = np.indices(shape)
        off_e = (cc - ic) * RES / 1000.0
        off_n = -(rr - ir) * RES / 1000.0
    else:
        d_fl = np.full(shape, 1e5)
        off_e = off_n = np.zeros(shape)
    F['d_fl'] = np.clip(d_fl / 1000.0, 0, 10) / 5.0
    pix = (P > 0) | (A > 0)
    d_pix = ndimage.distance_transform_edt(~pix, sampling=RES) if pix.any() else np.full(shape, 1e5)
    F['d_pix'] = np.clip(d_pix / 1000.0, 0, 10) / 5.0
    F['off_e'] = np.clip(off_e, -10, 10).astype(np.float32)
    F['off_n'] = np.clip(off_n, -10, 10).astype(np.float32)
    win = d_burn <= 10000.0
    if valid is not None:
        win &= valid
    F['inwin'] = win.astype(np.float32)
    if land is not None:
        fg = land['fg']
        F['nonburn'] = (fg == 0).astype(np.float32)
        for k, code in (('GR', 1), ('GS', 2), ('SH', 3), ('TU', 4), ('TL', 5)):
            F[k] = (fg == code).astype(np.float32)
        s = land['slope'].astype(np.float32)
        a = np.radians(land['aspect'].astype(np.float32) * 2.0)   # aspect = direction the slope faces (downhill)
        F['slope'] = s / 45.0
        F['up_e'] = -np.sin(np.radians(s)) * np.sin(a)             # uphill = opposite of aspect
        F['up_n'] = -np.sin(np.radians(s)) * np.cos(a)
        F['cc'] = land['cc'].astype(np.float32) / 100.0
        F['land_ok'] = np.ones(shape, np.float32)
    else:
        for k in ('nonburn', 'GR', 'GS', 'SH', 'TU', 'TL', 'slope', 'up_e', 'up_n', 'cc', 'land_ok'):
            F[k] = np.zeros(shape, np.float32)
    return F


VEC_PAIRS = [('off_e', 'off_n'), ('up_e', 'up_n')]
WX_VEC_IDX = [(0, 1), (8, 9)]       # d0 and d1 spread-toward vectors in the wx vector


def dihedral(arrs, wx, op):
    """Apply one of 8 dihedral ops to a dict of 2-D arrays (spatially + vector components) and to the wx vector.
    op bits: 1 = flip columns (E-W mirror), 2 = flip rows (N-S mirror), 4 = transpose (applied first)."""
    out = dict(arrs)
    wx = wx.copy()

    def vec(e, n):
        if op & 4:
            e, n = -n, -e
        if op & 1:
            e = -e
        if op & 2:
            n = -n
        return e, n
    for k, v in out.items():
        if op & 4:
            v = v.T
        if op & 1:
            v = v[:, ::-1]
        if op & 2:
            v = v[::-1, :]
        out[k] = np.ascontiguousarray(v)
    for ke, kn in VEC_PAIRS:
        if ke in out:
            out[ke], out[kn] = vec(out[ke], out[kn])
    for ie, inn in WX_VEC_IDX:
        wx[ie], wx[inn] = vec(wx[ie], wx[inn])
    return out, wx


def stack(F, wx):
    """Final channel stack (N_SPATIAL, H, W) from (cropped/augmented) fields and the wx vector (d1 wind)."""
    ue, un = float(wx[8]), float(wx[9])
    sp = np.hypot(ue, un)
    if sp > 1e-6:
        ue, un = ue / sp, un / sp
    else:
        ue, un = 0.0, 0.0
    along = F['off_e'] * ue + F['off_n'] * un
    across = np.abs(-F['off_e'] * un + F['off_n'] * ue)
    G = dict(F)
    G['along'] = np.clip(along, -10, 10) / 5.0
    G['across'] = np.clip(across, 0, 10) / 5.0
    return np.stack([G[k] for k in SPATIAL]).astype(np.float32)


def fire_scalars(area_km2, d_prev_m, has_prev, first_day, n_pix_pm, fl_frac):
    return np.array([np.log10(area_km2 + 0.01) / 2.0,
                     np.log1p(max(d_prev_m, 0.0) / 100.0) / 3.0 if has_prev else 0.0,
                     float(has_prev), float(first_day), np.log1p(n_pix_pm) / 5.0, fl_frac], np.float32)


def req_m(area_km2):
    return np.sqrt(max(area_km2, 0.0) * 1e6 / np.pi)


def make_inputs(now, prev, fl, pix_pm, pix_am, land, wx, has_prev, first, valid=None, crop=None, op=0):
    """Network inputs for one fire-day. Returns X (N_SPATIAL, H, W) float32, s (N_WX + N_FIRE) float32, and the
    (cropped) scored-candidate mask base 'inwin'. crop = (r0, r1, c0, c1) on the full grid, op = dihedral op."""
    now = np.asarray(now, bool)
    area = now.sum() * RES * RES / 1e6
    d_prev = (req_m(area) - req_m(prev.sum() * RES * RES / 1e6)) if (has_prev and prev is not None) else 0.0
    F = full_fields(now, prev if has_prev else None, fl, pix_pm, pix_am, land, valid)
    fl_b = F['fl'] > 0
    bnd = now & ~ndimage.binary_erosion(now, border_value=0)
    fl_frac = float(fl_b.sum()) / max(int(bnd.sum()), 1)
    npix = len(pix_pm['row']) if pix_pm is not None else 0
    if crop is not None:
        r0, r1, c0, c1 = crop
        F = {k: v[r0:r1, c0:c1] for k, v in F.items()}
    wx = np.asarray(wx, np.float32)
    if op:
        F, wx = dihedral(F, wx, op)
    X = stack(F, wx)
    s = np.concatenate([wx, fire_scalars(area, d_prev, has_prev, first, npix, min(fl_frac, 1.0))]).astype(np.float32)
    return X, s
