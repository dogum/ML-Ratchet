"""FireCast v1.0: next-day wildfire spread probability on a 90 m grid (numpy + scipy only).

    import firecast as fc
    net = fc.FireCastNP()                # firecast_weights.npz next to this file
    prob = fc.predict(net, day)          # P(cell burned by the next afternoon VIIRS overpass), calibrated on dev
    fc.nextday(day['now'], day['nxt'], prob, fc.CELL_KM2, thr=net.thr, window=fc.score_window(day['now']))

`day` is a dict (see README.md): now (bool HxW burned area at the issue overpass, 90 m, north up), prev (bool or None,
24 h earlier), fl (bool or None: FEDS active fire line), pix_pm / pix_am (VIIRS detections: row, col, ps, pt [km],
frp [MW] at the issue and the previous night overpass), land (optional LANDFIRE dict: fg, slope, aspect/2, cc),
wx (16 floats from feature_vector(gridMET day0, gridMET day1, grid_convergence_deg)), has_prev, first.
Feature code is generated from the training source (src/features.py, src/wxfeat.py); metrics from firedata.metrics.
"""
import numpy as np
from scipy import ndimage

CELL_KM2 = 90.0 * 90.0 / 1e6


RES = 90.0
SPATIAL = ['now', 'grew', 'fl', 'pix_pm', 'pix_am', 'd_burn', 'd_fl', 'd_pix', 'along', 'across', 'inwin', 'nonburn', 'GR', 'GS', 'SH', 'TU', 'TL', 'slope', 'up_e', 'up_n', 'cc', 'land_ok']
N_SPATIAL = len(SPATIAL)
N_FIRE = 6
N_WX = 16
VEC_PAIRS = [('off_e', 'off_n'), ('up_e', 'up_n')]
WX_VEC_IDX = [(0, 1), (8, 9)]


# ---- features (src/features.py) ----

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
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(max(r1, r0 + 1), ny), min(max(c1, c0 + 1), nx)
        if r0 >= ny or c0 >= nx or r1 <= 0 or c1 <= 0:
            continue
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


# ---- weather features (src/wxfeat.py) ----

DAY_KEYS = ['vs', 'th', 'rmin', 'tmmx', 'vpd', 'pr', 'fm100']


def feature_vector(d0, d1, gamma_deg):
    """Network scalar weather inputs from two gridMET-like day rows (dict-like with DAY_KEYS). gamma_deg = grid
    convergence (grid = true + gamma). Spread-toward vector in grid coords: (east, north)."""
    f = []
    for d in (d0, d1):
        vs = float(d['vs']) if np.isfinite(d['vs']) else 3.0
        th = float(d['th']) if np.isfinite(d['th']) else 0.0
        tow = np.radians(th + 180.0 + gamma_deg)
        f += [vs * np.sin(tow) / 5.0, vs * np.cos(tow) / 5.0, vs / 5.0,
              (_nz(d['rmin'], 20.0) - 20.0) / 15.0, (_nz(d['tmmx'], 28.0) - 28.0) / 8.0, (_nz(d['vpd'], 2.5) - 2.5) / 1.5,
              np.log1p(max(_nz(d['pr'], 0.0), 0.0)), (_nz(d['fm100'], 9.0) - 9.0) / 4.0]
    return np.array(f, np.float32)


def _nz(v, default):
    try:
        v = float(v)
    except Exception:
        return default
    return v if np.isfinite(v) else default


# ---- network (src/firecast_np.py; BatchNorm folded) ----

LAYERS = ['stem', 'e1a', 'e1b', 'e2a', 'e2b', 'e3a', 'e3b', 'e4a', 'e4b', 'd3', 'd2', 'd1', 'd0']


def conv3x3(x, W, b, stride=1):
    C, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    Ho, Wo = (H + 2 - 3) // stride + 1, (Wd + 2 - 3) // stride + 1
    out = np.zeros((W.shape[0], Ho * Wo), np.float32)
    for dy in range(3):
        for dx in range(3):
            xs = xp[:, dy:dy + stride * (Ho - 1) + 1:stride, dx:dx + stride * (Wo - 1) + 1:stride]
            out += W[:, :, dy, dx] @ xs.reshape(C, -1)
    out += b[:, None]
    return out.reshape(W.shape[0], Ho, Wo)


def relu(x):
    return np.maximum(x, 0, out=x)


def pool2(x):
    C, H, W = x.shape
    return x[:, :H // 2 * 2, :W // 2 * 2].reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))


def up2(x):
    return x.repeat(2, axis=1).repeat(2, axis=2)


class FireCastNP:
    def __init__(self, path=None):
        if path is None:   # the released weights next to this file
            import os
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'firecast_weights.npz')
        with np.load(path) as z:
            self.p = {k: z[k].astype(np.float32) for k in z.files}
        self.cal_T = float(self.p.get('cal_T', np.array(1.0)))
        self.cal_b = float(self.p.get('cal_b', np.array(0.0)))
        self.thr = float(self.p.get('thr', np.array(0.5)))

    def c(self, name, x, stride=1):
        return relu(conv3x3(x, self.p[name + '.w'], self.p[name + '.b'], stride))

    def forward(self, X, s):
        p = self.p
        X = np.asarray(X, np.float32)
        H, W = X.shape[1:]
        Hp, Wp = int(np.ceil(H / 16) * 16), int(np.ceil(W / 16) * 16)
        Xp = np.zeros((X.shape[0], Hp, Wp), np.float32)
        Xp[:, :H, :W] = X
        h = np.maximum(p['mlp0.w'] @ s + p['mlp0.b'], 0)
        h = np.maximum(p['mlp1.w'] @ h + p['mlp1.b'], 0)
        fb = [(p[f'film{i}.w'] @ h + p[f'film{i}.b'])[:, None, None] for i in range(9)]
        x0 = self.c('stem', Xp) + fb[0]
        x1 = self.c('e1b', self.c('e1a', x0, 2)) + fb[1]
        x2 = self.c('e2b', self.c('e2a', pool2(x1))) + fb[2]
        x3 = self.c('e3b', self.c('e3a', pool2(x2))) + fb[3]
        x4 = self.c('e4b', self.c('e4a', pool2(x3))) + fb[4]
        y = self.c('d3', np.concatenate([up2(x4), x3])) + fb[5]
        y = self.c('d2', np.concatenate([up2(y), x2])) + fb[6]
        y = self.c('d1', np.concatenate([up2(y), x1])) + fb[7]
        y = self.c('d0', np.concatenate([up2(y), x0])) + fb[8]
        o = np.tensordot(p['head.w'], y, axes=(0, 0)) + p['head.b'] + float(p['head_s.w'] @ h + p['head_s.b'][0])
        return o[:H, :W]

    def prob(self, X, s):
        return 1.0 / (1.0 + np.exp(-(self.forward(X, s) / self.cal_T + self.cal_b)))


# ---- scoring (firedata.metrics) ----

def average_precision(y, s):
    y = np.asarray(y, bool).ravel(); s = np.asarray(s, np.float64).ravel()
    P = y.sum()
    if P == 0: return np.nan
    o = np.argsort(-s, kind='mergesort'); y = y[o]; s = s[o]
    # group ties: precision/recall evaluated at distinct thresholds
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]; fp = (distinct + 1) - tp
    prec = tp / (tp + fp); rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def f1(y, yhat):
    y = np.asarray(y, bool); yhat = np.asarray(yhat, bool)
    tp = (y & yhat).sum(); fp = (~y & yhat).sum(); fn = (y & ~yhat).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan


def dice(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); d = a.sum() + b.sum()
    return float(2 * (a & b).sum() / d) if d else np.nan


def nextday(burned_now, burned_next, prob, cell_km2, thr=0.5, window=None):
    """Score one fire-day. burned_now/next: bool grids; prob: predicted P(burned by next) for every cell (values on
    already-burned cells are ignored; they count as burned). window: optional bool mask of scored cells."""
    now = np.asarray(burned_now, bool); nxt = np.asarray(burned_next, bool) | now
    W = np.ones_like(now) if window is None else np.asarray(window, bool)
    cand = W & ~now
    y = nxt[cand]; s = np.asarray(prob, np.float64)[cand]
    pred_new = s >= thr
    total_pred = now.copy(); total_pred[cand] = pred_new
    new_obs = y.sum(); new_pred = pred_new.sum()
    return dict(ap=average_precision(y, s), f1=f1(y, pred_new), dice_total=dice(total_pred & W, nxt & W),
                dice_new=dice(pred_new, y) if (new_obs + new_pred) else np.nan,
                new_obs_km2=float(new_obs * cell_km2), new_pred_km2=float(new_pred * cell_km2),
                area_err_km2=float((new_pred - new_obs) * cell_km2),
                log10_area_ratio=float(np.log10((new_pred + 1) / (new_obs + 1))),
                n_cand=int(cand.sum()), prevalence=float(y.mean()) if len(y) else np.nan)


def score_window(burned_now, res_m=90.0, km=10.0, dist=None):
    """Standard scored region: cells within `km` of the burned area at issue time (the fire grid itself is built from
    the FINAL perimeter + 30 %, so scoring the whole grid would reward knowing where the fire ends)."""
    from scipy import ndimage
    now = np.asarray(burned_now, bool)
    if dist is None: dist = ndimage.distance_transform_edt(~now, sampling=res_m)
    return dist <= km * 1000.0




def predict(net, day, land=True):
    """Calibrated next-day burn probability for every cell of the day's grid."""
    X, s = make_inputs(day['now'], day.get('prev'), day.get('fl'), day.get('pix_pm'), day.get('pix_am'),
                       day.get('land') if land else None, day['wx'], bool(day.get('has_prev', False)),
                       bool(day.get('first', False)))
    return net.prob(X, s)
