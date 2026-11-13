"""Trip log -> stored raw channels (physical units) -> network input. Shared by the bank, the real-data loader, the
release (vendored copy) and the browser port (web/tdet.js mirrors net_input)."""
import numpy as np

RAW = ['v', 'grade', 'T_amb', 'headwind', 'rho_air', 'fuel_rate', 'rpm_k', 'batt_kW', 'soc', 'aux_kW', 'valid']
FLAGS = ['has_fuel', 'fuel_est', 'has_rpm', 'has_batt', 'has_aux']
L = 1024


def raw_from_log(a, flags):
    """a: dict of 1 Hz contract arrays (logged). flags: dict of availability flags. returns (len(RAW), n) float32"""
    n = len(a['v'])
    v = np.nan_to_num(np.asarray(a['v'], np.float64))
    hw = np.asarray(a['wind_speed'], np.float64) * np.cos(np.deg2rad(np.asarray(a['wind_dir'], np.float64) - np.asarray(a['heading'], np.float64)))
    valid = np.asarray(a.get('valid', np.ones(n)), np.float64)
    def ch(x, on, scale=1.0):
        if not on: return np.zeros(n)
        x = np.asarray(x, np.float64) * scale
        bad = ~np.isfinite(x)
        return np.where(bad, 0.0, x), bad
    out = np.zeros((len(RAW), n), np.float32)
    out[0] = v
    out[1] = np.nan_to_num(np.asarray(a['grade'], np.float64)) * 100
    out[2] = np.nan_to_num(np.asarray(a['T_amb'], np.float64), nan=10.0)
    out[3] = np.nan_to_num(hw)
    out[4] = np.nan_to_num(np.asarray(a['rho_air'], np.float64), nan=1.22)
    bad_any = np.zeros(n, bool)
    if flags['has_fuel']:
        x, b = ch(a['fuel_rate'], True); out[5] = x; bad_any |= b
    if flags['has_rpm']:
        x, b = ch(a['engine_rpm'], True, 1e-3); out[6] = x; bad_any |= b
    if flags['has_batt']:
        P = np.asarray(a['batt_current'], np.float64) * np.asarray(a['batt_voltage'], np.float64) / 1000
        x, b = ch(P, True); out[7] = x; bad_any |= b
        s = np.asarray(a['soc'], np.float64)
        out[8] = np.where(np.isfinite(s), s, np.nanmedian(s) if np.isfinite(s).any() else 0.5)
    if flags['has_aux']:
        x, b = ch(a['aux_power'], True, 1e-3); out[9] = x
    out[10] = (valid > 0.5) & ~bad_any
    return out


def flags_vec(flags):
    return np.array([float(bool(flags[k])) for k in FLAGS], np.float32)


NIN = 14 + len(FLAGS)
FEAT = 1          # feature version: 2 adds three road-load physics channels (specific tractive, aero and rolling power)


def nin(feat=None):
    return NIN + (3 if (feat or FEAT) >= 2 else 0)


def net_input(raw, fl, Lw=L, feat=None):
    """raw: (len(RAW), n) -> (nin, Lw) float32 normalised, zero-padded; second output = time mask (Lw,)"""
    feat = feat or FEAT
    n = min(raw.shape[1], Lw)
    r = raw[:, :n].astype(np.float32)
    x = np.zeros((nin(feat), Lw), np.float32)
    v = r[0]
    a = np.zeros(n, np.float32)
    if n > 2: a[1:-1] = (v[2:] - v[:-2]) / 2
    va = r[10]
    x[0, :n] = v / 15
    x[1, :n] = a / 1.5
    x[2, :n] = r[1] / 3
    x[3, :n] = (r[2] - 10) / 12
    x[4, :n] = r[3] / 5
    x[5, :n] = (r[4] - 1.22) * 10
    x[6, :n] = r[5] * va
    x[7, :n] = np.log1p(np.maximum(r[5], 0) * 4) * va
    x[8, :n] = r[6] * va
    x[9, :n] = r[7] / 20 * va
    x[10, :n] = (r[8] - 0.5) * 2 * fl[3]
    x[11, :n] = r[9] / 2
    x[12, :n] = va
    x[13, :n] = 1.0                                   # inside the trip
    x[14:14 + len(FLAGS), :n] = fl[:, None]
    if feat >= 2:
        va2 = v + r[3]
        x[19, :n] = v * (1.03 * a + 9.81 * r[1] / 100) / 10          # specific tractive power, W/kg / 10
        x[20, :n] = 0.5 * r[4] * va2 * np.abs(va2) * v / 2000        # aero power per m2 of CdA, kW / 2
        x[21, :n] = 9.81 * v / 200                                    # rolling power per kg per unit Crr / 200
    sc = scalars(raw[:, :n], fl) if feat >= 3 else np.zeros(NSC, np.float32)
    return x, x[13], sc


# ---------------- trip-level classical estimates (feature version 3) ----------------
NSC = 16


def _smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode='same')


def _ridge(X, y):
    sc = np.sqrt((X ** 2).mean(0)) + 1e-9
    A = X / sc
    b = np.linalg.solve(A.T @ A + 1e-3 * len(y) * np.eye(X.shape[1]), A.T @ y)
    return b / sc


def scalars(raw, fl):
    """16 classical numbers per trip window (the same arithmetic in release/tripdetective.py and web/tdet.js):
    a least-squares fit of the logged source power to the road-load equation
        P_src = b0 v (1.03 a + g grade) + b1 g v + b2 0.5 rho v_air|v_air| v + b3      (b0 ~ mass x efficiency, ...)
    on traction seconds (5 s smoothing), its R2 and row count; idle fuel and rpm, high-load fuel, rpm per speed in
    cruise, engine-off share while moving, fuel and battery energy per km, battery power percentiles."""
    r = raw.astype(np.float64); n = r.shape[1]; out = np.zeros(NSC, np.float32)
    v = r[0]; ok = r[10] > 0.5
    a = np.zeros(n)
    if n > 2: a[1:-1] = (v[2:] - v[:-2]) / 2
    g = r[1] / 100; rho = r[4]; vw = v + r[3]
    P = np.zeros(n); have = False
    if fl[0] > 0.5: P = P + 0.25 * r[5] * 42600.0; have = True
    if fl[3] > 0.5: P = P + r[7] * 1000.0; have = True
    if have:
        Ps, vs, as_, gs = _smooth(P), _smooth(v), _smooth(a), _smooth(g)
        m = ok & (vs > 3) & (as_ > -0.3) & (Ps > 0)
        if m.sum() >= 30:
            X = np.stack([vs * (1.03 * as_ + 9.81 * gs), 9.81 * vs, 0.5 * rho * vw * np.abs(vw) * vs, np.ones(n)], 1)[m]
            y = Ps[m]; b = _ridge(X, y)
            res = y - X @ b; r2 = 1 - res.var() / max(y.var(), 1e-9)
            out[0] = np.arcsinh(b[0] / 500); out[1] = np.arcsinh(b[1] / 10); out[2] = np.arcsinh(b[2] / 0.3); out[3] = np.arcsinh(b[3] / 2000)
            out[4] = np.clip(r2, -1, 1); out[5] = np.log1p(m.sum()) / 7
    km = max((v * ok).sum() / 1000, 0.05)
    mv = ok & (v > 2)
    if fl[0] > 0.5:
        idle = ok & (v < 0.3) & ((r[6] > 0.3) if fl[2] > 0.5 else (r[5] > 0))
        if idle.sum() >= 5: out[6] = np.median(r[5][idle]) * 3
        if mv.sum() >= 10: out[8] = np.percentile(r[5][mv], 90) / 2
        out[12] = (r[5] * ok).sum() / km / 100
    if fl[2] > 0.5:
        idle = ok & (v < 0.3) & (r[6] > 0.3)
        if idle.sum() >= 5: out[7] = np.median(r[6][idle])
        cr = ok & (v > 11) & (v < 20) & (r[6] > 0.3)
        if cr.sum() >= 5: out[9] = np.median(r[6][cr] / v[cr]) * 10
        if mv.sum() >= 10: out[10] = np.mean(r[6][mv] < 0.1)
        out[11] = 1.0
    if fl[3] > 0.5:
        out[13] = (r[7] * ok).sum() / 3600 / km * 5
        if mv.sum() >= 10: out[14] = np.percentile(r[7][mv], 90) / 50; out[15] = np.percentile(r[7][mv], 10) / 50
    return out
