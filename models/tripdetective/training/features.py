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
                  # 3 adds 16 classical per-trip numbers (scalars); 4 (Phase D) = 3 + a
                  # source-power channel + the road-load regression's normal equations per window (pstats), which the
                  # network pools over a car's trips and solves (the physics-regression head)


def nin(feat=None):
    f = feat or FEAT
    return NIN + (3 if f >= 2 else 0) + (1 if f >= 4 else 0)


def nsc(feat=None):
    f = feat or FEAT
    return NSC + NPS if f >= 4 else (NSC if f >= 3 else 0)


def net_input(raw, fl, Lw=L, feat=None, sc=None):
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
    if feat >= 4:
        x[22, :n] = src_power(r, fl) / 20000.0 * va                    # wheel-equivalent source power, W / 20 kW
    if sc is not None:
        pass                                          # cached classical numbers (training)
    elif feat >= 4:
        sc = np.r_[scalars(raw[:, :n], fl), pstats(raw[:, :n], fl)].astype(np.float32)
    elif feat >= 3:
        sc = scalars(raw[:, :n], fl)
    else:
        sc = np.zeros(NSC, np.float32)
    return x, x[13], sc


# ---------------- trip-level classical estimates (feature version 3) ----------------
NSC = 16
NPS = 15          # pstats: normal equations of the road-load regression (10 unique X'X, 4 X'y, y'y), feature version 4
ETA_FUEL, ETA_BATT = 0.30, 0.90      # nominal source-to-wheel efficiencies that put fuel and battery power on one scale
XS = np.array([10.0, 100.0, 600.0, 1.0])   # column scales of the regressors (O(1) numbers); y in units of 20 kW
YS = 20000.0
IU = np.triu_indices(4)


def src_power(r, fl):
    """wheel-equivalent source power [W]: 0.30 x fuel chemical power + 0.90 x HV battery power (0 if neither logged)"""
    P = np.zeros(r.shape[1])
    if fl[0] > 0.5: P = P + ETA_FUEL * r[5].astype(np.float64) * 42600.0
    if fl[3] > 0.5: P = P + ETA_BATT * r[7].astype(np.float64) * 1000.0
    return P


def pstats(raw, fl):
    """per-window normal equations of   P_src = b0 v (1.03 a + g grade) + b1 g v + b2 0.5 rho v_air|v_air| v + b3
    on traction seconds (5 s smoothing, v > 3 m/s, a > -0.3 m/s2, P > 0), regressors scaled by XS, P by 20 kW, all
    divided by 1024. Summed over a car's windows and solved at car level (the network's physics-regression head)."""
    r = raw.astype(np.float64); n = r.shape[1]; out = np.zeros(NPS, np.float32)
    if fl[0] < 0.5 and fl[3] < 0.5: return out
    v = r[0]; ok = r[10] > 0.5
    a = np.zeros(n)
    if n > 2: a[1:-1] = (v[2:] - v[:-2]) / 2
    g = r[1] / 100; rho = r[4]; vw = v + r[3]
    Ps, vs, as_, gs = _smooth(src_power(r, fl)), _smooth(v), _smooth(a), _smooth(g)
    m = ok & (vs > 3) & (as_ > -0.3) & (Ps > 0)
    if m.sum() < 10: return out
    X = np.stack([vs * (1.03 * as_ + 9.81 * gs), 9.81 * vs, 0.5 * rho * vw * np.abs(vw) * vs, np.ones(n)], 1)[m] / XS
    y = Ps[m] / YS
    S = X.T @ X
    out[:10] = S[IU] / 1024; out[10:14] = (X.T @ y) / 1024; out[14] = (y @ y) / 1024
    return out


def psolve(ps, lam=1e-3):
    """pooled pstats (sum over windows) -> coefficients b (physical units: b0 kg-equivalent, b1 N/kg-ish, b2 m2, b3 W),
    R2, rows. numpy twin of model.TDNet.phys_head."""
    S = np.zeros((4, 4)); S[IU] = ps[:10]; S = S + S.T - np.diag(np.diag(S))
    rr = ps[10:14].astype(np.float64); yy = float(ps[14]); nrow = S[3, 3]
    if nrow * 1024 < 30: return None
    A = S + lam * np.diag(np.diag(S)) + 1e-9 * np.eye(4)
    b = np.linalg.solve(A, rr)
    sse = yy - 2 * b @ rr + b @ S @ b; sst = yy - rr[3] ** 2 / max(nrow, 1e-9)
    return dict(b=b * YS / XS, r2=1 - sse / max(sst, 1e-12), rows=nrow * 1024)


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


# ---------------- engine-stop constraint (v2.1, 15 Nov 2026) ----------------
# Physics: a conventional engine is geared to the wheels; it cannot stop while the car moves at more than 5 m/s. So a
# logged rpm under 100 at v > 5 m/s on more than OFF_TAU of the moving seconds rules out 'conv'. OFF_TAU is set from
# simulation only: over the 5,814 conventional trips of bank p2_train_v4 (OBD emulator with dropouts and holds) the
# per-trip share never exceeds 0.033 (99.9th percentile 0.018), so 0.05 never fires on a simulated conventional car.
OFF_TAU = 0.05
OFF_MIN_S = 30


def offstop_counts(x):
    """(engine-off moving seconds, moving seconds) of one network input x (nin, Lw); (0, 0) when rpm is not logged.
    Reads only the input channels (v/15, rpm x valid, valid, in-trip, has_rpm), so the browser computes it the same way."""
    if x[16, 0] < 0.5: return 0, 0
    mv = (x[0] > 5.0 / 15.0) & (x[12] > 0.5) & (x[13] > 0.5)
    return int((mv & (x[8] < 0.1)).sum()), int(mv.sum())


def offstop_fires(counts):
    """counts: list of (k, n) of one pooled group -> True when the engine stopped while moving beyond OFF_TAU"""
    K = sum(c[0] for c in counts); N = sum(c[1] for c in counts)
    return N >= OFF_MIN_S and K > OFF_TAU * N


CONV_STRATS = ('conv_rule', 'conv_opt', 'conv_stopstart')


def apply_offstop(arch, strat, fires, archs, strats):
    """zero 'conv' (and the conventional strategies) where the constraint fires, renormalise. arch (G, 7), strat (G, 14)"""
    arch = arch.copy(); strat = strat.copy()
    ic = archs.index('conv'); isc = [strats.index(k) for k in CONV_STRATS]
    for g in np.where(fires)[0]:
        arch[g, ic] = 0.0; arch[g] /= max(arch[g].sum(), 1e-12)
        strat[g, isc] = 0.0; strat[g] /= max(strat[g].sum(), 1e-12)
    return arch, strat
