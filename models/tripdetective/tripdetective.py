"""TripDetective: from a vehicle trip log alone, infer what the vehicle is and where its energy went, with error bars.
numpy + scipy only. Weights: tripdetective_weights.npz (BatchNorm folded into the convolutions).

    import tripdetective as TD
    td = TD.TripDetective()
    car = td.read_car([trip1, trip2, ...])      # each trip: dict of 1 Hz arrays (trip contract, SI units)
    car['architecture'], car['mass_kg'], car['displacement_L'], car['ledger_kJ_per_km'][0]

A trip dict needs v [m/s] and, if known: grade [rise/run], T_amb [C], wind_speed [m/s], wind_dir [deg FROM],
heading [deg], rho_air [kg/m3], and whatever the car logged: fuel_rate [g/s], engine_rpm, batt_current [A, + discharge],
batt_voltage [V], soc [0-1], aux_power [W]; optional valid (1 = logged). Missing channels: leave out or all-NaN.
Say whether fuel is estimated from the MAF / load (most OBD loggers) with fuel_estimated=True (default)."""
import json, os
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
L = 1024
RAW = ['v', 'grade', 'T_amb', 'headwind', 'rho_air', 'fuel_rate', 'rpm_k', 'batt_kW', 'soc', 'aux_kW', 'valid']
FLAGS = ['has_fuel', 'fuel_est', 'has_rpm', 'has_batt', 'has_aux']
NIN = 14 + len(FLAGS)
LED_SCALE = np.array([200., 150., 150., 150., 1500., 150.], np.float32)
VEDTYPE = {'conv': 'ICE', 'hev_powersplit': 'HEV', 'hev_parallel_p2': 'HEV', 'hev_series': 'HEV',
           'phev_powersplit': 'PHEV', 'phev_series': 'PHEV', 'bev': 'EV'}
CURB_OFFSET = 136.0


# ---------------- features (same as training/features.py) ----------------
def _arr(a, k, n, fill=np.nan):
    x = a.get(k)
    return np.full(n, fill, np.float64) if x is None else np.asarray(x, np.float64)


def flags_from_trip(a, fuel_estimated=True):
    n = len(a['v'])
    fin = lambda k: a.get(k) is not None and float(np.isfinite(np.asarray(a[k], float)).mean()) > 0.5
    return dict(has_fuel=fin('fuel_rate'), fuel_est=bool(fuel_estimated and fin('fuel_rate')), has_rpm=fin('engine_rpm'),
                has_batt=fin('batt_current') and fin('batt_voltage'), has_aux=fin('aux_power'))


def raw_from_log(a, flags):
    n = len(a['v'])
    v = np.nan_to_num(np.asarray(a['v'], np.float64))
    hw = _arr(a, 'wind_speed', n, 0.0) * np.cos(np.deg2rad(_arr(a, 'wind_dir', n, 0.0) - _arr(a, 'heading', n, 0.0)))
    valid = _arr(a, 'valid', n, 1.0)
    out = np.zeros((len(RAW), n), np.float32)
    out[0] = v; out[1] = np.nan_to_num(_arr(a, 'grade', n, 0.0)) * 100
    out[2] = np.nan_to_num(_arr(a, 'T_amb', n, 10.0), nan=10.0); out[3] = np.nan_to_num(hw)
    out[4] = np.nan_to_num(_arr(a, 'rho_air', n, 1.22), nan=1.22)
    bad = np.zeros(n, bool)
    def put(i, x, on):
        nonlocal bad
        if not on: return
        b = ~np.isfinite(x); out[i] = np.where(b, 0.0, x); bad |= b
    put(5, _arr(a, 'fuel_rate', n), flags['has_fuel'])
    put(6, _arr(a, 'engine_rpm', n) * 1e-3, flags['has_rpm'])
    if flags['has_batt']:
        put(7, _arr(a, 'batt_current', n) * _arr(a, 'batt_voltage', n) / 1000, True)
        s = _arr(a, 'soc', n); out[8] = np.where(np.isfinite(s), s, np.nanmedian(s) if np.isfinite(s).any() else 0.5)
    if flags['has_aux']:
        x = _arr(a, 'aux_power', n) * 1e-3; out[9] = np.where(np.isfinite(x), x, 0.0)
    out[10] = (valid > 0.5) & ~bad
    return out


def net_input(raw, fl, Lw=L, feat=1):
    n = min(raw.shape[1], Lw); r = raw[:, :n].astype(np.float32)
    x = np.zeros((NIN + (3 if feat >= 2 else 0), Lw), np.float32); v = r[0]; a = np.zeros(n, np.float32)
    if n > 2: a[1:-1] = (v[2:] - v[:-2]) / 2
    va = r[10]
    x[0, :n] = v / 15; x[1, :n] = a / 1.5; x[2, :n] = r[1] / 3; x[3, :n] = (r[2] - 10) / 12; x[4, :n] = r[3] / 5
    x[5, :n] = (r[4] - 1.22) * 10; x[6, :n] = r[5] * va; x[7, :n] = np.log1p(np.maximum(r[5], 0) * 4) * va
    x[8, :n] = r[6] * va; x[9, :n] = r[7] / 20 * va; x[10, :n] = (r[8] - 0.5) * 2 * fl[3]; x[11, :n] = r[9] / 2
    x[12, :n] = va; x[13, :n] = 1.0; x[14:14 + len(FLAGS), :n] = fl[:, None]
    if feat >= 2:      # road-load physics channels: specific tractive power, aero power per m2 CdA, rolling power per kg per Crr
        va2 = v + r[3]
        x[19, :n] = v * (1.03 * a + 9.81 * r[1] / 100) / 10
        x[20, :n] = 0.5 * r[4] * va2 * np.abs(va2) * v / 2000
        x[21, :n] = 9.81 * v / 200
    return x, x[13], (scalars(raw[:, :n], fl) if feat >= 3 else np.zeros(NSC, np.float32))


def windows(n, Lw=L, min_len=120):
    if n <= Lw: return [(0, n)]
    k = int(np.ceil(n / Lw)); e = np.linspace(0, n, k + 1).astype(int)
    return [(a, b) for a, b in zip(e[:-1], e[1:]) if b - a >= min_len]


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


# ---------------- network ----------------
def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def conv1d(x, w, b, stride):
    """x (C, T), w (O, C, K), zero padding K//2"""
    O, C, K = w.shape; p = K // 2; T = x.shape[1]
    xp = np.pad(x, ((0, 0), (p, p)))
    To = (T + 2 * p - K) // stride + 1
    idx = np.arange(To)[:, None] * stride + np.arange(K)[None, :]
    cols = xp[:, idx]                                  # (C, To, K)
    return np.einsum('ctk,ock->ot', cols, w, optimize=True) + b[:, None]


class TripDetective:
    def __init__(self, path=None):
        z = np.load(path or os.path.join(HERE, 'tripdetective_weights.npz'))
        self.W = {k: z[k] for k in z.files}
        self.meta = json.loads(str(self.W['meta']))
        self.nb = int(self.W['n_blocks'])
        self.sigk = self.meta.get('sigk', {}); self.feat = int(self.meta.get('feat', 1))

    def encode(self, x, m, sc=None):
        W = self.W
        T = int(m.sum()); T = min(L, max(32, int(np.ceil(T / 32) * 32)))
        h = x[:, :T].astype(np.float64); mk = m[:T].astype(np.float64)
        for i in range(self.nb):
            s = int(W[f'c{i}_s'])
            h = gelu(conv1d(h, W[f'c{i}_w'], W[f'c{i}_b'], s))
            if i > 0:
                mk = np.maximum(mk[0::2], np.r_[mk[1::2], np.zeros(len(mk) % 2)][:len(mk[0::2])])
            h = h[:, :len(mk)] * mk
        n = max(mk.sum(), 1.0)
        mean = (h * mk).sum(1) / n
        mx = (h * mk + (mk - 1) * 1e4).max(1)
        dur = m.sum() / 1024.0
        f = np.r_[mean, mx, dur, np.log(dur + 1e-3)]
        if self.feat >= 3: f = np.r_[f, sc]
        f = gelu(W['trip0_w'] @ f + W['trip0_b'])
        return W['trip1_w'] @ f + W['trip1_b']

    def heads(self, E):
        W = self.W
        z = np.mean(E, 0)
        z = np.r_[z, np.log(len(E))]
        z = gelu(W['car0_w'] @ z + W['car0_b']); z = gelu(W['car1_w'] @ z + W['car1_b'])
        sm = lambda u: np.exp(u - u.max()) / np.exp(u - u.max()).sum()
        arch = sm(W['arch_w'] @ z + W['arch_b']); strat = sm(W['strat_w'] @ z + W['strat_b'])
        r = W['reg_w'] @ z + W['reg_b']; nr = len(r) // 2
        mu, sig = r[:nr], np.exp(np.clip(r[nr:], -6, 3))
        led = []
        for e in E:
            u = gelu(W['led0_w'] @ np.r_[e, z] + W['led0_b']); u = W['led1_w'] @ u + W['led1_b']
            led.append((u[:6] * LED_SCALE, np.exp(np.clip(u[6:], -6, 3)) * LED_SCALE))
        return dict(arch=arch, strat=strat, mu=mu, sig=sig, led=led)

    def predict_raw(self, raws, flags):
        """raws: list of (len(RAW), n) arrays; flags: list of flag vectors -> network units (one pooled car)"""
        E = [self.encode(*net_input(r, np.asarray(f, np.float32), feat=self.feat)) for r, f in zip(raws, flags)]
        h = self.heads(E)
        return dict(arch=h['arch'][None], mu=h['mu'][None], sig=h['sig'][None], led_mu=np.stack([l[0] for l in h['led']]))

    def read_car(self, trips, fuel_estimated=True):
        """trips: list of trip dicts (same car). Returns physical estimates with 1-sigma error bars."""
        raws, fls, tid = [], [], []
        for t, a in enumerate(trips):
            fl = flags_from_trip(a, fuel_estimated)
            raw = raw_from_log(a, fl)
            for s, e in windows(raw.shape[1]):
                raws.append(raw[:, s:e]); fls.append(np.array([float(fl[k]) for k in FLAGS], np.float32)); tid.append(t)
        E = [self.encode(*net_input(r, f, feat=self.feat)) for r, f in zip(raws, fls)]
        h = self.heads(E)
        A = self.meta['archs']; S = self.meta['strats']; mu, sg = h['mu'], h['sig']
        km = self.sigk.get('mass_car' if len(trips) > 1 else 'mass_trip', 1.0)
        kd = self.sigk.get('disp_car' if len(trips) > 1 else 'disp_trip', 1.0)
        m = 1800 * np.exp(mu[0]); d = 2.5 * np.exp(mu[1])
        p = {a: float(x) for a, x in zip(A, h['arch'])}
        typ = {}
        for a, x in p.items(): typ[VEDTYPE[a]] = typ.get(VEDTYPE[a], 0) + x
        # ledger per trip: average the windows of each trip, weighting by length
        led = []
        for t in range(len(trips)):
            js = [j for j, tt in enumerate(tid) if tt == t]
            if not js: led.append(None); continue
            w = np.array([raws[j].shape[1] for j in js], float); w /= w.sum()
            mu_l = sum(w[i] * h['led'][j][0] for i, j in enumerate(js)); sg_l = np.sqrt(sum((w[i] * h['led'][j][1]) ** 2 for i, j in enumerate(js)))
            led.append({k: (float(a), float(b)) for k, a, b in zip(self.meta['ledger'], mu_l, sg_l)})
        return dict(architecture=A[int(np.argmax(h['arch']))], arch_probs=p, type_probs=typ,
                    strategy=S[int(np.argmax(h['strat']))], strategy_probs={s: float(x) for s, x in zip(S, h['strat'])},
                    mass_kg=float(m), mass_kg_1sigma=(float(m * np.exp(-sg[0] * km)), float(m * np.exp(sg[0] * km))),
                    curb_kg=float(m - CURB_OFFSET),
                    displacement_L=float(d) if typ.get('EV', 0) < 0.5 else None,
                    displacement_L_1sigma=(float(d * np.exp(-sg[1] * kd)), float(d * np.exp(sg[1] * kd))),
                    CdA_m2=float(0.75 * np.exp(mu[2])), CdA_1sigma_rel=float(sg[2]), Crr=float(0.008 * np.exp(mu[3])), Crr_1sigma_rel=float(sg[3]),
                    ev_share_moving=float(mu[4]), engine_on_threshold_kW=float(np.expm1(mu[5] + 2)), max_ev_speed_mps=float((mu[6] + 1.5) * 10),
                    ledger_kJ_per_km=led, n_trips=len(trips), n_windows=len(raws))

    def read_trip(self, trip, fuel_estimated=True):
        return self.read_car([trip], fuel_estimated)
