"""Route inputs for training: synthetic drives (speed profile with stops, hills, heading, weather), real VED
train_noise routes (unlabelled cars: speed, lidar grade, ASOS weather), and dynamometer cycles.

Synthetic speed profiles: a driver follows a list of road segments (target speed, length, stop at the end, dwell),
accelerating with a speed-dependent limit (a_max (1 - v/v_lim)), braking at a comfortable decel so it stops at the
segment end, cruising with AR(1) speed noise. Grade: smoothed random grade field along distance (terrain RMS),
heading changes at intersections, weather drawn per trip."""
from common import *
import numpy as np

KINDS_LD = {'urban': 0.24, 'suburban': 0.22, 'highway': 0.16, 'mixed': 0.16, 'rural': 0.1, 'congested': 0.07, 'hilly': 0.05}
KINDS = {'car': KINDS_LD, 'suv': KINDS_LD, 'pickup': KINDS_LD, 'van': dict(KINDS_LD, delivery=0.1),
         'bus': {'bus': 0.7, 'bus_express': 0.15, 'urban': 0.15},
         'truck_md': {'delivery': 0.45, 'suburban': 0.2, 'highway': 0.2, 'rural': 0.15},
         'truck_hd': {'longhaul': 0.5, 'regional': 0.3, 'highway': 0.1, 'delivery': 0.1}}
KMH = 1 / 3.6


def _pick(rng, w):
    k = list(w); p = np.array([w[x] for x in k], float); p /= p.sum()
    return k[int(rng.choice(len(k), p=p))]


def _segments(rng, kind, hd):
    """list of (v_target m/s, length m, stop bool, dwell s)."""
    S = []
    def add(n, vlo, vhi, llo, lhi, pstop, dlo, dhi):
        for _ in range(n):
            S.append((rng.uniform(vlo, vhi) * KMH, rng.uniform(llo, lhi), rng.random() < pstop, rng.uniform(dlo, dhi)))
    if kind == 'urban': add(int(rng.integers(6, 40)), 25, 55, 120, 600, 0.5, 3, 60)
    elif kind == 'suburban': add(int(rng.integers(5, 25)), 40, 75, 300, 1500, 0.35, 3, 50)
    elif kind == 'highway':
        add(1, 40, 60, 300, 800, 0.0, 0, 0); add(int(rng.integers(2, 10)), 95, 130 if not hd else 105, 2000, 12000, 0.0, 0, 0)
        if rng.random() < 0.4: add(int(rng.integers(1, 4)), 20, 60, 500, 2000, 0.1, 5, 30)
        add(1, 40, 60, 300, 800, 1.0, 0, 5)
    elif kind == 'mixed':
        add(int(rng.integers(3, 10)), 25, 55, 150, 600, 0.45, 3, 50); add(int(rng.integers(1, 5)), 90, 125, 1500, 8000, 0.0, 0, 0)
        add(int(rng.integers(3, 10)), 25, 55, 150, 600, 0.45, 3, 50)
    elif kind == 'rural': add(int(rng.integers(3, 15)), 60, 100, 1000, 5000, 0.12, 2, 20)
    elif kind == 'hilly': add(int(rng.integers(3, 15)), 40, 90, 500, 3000, 0.15, 2, 20)
    elif kind == 'congested': add(int(rng.integers(10, 50)), 8, 35, 40, 300, 0.6, 2, 30)
    elif kind == 'bus':
        add(int(rng.integers(12, 60)), 25, 50, 200, 600, 0.85, 8, 45)
    elif kind == 'bus_express': add(int(rng.integers(6, 25)), 40, 75, 500, 2500, 0.6, 8, 40)
    elif kind == 'delivery': add(int(rng.integers(6, 30)), 25, 60, 200, 1500, 0.7, 20, 300)
    elif kind == 'longhaul':
        add(1, 40, 60, 500, 1500, 0.0, 0, 0); add(int(rng.integers(2, 6)), 85, 105, 10000, 40000, 0.0, 0, 0); add(1, 30, 50, 500, 1000, 1.0, 0, 0)
    elif kind == 'regional': add(int(rng.integers(4, 15)), 50, 95, 2000, 10000, 0.2, 5, 60)
    else: raise ValueError(kind)
    return S


def speed_profile(rng, kind, hd=False, max_s=3600):
    segs = _segments(rng, kind, hd)
    a0 = rng.uniform(1.2, 2.6) if not hd else rng.uniform(0.5, 1.1)
    vlim = rng.uniform(35, 50) if not hd else rng.uniform(25, 35)
    b = rng.uniform(1.0, 2.6) if not hd else rng.uniform(0.6, 1.4)
    sig = rng.uniform(0.2, 0.9); rho_ar = 0.97
    v, x, out = 0.0, 0.0, [0.0]
    if rng.random() < 0.3 and kind not in ('bus',):
        v = segs[0][0] * rng.uniform(0.5, 1.0); out = [v]          # start mid-drive sometimes
    noise = 0.0
    if rng.random() < 0.5:
        out += [0.0] * int(rng.integers(1, 30)); v = 0.0 if out[-1] == 0 else v
    for (vt, L, stop, dwell) in segs:
        xs = 0.0
        while xs < L and len(out) < max_s:
            noise = rho_ar * noise + rng.normal(0, sig * 0.243)        # stationary sd = sig
            tgt = max(0.3 * vt, vt + noise * min(1.0, vt / 10.0))
            rem = L - xs
            if stop and rem <= (v * v) / (2 * b) + v * 0.5:
                vn = max(0.0, v - b * rng.uniform(0.85, 1.1))
                if vn < 0.3:
                    vn = 0.0
            else:
                amax = a0 * max(0.15, 1 - v / vlim)
                vn = v + float(np.clip(tgt - v, -b * 0.6, amax))
            d = 0.5 * (v + vn)
            xs += d; v = vn; out.append(v)
            if stop and v == 0.0 and L - xs < 40.0:
                break
        if stop and len(out) < max_s:
            v = 0.0; out += [0.0] * int(dwell)
    v = np.array(out[:max_s], float)
    return np.clip(v, 0, None)


def grade_profile(rng, x, terrain_rms, corr_m, slope=0.0, gmax=0.10):
    """smoothed random grade field along distance x (m)."""
    L = max(x[-1], 10.0)
    k = max(1, int(corr_m / 10))
    n = max(int(L / 10) + 2, 6 * k + 3)
    g = rng.normal(0, 1, n)
    ker = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2); ker /= ker.sum()
    g = np.convolve(g, ker, mode='same')
    g = g / (g.std() + 1e-9) * terrain_rms + slope
    gx = np.clip(g, -gmax, gmax)
    return np.interp(x, np.arange(n) * 10.0, gx)


def weather(rng):
    T = float(rng.uniform(-25, 40) if rng.random() < 0.45 else np.clip(rng.normal(12, 11), -28, 42))
    ws = float(min(rng.gamma(2.0, 1.8), 15.0)); wd = float(rng.uniform(0, 360))
    elev0 = float(rng.uniform(0, 1600) if rng.random() < 0.3 else rng.uniform(0, 400))
    u = rng.random()
    surf = 4 if (T < 1 and u < 0.25) else (2 if u < 0.15 + (0.1 if T > 1 else 0) else 0)
    return T, ws, wd, elev0, surf


def rho_air(T, elev):
    p = 101325.0 * (1 - 2.25577e-5 * elev) ** 5.25588
    return p / (287.05 * (T + 273.15))


def synth_trip(rng, cls='car', kind=None):
    hd = cls in HD
    kind = kind or _pick(rng, KINDS[cls])
    v = speed_profile(rng, kind, hd, max_s=7200 if hd else 2400)
    if len(v) < 60:
        v = np.concatenate([v, np.zeros(60 - len(v))])
    x = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]))])
    terr = {'urban': (0.004, 0.025), 'suburban': (0.005, 0.03), 'highway': (0.003, 0.025), 'mixed': (0.004, 0.03),
            'rural': (0.008, 0.04), 'hilly': (0.03, 0.06), 'congested': (0.002, 0.02), 'bus': (0.003, 0.035),
            'bus_express': (0.003, 0.03), 'delivery': (0.003, 0.03), 'longhaul': (0.004, 0.035), 'regional': (0.005, 0.04)}[kind]
    rms = float(np.exp(rng.uniform(np.log(terr[0]), np.log(terr[1]))))
    slope = float(rng.normal(0, 0.004))
    grade = grade_profile(rng, x, rms, rng.uniform(150, 1200), slope, gmax=0.07 if hd else 0.12)
    elevr = np.concatenate([[0], np.cumsum(grade[1:] * 0.5 * (v[1:] + v[:-1]))])
    T, ws, wd, elev0, surf = weather(rng)
    N = len(v)
    head = np.zeros(N); h = rng.uniform(0, 360)
    stops = np.where((v[1:] < 0.3) & (v[:-1] >= 0.3))[0]
    turn_pts = set(stops.tolist()) | set(rng.integers(0, N, size=max(1, N // 300)).tolist())
    for k in range(N):
        if k in turn_pts and kind not in ('highway', 'longhaul'):
            h = (h + rng.choice([-90, 0, 90, 180], p=[0.35, 0.3, 0.3, 0.05])) % 360
        h = (h + rng.normal(0, 0.5)) % 360
        head[k] = h
    Tarr = T + np.linspace(0, rng.normal(0, 1.0), N)
    wsa = np.clip(ws * (1 + 0.15 * np.sin(np.arange(N) / rng.uniform(200, 900) + rng.uniform(0, 6))), 0, None)
    elev = elev0 + elevr
    surface = np.full(N, surf, np.int8)
    if surf == 0 and kind == 'highway' and rng.random() < 0.3: surface[:] = 1
    if kind in ('rural', 'hilly') and rng.random() < 0.06: surface[:] = 3
    trip = dict(t=np.arange(N, dtype=float), v=v, grade=grade, elev=elev, heading=head, T_amb=Tarr, wind_speed=wsa,
                wind_dir=np.full(N, wd), rho_air=rho_air(Tarr, elev), surface=surface)
    return {k: (a.astype(np.float32) if a.dtype != np.int8 else a) for k, a in trip.items()}, kind
