"""Route inputs for the simulation bank. Three sources (no labelled vehicle is used):
  1. VED train_noise trips (17 unlabelled cars, 1,092 real Ann Arbor routes with real lidar grade and ASOS weather),
  2. EPA/other dynamometer cycles, cut into random windows, given the grade profile and weather of a random real route,
  3. synthetic stop-and-go / arterial / highway routes with a smooth random grade profile.
All are fragments that may start and end moving, like VED trips."""
from common import *
import pickle

POOL = f'{ROOT}/data/route_pool.pkl'
KEYS = ['v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface']
LIGHT_CYCLES = ['UDDS', 'HWFET', 'US06', 'SC03', 'NYCC', 'LA92', 'IM240', 'EUDC', 'JP_10_15', 'WLTC_3b']


def build_pool():
    import realdata as rd
    tn = []
    for m, a in rd.load_trips('train_noise', route_ok=True):
        n = len(a['v'])
        for s in range(0, n, L):
            e = min(n, s + L)
            if e - s < 120: continue
            tn.append({k: np.asarray(a[k][s:e], np.float32) for k in KEYS} | {'id': f"{m['trip_id']}:{s}"})
    cyc = {}
    for c in LIGHT_CYCLES:
        m, a = rd.cycles(c)
        cyc[c] = np.asarray(a['v'], np.float32)
    pickle.dump({'tn': tn, 'cyc': cyc}, open(POOL, 'wb'))
    print('pool', len(tn), 'train_noise fragments;', {k: len(v) for k, v in cyc.items()})


_P = None


def pool():
    global _P
    if _P is None:
        if not os.path.exists(POOL): build_pool()
        _P = pickle.load(open(POOL, 'rb'))
    return _P


def _weather_from(rng, n):
    """weather/heading/grade context from a random real route (by distance index)"""
    r = pool()['tn'][int(rng.integers(len(pool()['tn'])))]
    return r


def _grade_profile(rng, dist_m, sigma=None):
    """smooth random grade (rise/run) along distance: sum of a few random sinusoids, sigma ~1-2.5 %"""
    sigma = sigma if sigma is not None else rng.uniform(0.003, 0.025)
    g = np.zeros_like(dist_m)
    for _ in range(4):
        lam = rng.uniform(150, 1500)
        g += np.sin(2 * np.pi * dist_m / lam + rng.uniform(0, 2 * np.pi)) * rng.normal()
    g = g / (np.std(g) + 1e-9) * sigma
    return np.clip(g, -0.12, 0.12)


def _ctx_arrays(rng, v):
    """attach weather/heading from a random real route; grade either from real route (by distance), synthetic, or flat"""
    n = len(v); r = _weather_from(rng, n)
    d = np.cumsum(v)
    m = len(r['v']); dr = np.cumsum(r['v'])
    u = rng.random()
    if u < 0.45 and dr[-1] > 200:       # real grade by distance (wrap)
        grade = np.interp(d % dr[-1], dr, r['grade'])
    elif u < 0.85:
        grade = _grade_profile(rng, d)
    else:
        grade = np.zeros(n)
    elev = 260 + np.cumsum(grade * v)
    i = np.minimum(np.arange(n) * m // max(n, 1), m - 1)
    out = dict(v=v.astype(np.float32), grade=grade.astype(np.float32), elev=elev.astype(np.float32))
    for k in ['T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface']:
        out[k] = np.full(n, r[k][0], np.float32) if k == 'surface' else np.asarray(r[k][i], np.float32)
    # heading: a few straight legs with turns
    h = np.zeros(n); cur = rng.uniform(0, 360); t = 0
    while t < n:
        seg = int(rng.integers(30, 300)); h[t:t + seg] = cur; t += seg; cur = (cur + rng.choice([-90, 90, 180, 0, 45])) % 360
    out['heading'] = h.astype(np.float32)
    return out


def cycle_route(rng, maxlen=L):
    cyc = pool()['cyc']; name = list(cyc)[int(rng.integers(len(cyc)))]
    v = cyc[name]
    n = int(min(len(v), rng.integers(150, maxlen + 1)))
    s = int(rng.integers(0, len(v) - n + 1))
    vv = v[s:s + n] * rng.uniform(0.9, 1.1)          # mild speed scaling
    return _ctx_arrays(rng, vv), f'cycle:{name}'


def synthetic_route(rng, maxlen=L):
    """stop-and-go driving: cruise targets per segment, physically plausible accelerations, dwell at stops"""
    n = int(rng.integers(150, maxlen + 1))
    kind = rng.choice(['city', 'arterial', 'highway', 'mixed'], p=[0.35, 0.3, 0.15, 0.2])
    vmax = {'city': (8, 16), 'arterial': (12, 22), 'highway': (22, 34), 'mixed': (8, 32)}[kind]
    v = np.zeros(n); t = 0; cur = rng.uniform(0, vmax[1]) if rng.random() < 0.85 else 0.0
    acc = rng.uniform(0.8, 2.0); dec = rng.uniform(1.0, 2.5)
    while t < n:
        tgt = rng.uniform(*vmax); hold = int(rng.integers(10, 120 if kind != 'highway' else 400))
        while abs(cur - tgt) > 0.1 and t < n:
            step = acc * rng.uniform(0.5, 1.2) if tgt > cur else -dec * rng.uniform(0.4, 1.0)
            cur = min(cur + step, tgt) if step > 0 else max(cur + step, tgt); v[t] = cur; t += 1
        for _ in range(hold):
            if t >= n: break
            cur = max(0.0, cur + rng.normal(0, 0.25)); v[t] = cur; t += 1
        if rng.random() < {'city': 0.6, 'arterial': 0.4, 'highway': 0.05, 'mixed': 0.3}[kind]:
            while cur > 0 and t < n:
                cur = max(0.0, cur - dec * rng.uniform(0.5, 1.0)); v[t] = cur; t += 1
            dwell = int(rng.integers(3, 60))
            v[t:t + dwell] = 0; t += dwell; cur = 0.0
    v = np.convolve(v, np.ones(3) / 3, mode='same'); v[v < 0.05] = 0
    return _ctx_arrays(rng, v[:n]), f'synthetic:{kind}'


def real_route(rng):
    tn = pool()['tn']; r = tn[int(rng.integers(len(tn)))]
    return {k: r[k].copy() for k in KEYS}, f"tn:{r['id']}"


def draw_route(rng, p=(0.5, 0.25, 0.25)):
    u = rng.random()
    if u < p[0]: return real_route(rng)
    if u < p[0] + p[1]: return cycle_route(rng)
    return synthetic_route(rng)


def perturb_for_sim(route, rng):
    """the car experiences slightly different conditions from what the log/route data says:
    ASOS temperature +-1.5 C, wind at car height x0.4-0.9 with direction scatter, grade repeatability noise 0.3 %."""
    r = {k: np.array(v, copy=True) for k, v in route.items()}
    n = len(r['v'])
    r['T_amb'] = r['T_amb'] + rng.normal(0, 1.5)
    r['wind_speed'] = r['wind_speed'] * rng.uniform(0.4, 0.9)
    r['wind_dir'] = (r['wind_dir'] + rng.normal(0, 20)) % 360
    gn = np.convolve(rng.normal(0, 1, n + 40), np.ones(40) / np.sqrt(40), mode='same')[20:20 + n] * 0.003
    r['grade'] = (r['grade'] + gn).astype(np.float32)
    return r


if __name__ == '__main__':
    build_pool()
