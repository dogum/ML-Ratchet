"""pipeline.py (numpy only): read NDBC historical DART files and turn them into detided 1-minute series after an origin time.
Detiding: least-squares fit of 8 tidal constituents (M2 S2 N2 K2 K1 O1 P1 Q1) plus a quadratic trend to all samples
(15-min, 1-min, 15-s) in [t0 - 3 d, t0 + 3 d] EXCLUDING [t0 - 10 min, t0 + 26 h]; the fit is subtracted.
Samples are then put on a 1-minute grid (t0 .. t0 + 24 h): each minute takes the mean of the samples within +-30 s;
minutes with no sample are masked. The same code serves real events and the quiet noise windows."""
import os, gzip, numpy as np, datetime as dt
DDIR = os.environ.get('DART_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dart'))
PER_H = {'M2': 12.4206012, 'S2': 12.0, 'N2': 12.65834751, 'K2': 11.96723606, 'K1': 23.93447213, 'O1': 25.81933871, 'P1': 24.06588766, 'Q1': 26.868350}
_cache = {}

def read_year(station, year):
    key = (station, year)
    if key in _cache: return _cache[key]
    fn = os.path.join(DDIR, f'{station}t{year}.txt.gz')
    if not os.path.exists(fn):
        _cache[key] = None; return None
    rows = []
    with gzip.open(fn, 'rt') as f:
        for line in f:
            if line[:1].isdigit(): rows.append(line)
    if not rows:
        _cache[key] = None; return None
    a = np.loadtxt(rows, dtype=np.float64, usecols=range(8), ndmin=2)
    days = (np.array([f'{int(r[0]):04d}-{int(r[1]):02d}-{int(r[2]):02d}' for r in a[:, :3]], 'datetime64[D]') - np.datetime64('1970-01-01', 'D')).astype(np.int64)
    t = days * 86400.0 + a[:, 3] * 3600 + a[:, 4] * 60 + a[:, 5]
    h = a[:, 7]; typ = a[:, 6].astype(int)
    ok = (h < 9000) & (h > 100)
    out = (t[ok], h[ok], typ[ok])
    if len(_cache) > 40: _cache.clear()
    _cache[key] = out
    return out

def read_span(station, t0, t1):
    y0 = dt.datetime.fromtimestamp(t0, dt.timezone.utc).year; y1 = dt.datetime.fromtimestamp(t1, dt.timezone.utc).year
    T, H, Y = [], [], []
    for y in range(y0, y1 + 1):
        r = read_year(station, y)
        if r is None: continue
        m = (r[0] >= t0) & (r[0] <= t1); T.append(r[0][m]); H.append(r[1][m]); Y.append(r[2][m])
    if not T: return None
    T, H, Y = np.concatenate(T), np.concatenate(H), np.concatenate(Y)
    o = np.argsort(T, kind='stable'); return T[o], H[o], Y[o]

def tide_design(t, tref):
    th = (t - tref) / 3600.0
    cols = [np.ones_like(th), th / 72, (th / 72) ** 2]
    for p in PER_H.values():
        w = 2 * np.pi / p; cols += [np.cos(w * th), np.sin(w * th)]
    return np.stack(cols, 1)

def detided_minutes(station, t0, hours=24, robust=True):
    """Return (eta[m], mask[bool], info) on the minute grid t0 + 60 k, k = 0..60*hours. eta in metres."""
    r = read_span(station, t0 - 3 * 86400, t0 + 3 * 86400)
    n = 60 * hours + 1
    if r is None: return None
    T, H, Y = r
    fit = (T < t0 - 600) | (T > t0 + 26 * 3600)
    if fit.sum() < 100 or (T[fit] < t0).sum() < 50 or (T[fit] > t0).sum() < 50: return None
    A = tide_design(T[fit], t0); w = np.ones(fit.sum())
    for it in range(3 if robust else 1):
        coef = np.linalg.lstsq(A * w[:, None], H[fit] * w, rcond=None)[0]
        res = H[fit] - A @ coef
        s = 1.4826 * np.median(np.abs(res)) + 1e-4
        w = 1.0 / np.maximum(1.0, np.abs(res) / (4 * s))   # Huber-like down-weighting of spikes
    win = (T >= t0 - 30) & (T <= t0 + hours * 3600 + 30)
    Tw, Rw = T[win], H[win] - tide_design(T[win], t0) @ coef
    k = np.round((Tw - t0) / 60.0).astype(int); ok = (k >= 0) & (k < n) & (np.abs(Tw - t0 - 60 * k) <= 30)
    s = np.zeros(n); c = np.zeros(n)
    np.add.at(s, k[ok], Rw[ok]); np.add.at(c, k[ok], 1)
    eta = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    mask = c > 0
    # despike gross glitches only: points more than 0.1 m + 0.5 |median| away from the 5-sample running median
    idx = np.nonzero(mask)[0]
    if len(idx) > 10:
        v = eta[idx]; med = np.array([np.median(v[max(0, i - 2):i + 3]) for i in range(len(v))])
        bad = np.abs(v - med) > 0.1 + 0.5 * np.abs(med)
        mask[idx[bad]] = False; eta[idx[bad]] = 0.0
    info = dict(tide_rms_fit=float(np.sqrt(np.mean(res ** 2))), n_fit=int(fit.sum()),
                frac_1min=float(np.mean(mask)), types=np.bincount(Y[win], minlength=4)[1:].tolist())
    return eta.astype(np.float32), mask, info
