"""pipeline.py (numpy + scipy): read NDBC historical DART files and turn them into detided 1-minute series after an
origin time, exactly as training/dart.py did for the training noise, the dev set, the exam and examples/ (this file is
that code with the data folder taken from $DART_DIR, default ./dart next to this file).
Processing is causal except for a short look-ahead, so it can run in near real time: the tide fit uses only the 3 days
before the origin and the high-pass is causal, but the glitch removal reads up to 34 LATER observed samples (a 9-sample
running median, centred, and a 61-sample robust scale of its deviations, centred) and gap filling interpolates to the
next observed sample. A minute's value and mask are final once 34 more observed samples have arrived (about 34 min at
1-minute sampling, longer across gaps). Training, dev, exam and examples/ all used this processing as written (22 Nov
2026: documented, not changed, so inference stays identical to training):
 1. Detide: least-squares fit of 8 tidal constituents (M2 S2 N2 K2 K1 O1 P1 Q1) plus a quadratic trend to all samples
    (15-min, 1-min, 15-s) in the 3 days BEFORE the origin [t0 - 3 d, t0 - 10 min]; the prediction is subtracted.
 QC: records whose pre-event tide-fit residual exceeds 3 cm rms, or with > 5 % of pre-event samples flagged as
    outliers by the robust fit, are rejected, and
    single samples more than 3 m from the tide prediction are dropped.
 2. Minute grid from t0 - 12 h to t0 + 24 h: each minute is the mean of samples within +-30 s; gross glitches are
    removed (> 6 local robust sigmas + 1 cm from a 9-sample running median, or > 0.5 m from it).
 3. High-pass: gaps filled by linear interpolation, then a causal 2nd-order Butterworth high-pass with a 3-hour cutoff
    (removes the tide-prediction error and drift; tsunami periods of 5-60 min pass). The same filter is applied to
    every Green's function (tsim.GFBank, release), so simulations, basis channels and forecasts see the same processing.
Returned: t0 .. t0 + 24 h; minutes with no real sample are masked. The same code serves real events and noise windows."""
import os, gzip, numpy as np, datetime as dt
from scipy.signal import butter, sosfilt, sosfilt_zi
SOS = butter(2, 1.0 / 180.0, 'highpass', fs=1.0, output='sos')   # per-minute samples, 3 h cutoff
PRE = 720

def highpass(x):
    """Causal 3-h high-pass, started at rest (the series' first value is taken as the initial level)."""
    x = np.asarray(x, np.float64)
    return sosfilt(SOS, x, zi=sosfilt_zi(SOS) * x[0])[0]

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
    r = read_span(station, t0 - 3 * 86400, t0 + hours * 3600 + 60)
    n = 60 * hours + 1; N = PRE + n
    if r is None: return None
    T, H, Y = r
    fit = T < t0 - 600
    if fit.sum() < 100 or (T[fit] < t0 - 2 * 86400).sum() < 20: return None
    A = tide_design(T[fit], t0); w = np.ones(fit.sum())
    for it in range(3 if robust else 1):
        coef = np.linalg.lstsq(A * w[:, None], H[fit] * w, rcond=None)[0]
        res = H[fit] - A @ coef
        s = 1.4826 * np.median(np.abs(res)) + 1e-4
        w = 1.0 / np.maximum(1.0, np.abs(res) / (4 * s))
    qc_frac = float(np.mean(w < 1))
    if np.sqrt(np.mean((res * (w >= 1)) ** 2)) > 0.03 or qc_frac > 0.05: return None   # QC: jumps/glitches before the event
    tg0 = t0 - PRE * 60
    win = (T >= tg0 - 30) & (T <= t0 + hours * 3600 + 30)
    Tw, Rw = T[win], H[win] - tide_design(T[win], t0) @ coef
    g = np.abs(Rw) < 3.0; Tw, Rw = Tw[g], Rw[g]                       # QC: no deep-ocean tsunami exceeds 3 m
    k = np.round((Tw - tg0) / 60.0).astype(int); ok = (k >= 0) & (k < N) & (np.abs(Tw - tg0 - 60 * k) <= 30)
    sm = np.zeros(N); c = np.zeros(N)
    np.add.at(sm, k[ok], Rw[ok]); np.add.at(c, k[ok], 1)
    eta = np.where(c > 0, sm / np.maximum(c, 1), 0.0); mask = c > 0
    idx = np.nonzero(mask)[0]
    if len(idx) < 30 or mask[PRE:].sum() < 10: return None
    v = eta[idx]; med = np.array([np.median(v[max(0, i - 4):i + 5]) for i in range(len(v))]); dev = np.abs(v - med)
    loc = np.array([np.median(dev[max(0, i - 30):i + 31]) for i in range(len(v))]) * 1.4826
    bad = (dev > 6 * loc + 0.01) | (dev > 0.5)
    mask[idx[bad]] = False
    idx = np.nonzero(mask)[0]
    filled = np.interp(np.arange(N), idx, eta[idx])
    hp = highpass(filled)
    out = np.where(mask, hp, 0.0)[PRE:].astype(np.float32); m = mask[PRE:]
    info = dict(tide_rms_fit=float(np.sqrt(np.mean(res ** 2))), n_fit=int(fit.sum()), frac_1min=float(np.mean(m)),
                types=np.bincount(Y[win], minlength=4)[1:].tolist(), n_despiked=int(bad.sum()), qc_frac=qc_frac)
    return out, m, info
