"""StarQuake signal processing, shared by the simulator, the real-star reader, the release and (ported) the web page.

light curve (ppm on the Kepler long-cadence grid, NaN in gaps)
  -> 10-day running-mean high-pass, 5-sigma clip
  -> power density spectrum (ppm^2/uHz, single-sided, Parseval-calibrated for the duty cycle)
  -> features: (1) log10 power on 1024 log-spaced bins, 3-283 uHz (finds the hump: numax)
               (2) a 48 x 110 'comb map': for 48 trial centres c (6-260 uHz) the autocorrelation of the spectrum within
                   +-3.5 expected large separations of c, at lags 0.55-1.65 x the expected separation (finds Delta nu)
"""
import numpy as np
DT = 1765.4637                 # Kepler long cadence (s)
NYQ = 1e6 / (2 * DT)           # 283.2 uHz
NB = 1024; F_LO, F_HI = 3.0, 283.0
EDGES = np.exp(np.linspace(np.log(F_LO), np.log(F_HI), NB + 1)); FC = np.sqrt(EDGES[1:] * EDGES[:-1])
NC = 48; CEN = np.exp(np.linspace(np.log(6.0), np.log(260.0), NC))
STEP = 100; HALF = 3.5; NW = int(2 * HALF * STEP)         # window grid: 700 points, step dnu_s/100
LAGS = np.arange(56, 166)                                  # lag r = LAGS/100 = 0.56 ... 1.65 dnu_s
NL = len(LAGS)

def dnu_s(numax): return 0.276 * np.asarray(numax) ** 0.751      # Mosser et al. 2012, Table 2

def highpass(flux, days=10.0, clip=5.0):
    """flux: ppm on a regular cadence grid with NaN in gaps. Returns filtered ppm (NaN in gaps and clipped points)."""
    m = np.isfinite(flux); x = np.where(m, flux, 0.0); w = int(round(days * 86400 / DT)) | 1
    k = np.ones(w); cs = lambda a: np.convolve(a, k, 'same')
    sm = cs(x) / np.maximum(cs(m.astype(float)), 1); y = np.where(m, x - sm, np.nan)
    med = np.nanmedian(y); mad = 1.4826 * np.nanmedian(np.abs(y - med))
    y[np.abs(y - med) > clip * mad] = np.nan; return y

def psd(y):
    """y: ppm on the cadence grid, NaN in gaps -> (freq uHz, PSD ppm^2/uHz) up to Nyquist."""
    m = np.isfinite(y); x = np.where(m, y - np.nanmean(y), 0.0); n = len(x); nv = max(m.sum(), 1)
    X = np.fft.rfft(x); P = 2 * DT * np.abs(X) ** 2 / nv * 1e-6
    f = np.arange(len(P)) * 1e6 / (n * DT); return f[1:], P[1:]

def rebin(f, P, edges):
    """Area-preserving mean of P between consecutive edges (works whether the native grid is finer or coarser)."""
    df = f[1] - f[0]; e = np.concatenate([[f[0] - df / 2], f + df / 2]); C = np.concatenate([[0], np.cumsum(P * df)])
    a = np.interp(edges, e, C); return np.diff(a) / np.diff(edges)

def features(f, P):
    """-> F1 (NB,) float32, F2 (NC, NL) float32"""
    F1 = np.log10(np.maximum(rebin(f, P, EDGES), 1e-6)).astype(np.float32)
    F2 = np.zeros((NC, NL), np.float32); fmax = f[-1]; t = np.linspace(-1, 1, NW)
    V = np.vander(t, 3)
    for j, c in enumerate(CEN):
        d = dnu_s(c); e = c + (np.arange(NW + 1) - NW / 2) * d / STEP; ok = (e[:-1] > f[0]) & (e[1:] < fmax)
        if ok.sum() < NW // 2: continue
        y = rebin(f, P, np.clip(e, f[0], fmax)); y = np.log(np.maximum(y, 1e-12))
        coef = np.linalg.lstsq(V[ok], y[ok], rcond=None)[0]; z = np.where(ok, np.exp(y - V @ coef) - 1, 0.0)
        z = z - z[ok].mean() * ok; Z = np.fft.rfft(z, 1024); R = np.fft.irfft(np.abs(Z) ** 2, 1024)
        F2[j] = (R[LAGS] / max(R[0], 1e-12)).astype(np.float32)
    return F1, F2

BAD = 1 | 2 | 4 | 8 | 16 | 32 | 128 | 1024    # attitude tweak, safe mode, coarse/earth point, argabrightening, desat, manual exclude, impulsive outlier

def fill_short(y, maxgap=3):
    """Linearly interpolate gaps of <= maxgap cadences (Kepler's many single-cadence gaps leak low-frequency power
    across the whole spectrum through the window function)."""
    m = np.isfinite(y); idx = np.flatnonzero(m)
    if len(idx) < 2: return y
    d = np.diff(np.concatenate([[1], m.astype(int), [1]])); st = np.flatnonzero(d == -1); en = np.flatnonzero(d == 1)
    out = y.copy(); fill = np.zeros(len(y), bool)
    for a, b in zip(st, en):
        if 0 < a and b < len(y) and b - a <= maxgap: fill[a:b] = True
    out[fill] = np.interp(np.flatnonzero(fill), idx, y[idx]); return out

def grid_lc(cad, flux, quarter, q):
    """Real Kepler PDCSAP light curve -> ppm on the cadence grid (per-quarter median normalisation), NaN in gaps."""
    c0 = cad.min(); n = cad.max() - c0 + 1; out = np.full(n, np.nan)
    good = np.isfinite(flux) & ((q & BAD) == 0)
    for qq in np.unique(quarter):
        s = good & (quarter == qq)
        if s.sum() < 50: continue
        out[cad[s] - c0] = 1e6 * (flux[s] / np.median(flux[s]) - 1)
    return out

def read_npz(path):
    z = np.load(path); return grid_lc(z['cad'], z['flux'].astype(np.float64), z['quarter'], z['q'])

def scalars(y):
    """observing length (days) and duty cycle -> 2 network scalars"""
    m = np.isfinite(y); idx = np.flatnonzero(m); T = (idx[-1] - idx[0] + 1) * DT / 86400 if len(idx) else 1.0
    return np.array([np.log10(max(T, 1.0)) - 2.5, m.mean()], np.float32), T

def prepare(y):
    """ppm light curve on the cadence grid -> (F1, F2, scalars, (f, P), T_days)"""
    yh = highpass(fill_short(y)); f, P = psd(yh); F1, F2 = features(f, P); s, T = scalars(yh); return F1, F2, s, (f, P), T
