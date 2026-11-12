"""Classical pipeline in the style of SYD (Huber et al. 2009), on the same processed spectrum StarQuake reads.
1. Find the power excess: comb strength (autocorrelation peak near the expected separation) at 48 trial centres.
2. Fit the background: white noise + two Harvey-like components (exponent 4) + a Gaussian hump, in log power.
3. numax: peak of the background-subtracted spectrum smoothed with a Gaussian of FWHM 4 x expected dnu.
   (v2, short pieces < 300 d: the background+hump fit is started from 10 trial humps and the best fit kept.)
4. dnu: highest autocorrelation peak of the background-divided spectrum around numax, within 0.75-1.33 x expected,
   refined by a parabola through the peak."""
import numpy as np
from scipy.optimize import least_squares
import sq

def collapsed(F2):
    r = sq.LAGS / sq.STEP; m = (r > 0.8) & (r < 1.25); s = F2[:, m].max(1)
    return np.convolve(s, np.ones(3) / 3, 'same')

def bg_model(p, f, with_gauss=True):
    lw, la, lb1, lb2, lh, lnm, ls = p; a, b1, b2 = np.exp(la), np.exp(lb1), np.exp(lb2); eta = np.sinc(f / (2 * sq.NYQ)) ** 2
    B = np.exp(lw) + eta * sq_h(f, a, b1) + eta * sq_h(f, a, b2)
    if with_gauss: B = B + eta * np.exp(lh) * np.exp(-(f - np.exp(lnm)) ** 2 / (2 * np.exp(ls) ** 2))
    return B

def sq_h(f, a, b): return 2 * np.sqrt(2) / np.pi * a * a / b / (1 + (f / b) ** 4)

def fit_bg(x, y, W, nm0):
    p0 = [W * np.log(10), np.log(3382 * nm0 ** -0.609), np.log(0.317 * nm0 ** 0.97), np.log(0.948 * nm0 ** 0.992), np.log(max(2.0e7 * nm0 ** -2.38, 1)), np.log(nm0), np.log(0.66 * nm0 ** 0.88 / 2.355)]
    lo = [p0[0] - 5, p0[1] - 3, p0[2] - 1.5, p0[3] - 1.5, p0[4] - 6, np.log(nm0 / 1.8), np.log(0.2 * np.exp(p0[6]))]
    hi = [p0[0] + 5, p0[1] + 3, p0[2] + 1.5, p0[3] + 1.5, p0[4] + 4, np.log(min(nm0 * 1.8, 290)), np.log(4 * np.exp(p0[6]))]
    p0 = np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    try:
        r = least_squares(lambda p: np.log10(bg_model(p, x)) - y, p0, bounds=(lo, hi), max_nfev=400); return r.x, float(r.cost)
    except Exception:
        return p0, np.inf

def measure(f, P, F1=None, F2=None, multistart=None):
    """multistart None = auto: the comb's pick starts the fit for a year or more of data; shorter pieces use the
    multi-start global fit (chosen on the 18 dev stars: v2 cut the 54-day numax error from 1.7 to 0.05 in ln)."""
    if F1 is None: F1, F2 = sq.features(f, P)
    if multistart is None: multistart = 1e6 / (f[1] - f[0]) / 86400 < 300
    st = collapsed(F2); j = int(np.argmax(st)); nm0 = float(sq.CEN[j])
    x = sq.FC; y = F1.astype(float); W = np.median(y[-40:])
    if multistart:      # v2: global fit from 9 starting humps (plus the comb's pick); keep the best-fitting hump
        best = None
        for s0 in [nm0] + list(np.exp(np.linspace(np.log(7), np.log(240), 9))):
            p, cost = fit_bg(x, y, W, s0)
            if best is None or cost < best[1]: best = (p, cost)
        nm0 = float(np.exp(best[0][5]))
    p, _ = fit_bg(x, y, W, nm0)
    B = bg_model(p, f, with_gauss=False); R = P / B; E = P - B
    d0 = float(sq.dnu_s(np.exp(p[5]))); df = f[1] - f[0]; sg = 4 * d0 / 2.355 / df; k = np.arange(-int(4 * sg) - 1, int(4 * sg) + 2)
    g = np.exp(-k ** 2 / (2 * sg ** 2)); g /= g.sum(); S = np.convolve(E, g, 'same')
    m = (f > nm0 / 1.8) & (f < min(nm0 * 1.8, f[-1])); i = np.flatnonzero(m)[np.argmax(S[m])]
    numax = f[i]
    if 0 < i < len(f) - 1:
        a, b, c = S[i - 1], S[i], S[i + 1]; den = a - 2 * b + c; numax = f[i] + (0.5 * (a - c) / den * df if den < 0 else 0)
    d0 = float(sq.dnu_s(numax)); half = max(2.5 * d0, 0.6 * 0.66 * numax ** 0.88)
    w = (f > numax - half) & (f < numax + half); z = R[w] - R[w].mean() if w.sum() > 8 else np.zeros(0); n = len(z)
    if n < 8: return dict(numax=float(numax), dnu=d0, nm0=nm0, bg=[float(v) for v in p])
    Z = np.fft.rfft(z, 2 * n); ac = np.fft.irfft(np.abs(Z) ** 2, 2 * n)[:n]; ac /= ac[0]
    lag = np.arange(n) * df; mm = (lag > 0.75 * d0) & (lag < 1.33 * d0)
    if mm.sum() < 3: return dict(numax=float(numax), dnu=d0, nm0=nm0, bg=p.tolist())
    j = np.flatnonzero(mm)[np.argmax(ac[mm])]; dnu = lag[j]
    if 0 < j < n - 1:
        a, b, c = ac[j - 1], ac[j], ac[j + 1]; den = a - 2 * b + c; dnu = lag[j] + (0.5 * (a - c) / den * df if den < 0 else 0)
    return dict(numax=float(numax), dnu=float(dnu), nm0=nm0, bg=[float(v) for v in p])
