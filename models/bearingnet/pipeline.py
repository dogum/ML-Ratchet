"""BearingNet front end (numpy/scipy only; copied verbatim into the release).
One 2.73 s window at 12 kHz -> envelope spectra in 5 demodulation bands, each read on a log-frequency axis aligned to each
fault hypothesis (shaft, BPFO, BPFI, BSF) computed from the bearing geometry and a rough nominal shaft speed.
On that axis harmonics of the hypothesis sit at v = log2(1), log2(2), ... whatever the bearing, so the network learns a
bearing-agnostic 'comb detector'. A speed error shifts every strip by the same amount (a translation the CNN can read)."""
import numpy as np, scipy.signal as ss
FS = 12000; NWIN = 32768; DF = FS / NWIN
V = np.arange(-2.0, 2.6 + 1e-9, 1 / 64)            # log2(f / f_hypothesis), 295 cells
HYP = ('shaft', 'BPFO', 'BPFI', 'BSF')
FIXED_BANDS = ((500, 5700), (500, 2000), (2000, 3800), (3800, 5700))
NB = len(FIXED_BANDS) + 1                           # + the spectral-kurtosis band
NCH = NB * len(HYP) + 1                             # envelope strips + raw spectrum strip on the shaft axis
NSC = NB + 3
_FR = np.fft.rfftfreq(NWIN, 1 / FS); _HANN = np.hanning(NWIN)

def orders(n, d, D, phi=0.0):
    r = d / D * np.cos(np.radians(phi))
    return dict(FTF=0.5 * (1 - r), BPFO=n / 2 * (1 - r), BPFI=n / 2 * (1 + r), BSF=D / (2 * d) * (1 - r * r))

def to_fs(x, fs):
    """resample any recording to 12 kHz (polyphase, anti-aliased)"""
    from fractions import Fraction
    if abs(fs - FS) < 1e-6: return np.asarray(x, float)
    q = Fraction(FS / fs).limit_denominator(2000); return ss.resample_poly(np.asarray(x, float), q.numerator, q.denominator)

def sk_band(x, lo=500, hi=5700):
    """simplified kurtogram: spectral kurtosis from a 256-point STFT; the (centre, width) band of highest mean SK"""
    f, t, S = ss.stft(x, FS, nperseg=256, noverlap=192, boundary=None, padded=False)
    P = np.abs(S) ** 2; sk = (P ** 2).mean(1) / np.maximum(P.mean(1) ** 2, 1e-30) - 2
    best = (-1e9, (lo, hi))
    for bw in (600, 1200, 2400):
        for c in np.arange(lo + bw / 2, hi - bw / 2 + 1, 150):
            m = (f >= c - bw / 2) & (f <= c + bw / 2); s = sk[m].mean()
            if s > best[0]: best = (s, (c - bw / 2, c + bw / 2))
    return best[1]

def _gain(lo, hi, taper=60.0):
    g = np.clip((_FR - lo + taper) / taper, 0, 1) * np.clip((hi + taper - _FR) / taper, 0, 1); return g

def envelope_spectrum(X, lo, hi):
    """squared-envelope spectrum of the band [lo, hi] Hz from the window's rfft X; also the band signal's kurtosis"""
    g = _gain(lo, hi); Z = np.zeros(NWIN, complex); Z[:len(X)] = 2 * X * g; z = np.fft.ifft(Z)      # analytic band signal
    e = np.abs(z) ** 2; k = float(np.mean(e ** 2) / max(np.mean(e) ** 2, 1e-30))                  # = kurtosis of the real band signal x 2 (Gaussian -> 2)
    ses = np.abs(np.fft.rfft((e - e.mean()) * _HANN)); return ses, k

def strip(spec, f0, half=1 / 128):
    """read spectrum `spec` (bins DF apart) at f0 * 2**V: max over the bins inside each 1/64-octave cell, or linear
    interpolation where a cell is narrower than a bin"""
    fc = f0 * 2.0 ** V; lo = np.clip(np.ceil(fc * 2 ** -half / DF).astype(int), 1, len(spec) - 1)
    hi = np.clip(np.floor(fc * 2 ** half / DF).astype(int), 1, len(spec) - 1)
    out = np.interp(fc / DF, np.arange(len(spec)), spec)
    for i in np.nonzero(hi >= lo)[0]: out[i] = max(out[i], spec[lo[i]:hi[i] + 1].max())
    return out

def window_features(x, geom, f_nom, band=None):
    """x: NWIN samples at 12 kHz. geom: dict(n, d, D[, phi]). f_nom: rough shaft speed in Hz.
    returns (NCH, len(V)) float32 map, (NSC,) scalars, and the SK band used"""
    x = np.asarray(x, float) - np.mean(x); o = orders(**geom); X = np.fft.rfft(x)
    band = band or sk_band(x); bands = list(FIXED_BANDS) + [band]
    F = np.zeros((NCH, len(V)), np.float32); ks = []
    fh = {'shaft': f_nom, 'BPFO': o['BPFO'] * f_nom, 'BPFI': o['BPFI'] * f_nom, 'BSF': o['BSF'] * f_nom}
    for b, (lo, hi) in enumerate(bands):
        ses, k = envelope_spectrum(X, lo, hi); ks.append(k)
        for h, name in enumerate(HYP):
            s = np.log10(strip(ses, fh[name]) + 1e-12); F[b * len(HYP) + h] = s - np.median(s)
    raw = np.abs(np.fft.rfft(x * _HANN)); s = np.log10(strip(raw, f_nom) + 1e-12); F[-1] = s - np.median(s)
    r = geom['d'] / geom['D'] * np.cos(np.radians(geom.get('phi', 0.0)))
    sc = np.array([np.log10(max(k, 1e-3) / 2) for k in ks] + [np.log2(f_nom / 25.0), geom['n'] / 10.0, 4 * r], np.float32)
    return np.clip(F, -2, 5), sc, band

def pad_short(x):
    """records shorter than one window (some CWRU 48 kHz files are 1.3-2.7 s) are extended by mirror reflection"""
    x = np.asarray(x, float)
    while len(x) < NWIN: x = np.concatenate([x, x[::-1]])
    return x

def windows(x, max_win=8):
    """start indices of up to max_win half-overlapping windows covering the recording"""
    n = len(x)
    if n < NWIN: return [0]          # short record: pad_short() first
    st = list(range(0, n - NWIN + 1, NWIN // 2))
    if st[-1] != n - NWIN: st.append(n - NWIN)
    if len(st) > max_win: st = [st[i] for i in np.linspace(0, len(st) - 1, max_win).round().astype(int)]
    return st
