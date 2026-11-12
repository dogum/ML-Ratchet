"""Physics + data pipeline: LIGO strain I/O, PSDs, whitening, GR waveform bank, injections."""
import numpy as np, h5py, json, glob, os
from scipy.signal import welch
from scipy.signal.windows import tukey

FS_IN, FS = 4096, 2048          # raw sample rate, model sample rate
WIN = 4096                       # model window: 2 s at 2048 Hz
BAND = (20.0, 900.0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- real data
def load_strain(path):
    with h5py.File(path) as h:
        return h['strain/Strain'][:].astype(np.float64), int(h['meta/GPSstart'][()])

def events():
    return json.load(open(os.path.join(ROOT, 'data_raw/BBH_events_v3.json')))

def psd(x, fs=FS_IN, seg=4.0):
    f, p = welch(x, fs=fs, nperseg=int(seg * fs), noverlap=int(seg * fs / 2), window='hann', average='median')
    return f, p

def band_mask(f, lo=BAND[0], hi=BAND[1], taper=8.0):
    m = np.zeros_like(f)
    m[(f >= lo) & (f <= hi)] = 1.0
    a = (f > lo - taper) & (f < lo); m[a] = 0.5 - 0.5 * np.cos(np.pi * (f[a] - (lo - taper)) / taper)
    b = (f > hi) & (f < hi + taper); m[b] = 0.5 + 0.5 * np.cos(np.pi * (f[b] - hi) / taper)
    return m

def whiten_dft(X, n_in, fs_in, S_at_f):
    """X: rfft of a length-n_in series at fs_in. S_at_f: one-sided PSD at rfft freqs.
    Returns whitened, band-limited series at FS (2048 Hz), length n_in*FS/fs_in, unit-variance noise."""
    f = np.fft.rfftfreq(n_in, 1 / fs_in)
    W = X / np.sqrt(S_at_f * fs_in / 2.0) * band_mask(f)
    n_out = n_in * FS // fs_in
    W = W[: n_out // 2 + 1]
    y = np.fft.irfft(W, n=n_out) * (n_out / n_in)
    return y / NORM

def _norm_const():
    # amplitude after whitening unit white noise -> sets NORM so whitened noise has unit variance
    n = 2 ** 16
    f = np.fft.rfftfreq(n, 1 / FS_IN)
    m = band_mask(f)[: (n * FS // FS_IN) // 2 + 1]
    # white noise of variance s2 has S = 2 s2 / fs ; whitened bins have E|W|^2 = n ; irfft scaling
    n_out = n * FS // FS_IN
    var = (np.sum(m[1:-1] ** 2) * 2 + m[0] ** 2 + m[-1] ** 2) * n / n_out ** 2 * (n_out / n) ** 2
    return np.sqrt(var)
NORM = 1.0
NORM = _norm_const()
PHI = None  # in-band fraction for SNR, set below

def whiten(x, fs_in=FS_IN, psd_fp=None):
    """Whiten a raw strain series with its own Welch PSD (or a supplied (f, p))."""
    n = len(x)
    if psd_fp is None: psd_fp = psd(x, fs_in)
    fp, pp = psd_fp
    f = np.fft.rfftfreq(n, 1 / fs_in)
    S = np.interp(f, fp, pp)
    X = np.fft.rfft(x * tukey(n, alpha=1.0 / 8))
    return whiten_dft(X, n, fs_in, S)

def snr_factor():
    f = np.fft.rfftfreq(WIN, 1 / FS)
    m = band_mask(f)
    return float(np.mean(m[1:] ** 2))       # rho^2 = PHI * sum(s^2) for unit-variance band-limited noise
PHI = snr_factor()

def bandlimited_noise(rng, shape):
    """Unit-variance white noise limited to BAND, the same spectrum whitened detector noise has."""
    z = rng.standard_normal(shape).astype(np.float32)
    Z = np.fft.rfft(z, axis=-1)
    f = np.fft.rfftfreq(shape[-1], 1 / FS)
    m = band_mask(f).astype(np.float32)
    y = np.fft.irfft(Z * m, n=shape[-1], axis=-1)
    return (y / np.sqrt(np.mean(m[1:] ** 2))).astype(np.float32)

# ---------------------------------------------------------------- physics
def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

def masses_from_mc_q(mc, q):
    # q = m2/m1 <= 1
    eta = q / (1 + q) ** 2
    M = mc / eta ** 0.6
    m1 = M / (1 + q); m2 = q * m1
    return m1, m2

# ---------------------------------------------------------------- waveform bank
SEG_PRE, SEG_POST = 2.6, 0.3                    # seconds before / after merger kept in the bank
SEG = int((SEG_PRE + SEG_POST) * FS)            # samples
PEAK = int(SEG_PRE * FS)

def training_psds():
    """PSDs (f, p) per detector: LIGO O1/O2 noise estimated around the four events, plus design-curve variants."""
    import lalsimulation as ls
    out = {'H1': [], 'L1': []}
    for name, ev in events().items():
        for det in ('H1', 'L1'):
            s, _ = load_strain(os.path.join(ROOT, 'data_raw', ev['fn_' + det]))
            out[det].append(psd(s))
    f = np.arange(0, 2048.25, 0.25)
    for fn in (ls.SimNoisePSDaLIGOZeroDetHighPower, ls.SimNoisePSDaLIGOZeroDetLowPower, ls.SimNoisePSDaLIGOBHBH20Deg):
        p = np.array([fn(max(x, 10.0)) for x in f])
        for det in ('H1', 'L1'): out[det].append((f, p))
    return out

def make_bank(n, seed=1, psds=None):
    import lal, lalsimulation as ls
    rng = np.random.default_rng(seed)
    psds = psds or training_psds()
    T, fs_in = 16, FS_IN
    n_in = T * fs_in; df = 1.0 / T
    f = np.fft.rfftfreq(n_in, 1 / fs_in)
    Scache = {d: [np.interp(f, fp, pp) for fp, pp in psds[d]] for d in psds}
    bank = np.zeros((n, 2, 2, SEG), np.float16)   # [wf, det(H,L), pol(+,x), t]
    par = np.zeros((n, 7), np.float32)            # mc, q, m1, m2, chi1, chi2, cos_iota
    i = 0
    while i < n:
        mc = np.exp(rng.uniform(np.log(5.5), np.log(45.0))); q = rng.uniform(0.2, 1.0)
        m1, m2 = masses_from_mc_q(mc, q)
        if m1 > 110 or m2 < 4.0: continue
        c1, c2 = rng.uniform(-0.85, 0.85, 2); ci = rng.uniform(-1, 1)
        hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, c1, 0, 0, c2,
                                                400e6 * lal.PC_SI, float(np.arccos(ci)), 0.0, 0, 0, 0, df, 20.0, fs_in / 2, 20.0, None, ls.IMRPhenomD)
        Hp = hp.data.data[: len(f)] * fs_in; Hc = hc.data.data[: len(f)] * fs_in      # DFT ~ h~ / dt
        shift = np.exp(-2j * np.pi * f * (T / 2))                                         # move merger to mid-array
        for d, det in enumerate(('H1', 'L1')):
            S = Scache[det][rng.integers(len(Scache[det]))]
            wp = whiten_dft(Hp * shift, n_in, fs_in, S); wc = whiten_dft(Hc * shift, n_in, fs_in, S)
            if d == 0:
                env = wp ** 2 + wc ** 2
                pk = int(np.argmax(env))
            seg_p = wp[pk - PEAK: pk - PEAK + SEG]; seg_c = wc[pk - PEAK: pk - PEAK + SEG]
            sc = 1.0 / max(np.sqrt(PHI * np.sum(seg_p ** 2 + seg_c ** 2) / 2), 1e-30)   # store at ~unit SNR
            bank[i, d, 0] = seg_p * sc; bank[i, d, 1] = seg_c * sc
        par[i] = (mc, q, m1, m2, c1, c2, ci)
        i += 1
    return bank, par

def sky_table(n, seed=2):
    import lal
    rng = np.random.default_rng(seed)
    H = lal.CachedDetectors[lal.LHO_4K_DETECTOR]; L = lal.CachedDetectors[lal.LLO_4K_DETECTOR]
    out = np.zeros((n, 6), np.float32)       # FpH, FxH, FpL, FxL, dtH, dtL (s, relative to geocenter)
    gps = lal.LIGOTimeGPS(1126259462)
    for k in range(n):
        ra = rng.uniform(0, 2 * np.pi); dec = np.arcsin(rng.uniform(-1, 1)); psi = rng.uniform(0, np.pi)
        gmst = rng.uniform(0, 2 * np.pi)
        fpH, fxH = lal.ComputeDetAMResponse(H.response, ra, dec, psi, gmst)
        fpL, fxL = lal.ComputeDetAMResponse(L.response, ra, dec, psi, gmst)
        # time delay: use RA relative to gmst so the geometry matches the response
        raE = ra  # response above takes gmst explicitly; for the delay rotate into the frame at `gps`
        g0 = lal.GreenwichMeanSiderealTime(gps)
        dH = lal.TimeDelayFromEarthCenter(H.location, ra - gmst + g0, dec, gps)
        dL = lal.TimeDelayFromEarthCenter(L.location, ra - gmst + g0, dec, gps)
        out[k] = (fpH, fxH, fpL, fxL, dH, dL)
    return out
