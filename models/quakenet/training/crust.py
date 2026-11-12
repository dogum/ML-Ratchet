"""Layered structure that a 1-D global model leaves out, applied above 0.04 Hz (1 sps):
  ocean   most shallow quakes are offshore (82% in the 1990-2011 catalog). Energy that reaches the sea floor rings in the
          water column: copies every tau = 2 H / 1.5 km/s (H the water depth), alternating in sign and decaying by r per
          round trip (pwP and its multiples).
  site    the crust and sediment under the station: a P-to-S conversion onto the radial horizontal a few seconds after
          each arrival, and reverberation in a soft surface layer on all three components."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
HP = butter(2, 0.04, 'high', fs=1.0, output='sos')

def shift(x, d):
    t = np.arange(x.shape[-1]); return np.interp(t - d, t, x, left=0, right=0)

def train(x, tau, a, r, n=6):
    """sum over n of a (-r)^(k-1) x(t - k tau), alternating sign: free-surface reflection (-1) times sea-floor reflection r"""
    out = np.zeros_like(x)
    for k in range(1, n + 1): out += a * (-r) ** (k - 1) * shift(x, k * tau)
    return -out

def apply(rng, X, baz, depth):
    Xh = sosfiltfilt(HP, X, axis=1); Y = X.copy(); info = {}
    if rng.random() < (0.8 if depth <= 70 else 0.7):                    # offshore source
        H = float(np.exp(rng.uniform(np.log(0.3), np.log(6.0)))); tau = 2 * H / 1.5 * 0.97; a = rng.uniform(0.15, 0.5); r = rng.uniform(0.4, 0.8)
        Y += np.stack([train(x, tau, a, r) for x in Xh]); info['water_km'] = H
    tps = rng.uniform(2.5, 6.5); aps = rng.uniform(0.05, 0.3)            # Moho P-to-S at the station, onto the radial component
    Rp = aps * shift(Xh[0], tps); Y[1] += -Rp * np.cos(baz); Y[2] += -Rp * np.sin(baz)
    if rng.random() < 0.5:                                                # soft sediment under the station
        ts = rng.uniform(0.5, 3.0); rs = rng.uniform(0.2, 0.5)
        Y += np.stack([sum(((-rs) ** k) * shift(x, k * ts) for k in range(1, 5)) for x in Xh])        # bottom reflection rs, free surface -1
    return Y, info

def damp_short_sw(rng, X, tS, scale=None):
    """Real surface waves carry much less 9-18 s energy than ak135's (dev quakes: 9-12 s x0.2, 12-18 s x0.5; 18 s and
    longer, and under 8 s, unchanged; calib_swq.py). Applied after S with a 60 s taper; the depletion scales by a random factor per record."""
    s = rng.uniform(0.5, 1.5) if scale is None else scale; L = X.shape[1]; n = 1 << int(np.ceil(np.log2(2 * L))); f = np.fft.rfftfreq(n, 1.0)
    logg = np.interp(f, [0, 1 / 20, 1 / 15, 1 / 10.5, 1 / 8, 1 / 6, 0.5], [0, 0, -0.3, -0.7, -0.1, 0, 0]) * s
    Y = np.fft.irfft(np.fft.rfft(X, n, axis=1) * 10 ** logg, n, axis=1)[:, :L]
    w = np.clip((np.arange(L) - tS) / 60.0, 0, 1); w = 0.5 - 0.5 * np.cos(np.pi * w)
    return X * (1 - w) + Y * w

def smear_short_sw(rng, X, tS, frac=None, tau=None):
    """Real short-period surface waves arrive smeared (multipathing, off-path scattering, Love-Rayleigh coupling): after S,
    a fraction of the energy under 25 s is replaced by the same band convolved with decaying random noise across the three
    components, at equal energy."""
    from scipy.signal import fftconvolve
    f = rng.uniform(0.3, 0.9) if frac is None else frac; tau = rng.uniform(20, 80) if tau is None else tau
    sos = butter(4, 1 / 25, 'high', fs=1.0, output='sos'); Xs = sosfiltfilt(sos, X, axis=1); L = X.shape[1]
    w = np.clip((np.arange(L) - tS) / 60.0, 0, 1); w = 0.5 - 0.5 * np.cos(np.pi * w); Xs = Xs * w
    t = np.arange(int(5 * tau)); env = np.exp(-t / tau); env /= np.sqrt(np.sum(env ** 2) * 3)
    Y = np.stack([sum(fftconvolve(Xs[j], rng.standard_normal(len(t)) * env)[:L] for j in range(3)) for _ in range(3)])
    Y = sosfiltfilt(sos, Y, axis=1) * w; Y *= np.sqrt(np.sum(Xs ** 2) / max(np.sum(Y ** 2), 1e-30))
    return X + (np.sqrt(1 - f) - 1) * Xs + np.sqrt(f) * Y      # tested with twin.py: no effect on the shallow-depth bias; not used in training

def _rt(X, baz):
    N, E = X[1], X[2]; return -N * np.cos(baz) - E * np.sin(baz), N * np.sin(baz) - E * np.cos(baz)

def _ne(R, T, baz):
    return -R * np.cos(baz) + T * np.sin(baz), -R * np.sin(baz) - T * np.cos(baz)

def path_surface(rng, X, tS, dist_deg, baz, love=True, disp=True, s_love=None, s_disp=None):
    """Two path effects measured on shallow dev quakes against synthetics of the same quakes (calib_love.py), after S:
      love  transverse (Love) energy at 12-25 s is about half of ak135's (x0.5), longer periods unchanged
      disp  short-period Rayleigh waves (Z, R) run slower than ak135 relative to their long periods: an extra group delay of
            17 s per 1000 km at 9-25 s, 4 s at 25-35 s, none beyond 35 s, scaled per record by a lognormal (paths differ)"""
    L = X.shape[1]; n = 1 << int(np.ceil(np.log2(2 * L))); f = np.fft.rfftfreq(n, 1.0)
    w = np.clip((np.arange(L) - tS) / 60.0, 0, 1); w = 0.5 - 0.5 * np.cos(np.pi * w)
    Z, R, T = X[0].copy(), *_rt(X, baz)
    if love:
        s = rng.uniform(0.5, 1.5) if s_love is None else s_love
        g = 10 ** (np.interp(f, [0, 1 / 30, 1 / 25, 1 / 12, 1 / 9, 0.5], [0, 0, -0.3, -0.3, 0, 0]) * s)
        T = T * (1 - w) + np.fft.irfft(np.fft.rfft(T, n) * g, n)[:L] * w
    if disp:
        s = float(np.exp(rng.normal(0, 0.8))) if s_disp is None else s_disp
        gd = np.interp(f, [0, 1 / 40, 1 / 30, 1 / 25, 1 / 9, 0.5], [0, 0, 4, 17, 17, 17]) * s * dist_deg * 111.19 / 1000     # group delay, s
        ph = np.exp(-2j * np.pi * np.concatenate([[0], np.cumsum(gd[1:] * np.diff(f))]))                              # phase = -2 pi int g df
        Zd = np.fft.irfft(np.fft.rfft(Z, n) * ph, n)[:L]; Rd = np.fft.irfft(np.fft.rfft(R, n) * ph, n)[:L]
        Z = Z * (1 - w) + Zd * w; R = R * (1 - w) + Rd * w
    N, E = _ne(R, T, baz); return np.stack([Z, N, E])

ZRT_NODES = [0, 1 / 30, 1 / 21.5, 1 / 15, 1 / 10.5, 1 / 8, 1 / 6, 0.5]
ZRT_LOGGAIN = {'Z': [0, 0, 0, -0.3, -0.7, -0.1, 0, 0], 'R': [0, 0, 0, -0.06, -0.19, -0.05, 0, 0], 'T': [0, 0, -0.32, -0.31, -0.22, -0.05, 0, 0]}

def surface_zrt(rng, X, tS, dist_deg, baz, scale=None, s_disp=None, disp=True):
    """v7: replaces damp_short_sw (which applied the vertical's depletion to all three components). After S, each component
    gets its own measured real/ak135 gain (calib_love.py, shallow dev quakes): vertical 9-12 s x0.2, 12-18 s x0.5;
    radial nearly unchanged; transverse (Love) x0.5 at 12-25 s. Plus the extra short-period Rayleigh group delay of
    path_surface. One per-record scale U(0.5, 1.5) on all gains."""
    L = X.shape[1]; n = 1 << int(np.ceil(np.log2(2 * L))); f = np.fft.rfftfreq(n, 1.0); s = rng.uniform(0.5, 1.5) if scale is None else scale
    w = np.clip((np.arange(L) - tS) / 60.0, 0, 1); w = 0.5 - 0.5 * np.cos(np.pi * w)
    R, T = _rt(X, baz); out = []
    for k, x in (('Z', X[0]), ('R', R), ('T', T)):
        y = np.fft.irfft(np.fft.rfft(x, n) * 10 ** (np.interp(f, ZRT_NODES, ZRT_LOGGAIN[k]) * s), n)[:L]; out.append(x * (1 - w) + y * w)
    Z, R, T = out; Y = np.stack([Z, *_ne(R, T, baz)])
    return path_surface(rng, Y, tS, dist_deg, baz, love=False, disp=disp, s_disp=s_disp) if disp else Y
