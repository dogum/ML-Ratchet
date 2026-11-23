"""Scattering coda for a 1-D synthetic. The real Earth is lumpy, so every arrival keeps ringing: each output component gets
the three input components convolved with random noise under an exponential envelope (a: coda amplitude, tau: decay in s,
1 sps), applied above 0.04 Hz only, which also leaks vertical P energy into the horizontals the way scattering does."""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfiltfilt
HP = butter(2, 0.04, 'high', fs=1.0, output='sos')   # scattering is a short-period effect: long-period surface waves pass through

def scatter(rng, X, a, tau, L=None):
    L = L or int(min(6 * tau, 1200)); t = np.arange(L); env = np.exp(-t / tau); env[:2] = 0
    out = X.copy(); Xh = sosfiltfilt(HP, X, axis=1)
    for c in range(3):
        h = rng.standard_normal((3, L)) * env * a / np.sqrt(3)
        out[c] += sum(fftconvolve(Xh[j], h[j])[:X.shape[1]] for j in range(3))
    return out
