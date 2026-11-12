import numpy as np
def cwt_energy(x, fs, freqs, Q=7.0):
    """Morlet scalogram |W|^2, normalised so whitened Gaussian noise averages 1."""
    n = len(x); X = np.fft.fft(x); f = np.fft.fftfreq(n, 1 / fs)
    out = np.zeros((len(freqs), n), np.float32)
    for i, f0 in enumerate(freqs):
        sig_f = f0 / Q
        G = np.exp(-0.5 * ((f - f0) / sig_f) ** 2) * (f > 0)
        w = np.fft.ifft(X * G)
        e = np.abs(w) ** 2
        out[i] = e / np.mean(e[n // 8: -n // 8]) if np.mean(e) > 0 else e
    return out
