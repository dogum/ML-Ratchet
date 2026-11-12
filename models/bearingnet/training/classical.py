"""Classical baseline (Randall & Antoni 2011, sec. 3-4): pick the demodulation band by spectral kurtosis, square-envelope
spectrum, then look for lines at the theoretical defect frequencies. Speed from a harmonic comb on the raw spectrum
within +-5% of the nominal speed. Score per hypothesis = mean over harmonics 1-3 of log10(peak in +-1.5% / local median).
Decision: healthy if the best score < tau (tau fitted on the dev set only), else the best-scoring fault."""
import numpy as np, features as FE
F = np.fft.rfftfreq(FE.NWIN, 1 / FE.FS)

def speed(x, f_nom):
    raw = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))); lr = np.log10(raw + 1e-12)
    cand = f_nom * np.linspace(0.95, 1.05, 401); best = None
    for f in cand:
        s = 0.0
        for k in range(1, 9):
            j = int(round(k * f / FE.DF)); s += lr[j - 1:j + 2].max()
        if best is None or s > best[0]: best = (s, f)
    return best[1]

def scores(x, geom, f):
    x = np.asarray(x, float) - np.mean(x); lo, hi = FE.sk_band(x); ses, _ = FE.envelope_spectrum(np.fft.rfft(x), lo, hi); o = FE.orders(**geom); out = []
    for fd in (o['BPFO'] * f, o['BPFI'] * f, 2 * o['BSF'] * f):
        v = []
        for h in (1, 2, 3):
            m = (F > h * fd * 0.985) & (F < h * fd * 1.015); w = (F > h * fd * 0.85) & (F < h * fd * 1.15)
            v.append(np.log10(ses[m].max() / np.median(ses[w])))
        out.append(np.mean(v))
    return np.array(out)

def decide(sc, tau):
    sc = np.asarray(sc); return np.where(sc.max(-1) < tau, 0, sc.argmax(-1) + 1)
