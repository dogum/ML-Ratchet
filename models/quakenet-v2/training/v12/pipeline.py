"""QuakeNet's reading pipeline, the same for simulated and real seismograms: three components (Z, N, E) of ground
velocity at 1 sample/s, band 0.008-0.35 Hz. A short-term/long-term average trigger finds the P wave near where it is
expected; the window runs from 60 s before the pick to 2,400 s after. Each window is divided by its largest component
standard deviation; that scale's log10 goes to the network as a number (it carries the size of the quake)."""
import numpy as np
from obspy.signal.trigger import recursive_sta_lta
from scipy.signal import butter, sosfiltfilt
PRE, POST, FS = 60, 2400, 1.0; L = PRE + POST
SOS_BAND = butter(4, [0.008, 0.35], 'band', fs=2.0, output='sos')           # for 2 sps synthetics before decimation
SOS_PICK = butter(2, [0.05, 0.3], 'band', fs=1.0, output='sos')

def pick(z, guess, search=30):
    """P pick on the vertical: the first sample within guess +- search where STA/LTA (5 s / 60 s) exceeds 3, or its maximum"""
    zf = sosfiltfilt(SOS_PICK, z); c = recursive_sta_lta(zf, 5, 60); lo, hi = max(61, guess - search), min(len(z) - 1, guess + search)
    seg = c[lo:hi]
    if len(seg) == 0: return guess, 0.0
    above = np.where(seg > 3.0)[0]; i = lo + (above[0] if len(above) else int(np.argmax(seg))); return int(i), float(seg.max())

def window(X, p):
    out = np.zeros((3, L), np.float32); a, b = p - PRE, p + POST; s0, s1 = max(a, 0), min(b, X.shape[1]); out[:, s0 - a:s1 - a] = X[:, s0:s1]; return out

def normalize(W):
    sc = float(max(W.std(1).max(), 1e-12)); return (W / sc).astype(np.float32), np.log10(sc)

def read(X, guess):
    """X: (3, n) Z N E velocity (m/s) at 1 sps; guess: expected P sample. -> normalized window, log10 scale, pick, STA/LTA peak"""
    p, peak = pick(X[0], guess); W, ls = normalize(window(X, p)); return W, ls, p, peak
