"""A simple QRS detector (Pan-Tompkins style): 5-15 Hz band-pass, derivative, squaring, 150 ms moving integration,
peaks at least 250 ms apart above an adaptive threshold, then moved to the largest |0.5-40 Hz| deflection within +-60 ms.
Used for records without beat annotations (QT database) and reported against the MIT-BIH reference positions."""
import numpy as np, scipy.signal as ss

def detect(x, fs):
    x = np.asarray(x, float); b = ss.butter(2, [5, 15], 'bandpass', fs=fs, output='sos'); y = ss.sosfiltfilt(b, x)
    e = np.convolve(np.gradient(y) ** 2, np.ones(int(0.15 * fs)) / int(0.15 * fs), 'same')
    win = int(8 * fs); lvl = np.array([np.percentile(e[max(0, i - win):i + win], 98) for i in range(0, len(e), win // 4)])
    thr = 0.25 * np.repeat(lvl, win // 4)[:len(e)]
    pk, _ = ss.find_peaks(e, height=thr, distance=int(0.25 * fs))
    bp = ss.sosfiltfilt(ss.butter(2, [0.5, 40], 'bandpass', fs=fs, output='sos'), x); h = int(0.06 * fs)
    return np.array([max(p - h, 0) + int(np.argmax(np.abs(bp[max(p - h, 0):p + h]))) for p in pk])

def match(ref, det, fs, tol=0.15):
    """sensitivity and positive predictivity of detections vs reference positions (tolerance 150 ms)"""
    j = np.searchsorted(det, ref); hit = np.zeros(len(ref), bool); used = np.zeros(len(det), bool)
    for i, r in enumerate(ref):
        for k in (j[i] - 1, j[i]):
            if 0 <= k < len(det) and not used[k] and abs(det[k] - r) <= tol * fs: hit[i] = used[k] = True; break
    return int(hit.sum()), len(ref), int(used.sum()), len(det)
