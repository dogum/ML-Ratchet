"""BeatNet front end, identical for simulated and real records (numpy + scipy only; copied into the release).

  1. resample to 360 Hz, band-pass 0.5-40 Hz (zero phase), decimate to 180 Hz
  2. for every annotated beat: a 2.5 s window centred on the beat (450 samples), divided by the record's local QRS size
     (median peak-to-peak within +-80 ms of the 40 neighbouring beats), and a template = median window of those neighbours
  3. rhythm features from the beat positions: pre-RR, post-RR, local mean RR (+-10 beats), RR irregularity."""
import numpy as np, scipy.signal as ss

FS0, FS = 360, 180
HALF = 225          # window = 2 * HALF samples at 180 Hz = 2.5 s
NB = 20             # neighbours each side for the template and the amplitude scale
SOS = ss.butter(2, [0.5, 40], 'bandpass', fs=FS0, output='sos')


def to360(x, fs):
    if fs == FS0: return np.asarray(x, float)
    from fractions import Fraction
    f = Fraction(FS0, int(round(fs))).limit_denominator(1000); return ss.resample_poly(np.asarray(x, float), f.numerator, f.denominator)


def filt(x360):
    return ss.sosfiltfilt(SOS, x360)[::2]


def windows(y, pos180):
    """y: filtered 180 Hz signal; pos180: beat sample positions at 180 Hz -> (n, 2*HALF)"""
    yp = np.pad(y, (HALF, HALF)); idx = np.asarray(pos180)[:, None] + np.arange(2 * HALF)[None, :]; return yp[idx]


def features(x, fs, pos, keep=None):
    """x: one ECG lead (mV), fs: its sampling rate, pos: beat positions in samples at fs (all beats of the record, sorted).
    keep: indices of the beats to return (default all). Returns dict(w (n,450), tpl (n,450), rr (n,6), scale (n,))."""
    pos = np.asarray(pos); n = len(pos); keep = np.arange(n) if keep is None else np.asarray(keep)
    y = filt(to360(x, fs)); p180 = np.round(np.asarray(pos) * (FS / fs)).astype(int)
    W = windows(y, p180).astype(np.float32); c = HALF; q = int(0.08 * FS)
    ptp = W[:, c - q:c + q].max(1) - W[:, c - q:c + q].min(1)
    t = pos / fs; rr = np.diff(t); pre = np.concatenate([[rr[0] if n > 1 else 1.0], rr]); post = np.concatenate([rr, [rr[-1] if n > 1 else 1.0]])
    out_w, out_t, out_r, out_s = [], [], [], []
    for i in keep:
        lo, hi = max(0, i - NB), min(n, i + NB + 1); nb = [j for j in range(lo, hi) if j != i] or [i]
        s = max(float(np.median(ptp[nb])), 0.05); tpl = np.median(W[nb], 0)
        k0, k1 = max(0, i - 10), min(n - 1, i + 10); loc = float(np.mean(rr[k0:k1])) if k1 > k0 else 1.0
        lr = np.log(rr[k0:k1]) if k1 > k0 else np.zeros(1); irr = float(np.median(np.abs(np.diff(lr)))) if len(lr) > 2 else 0.0
        out_w.append(W[i] / s); out_t.append(tpl / s); out_s.append(s)
        out_r.append([np.log(pre[i] / loc), np.log(post[i] / loc), np.log(loc / 0.8), np.log(pre[i] / post[i]), irr * 5, np.log(ptp[i] / s)])
    return dict(w=np.array(out_w, np.float32), tpl=np.array(out_t, np.float32), rr=np.array(out_r, np.float32), scale=np.array(out_s, np.float32))
