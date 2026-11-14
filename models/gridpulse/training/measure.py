"""Measurement model: simulated deviation (0.1 s, from the loss at t = 0) -> NESO-like 1 s series with real noise,
then the SAME onset rule griddata used on the real data (steepest 1 s fall, walk back while falling >= 10 mHz/s),
then the network window and the measured-style labels."""
import numpy as np

PRE, POST = 20, 60          # network window: t0-20 .. t0+60 (81 samples)
K0 = 66                     # noise segment index of event time k = 0  (segment covers k = -66 .. +73)


def to_1s(dev, phase, mode="mean", tau=0.0, dt=0.1, nk=74):
    """dev: (B, n) deviation [Hz] at t = 0, dt, ... (0 before t = 0). phase: (B,) in [0,1).
    Returns (B, nk) 1 s values at t_k = k - 1 + phase, k = 0..nk-1 (index k <-> event-relative second k-1+phase).
    mode 'sample' = instantaneous value, 'mean' = average over the preceding second. tau: extra sensor lag [s]."""
    B, n = dev.shape
    t = np.arange(n) * dt
    x = dev.astype(np.float64)
    if tau > 0:
        a = 1 - np.exp(-dt / tau); y = np.zeros_like(x); acc = np.zeros(B)
        for i in range(n):
            acc = acc + a * (x[:, i] - acc); y[:, i] = acc
        x = y
    tk = np.arange(nk)[None, :] - 1 + phase[:, None]                     # (B, nk)
    out = np.zeros((B, nk))
    if mode == "sample":
        for b in range(B):
            out[b] = np.interp(tk[b], t, x[b], left=0.0, right=x[b, -1])
    else:
        C = np.concatenate([np.zeros((B, 1)), np.cumsum((x[:, 1:] + x[:, :-1]) * 0.5 * dt, 1)], 1)
        for b in range(B):
            hi = np.interp(np.clip(tk[b], 0, t[-1]), t, C[b]); lo = np.interp(np.clip(tk[b] - 1, 0, t[-1]), t, C[b])
            out[b] = hi - lo
    return out


def compose(dev1s, noise, scale=1.0):
    """Place 1 s deviation (k = 0 .. nk-1 <-> seconds -1+phase ..) into noise segments (B, 140): event second 0 sits at
    segment index K0. Returns (B, 140) deviation-from-nominal series (noise + signal)."""
    B, nk = dev1s.shape
    out = noise.astype(np.float64) * scale
    out[:, K0 - 1:K0 - 1 + nk] += dev1s
    out[:, K0 - 1 + nk:] += dev1s[:, -1:]
    return out


def onset(x, lo=K0 - 3, hi=K0 + 8):
    """griddata's onset rule on 1 s series x (B, L): steepest 1 s fall in [lo, hi], walk back while d1 <= -0.01."""
    d1 = np.diff(x, axis=1, prepend=x[:, :1])
    i = lo + np.argmin(d1[:, lo:hi], axis=1)
    j = i.copy()
    for _ in range(10):
        step = (d1[np.arange(len(j)), j - 1] <= -0.01) & (i - j < 10)
        j = np.where(step, j - 1, j)
    return np.clip(j - 1, 60, x.shape[1] - 66)


def window(x, t0):
    """Network window and measured-style metrics from series x (B, L) with onset index t0 (B,)."""
    B = len(x)
    idx = t0[:, None] + np.arange(-60, 0)[None, :]
    f0 = x[np.arange(B)[:, None], idx].mean(1)
    w = x[np.arange(B)[:, None], t0[:, None] + np.arange(-PRE, POST + 1)[None, :]] - f0[:, None]
    post = x[np.arange(B)[:, None], t0[:, None] + np.arange(0, 61)[None, :]] - f0[:, None]
    nadir = post.min(1)
    settle = x[np.arange(B)[:, None], t0[:, None] + np.arange(55, 66)[None, :]].mean(1) - f0
    d10 = (x[:, 10:] - x[:, :-10])
    fall10 = np.array([d10[b, max(t0[b] - 10, 0):t0[b] + 20].min() for b in range(B)])
    return dict(w=w.astype(np.float32), f0=f0, nadir=nadir, settle=settle, fall10=fall10)
