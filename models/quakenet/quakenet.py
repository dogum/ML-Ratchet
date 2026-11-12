"""QuakeNet: one seismometer, one earthquake. From 41 minutes of three-component ground motion at a single station
(1 minute before the P wave to 40 after), how big was the quake, how far away, how deep, and from which direction?

Learned from physics alone (ak135 Green's functions, real station noise), checked on 3,105 real records it never saw.
  mw       moment magnitude, with a 68% error bar mw_sig
  dist     epicentral distance in degrees (valid 25-95), dist_sig
  depth    km; logz_sig is the 68% error bar on log10(depth)
  baz      back-azimuth: the compass direction from the station toward the quake, degrees; kappa its von Mises concentration
           (roughly, the 68% half-width is 57 / sqrt(kappa) degrees)
numpy and scipy only. See demo.py and MODEL_CARD.md."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import pipeline as PL

def _gelu(x):
    a = np.abs(x) * 0.7071067811865476; t = 1 / (1 + 0.3275911 * a)
    erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1 + np.sign(x) * erf)

def _conv(x, w, b, stride=1):
    k = w.shape[2]; p = k // 2; xp = np.pad(x, ((0, 0), (0, 0), (p, p))); win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)[:, :, ::stride]
    return np.einsum('bclk,ock->bol', win, w, optimize=True) + b[None, :, None]

class QuakeNet:
    def __init__(self, path=os.path.join(HERE, 'quakenet_weights.npz')):
        z = np.load(path); self.meta = json.loads(str(z['meta'])); self.W = {k: z[k].astype(np.float32) for k in z.files if k != 'meta'}
        self.mu, self.sd = np.array(self.meta['mu_y']), np.array(self.meta['sd_y'])

    def forward(self, x, s):
        """x: (B, 3, 2460) normalized window; s: (B, 2) scalars -> raw outputs (B, 8)"""
        env = np.log10(np.stack([np.convolve(np.pad(np.abs(c), 4), np.ones(9) / 9, 'valid') for c in np.abs(x).reshape(-1, x.shape[-1])]).reshape(x.shape) + 1e-3)
        h = np.concatenate([x, env], 1).astype(np.float32); W = self.W
        for i in range(self.meta['n_blocks']):
            y = _gelu(_conv(h, W[f'b{i}_c1_w'], W[f'b{i}_c1_b'], 2)); y = _conv(y, W[f'b{i}_c2_w'], W[f'b{i}_c2_b'])
            h = _gelu(y + _conv(h[:, :, ::2], W[f'b{i}_sk_w'], W[f'b{i}_sk_b']))
        z = np.concatenate([h.mean(-1), h.max(-1), s], 1)
        for j in range(self.meta['n_head']):
            z = z @ W[f'h{j}_w'].T + W[f'h{j}_b']
            if j < self.meta['n_head'] - 1: z = _gelu(z)
        return z

    def predict_windows(self, W, ls, peak, bs=32, calibrated=True):
        s = np.stack([(np.asarray(ls) + 6) / 2, np.log10(np.maximum(np.asarray(peak, float), 1.0))], 1).astype(np.float32)
        o = np.concatenate([self.forward(np.asarray(W[i:i + bs], np.float32), s[i:i + bs]) for i in range(0, len(W), bs)])
        f = self.meta.get('sigma_factor', {}); F = np.array([f.get('mw', 1), f.get('dist', 1), f.get('logz', 1)])   # error bars widened to real-Earth size on the dev records
        mu = o[:, :3] * self.sd + self.mu; sg = np.exp(np.clip(o[:, 3:6], -6, 3)) * self.sd * (F if calibrated else 1); v = o[:, 6:8]
        kap = np.sqrt((v ** 2).sum(1)) / (f.get('baz', 1) ** 2 if calibrated else 1)
        return dict(mw=mu[:, 0], mw_sig=sg[:, 0], dist=mu[:, 1], dist_sig=sg[:, 1], depth=10 ** mu[:, 2], logz=mu[:, 2], logz_sig=sg[:, 2],
                    baz=np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360, kappa=kap)

    def read(self, zne, p_guess):
        """zne: (3, n) vertical, north, east ground velocity in m/s at 1 sample/s, instrument response removed, band 0.008-0.35 Hz;
        p_guess: the sample where you expect (or saw) the P wave. Returns one dict of floats, plus the pick it used."""
        W, ls, p, peak = PL.read(np.asarray(zne, float), int(p_guess)); r = self.predict_windows(W[None], [ls], [peak])
        out = {k: float(v[0]) for k, v in r.items()}; out['p_pick'] = int(p); out['p_strength'] = float(peak); return out
