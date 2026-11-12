"""SwellCast: one wave buoy, one distant storm. From 10 days of a buoy's wave spectra (energy density plus the first
directional moments, as NOAA NDBC publishes them), how far away was the storm that made a swell, when did it blow,
from which direction did the swell come, and how big were its seas?

Learned from physics alone (simulated swell injected into real NDBC buoy records), checked on real hurricanes and
typhoons it never saw (IBTrACS best tracks as truth).
  D_km, logD_sig     distance to the storm (km); 68% error bar on log10(D)
  t0, t0_sig         storm peak time, hours relative to the window centre (68% error bar in hours)
  dir, kappa         compass direction the swell comes FROM (deg); von Mises concentration (68% half-width ~ 57/sqrt(kappa) deg)
  hs, loghs_sig      significant wave height in the storm (m), 68% error bar on log10
  p_swell, detected  network score that a storm swell arrives near the window centre; detected = score above the threshold
                     that gave 10% false alarms on quiet 2024 buoy windows
numpy only. See demo.py and MODEL_CARD.md."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FGRID = 0.0325 + 0.005 * np.arange(34)          # Hz
DT_H, HALF_H, NT = 3, 120, 80                   # 3-hour steps, 240-hour window

def _gelu(x):
    a = np.abs(x) * 0.7071067811865476; t = 1 / (1 + 0.3275911 * a)
    erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1 + np.sign(x) * erf)

def _conv(x, w, b, stride=1):
    """x (B, C, H, W), w (O, C, k, k), padding k//2"""
    k = w.shape[2]; p = k // 2; xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
    return np.einsum('bchwij,ocij->bohw', win, w, optimize=True) + b[None, :, None, None]

def features(X):
    """X: (B, 3, 80, 34) raw [E m^2/Hz, r1 cos(a1), r1 sin(a1)] with NaN for missing -> (B, 5, 34, 80)"""
    X = np.asarray(X, np.float32); E = X[:, 0]; m = np.isfinite(E)
    L = np.log10(np.maximum(np.where(m, E, 0.005), 0.005)); L = np.where(m, L, np.nan)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); med = np.nanmedian(L, axis=1, keepdims=True)
    med = np.where(np.isfinite(med), med, -2.3)
    Ln = np.where(m, (L + 0.5) / 1.2, 0.0); A = np.where(m, (L - med) / 0.6, 0.0)
    C = np.where(m, np.nan_to_num(X[:, 1]), 0); S = np.where(m, np.nan_to_num(X[:, 2]), 0)
    return np.ascontiguousarray(np.stack([Ln, A, C, S, m.astype(np.float32)], 1).transpose(0, 1, 3, 2)).astype(np.float32)

class SwellCast:
    def __init__(self, path=os.path.join(HERE, 'swellcast_weights.npz')):
        z = np.load(path); self.meta = json.loads(str(z['meta'])); self.W = {k: z[k].astype(np.float32) for k in z.files if k != 'meta'}
        self.mu, self.sd = np.array(self.meta['mu_y']), np.array(self.meta['sd_y'])

    def forward(self, f):
        W = self.W; h = _gelu(_conv(f, W['stem_w'], W['stem_b']))
        for i in range(self.meta['n_blocks']):
            y = _gelu(_conv(h, W[f'b{i}_c1_w'], W[f'b{i}_c1_b'], 2)); y = _conv(y, W[f'b{i}_c2_w'], W[f'b{i}_c2_b'])
            h = _gelu(y + _conv(h, W[f'b{i}_sk_w'], W[f'b{i}_sk_b'], 2))
        h = _gelu(_conv(h, W['red_w'], W['red_b'])); z = h.reshape(len(h), -1)
        for j in range(self.meta['n_head']):
            z = z @ W[f'h{j}_w'].T + W[f'h{j}_b']
            if j < self.meta['n_head'] - 1: z = _gelu(z)
        return z

    def predict(self, X, bs=32, calibrated=True):
        """X: (B, 3, 80, 34) windows -> dict of arrays"""
        o = np.concatenate([self.forward(features(X[i:i + bs])) for i in range(0, len(X), bs)])
        f = self.meta.get('sigma_factor', {}) if calibrated else {}
        mu = o[:, :3] * self.sd + self.mu; sg = np.exp(np.clip(o[:, 3:6], -6, 3)) * self.sd; v = o[:, 6:8]
        return dict(D_km=10 ** mu[:, 0], logD=mu[:, 0], logD_sig=sg[:, 0] * f.get('logD', 1), t0=mu[:, 1], t0_sig=sg[:, 1] * f.get('t0', 1),
                    hs=10 ** mu[:, 2], loghs=mu[:, 2], loghs_sig=sg[:, 2] * f.get('loghs', 1),
                    dir=np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360, kappa=np.sqrt((v ** 2).sum(1)) / f.get('dir', 1) ** 2,
                    p_swell=1 / (1 + np.exp(-o[:, 8])), detected=1 / (1 + np.exp(-o[:, 8])) > self.meta.get('det_thr', 0.5))

    def read(self, window):
        """window: (3, 80, 34) from pipeline.window(...). Returns one dict of floats."""
        r = self.predict(np.asarray(window)[None]); return {k: float(v[0]) for k, v in r.items()}
