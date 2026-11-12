"""BeatNet: heartbeat classes from one ECG lead, learned only from a simulated heart.

For every beat of a modified-lead-II ECG it returns the AAMI EC57 class
  N  normal (including bundle-branch-block beats), S  supraventricular ectopic, V  ventricular ectopic, F  fusion,
with class probabilities, the QRS width in ms with a 68% error bar, and the probability that the beat sits in atrial
fibrillation. Heart rate and RR come from the beat positions. NOT A MEDICAL DEVICE: a research and teaching model; do not use it to make clinical decisions.

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

class BeatNet:
    def __init__(self, path=os.path.join(HERE, 'beatnet_weights.npz')):
        z = np.load(path); self.meta = json.loads(str(z['meta'])); self.W = {k: z[k].astype(np.float32) for k in z.files if k != 'meta'}

    def forward(self, x, r):
        """x: (B, cin, 450) normalised windows; r: (B, 6) rhythm features -> raw outputs (B, 6): 4 logits, QRS mu, log-sigma"""
        h = np.asarray(x, np.float32); W = self.W
        for i, s in enumerate(self.meta['strides']):
            y = _gelu(_conv(h, W[f'b{i}_c1_w'], W[f'b{i}_c1_b'], s)); y = _conv(y, W[f'b{i}_c2_w'], W[f'b{i}_c2_b'])
            h = _gelu(y + _conv(h[:, :, ::s], W[f'b{i}_sk_w'], W[f'b{i}_sk_b']))
        m = h.shape[-1] // 2; k = self.meta['ctr'] // 2
        z = np.concatenate([h[:, :, m - k:m + k + 1].reshape(len(h), -1), h.mean(-1), h.max(-1), np.asarray(r, np.float32)], 1)
        for j in range(self.meta['n_head']):
            z = z @ W[f'h{j}_w'].T + W[f'h{j}_b']
            if j < self.meta['n_head'] - 1: z = _gelu(z)
        return z

    def predict_features(self, F, bs=256, prior=True):
        x = F['w'][:, None] if self.meta['cin'] == 1 else np.stack([F['w'], F['tpl']], 1)
        o = np.concatenate([self.forward(x[i:i + bs], np.nan_to_num(F['rr'][i:i + bs])) for i in range(0, len(x), bs)])
        lp = np.array(self.meta.get('logprior', [0, 0, 0, 0])) if prior else 0; z = o[:, :4] + lp; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        c = self.meta.get('qrs_cal', {'a': 0.0, 'b': 1.0, 'sig_factor': 1.0})
        q = np.exp(o[:, 4] * self.meta['qsd'] + self.meta['qmu']); qs = np.exp(np.clip(o[:, 5], -4, 2)) * self.meta['qsd'] * q
        out = dict(probs=p, label=np.array(self.meta['classes'])[p.argmax(1)], qrs_ms=c['a'] + c['b'] * q, qrs_sig_ms=qs * c['b'] * c['sig_factor'], raw=o)
        if o.shape[1] > 6: out['af_prob'] = 1 / (1 + np.exp(-o[:, 6]))      # probability that the beat sits in atrial fibrillation
        return out

    def classify(self, ecg, fs, beats=None):
        """ecg: one lead (modified lead II), mV; fs: Hz; beats: sample positions of the beats (reference annotations if you
        have them; otherwise a simple QRS detector finds them). Returns per-beat results and rhythm numbers."""
        beats = PL.detect(ecg, fs) if beats is None else np.asarray(beats); F = PL.features(ecg, fs, beats); out = self.predict_features(F)
        rr = np.diff(beats) / fs; out.update(beats=beats, heart_rate_bpm=60 / np.median(rr) if len(rr) else np.nan, rr_s=rr)
        if 'af_prob' in out: out['af_share'] = float(np.mean(out['af_prob'] > 0.5))   # share of beats read as atrial fibrillation
        return out
