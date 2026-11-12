"""BearingNet: which rolling-bearing fault is in this vibration recording?
From a few seconds of accelerometer signal on a rotating machine, plus the bearing's geometry and a rough shaft speed,
it returns the condition of that bearing (healthy N, outer-race fault OR, inner-race fault IR, rolling-element/ball fault B)
with probabilities, a refined shaft speed with an error bar, and a (weak) defect-size estimate.
Trained only on physics simulations injected into real healthy-machine vibration. numpy and scipy only.

    import bearingnet as BNet
    net = BNet.BearingNet()
    r = net.read(x, fs=12000, bearing=dict(n=9, d=7.94, D=39.04), rpm_nominal=1750)
    r['condition'], r['p'], r['rpm'], r['rpm_sig'], r['size_mm']
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import pipeline as PL
BEARINGS = {'SKF6205': dict(n=9, d=0.3126 * 25.4, D=1.537 * 25.4), 'SKF6203': dict(n=8, d=0.2656 * 25.4, D=1.122 * 25.4),
            'PB6203': dict(n=8, d=6.75, D=29.05)}

def _gelu(x):
    a = np.abs(x) * 0.7071067811865476; t = 1 / (1 + 0.3275911 * a)
    erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1 + np.sign(x) * erf)

def _conv(x, w, b, stride=1):
    k = w.shape[2]; p = k // 2; xp = np.pad(x, ((0, 0), (0, 0), (p, p))); win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)[:, :, ::stride]
    return np.einsum('bclk,ock->bol', win, w, optimize=True) + b[None, :, None]

class BearingNet:
    def __init__(self, path=os.path.join(HERE, 'bearingnet_weights.npz')):
        z = np.load(path); self.meta = json.loads(str(z['meta'])); self.W = {k: z[k].astype(np.float32) for k in z.files if k != 'meta'}

    def _trunk(self, x, p, n):
        W = self.W; h = _gelu(_conv(x, W[p + 'stem_w'], W[p + 'stem_b'], 2))
        for i in range(n):
            y = _gelu(_conv(h, W[f'{p}b{i}_c1_w'], W[f'{p}b{i}_c1_b'], 2)); y = _conv(y, W[f'{p}b{i}_c2_w'], W[f'{p}b{i}_c2_b'])
            h = _gelu(y + _conv(h[:, :, ::2], W[f'{p}b{i}_sk_w'], W[f'{p}b{i}_sk_b']))
        return np.concatenate([h.mean(-1), h.max(-1)], 1)

    def _mlp(self, z, p, n):
        for j in range(n):
            z = z @ self.W[f'{p}{j}_w'].T + self.W[f'{p}{j}_b']
            if j < n - 1: z = _gelu(z)
        return z

    def forward(self, F, S):
        """F: (B, 21, 295) hypothesis-aligned envelope strips, S: (B, 8) scalars -> raw outputs (B, 8):
        4 class logits (N, OR, IR, B), speed (mean, log sigma), size (mean, log sigma)"""
        m = self.meta; F = np.asarray(F, np.float32); S = np.asarray(S, np.float32); B = len(F)
        if m.get('arch', 'base') == 'base':
            return self._mlp(np.concatenate([self._trunk(F, '', m['n_blocks']), S], 1), 'h', m['n_head'])
        nb, nh = m['nb'], m['nh']                         # hypothesis-shared trunk: same weights for the shaft, BPFO, BPFI, BSF strips
        e = F[:, :nb * nh].reshape(B, nb, nh, -1).transpose(0, 2, 1, 3).reshape(B * nh, nb, -1)
        g = self._trunk(e, '', m['n_blocks']).reshape(B, nh, -1); r = self._trunk(F[:, -1:], 'r', m['n_rblocks'])
        lf = np.concatenate([self._mlp(np.concatenate([g[:, k], g[:, 0], np.repeat(self.W['emb'][k - 1][None], B, 0), S], 1), 'f', m['n_fault']) for k in (1, 2, 3)], 1)
        o = self._mlp(np.concatenate([g.reshape(B, -1), r, S], 1), 'h', m['n_rest'])
        return np.concatenate([o[:, :1], lf, o[:, 1:]], 1)

    def read(self, x, fs, bearing, rpm_nominal, max_win=8, calibrated=True):
        """x: 1-D acceleration (any unit) at fs Hz, at least ~2.7 s (shorter is mirror-padded). bearing: dict(n, d, D[, phi])
        (d, D in any one unit) or a name in BEARINGS. rpm_nominal: the speed you believe, within about +-5%."""
        g = BEARINGS[bearing] if isinstance(bearing, str) else bearing; f_nom = rpm_nominal / 60.0
        x = PL.pad_short(PL.to_fs(x, fs)); F, S = [], []
        for s in PL.windows(x, max_win):
            f, sc, _ = PL.window_features(x[s:s + PL.NWIN], g, f_nom); F.append(f); S.append(sc)
        o = self.forward(np.stack(F), np.stack(S)); m = self.meta; cal = m.get('calib', {}) if calibrated else {}
        lp = o[:, :4] / cal.get('temp', 1.0); lp = lp - lp.max(1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(1, keepdims=True)); lp = lp.mean(0); lp[0] += cal.get('nbias', 0.0)
        p = np.exp(lp - lp.max()); p /= p.sum()
        sp, sps = o[:, 4] * m['sp_scale'], np.exp(np.clip(o[:, 5], -5, 3)) * m['sp_scale']; w = 1 / sps ** 2
        spm = float((sp * w).sum() / w.sum()); sps_r = float(np.sqrt(len(w) / w.sum())) * cal.get('speed', 1.0)
        lz, lzs = o[:, 6] * m['sz_sd'] + m['sz_mu'], np.exp(np.clip(o[:, 7], -5, 3)) * m['sz_sd']; w2 = 1 / lzs ** 2
        lzm = float((lz * w2).sum() / w2.sum()); lzs_r = float(np.sqrt(len(w2) / w2.sum())) * cal.get('size', 1.0)
        return dict(condition=m['classes'][int(p.argmax())], p={c: float(v) for c, v in zip(m['classes'], p)}, rpm=float(rpm_nominal * 2 ** spm),
                    rpm_sig=float(rpm_nominal * 2 ** spm * sps_r * np.log(2)), size_mm=float(10 ** lzm), log10_size_sig=lzs_r, n_windows=len(F))
