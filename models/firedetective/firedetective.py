"""FireDetective numpy inference (no torch). Loads weights/firedetective_v1.npz and calib.json.

    import firedetective as fdet
    m = fdet.load()                       # from this folder
    out = m.predict(seq, seq_hidden, static)   # arrays [N, 12, 25], [N, 12, 25], [N, 17] (see feats.py)

Outputs (dict of arrays, one value per fire):
  hs, hs_sd        MTBS high-severity fraction of the burned area (dev-calibrated map of the simulated crown-fire
                   fraction) and its 1-sigma (dev widen-only factor applied)
  crown            simulated crown-fire fraction (passive + active) of the burned area, as the simulator defines it
  wind, wind_lo, wind_hi   daily-mean 10 m wind (m/s, gridMET definition) inferred WITHOUT the wind input; 68 % range
  lfmc, lfmc_sd    live fuel moisture (% dry mass) the fire behaviour implies
  ros_adj          effective rate-of-spread multiplier relative to uncalibrated Rothermel (simulator label)
  m1               effective afternoon 1-h dead fuel moisture (%)
  wind_regime      share of the burned area where wind (not slope) dominated the spread
  p_spot, p_ia     probabilities that spotting was active / an initial-attack line was built (simulator labels)
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REG = ['crown', 'wind', 'lfmc', 'ros', 'm1', 'mfac', 'regime', 'wf']


def _conv1d(x, w, b):
    # x [B, C, T], w [O, C, K], same padding
    K = w.shape[2]; p = K // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p)))
    B, C, T = x.shape
    out = np.zeros((B, w.shape[0], T), np.float64)
    for k in range(K):
        out += np.einsum('bct,oc->bot', xp[:, :, k:k + T], w[:, :, k])
    return out + b[None, :, None]


def relu(x):
    return np.maximum(x, 0)


class FireDetective:
    def __init__(self, W, calib):
        self.W = {k: np.asarray(v, np.float64) for k, v in W.items()}
        self.cal = calib

    def raw(self, seq, static):
        """Seed ensemble: mean of member means; variance = mean member variance + spread of member means."""
        ids = sorted({k.split('.')[0] for k in self.W if '.' in k})
        outs = [self._raw1(seq, static, i) for i in ids]
        mu = np.stack([o[0] for o in outs])
        var = np.stack([o[1] ** 2 for o in outs]).mean(0) + mu.var(0)
        p = np.stack([o[2] for o in outs]).mean(0)
        return mu.mean(0), np.sqrt(var), p

    def _raw1(self, seq, static, mid):
        W = {k[len(mid) + 1:]: v for k, v in self.W.items() if k.startswith(mid + '.')}
        W.update({k: v for k, v in self.W.items() if '.' not in k})
        S = np.asarray(seq, np.float64)
        mask = (S[..., 0] > 0).astype(np.float64)
        X = (S - W['seq_mu']) / W['seq_sd'] * mask[..., None]
        T = (np.asarray(static, np.float64) - W['st_mu']) / W['st_sd']
        h = relu(X @ W['emb.weight'].T + W['emb.bias']) * mask[..., None]
        h = h.transpose(0, 2, 1)
        h = relu(_conv1d(h, W['c1.weight'], W['c1.bias'])) * mask[:, None, :]
        h = relu(_conv1d(h, W['c2.weight'], W['c2.bias'])) * mask[:, None, :]
        n = np.maximum(mask.sum(1, keepdims=True), 1)
        mean = h.sum(2) / n
        mx = (h - 1e4 * (1 - mask[:, None, :])).max(2)
        s = relu(T @ W['st.weight'].T + W['st.bias'])
        z = relu(np.concatenate([mean, mx, s], 1) @ W['f1.weight'].T + W['f1.bias'])
        z = relu(z @ W['f2.weight'].T + W['f2.bias'])
        mu = z @ W['mu.weight'].T + W['mu.bias']
        ls = np.clip(z @ W['ls.weight'].T + W['ls.bias'], -5, 3)
        bl = z @ W['bn.weight'].T + W['bn.bias']
        return mu, np.exp(ls), 1 / (1 + np.exp(-bl))

    def predict(self, seq, seq_hidden, static):
        c = self.cal
        mu, sd, pb = self.raw(seq, static)
        muh, sdh, _ = self.raw(seq_hidden, static)
        e = np.exp(mu[:, 0])
        crown = (1.02 * e - 0.02) / (1 + e)
        dcdy = 1.04 * e / (1 + e) ** 2
        hs = c['hs_map']['a'] + c['hs_map']['b'] * crown
        hs_sd = abs(c['hs_map']['b']) * dcdy * sd[:, 0] * c['hs_widen']
        wsd = sdh[:, 1] * c['wind_widen']
        return dict(hs=np.clip(hs, 0, 1), hs_sd=hs_sd, crown=crown, wind=np.exp(muh[:, 1]),
                    wind_lo=np.exp(muh[:, 1] - wsd), wind_hi=np.exp(muh[:, 1] + wsd),
                    lfmc=mu[:, 2] * 100, lfmc_sd=sd[:, 2] * 100 * c['lfmc_widen'], ros_adj=np.exp(mu[:, 3]),
                    m1=np.exp(mu[:, 4]), wind_regime=np.clip(mu[:, 6], 0, 1), p_spot=pb[:, 0], p_ia=pb[:, 1])


def load(folder=HERE):
    W = dict(np.load(os.path.join(folder, 'weights', 'firedetective_v1.npz')))
    cal = json.load(open(os.path.join(folder, 'weights', 'calib.json')))
    return FireDetective(W, cal)
