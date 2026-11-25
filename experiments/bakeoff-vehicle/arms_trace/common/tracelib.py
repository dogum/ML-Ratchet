"""Vehicle Bake-off, arms_trace: the ONE shared input pipeline + numpy inference for the four trace learners.

Every arm (1-D CNN, GRU, transformer, S4D) gets exactly the same thing; only the sequence core differs:
  trace    the harness's 1 Hz trace (L, 8) -> the shared 5-s resampling (HARNESS.md section 4: block means of the 8
           channels + mean v^2, mean v^3, summed positive kinetic-energy increment per unit mass) -> 12 scaled channels
           per 5-s block (the 11 + the block's valid-second fraction)
  statics  spec (49, standardised with the train bank's mean / sd), the 5 road-load numbers (log m, log CdA, log Crr,
           Crr speed coefficient, rotating-mass fraction; standardised), log trip distance and duration (from the
           trace), and log(E_phys / A0) when the rung uses the physics anchor (0 otherwise) -> 57 values, repeated at
           every block (early fusion) and fed again to the head
  stem     Linear(69 -> d) + a residual pointwise MLP (d -> 2d -> d, GELU): the input-side nonlinearity the River
           Bake-off found S4D needs, given to every arm
  core     cnn | gru | transformer | s4d over the blocks (non-causal: the whole trip is known)
  pool     mean over the valid blocks (an intensive summary); traces longer than 1,440 blocks (7,200 s; only real
           heavy-duty legs) are cut into <= 1,440-block chunks, each chunk is encoded, and the chunk means are
           averaged weighted by their block counts (mean = the rule for intensive quantities)
  head     [pool | statics] -> MLP -> 5 log-quantiles of log(E / anchor) (median + cumulative softplus, monotone)
           + 5 log component ratios (aero, rolling, brakes, aux, powertrain losses) / anchor
  anchor   'dumb': A0 = the train bank's median energy per km for (architecture, class) x trip distance (no physics);
           'phys': E_phys = EnergyNet's generic road-load estimate recomputed from the trace inputs
           (bake.e_phys_from_trace, the harness function; declared feature engineering)
numpy only (float64). Encodings of the trace (5-s blocks) are cached on trace_key (EPA rows repeat 5 cycles).
"""
import json
import os

import numpy as np

QKEYS = ['q05', 'q25', 'q50', 'q75', 'q95']
BLOCK = 5
CHUNK = 1440                 # blocks (7,200 s): the sim bank's maximum trace length
N_DYN = 12
N_ST = 57
COMP_KEYS = ['aero', 'rolling', 'brake', 'aux', 'losses']


# ------------------------------------------------------------------ shared input pipeline
def block_means(tr, k=BLOCK):
    """= bake.block_means (the harness's shared resampling), + the valid-second count per block."""
    tr = np.asarray(tr, np.float64)
    L = len(tr); nb = -(-L // k)
    pad = nb * k - L
    v = tr[:, 0]
    dke = np.maximum(np.diff(v * v, prepend=v[0] * v[0]) / 2.0, 0.0)
    X = np.concatenate([tr, (v * v)[:, None], (v ** 3)[:, None], dke[:, None]], 1)
    if pad:
        X = np.concatenate([X, np.full((pad, X.shape[1]), np.nan)], 0)
    X = X.reshape(nb, k, -1)
    out = np.nanmean(X[:, :, :10], 1)
    cnt = np.full(nb, float(k)); cnt[-1] = k - pad
    return np.concatenate([out, np.nansum(X[:, :, 10:], 1), cnt[:, None]], 1)


def dyn_features(tr):
    """(L, 8) 1 Hz trace -> (nb, 12) scaled block channels."""
    b = block_means(tr)
    f = np.empty((len(b), N_DYN))
    f[:, 0] = b[:, 0] / 10.0                        # v [m/s]
    f[:, 1] = np.clip(b[:, 1], -0.2, 0.2) * 10.0    # grade
    f[:, 2] = np.clip(b[:, 2] / 100.0, -20, 20)     # elevation relative to the start [m]
    f[:, 3] = b[:, 3] / 20.0                        # T_amb [C]
    f[:, 4] = b[:, 4] / 5.0                         # headwind [m/s]
    f[:, 5] = b[:, 5] / 5.0                         # crosswind
    f[:, 6] = (b[:, 6] - 1.2) * 10.0                # rho_air
    f[:, 7] = b[:, 7] / 2.0                         # surface code
    f[:, 8] = b[:, 8] / 100.0                       # mean v^2
    f[:, 9] = b[:, 9] / 1000.0                      # mean v^3
    f[:, 10] = b[:, 10] / 20.0                      # positive kinetic-energy increments per unit mass [m2/s2]
    f[:, 11] = b[:, 11] / BLOCK                     # valid-second fraction of the block
    return f


def trip_dist_dur(tr):
    return max(float(np.sum(np.asarray(tr, np.float64)[:, 0])), 1.0), float(len(tr))


def anchor_dumb(W, arch, cls, dist):
    rate = W['dumb_rate'][np.asarray(arch, int), np.asarray(cls, int)]
    return rate * np.maximum(dist, 100.0)


def rl_feats(rl):
    rl = np.asarray(rl, np.float64)
    return np.stack([np.log(np.maximum(rl[:, 0], 1.0)), np.log(np.maximum(rl[:, 1], 1e-3)),
                     np.log(np.maximum(rl[:, 2], 1e-5)), rl[:, 3] * 100.0, rl[:, 4]], 1)


def statics(W, cfg, spec, rl, dist, dur, A0, Eph):
    """-> (n, 57) standardised statics."""
    s = (np.asarray(spec, np.float64) - W['sp_mu']) / W['sp_sd']
    r = (rl_feats(rl) - W['rl_mu']) / W['rl_sd']
    o = np.stack([np.log(dist / 5000.0), np.log(dur / 900.0)], 1)
    ph = np.log(Eph / A0)[:, None] if cfg['anchor'] == 'phys' else np.zeros((len(s), 1))
    return np.clip(np.concatenate([s, r, o, ph], 1), -8, 8)


# ------------------------------------------------------------------ numpy layers
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def _gelu(x):          # tanh approximation (the torch nets use approximate='tanh')
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _lin(x, W, p):
    return x @ W[p + '.weight'].T + W[p + '.bias']


def stem(W, z):
    x = _lin(z, W, 'inp')
    return x + _lin(_gelu(_lin(x, W, 'stem1')), W, 'stem2')


# ------------------------------------------------------------------ the four cores (numpy mirrors of train/nets.py)
# Every core takes x (B, T, d) of equal-length sequences (no padding at inference) and returns (B, T, d).
def conv_same(x, w, b, dil):
    """x (B, T, Cin), w (Cout, Cin, k) torch Conv1d layout, zero 'same' padding (k odd), dilation dil."""
    Bn, T, Ci = x.shape
    k = w.shape[2]
    pad = (k - 1) // 2 * dil
    xp = np.concatenate([np.zeros((Bn, pad, Ci)), x, np.zeros((Bn, pad, Ci))], 1)
    y = np.zeros((Bn, T, w.shape[0])) + b
    for j in range(k):
        y += xp[:, j * dil:j * dil + T] @ w[:, :, j].T
    return y


def core_cnn(W, x, cfg):
    for l, dil in enumerate(cfg['dilations']):
        p = f'blocks.{l}.'
        y = _ln(x, W[p + 'ln.weight'], W[p + 'ln.bias'])
        y = _gelu(conv_same(y, W[p + 'conv.weight'], W[p + 'conv.bias'], dil))
        x = x + _lin(y, W, p + 'mix')
    return x


def _gru_dir(W, x, p):
    Bn, T, _ = x.shape
    Wi, Wh = W[p + '.weight_ih_l0'], W[p + '.weight_hh_l0']
    bi, bh = W[p + '.bias_ih_l0'], W[p + '.bias_hh_l0']
    H = Wh.shape[1]
    xs = x @ Wi.T + bi
    WhT = Wh.T
    h = np.zeros((Bn, H)); out = np.empty((Bn, T, H))
    for t in range(T):
        gh = h @ WhT + bh
        gi = xs[:, t]
        r = _sig(gi[:, :H] + gh[:, :H])
        z = _sig(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1.0 - z) * n + z * h
        out[:, t] = h
    return out


def core_gru(W, x, cfg):
    f = _gru_dir(W, x, 'gru')
    b = _gru_dir(W, x[:, ::-1], 'grub')[:, ::-1]
    return x + _lin(np.concatenate([f, b], -1), W, 'gmix')


def sinusoid(T, d):
    pos = np.arange(T)[:, None].astype(np.float64)
    i = np.arange(0, d, 2).astype(np.float64)
    ang = pos / (10000.0 ** (i / d))
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang[:, :d // 2])
    return pe


def core_transformer(W, x, cfg):
    """patch tokens of p blocks (zero-padded to a multiple of p), non-causal pre-LN encoder, tokens repeated back."""
    d, nh, nl, p = cfg['d'], cfg['heads'], cfg['layers'], cfg['patch']
    Bn, T, _ = x.shape
    padn = (-T) % p
    K = (T + padn) // p
    xp = np.concatenate([x, np.zeros((Bn, padn, d))], 1).reshape(Bn, K, p * d)
    t = _lin(xp, W, 'pemb') + sinusoid(K, d)[None]
    dh = d // nh
    for l in range(nl):
        pre = f'tblocks.{l}.'
        y = _ln(t, W[pre + 'ln1.weight'], W[pre + 'ln1.bias'])
        qkv = _lin(y, W, pre + 'qkv')
        att = np.empty((Bn, K, d))
        for hh in range(nh):
            q = qkv[:, :, hh * dh:(hh + 1) * dh]
            k = qkv[:, :, d + hh * dh:d + (hh + 1) * dh]
            v = qkv[:, :, 2 * d + hh * dh:2 * d + (hh + 1) * dh]
            s = np.einsum('bqc,bkc->bqk', q, k) / np.sqrt(dh)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            att[:, :, hh * dh:(hh + 1) * dh] = np.einsum('bqk,bkc->bqc', s, v)
        t = t + _lin(att, W, pre + 'proj')
        y = _ln(t, W[pre + 'ln2.weight'], W[pre + 'ln2.bias'])
        t = t + _lin(_gelu(_lin(y, W, pre + 'ff1')), W, pre + 'ff2')
    return x + np.repeat(t, p, 1)[:, :T]


def s4d_kernel(W, p, T, j):
    dt = np.exp(W[p + 'log_dt'])                                   # (H,)
    A = -np.exp(W[p + 'log_A_re']) + 1j * W[p + 'A_im']            # (H, N2)
    C = W[p + 'C'][j, ..., 0] + 1j * W[p + 'C'][j, ..., 1]         # (H, N2)
    dtA = dt[:, None] * A
    Cb = C * (np.exp(dtA) - 1.0) / A
    return 2.0 * np.real(np.einsum('hn,hnl->hl', Cb, np.exp(dtA[:, :, None] * np.arange(T)[None, None, :])))


def _kernel_cached(W, p, T, j):
    """the kernel at lags 0..CHUNK-1, computed once per layer and direction (lag l depends on l only), sliced to T."""
    key = f'_kernel.{p}{j}'
    if key not in W:
        W[key] = s4d_kernel(W, p, CHUNK, j)
    return W[key][:, :T]


def _fftconv(y, K, n):
    u = np.fft.rfft(y, n=n, axis=1)
    k = np.fft.rfft(K, n=n, axis=1).T[None]
    return np.fft.irfft(u * k, n=n, axis=1)[:, :y.shape[1]]


def core_s4d(W, x, cfg):
    """bidirectional diagonal state space (S4D-Lin init, ZOH): a forward and a backward kernel per channel."""
    Bn, T, _ = x.shape
    n = 1 << (2 * T - 1).bit_length()
    for l in range(cfg['layers']):
        p = f'blocks.{l}.'
        y = _ln(x, W[p + 'ln.weight'], W[p + 'ln.bias'])
        yf = _fftconv(y, _kernel_cached(W, p, T, 0), n)
        yb = _fftconv(y[:, ::-1], _kernel_cached(W, p, T, 1), n)[:, ::-1]
        y = _gelu(yf + yb + y * W[p + 'D'])
        g = _lin(y, W, p + 'out')
        H = g.shape[-1] // 2
        x = x + g[..., :H] * _sig(g[..., H:])
    return x


CORES = {'cnn': core_cnn, 'gru': core_gru, 'transformer': core_transformer, 's4d': core_s4d}


def encode(W, cfg, dyn, st):
    """dyn (B, T, 12), st (B, 57) -> pooled (B, d): stem -> core -> final LN -> mean over blocks."""
    z = np.concatenate([dyn, np.repeat(st[:, None, :], dyn.shape[1], 1)], 2)
    x = stem(W, z)
    h = CORES[cfg['arch']](W, x, cfg)
    h = _ln(h, W['lnf.weight'], W['lnf.bias'])
    return h.mean(1)


def head(W, pooled, st):
    o = _lin(_gelu(_lin(np.concatenate([pooled, st], 1), W, 'hd1')), W, 'hd2')
    q50 = o[:, 0]
    q25 = q50 - _softplus(o[:, 1]); q05 = q25 - _softplus(o[:, 2])
    q75 = q50 + _softplus(o[:, 3]); q95 = q75 + _softplus(o[:, 4])
    return np.stack([q05, q25, q50, q75, q95], 1), o[:, 5:10]


# ------------------------------------------------------------------ the model
class Model:
    def __init__(self, path, group_max=64):
        zf = np.load(path, allow_pickle=False)
        self.cfg = json.loads(str(zf['cfg_json']))
        self.W = {k: zf[k].astype(np.float64) for k in zf.files if k != 'cfg_json'}
        self.group_max = group_max
        self._dyn = {}                  # trace_key -> (dyn (nb, 12), dist, dur)

    def _trace(self, key, tr):
        if key is not None and key in self._dyn:
            return self._dyn[key]
        r = (dyn_features(tr),) + trip_dist_dur(tr)
        if key is not None:
            if len(self._dyn) > 4096:
                self._dyn.clear()
            self._dyn[key] = r
        return r

    def anchors(self, batch, dist):
        A0 = anchor_dumb(self.W, batch['arch'], batch['cls'], dist)
        if self.cfg['anchor'] == 'phys':
            import bake                                  # the harness (E_phys from the trace-track inputs)
            Eph = np.maximum(bake.e_phys_from_trace(batch), 1e-3)
            return A0, Eph, Eph
        return A0, A0, A0

    def predict(self, batch):
        n = int(batch['n'])
        keys = batch.get('trace_key')
        enc = [self._trace(None if keys is None else str(keys[i]), batch['traces'][i]) for i in range(n)]
        dist = np.array([e[1] for e in enc]); dur = np.array([e[2] for e in enc])
        A0, Eph, anc = self.anchors(batch, dist)
        st = statics(self.W, self.cfg, batch['spec'], batch['rl'], dist, dur, A0, Eph)
        # sequences (chunks of <= CHUNK blocks), grouped by exact length: no padding at inference
        items = []                       # (row, n_blocks, dyn chunk)
        for i, e in enumerate(enc):
            dyn = e[0]
            for a in range(0, len(dyn), CHUNK):
                items.append((i, dyn[a:a + CHUNK]))
        d = self.W['lnf.weight'].shape[0]
        pooled = np.zeros((n, d)); wsum = np.zeros(n)
        bylen = {}
        for j, (i, dy) in enumerate(items):
            bylen.setdefault(len(dy), []).append(j)
        for T, js in bylen.items():
            for a in range(0, len(js), self.group_max):
                g = js[a:a + self.group_max]
                rows = np.array([items[j][0] for j in g])
                dyn = np.stack([items[j][1] for j in g])
                P = encode(self.W, self.cfg, dyn, st[rows])
                np.add.at(pooled, rows, P * T)
                np.add.at(wsum, rows, T)
        pooled /= wsum[:, None]
        q, c = head(self.W, pooled, st)
        Q = anc[:, None] * np.exp(q)
        out = {'pred': Q[:, 2].copy()}
        for j, k in enumerate(QKEYS):
            out[k] = Q[:, j]
        out['components'] = anc[:, None] * np.exp(c)
        return out

    def predict_many(self, batches):
        return [self.predict(b) for b in batches]
