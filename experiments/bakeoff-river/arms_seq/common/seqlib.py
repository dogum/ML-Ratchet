"""River Bake-off, arms_seq: the ONE shared input pipeline + numpy inference for the four sequence learners.

Every arm (LSTM, transformer, state-space, TCN) gets exactly the same thing:
  inputs   FlowCast's 11 scaled daily features (P/10, log1p P, PET/5, T_mean/T_min/T_max /10, SRAD/150, vp/1000,
           wind/3, sin/cos day of year) + FlowCast's 18-value static vector standardised with the train bank's mean/sd,
           repeated at every day -> 29 channels per day
  window   730 days = 365 warm-up + 365 target days; inference in 365-day target blocks, each with its own 365-day
           warm-up (FlowCast's predict_arrays)
  head     Linear(H -> 6): mean = softplus, quantiles 5/25/50/75/95 % = cumulative softplus (monotone), all in units
           of the basin's mean precipitation p_mean = exp(static[11]); multiplied back to mm/day
  causal   every arm sees only forcing up to and including day t when predicting day t (as the LSTM does)
Only the sequence core differs: lstm | transformer | ssm | tcn. numpy only (float64).
"""
import os

import numpy as np

WARM = 365
L = 730
QKEYS = ['q05', 'q25', 'q50', 'q75', 'q95']


# ------------------------------------------------------------------ shared input pipeline
def dyn_features(F, doy):
    """FlowCast's 11 daily features. F (T, 8) raw forcing (C.DYN order), doy (T,)."""
    F = np.asarray(F, np.float32)
    P = np.maximum(F[:, 0], 0)
    out = np.empty((len(F), 11), np.float32)
    out[:, 0] = P / 10.0; out[:, 1] = np.log1p(P); out[:, 2] = F[:, 1] / 5.0
    out[:, 3] = F[:, 2] / 10.0; out[:, 4] = F[:, 3] / 10.0; out[:, 5] = F[:, 4] / 10.0
    out[:, 6] = F[:, 5] / 150.0; out[:, 7] = F[:, 6] / 1000.0; out[:, 8] = F[:, 7] / 3.0
    w = 2 * np.pi * np.asarray(doy, np.float32) / 365.25
    out[:, 9] = np.sin(w); out[:, 10] = np.cos(w)
    return out


def blocks_of(batch):
    """-> X (nb, 730, 11) float32, list of (a, b) target-index blocks. Exactly FlowCast.predict_arrays' blocking."""
    F = np.asarray(batch['forcing'], np.float32)
    doy = np.asarray(batch['doy'])
    n_target = int(batch['n_target'])
    assert int(batch['n_warm']) == WARM
    blocks, a = [], 0
    while a < n_target:
        b = min(a + 365, n_target); blocks.append((a, b)); a = b
    X = []
    for a, b in blocks:
        Fw = F[a:a + WARM + (b - a)]
        if len(Fw) < L:
            Fw = np.concatenate([Fw, np.repeat(Fw[-1:], L - len(Fw), 0)])
        X.append(dyn_features(Fw, (doy[a] - 1 + np.arange(L)) % 365 + 1))
    return np.stack(X), blocks


def inputs(W, X, st):
    """X (B, T, 11), st (B, 18) -> z (B, T, 29): daily features + standardised statics repeated."""
    s = (np.asarray(st, np.float64) - W['st_mu']) / W['st_sd']
    return np.concatenate([np.asarray(X, np.float64), np.repeat(s[:, None, :], X.shape[1], 1)], 2)


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


def head(W, h):
    out = h @ W['head.weight'].T + W['head.bias']
    mean = _softplus(out[..., 0])
    q = np.cumsum(np.concatenate([_softplus(out[..., 1:2]) * 0.2, _softplus(out[..., 2:])], -1), -1)
    return mean, q


# ------------------------------------------------------------------ the four cores (numpy mirrors of train/nets.py)
def core_lstm(W, z, cfg):
    Bn, T, _ = z.shape
    H = W['lstm.weight_hh_l0'].shape[1]
    xs = z @ W['lstm.weight_ih_l0'].T + (W['lstm.bias_ih_l0'] + W['lstm.bias_hh_l0'])
    Wh = W['lstm.weight_hh_l0'].T
    h = np.zeros((Bn, H)); c = np.zeros((Bn, H)); hs = np.empty((Bn, T, H))
    for t in range(T):
        g = xs[:, t] + h @ Wh
        i, f, gg, o = _sig(g[:, :H]), _sig(g[:, H:2 * H]), np.tanh(g[:, 2 * H:3 * H]), _sig(g[:, 3 * H:])
        c = f * c + i * gg
        h = o * np.tanh(c)
        hs[:, t] = h
    return hs


def sinusoid(T, d):
    pos = np.arange(T)[:, None].astype(np.float64)
    i = np.arange(0, d, 2).astype(np.float64)
    ang = pos / (10000.0 ** (i / d))
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang[:, :d // 2])
    return pe


def core_transformer(W, z, cfg):
    d, nh, nl = cfg['d'], cfg['heads'], cfg['layers']
    p = cfg.get('patch', 1)
    e = _lin(z, W, 'inp')
    Bn, T, _ = e.shape
    if p > 1:      # causal patched tokens (see train/nets.py TransformerNet.core)
        padn = (-T) % p
        K = (T + padn) // p
        ep = np.concatenate([np.zeros((Bn, padn, d)), e], 1).reshape(Bn, K, p * d)
        x = _lin(ep, W, 'pemb') + sinusoid(K, d)[None]
    else:
        x = e + sinusoid(T, d)[None]
    Tt = x.shape[1]
    dh = d // nh
    mask = np.triu(np.full((Tt, Tt), -np.inf), 1)
    for l in range(nl):
        pre = f'blocks.{l}.'
        y = _ln(x, W[pre + 'ln1.weight'], W[pre + 'ln1.bias'])
        qkv = _lin(y, W, pre + 'qkv')                       # (B, T, 3d)
        att = np.empty((Bn, Tt, d))
        for bi in range(Bn):
            for hh in range(nh):
                q = qkv[bi, :, hh * dh:(hh + 1) * dh]
                k = qkv[bi, :, d + hh * dh:d + (hh + 1) * dh]
                v = qkv[bi, :, 2 * d + hh * dh:2 * d + (hh + 1) * dh]
                s = q @ k.T / np.sqrt(dh) + mask
                s = np.exp(s - s.max(1, keepdims=True))
                s /= s.sum(1, keepdims=True)
                att[bi, :, hh * dh:(hh + 1) * dh] = s @ v
        x = x + _lin(att, W, pre + 'proj')
        y = _ln(x, W[pre + 'ln2.weight'], W[pre + 'ln2.bias'])
        x = x + _lin(_gelu(_lin(y, W, pre + 'ff1')), W, pre + 'ff2')
    if p == 1:
        return _ln(x, W['lnf.weight'], W['lnf.bias'])
    ctx = np.concatenate([np.repeat(W['c0'][None, None], Bn, 0), x[:, :-1]], 1)
    ctx = np.repeat(ctx, p, 1)[:, padn:]
    loc = causal_conv_depthwise(e, W['local.weight'], W['local.bias'])
    return _ln(e + ctx + loc, W['lnf.weight'], W['lnf.bias'])


def causal_conv_depthwise(x, w, b):
    """x (B, T, C), w (C, 1, k): per-channel causal conv (torch groups=C, left pad k-1)."""
    k = w.shape[2]
    T = x.shape[1]
    xp = np.concatenate([np.zeros((x.shape[0], k - 1, x.shape[2])), x], 1)
    y = np.zeros_like(x) + b
    for j in range(k):
        y += xp[:, j:j + T] * w[:, 0, j]
    return y


def s4d_kernel(W, p, T):
    dt = np.exp(W[p + 'log_dt'])                                   # (H,)
    A = -np.exp(W[p + 'log_A_re']) + 1j * W[p + 'A_im']            # (H, N2)
    C = W[p + 'C'][..., 0] + 1j * W[p + 'C'][..., 1]               # (H, N2)
    dtA = dt[:, None] * A
    Cb = C * (np.exp(dtA) - 1.0) / A
    K = 2.0 * np.real(np.einsum('hn,hnl->hl', Cb, np.exp(dtA[:, :, None] * np.arange(T)[None, None, :])))
    return K                                                       # (H, T)


def _silu(x):
    return x * _sig(x)


def mamba_layer(W, p, x, cfg):
    """Selective diagonal SSM with scalar decay per head (train/nets.py MambaBlock), as a plain recurrence over days:
    state_t = exp(dt_t A) state_{t-1} + dt_t x_t B_t^T ; y_t = state_t C_t + D x_t ; out = W_o (y * silu(z))."""
    Bn, T, d = x.shape
    nh, N = cfg['heads'], cfg['N']
    P = d // nh
    zx = _lin(_ln(x, W[p + 'ln.weight'], W[p + 'ln.bias']), W, p + 'in_proj')
    z, xs = zx[..., :d], zx[..., d:2 * d]
    Bm, Cm, dt = zx[..., 2 * d:2 * d + N], zx[..., 2 * d + N:2 * d + 2 * N], zx[..., 2 * d + 2 * N:]
    dt = _softplus(dt + W[p + 'dt_bias'])                     # (B, T, nh)
    A = -np.exp(W[p + 'A_log'])
    X = xs.reshape(Bn, T, nh, P)
    dA = np.exp(dt * A)                                        # (B, T, nh)
    h = np.zeros((Bn, nh, P, N))
    Y = np.empty((Bn, T, nh, P))
    for t in range(T):
        h = h * dA[:, t, :, None, None] + (dt[:, t, :, None] * X[:, t])[..., None] * Bm[:, t, None, None, :]
        Y[:, t] = np.einsum('bhpn,bn->bhp', h, Cm[:, t])
    Y = Y + X * W[p + 'D'][:, None]
    return x + _lin(Y.reshape(Bn, T, d) * _silu(z), W, p + 'out_proj')


def core_ssm(W, z, cfg):
    nl = cfg['layers']
    Bn, T, _ = z.shape
    x = _lin(z, W, 'inp')
    if cfg.get('pre_mlp'):
        x = x + _lin(_gelu(_lin(x, W, 'pre1')), W, 'pre2')
    if cfg.get('mixer', 's4d') == 'mamba':
        for l in range(nl):
            x = mamba_layer(W, f'blocks.{l}.', x, cfg)
        return _ln(x, W['lnf.weight'], W['lnf.bias'])
    n = 1 << (2 * T - 1).bit_length()          # FFT length >= 2T-1 (exact linear convolution); power of 2
    for l in range(nl):
        p = f'blocks.{l}.'
        y = _ln(x, W[p + 'ln.weight'], W[p + 'ln.bias'])
        K = s4d_kernel(W, p, T)
        u = np.fft.rfft(y, n=n, axis=1)                            # (B, n/2+1, H)
        k = np.fft.rfft(K, n=n, axis=1).T[None]                    # (1, n/2+1, H)
        y = np.fft.irfft(u * k, n=n, axis=1)[:, :T] + y * W[p + 'D']
        y = _gelu(y)
        g = _lin(y, W, p + 'out')
        H = g.shape[-1] // 2
        x = x + g[..., :H] * _sig(g[..., H:])
    return _ln(x, W['lnf.weight'], W['lnf.bias'])


def causal_conv(x, w, b, dil):
    """x (B, T, Cin), w (Cout, Cin, k) torch Conv1d layout, left-padded by (k-1)*dil."""
    Bn, T, Ci = x.shape
    k = w.shape[2]
    pad = (k - 1) * dil
    xp = np.concatenate([np.zeros((Bn, pad, Ci)), x], 1)
    y = np.zeros((Bn, T, w.shape[0])) + b
    for j in range(k):
        y += xp[:, j * dil:j * dil + T] @ w[:, :, j].T
    return y


def core_tcn(W, z, cfg):
    Bn, T, _ = z.shape
    x = _lin(z, W, 'inp')
    for l, dil in enumerate(cfg['dilations']):
        p = f'blocks.{l}.'
        y = _ln(x, W[p + 'ln.weight'], W[p + 'ln.bias'])
        y = causal_conv(y, W[p + 'conv.weight'], W[p + 'conv.bias'], dil)
        if cfg.get('gated'):                       # WaveNet gate: tanh(a) x sigmoid(b)
            c = y.shape[-1] // 2
            y = np.tanh(y[..., :c]) * _sig(y[..., c:])
        else:
            y = _gelu(y)
        x = x + _lin(y, W, p + 'mix')
    return _ln(x, W['lnf.weight'], W['lnf.bias'])


CORES = {'lstm': core_lstm, 'transformer': core_transformer, 'ssm': core_ssm, 'tcn': core_tcn}


def forward(W, cfg, X, st):
    """X (B, 730, 11), st (B, 18) -> mean (B, T), q (B, T, 5) in units of p_mean."""
    z = inputs(W, X, st)
    return head(W, CORES[cfg['arch']](W, z, cfg))


class Model:
    def __init__(self, path, chunk=16):
        zf = np.load(path, allow_pickle=False)
        import json
        self.cfg = json.loads(str(zf['cfg_json']))
        self.W = {k: zf[k].astype(np.float64) for k in zf.files if k != 'cfg_json'}
        self.chunk = chunk if self.cfg['arch'] != 'transformer' else 8
        self._cache = {}

    def _run(self, Xs, Ss):
        means, qs = [], []
        for a in range(0, len(Xs), self.chunk):
            m, q = forward(self.W, self.cfg, Xs[a:a + self.chunk], Ss[a:a + self.chunk])
            means.append(m); qs.append(q)
        return np.concatenate(means), np.concatenate(qs)

    def predict_many(self, batches):
        Xs, Ss, owner = [], [], []
        for j, b in enumerate(batches):
            X, blocks = blocks_of(b)
            Xs.append(X); Ss.append(np.repeat(np.asarray(b['static'], np.float64)[None], len(blocks), 0))
            owner.append(blocks)
        m, q = self._run(np.concatenate(Xs), np.concatenate(Ss))
        outs, k = [], 0
        for b, blocks in zip(batches, owner):
            pm = float(np.exp(np.float64(np.asarray(b['static'], np.float32)[11])))
            mm = np.concatenate([m[k + i, WARM:WARM + bb - aa] for i, (aa, bb) in enumerate(blocks)])
            qq = np.concatenate([q[k + i, WARM:WARM + bb - aa] for i, (aa, bb) in enumerate(blocks)])
            k += len(blocks)
            qq = np.maximum(qq, 0.0) * pm
            o = {'mean': mm * pm}
            for j, key in enumerate(QKEYS):
                o[key] = qq[:, j]
            outs.append(o)
        return outs

    def predict(self, batch):
        return self.predict_many([batch])[0]


def find_common():
    return os.path.dirname(os.path.abspath(__file__))
