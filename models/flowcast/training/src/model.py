"""FlowCast network: an LSTM over daily forcing with static basin attributes repeated at every step.
Output (per day), in units of the basin's mean precipitation (Q / p_mean): the mean flow (NSE loss) and the
5/25/50/75/95 % quantiles (pinball loss), built monotone with cumulative softplus."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

QS = [0.05, 0.25, 0.5, 0.75, 0.95]


class FlowNet(nn.Module):
    def __init__(self, n_dyn, n_st, H=64, dropout=0.1):
        super().__init__()
        self.n_dyn, self.n_st, self.H = n_dyn, n_st, H
        self.lstm = nn.LSTM(n_dyn + n_st, H, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(H, 1 + len(QS))
        self.register_buffer('st_mu', torch.zeros(n_st))
        self.register_buffer('st_sd', torch.ones(n_st))

    def forward(self, x, st):
        s = (st - self.st_mu) / self.st_sd
        z = torch.cat([x, s[:, None, :].expand(-1, x.shape[1], -1)], -1)
        h, _ = self.lstm(z)
        o = self.head(self.drop(h))
        mean = Fn.softplus(o[..., 0])
        q = torch.cumsum(torch.cat([Fn.softplus(o[..., 1:2]) * 0.2, Fn.softplus(o[..., 2:])], -1), -1)
        return mean, q


def loss_fn(mean, q, y, sig, mask=None):
    """NSE* (Kratzert 2019) on the mean + per-basin-normalised pinball loss on the quantiles."""
    w = 1.0 / (sig + 0.1)
    if mask is None:
        mask = torch.ones_like(y)
    n = mask.sum().clamp(min=1)
    nse = (((mean - y) * w[:, None]) ** 2 * mask).sum() / n
    taus = torch.tensor(QS, dtype=y.dtype)
    e = y[..., None] - q
    pin = torch.maximum(taus * e, (taus - 1) * e).mean(-1) * w[:, None]
    pin = (pin * mask).sum() / n
    return nse + pin, nse.detach(), pin.detach()


# ------------------------------------------------------------------ numpy inference (release + checks)
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def np_forward(W, x, st):
    """W: dict of numpy weights (export.py); x (T, n_dyn), st (n_st,). Returns mean (T,), q (T, 5)."""
    s = (st - W['st_mu']) / W['st_sd']
    T = x.shape[0]
    H = W['w_hh'].shape[1]
    Wi = W['w_ih']; Wh = W['w_hh']; b = W['b_ih'] + W['b_hh']
    xs = np.concatenate([x, np.repeat(s[None, :], T, 0)], 1) @ Wi.T + b
    h = np.zeros(H); c = np.zeros(H)
    hs = np.empty((T, H))
    for t in range(T):
        g = xs[t] + Wh @ h
        i, f, gg, o = _sig(g[:H]), _sig(g[H:2 * H]), np.tanh(g[2 * H:3 * H]), _sig(g[3 * H:])
        c = f * c + i * gg
        h = o * np.tanh(c)
        hs[t] = h
    out = hs @ W['head_w'].T + W['head_b']
    mean = _softplus(out[:, 0])
    q = np.cumsum(np.concatenate([_softplus(out[:, 1:2]) * 0.2, _softplus(out[:, 2:])], 1), 1)
    return mean, q
