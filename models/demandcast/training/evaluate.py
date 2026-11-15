"""Prediction and metrics shared by the ratchet, the exam and the page."""
import json, numpy as np, torch
from common import *
import model as M
from train import XK, PK, masks

torch.set_num_threads(1)


def load(name):
    ck = torch.load(f'{ROOT}/ckpt/{name}.pt', weights_only=False)
    nets = {}
    for k in ('d', 'w', 's'):
        sd = ck[k]
        net = M.Net(sd['xm'].shape[0], ck[k + '_hidden']); net.load_state_dict(sd); net.eval()
        nets[k] = net
    return nets


def predict(nets, B):
    out = {}
    with torch.no_grad():
        for k, net in nets.items():
            X = torch.tensor(np.nan_to_num(B[XK[k]].astype(np.float32))[:, :net.xm.shape[0]])   # older rungs: first features
            P = torch.tensor(B[PK[k]].astype(np.float32)) if k != 'd' else None
            mu, sig = M.head(k, net, X, P)
            mu, sig = mu.numpy(), sig.numpy()
            if k != 'd':
                mu = np.clip(mu, 0, 1.2)
            if k == 's':
                night = B['night'] > 0.5
                mu = np.where(night, 0.0, mu); sig = np.where(night, M.FLOOR['s'], sig)
            out[k] = (mu, sig)
    return out


def errors(B, pred, kind, rows=None, scale=1.0, use_prior=False):
    """Returns dict: err (MAPE for demand, nMAE for CF, capacity-weighted if real), cov68, cov95, n."""
    y, m = masks(B, kind)
    if kind == 's':
        m = np.isfinite(B['ys']) & (B['hass'][:, None] if 'hass' in B else True)
        y = np.nan_to_num(B['ys'])
    m = m.astype(bool)
    if rows is not None: m &= rows[:, None]
    mu, sig = pred[kind]
    if use_prior:
        mu = np.zeros_like(mu) if kind == 'd' else B[PK[kind]]
    sig = sig * scale
    if m.sum() == 0: return dict(err=np.nan, cov68=np.nan, cov95=np.nan, n=0)
    z = np.abs(y - mu) / sig
    if kind == 'd':
        err = np.mean(np.abs(np.exp(mu - y) - 1)[m]) * 100
    else:
        ck = 'capw' if kind == 'w' else 'caps'
        w = np.broadcast_to(B[ck][:, None], y.shape) if ck in B else np.ones_like(y)
        err = np.sum((np.abs(mu - y) * w)[m]) / max(np.sum((y * w)[m]), 1e-9) * 100
    if kind == 's':
        mm = m & (B['night'] < 0.5)
        cov68 = float(np.mean((z <= 1)[mm])); cov95 = float(np.mean((z <= 1.96)[mm]))
    else:
        cov68 = float(np.mean((z <= 1)[m])); cov95 = float(np.mean((z <= 1.96)[m]))
    return dict(err=float(err), cov68=cov68, cov95=cov95, n=int(m.sum()))


def sim_score(B, pred):
    """Frozen-val objective: mean over outputs of skill = 1 - err / err(physics prior)."""
    out = {}
    for k in ('d', 'w', 's'):
        e = errors(B, pred, k); e0 = errors(B, pred, k, use_prior=True)
        out[k] = dict(err=e['err'], prior_err=e0['err'], skill=1 - e['err'] / e0['err'], cov68=e['cov68'], cov95=e['cov95'])
    out['objective'] = float(np.mean([out[k]['skill'] for k in ('d', 'w', 's')]))
    return out


def sigma_scale(B, pred, kind, rows=None):
    """Widen-only factor so that 68.3 % of dev hours fall within +-1 sigma."""
    y, m = masks(B, kind)
    m = m.astype(bool)
    if rows is not None: m &= rows[:, None]
    mu, sig = pred[kind]
    z = np.abs(y - mu)[m] / sig[m]
    return float(max(1.0, np.quantile(z, 0.683)))
