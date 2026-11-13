"""TripDetective network: a 1-D CNN reads each trip (or window); trip embeddings of the same car are averaged
(set pooling), so one trip or many give one answer, and more trips give a tighter one.
Heads on the pooled car embedding: architecture, strategy, log mass, log displacement, log CdA, log Crr, engine
on/off behaviour (EV share, on-threshold power, max EV speed); per-trip heads (trip embedding + car embedding): the
energy ledger in kJ/km. Every regression has a mean and a log-sigma (Gaussian NLL)."""
import torch, torch.nn as nn, torch.nn.functional as Fn
import numpy as np
from features import NIN, nin, NSC

REG = ['mass', 'disp', 'CdA', 'Crr', 'ev_frac', 'P_on', 'v_ev']
LED_SCALE = np.array([200., 150., 150., 150., 1500., 150.], np.float32)


def reg_targets(meta):
    """vehicle-level regression targets in network units + mask"""
    y = np.full(len(REG), np.nan, np.float32)
    y[0] = np.log(meta['mass'] / 1800); y[2] = np.log(meta['CdA'] / 0.75); y[3] = np.log(meta['Crr'] / 0.008)
    if meta.get('disp') and np.isfinite(meta['disp']): y[1] = np.log(meta['disp'] / 2.5)
    for i, k in [(4, 'ev_frac'), (5, 'P_on_kW'), (6, 'v_ev')]:
        x = meta.get(k)
        if x is not None and np.isfinite(x):
            y[i] = x if k == 'ev_frac' else (np.log1p(max(x, 0)) - 2 if k == 'P_on_kW' else x / 10 - 1.5)
    return y


def from_reg(i, mu):
    k = REG[i]
    if k == 'mass': return 1800 * np.exp(mu)
    if k == 'disp': return 2.5 * np.exp(mu)
    if k == 'CdA': return 0.75 * np.exp(mu)
    if k == 'Crr': return 0.008 * np.exp(mu)
    if k == 'ev_frac': return mu
    if k == 'P_on': return np.expm1(mu + 2)
    if k == 'v_ev': return (mu + 1.5) * 10


class Block(nn.Module):
    def __init__(s, ci, co, k=5, stride=2):
        super().__init__()
        s.c = nn.Conv1d(ci, co, k, stride=stride, padding=k // 2); s.bn = nn.BatchNorm1d(co)

    def forward(s, x):
        return Fn.gelu(s.bn(s.c(x)))


class TDNet(nn.Module):
    def __init__(s, n_arch=7, n_strat=14, width=(32, 48, 64, 96, 128, 160), emb=128, feat=1):
        super().__init__()
        s.feat = feat
        s.stem = Block(nin(feat), width[0], k=7, stride=1)
        s.blocks = nn.ModuleList([Block(width[i], width[i + 1]) for i in range(len(width) - 1)])
        s.down = 2 ** (len(width) - 1)
        s.nsc = NSC if feat >= 3 else 0          # feature version 3: classical per-trip estimates join the trip embedding
        s.trip = nn.Sequential(nn.Linear(2 * width[-1] + 2 + s.nsc, 192), nn.GELU(), nn.Linear(192, emb))
        s.car = nn.Sequential(nn.Linear(emb + 1, 192), nn.GELU(), nn.Linear(192, emb), nn.GELU())
        s.h_arch = nn.Linear(emb, n_arch); s.h_strat = nn.Linear(emb, n_strat)
        s.h_reg = nn.Linear(emb, 2 * len(REG))
        s.h_led = nn.Sequential(nn.Linear(2 * emb, 128), nn.GELU(), nn.Linear(128, 12))

    def encode(s, x, mask, sc=None):
        """x (N, NIN, T), mask (N, T), sc (N, NSC) classical scalars (feat 3) -> trip embeddings (N, emb)"""
        m = mask[:, None]
        h = s.stem(x) * m                       # activations outside the trip are zeroed after every block, so the
        for b in s.blocks:                      # answer does not depend on how much padding a batch carries
            m = Fn.max_pool1d(m, 2, 2, ceil_mode=True)
            h = b(h)[:, :, :m.shape[2]] * m
        n = m.sum(2).clamp(min=1)
        mean = (h * m).sum(2) / n
        mx = (h * m + (m - 1) * 1e4).max(2).values
        dur = mask.sum(1, keepdim=True) / 1024.0
        parts = [mean, mx, dur, torch.log(dur + 1e-3)] + ([sc] if s.nsc else [])
        return s.trip(torch.cat(parts, 1))

    def forward(s, x, mask, group, n_groups, sc=None):
        """group (N,) car index of each trip -> dict of outputs per car and per trip"""
        e = s.encode(x, mask, sc)
        cnt = torch.zeros(n_groups, device=x.device).index_add_(0, group, torch.ones_like(group, dtype=x.dtype))
        z = torch.zeros(n_groups, e.shape[1], device=x.device).index_add_(0, group, e) / cnt[:, None]
        z = s.car(torch.cat([z, torch.log(cnt)[:, None]], 1))
        r = s.h_reg(z)
        led = s.h_led(torch.cat([e, z[group]], 1))
        return dict(arch=s.h_arch(z), strat=s.h_strat(z), mu=r[:, :len(REG)], ls=r[:, len(REG):].clamp(-6, 3),
                    led_mu=led[:, :6], led_ls=led[:, 6:].clamp(-6, 3))
