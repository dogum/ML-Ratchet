"""GridPulse v1.0 (numpy + scipy only).
From a 1 s power-system frequency record after a sudden loss of generation, estimate MW lost, system inertia,
frequency response (MW/Hz at NERC point B), governor headroom and FFR (weak), and forecast the nadir and the
frequency at 60 s, each with a 1-sigma band. GB (50 Hz) and ERCOT (60 Hz) as conditioned inputs.
Also GridPulse-Summary for ERCOT-style reports (delta f, inertia, load, IBR share).
Trained only on GridSim simulations in real NESO quiet noise. See MODEL_CARD.md."""
import os
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
PRE, POST = 20, 60


def onset(f):
    """griddata's onset rule on a 1 s series: steepest 1 s fall, walk back while falling >= 10 mHz/s; returns the
    index of the last sample before the fall."""
    f = np.asarray(f, float)
    d1 = np.diff(f, prepend=f[0]); i = int(np.nanargmin(d1)); j = i
    while j - 1 >= 1 and d1[j - 1] <= -0.01 and i - j < 10:
        j -= 1
    return j - 1


def window(f, i0):
    """81-sample window -20..+60 s around onset index i0, as deviation from the 60 s pre-event mean."""
    f = np.asarray(f, float)
    if i0 < 60 or i0 + POST >= len(f):
        raise ValueError('need >= 60 s before and 60 s after the onset at 1 s resolution')
    f0 = np.nanmean(f[i0 - 60:i0])
    w = f[i0 - PRE:i0 + POST + 1] - f0
    if np.isnan(w).any():
        idx = np.arange(len(w)); ok = ~np.isnan(w); w = np.interp(idx, idx[ok], w[ok])
    return w, f0


class GridPulse:
    def __init__(self, weights=os.path.join(HERE, 'gridpulse_weights.npz'), ercot=os.path.join(HERE, 'ercot_head.npz')):
        self.W = dict(np.load(weights)); self.ctx = bool(self.W['ctx'])
        self.E = dict(np.load(ercot)) if os.path.exists(ercot) else None

    def _features(self, w, N, is_gb, f0, demand):
        w = np.asarray(w, float) * (50.0 / np.asarray(f0, float))[:, None] * 10
        tt = np.arange(-PRE, POST + 1)[None, :]
        x = np.where(tt <= np.asarray(N)[:, None], w, 0.0); N = np.asarray(N, float)
        F = np.c_[x, np.asarray(is_gb, float), N / 60, np.log(N) / 4]
        if self.ctx:
            d = np.asarray(demand, float); k = np.isfinite(d)
            F = np.c_[F, np.where(k, np.log(np.where(k, d, 30000.0) / 30000.0), 0.0), k.astype(float)]
        return F

    def forward(self, w, N, is_gb, f0, demand):
        W = self.W
        h = (self._features(w, N, is_gb, f0, demand) - W['xm']) / W['xs']; i = 0
        while f'b{i}_w' in W:
            h = h @ W[f'b{i}_w'].T + W[f'b{i}_b']; h = 0.5 * h * (1 + erf(h / np.sqrt(2))); i += 1
        mu = h @ W['mu_w'].T + W['mu_b']; ls = np.clip(h @ W['ls_w'].T + W['ls_b'], -6, 3)
        return mu * W['ys'] + W['ym'], np.exp(ls) * W['ys'] * W['widen']

    def read_windows(self, w, N=60, system='GB', demand_MW=np.nan):
        """w: (n, 81) windows (Hz deviation, -20..+60 s). Returns dict of arrays."""
        w = np.atleast_2d(w); n = len(w); f0 = 50.0 if system == 'GB' else 60.0
        mu, sg = self.forward(w, np.full(n, N), np.full(n, system == 'GB'), np.full(n, f0), np.broadcast_to(demand_MW, (n,)))
        r = {}
        for i, (k, s) in enumerate([('loss_MW', 1), ('E_GWs', 1e-3), ('beta_MW_Hz', 1), ('headroom_MW', 1)]):
            r[k] = np.exp(mu[:, i]) * s; r[k + '_lo'] = np.exp(mu[:, i] - sg[:, i]) * s; r[k + '_hi'] = np.exp(mu[:, i] + sg[:, i]) * s
        r['ffr_MW'] = np.exp(mu[:, 4]) - 100
        k = f0 / 50 / 10
        r['nadir_Hz'], r['nadir_sigma_Hz'] = mu[:, 5] * k, sg[:, 5] * k
        r['settle_Hz'], r['settle_sigma_Hz'] = mu[:, 6] * k, sg[:, 6] * k
        return r

    def read(self, t, f, system='GB', demand_MW=np.nan, N=60, onset_index=None):
        """t [s], f [Hz] at 1 s. Onset found with griddata's rule unless given. N = seconds after onset to use.
        nadir_Hz / settle_Hz are deviations from the pre-event mean (settle = mean of 55-65 s)."""
        f = np.asarray(f, float)
        i0 = onset(f) if onset_index is None else onset_index
        w, f0 = window(f, i0)
        r = {k: float(v[0]) for k, v in self.read_windows(w[None], N, system, demand_MW).items()}
        r.update(onset_t=float(np.asarray(t)[i0]), f_pre_Hz=float(f0), N=N)
        return r

    def ercot_summary(self, delta_f_Hz, inertia_MWs, load_MW, ibr_share, mw_loss=None):
        """GridPulse-Summary: MW lost from (delta f, inertia, load, IBR share); and, if mw_loss is given, the forecast
        drop and inertia from the other numbers. Returns (value, 1-sigma factor) pairs."""
        def pred(k, X):
            W = {a.split('__')[1]: self.E[a] for a in self.E if a.startswith(k + '__')}
            g = lambda h: 0.5 * h * (1 + erf(h / np.sqrt(2)))
            h = (np.asarray(X, np.float32)[None] - W['xm']) / W['xs']
            h = g(h @ W['b.0.weight'].T + W['b.0.bias']); h = g(h @ W['b.2.weight'].T + W['b.2.bias'])
            o = h @ W['o.weight'].T + W['o.bias']
            return float(np.exp(o[0, 0] * W['ys'] + W['ym'])), float(np.exp(np.exp(np.clip(o[0, 1], -6, 3)) * W['ys']))
        ldf, lE, ll = np.log(max(delta_f_Hz, 0.005)), np.log(inertia_MWs / 1e3), np.log(load_MW / 1e3)
        out = {'loss_MW': pred('loss', [ldf, lE, ll, ibr_share])}
        if mw_loss:
            out['delta_f_Hz'] = pred('dfnad', [np.log(mw_loss), lE, ll, ibr_share])
            out['inertia_GWs'] = pred('E', [ldf, np.log(mw_loss), ll, ibr_share])
        return out
