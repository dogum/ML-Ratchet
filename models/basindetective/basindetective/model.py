"""BasinDetective network, numpy/scipy only (a GELU MLP with Gaussian heads, or an ensemble of them)."""
import math

import numpy as np

REG_TARGETS = ['d_logq', 'd_rr', 'd_bfi', 'd_logq5', 'd_logq95', 'd_hfd',
               'log_kslow', 'gw_share', 'log_soilcap', 'log_pmult', 'log_dor']
CLS_TARGET = 'reg'
UNITS = {'d_logq': 'log ratio of mean flow', 'd_rr': 'runoff ratio change', 'd_bfi': 'BFI change',
         'd_logq5': 'log ratio of Q5 (low flow)', 'd_logq95': 'log ratio of Q95 (high flow)',
         'd_hfd': 'days (half-flow date shift)', 'log_kslow': 'ln(days)', 'gw_share': 'fraction',
         'log_soilcap': 'ln(mm)', 'log_pmult': 'ln(true P / forcing P)', 'log_dor': 'log10(storage / annual flow)'}


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def gelu_exact(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


class Detective:
    """numpy forward pass. W = dict from an .npz written by train.export(), or an ensemble file written by
    pack_ensemble() (members m0_*, m1_*, ...: mean of the members' means, sigma^2 = mean member sigma^2 +
    spread of the means, probability = mean of the members')."""

    def __init__(self, path, _W=None):
        if _W is None:
            z = np.load(path, allow_pickle=False)
            _W = {k: z[k] for k in z.files}
        self.W = _W
        self.members = None
        if 'n_members' in self.W:
            n = int(self.W['n_members'])
            self.members = []
            for i in range(n):
                Wi = {k[len(f'm{i}_'):]: v for k, v in self.W.items() if k.startswith(f'm{i}_')}
                Wi['feat_names'] = self.W['feat_names']
                Wi['targets'] = self.W['targets']
                self.members.append(Detective(None, Wi))
        self.names = [str(s) for s in self.W['feat_names']]
        self.nl = int(self.W['n_layers']) if 'n_layers' in self.W else 0
        self.targets = [str(s) for s in self.W['targets']]
        self.sig_scale = self.W['sig_scale'] if 'sig_scale' in self.W else np.ones(len(self.targets))
        self.cls_temp = float(self.W['cls_temp']) if 'cls_temp' in self.W else 1.0

    def norm(self, X):
        if self.members:
            return self.members[0].norm(X)
        X = np.asarray(X, np.float64)
        Z = (X - self.W['x_med']) / self.W['x_iqr']
        Z = np.clip(Z, -6, 6)
        return np.where(np.isfinite(Z), Z, 0.0)

    def raw(self, X):
        h = self.norm(X)
        for i in range(self.nl):
            h = gelu_exact(h @ self.W[f'w{i}'].T + self.W[f'b{i}'])
        return h @ self.W['wo'].T + self.W['bo']

    def forward(self, X):
        if self.members:
            outs = [m.forward(X) for m in self.members]
            mus = np.stack([o[0] for o in outs])
            sds = np.stack([o[1] for o in outs])
            mu = mus.mean(0)
            sd = np.sqrt((sds ** 2).mean(0) + mus.var(0)) * self.sig_scale
            p = np.stack([o[2] for o in outs]).mean(0)
            if self.cls_temp != 1.0:
                lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
                p = 1.0 / (1.0 + np.exp(-lg / self.cls_temp))
            return mu, sd, p
        o = self.raw(X)
        nt = len(self.targets)
        mu_z, ls = o[:, :nt], o[:, nt:2 * nt]
        logit = o[:, 2 * nt]
        mu = mu_z * self.W['y_sd'] + self.W['y_mu']
        sd = np.exp(np.clip(ls, -7, 5)) * self.W['y_sd'] * self.sig_scale
        p = 1.0 / (1.0 + np.exp(-logit / self.cls_temp))
        return mu, sd, p

    def predict(self, X):
        mu, sd, p = self.forward(np.atleast_2d(X))
        out = []
        for i in range(mu.shape[0]):
            d = {t: (float(mu[i, j]), float(sd[i, j])) for j, t in enumerate(self.targets)}
            d['p_regulated'] = float(p[i])
            out.append(d)
        return out


