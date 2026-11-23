"""NeuralFoil (Sharpe & Hansman 2025, MIT licence) forward pass in plain numpy, for the exam baseline.

Re-implements neuralfoil.main.get_aero_from_kulfan_parameters (v0.3.3) and AeroSandbox's least-squares
Kulfan fit (N1 = 0.5, N2 = 1, 8 weights a side, LE term x(1-x)^8.5, TE thickness >= 0), so it runs without
AeroSandbox/casadi. Weights: the nn-*.npz files shipped in the neuralfoil 0.3.3 wheel (PyPI).
Only CL, CD, CM, Top_Xtr, Bot_Xtr and analysis_confidence are returned.
"""
import os
import numpy as np
from math import comb

WDIR = os.environ.get('NEURALFOIL_WEIGHTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tmp',
                                                         'nf', 'neuralfoil', 'nn_weights_and_biases'))
_eps = 10 / np.finfo(np.array(1.0).dtype).max
_ln_eps = np.log(_eps)


def kulfan_fit(Y, nw=8):
    """AeroSandbox get_kulfan_parameters(method='least_squares') on normalised Selig-order coordinates."""
    x = Y[:, 0]; y = Y[:, 1]
    le = np.argmin(x); is_upper = np.arange(len(x)) <= le
    C = np.sqrt(np.clip(x, 0, None)) * (1 - x)
    N = nw - 1
    S = np.stack([comb(N, i) * x ** i * (1 - x) ** (N - i) for i in range(N + 1)], 0)   # (nw, n)
    CS = (C[None, :] * S).T
    A = np.concatenate([np.where(is_upper[:, None], 0, CS), np.where(is_upper[:, None], CS, 0),
                        (x * np.maximum(1 - x, 0) ** (nw + 0.5))[:, None],
                        np.where(is_upper, x / 2, -x / 2)[:, None]], axis=1)
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    lw, uw, lew, te = sol[:nw], sol[nw:2 * nw], sol[-2], sol[-1]
    if te < 0:
        sol = np.linalg.lstsq(A[:, :-1], y, rcond=None)[0]
        lw, uw, lew, te = sol[:nw], sol[nw:2 * nw], sol[-1], 0.0
    return dict(lower_weights=lw, upper_weights=uw, leading_edge_weight=lew, TE_thickness=te)


class NeuralFoil:
    def __init__(self, size='xxxlarge'):
        self.p = dict(np.load(os.path.join(WDIR, f'nn-{size}.npz')))
        self.layers = sorted({int(k.split('.')[1]) for k in self.p})
        d = dict(np.load(os.path.join(WDIR, 'scaled_input_distribution.npz')))
        self.mean = d['mean_inputs_scaled']; self.icov = d['inv_cov_inputs_scaled']; self.nin = len(self.mean)

    def _net(self, x):
        x = x.T
        for k, i in enumerate(self.layers):
            x = self.p[f'net.{i}.weight'] @ x + self.p[f'net.{i}.bias'][:, None]
            if k < len(self.layers) - 1:
                x = x / (1 + np.exp(-x))            # swish
        return x.T

    def _maha(self, x):
        xm = x - self.mean[None, :]
        return np.sum(xm @ self.icov * xm, axis=1)

    def __call__(self, kp, alpha, Re, n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0):
        alpha = np.atleast_1d(np.asarray(alpha, float)); n = len(alpha)
        Re = np.broadcast_to(np.asarray(Re, float), (n,))
        rows = [*[np.full(n, kp['upper_weights'][i]) for i in range(8)],
                *[np.full(n, kp['lower_weights'][i]) for i in range(8)],
                np.full(n, kp['leading_edge_weight']), np.full(n, kp['TE_thickness'] * 50),
                np.sin(np.radians(2 * alpha)), np.cos(np.radians(alpha)), 1 - np.cos(np.radians(alpha)) ** 2,
                (np.log(Re) - 12.5) / 3.5, np.full(n, (n_crit - 9) / 4.5), np.full(n, xtr_upper),
                np.full(n, xtr_lower)]
        x = np.stack(rows, 1)
        y = self._net(x); y[:, 0] -= self._maha(x) / (2 * self.nin)
        xf = x.copy(); xf[:, :8] = -x[:, 8:16]; xf[:, 8:16] = -x[:, :8]; xf[:, 16] = -x[:, 16]
        xf[:, 18] = -x[:, 18]; xf[:, 23] = x[:, 24]; xf[:, 24] = x[:, 23]
        yf = self._net(xf); yf[:, 0] -= self._maha(xf) / (2 * self.nin)
        yu = yf.copy(); yu[:, 1] = -yf[:, 1]; yu[:, 3] = -yf[:, 3]; yu[:, 4] = yf[:, 5]; yu[:, 5] = yf[:, 4]
        y = (y + yu) / 2
        conf = 1 / (1 + np.exp(-np.clip(y[:, 0], _ln_eps, -_ln_eps)))
        return dict(CL=y[:, 1] / 2, CD=np.exp((y[:, 2] - 2) * 2), CM=y[:, 3] / 20,
                    Top_Xtr=np.clip(y[:, 4], 0, 1), Bot_Xtr=np.clip(y[:, 5], 0, 1), confidence=conf)
