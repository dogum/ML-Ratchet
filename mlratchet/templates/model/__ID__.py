"""__NAME__ inference — numpy only, so it runs anywhere.

Train in any framework, then export: fold BatchNorm into the convolutions
(``mlratchet.nn.fold_batchnorm``), save the weights to ``__ID___weights.npz`` with a JSON ``meta``
entry, and check numpy == framework to about 1e-5 (``training/export.py``).

This module deliberately does not import ``mlratchet`` — a released model is one folder you can copy.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _erf(x):
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return np.sign(x) * y


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))


def _conv1d(x, w, b, stride=1, pad=0):
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    n, c, L = x.shape
    o, _, k = w.shape
    out = (L - k) // stride + 1
    s0, s1, s2 = x.strides
    cols = np.lib.stride_tricks.as_strided(x, (n, c, out, k), (s0, s1, s2 * stride, s2), writeable=False)
    return np.einsum("nclk,ock->nol", cols, w, optimize=True) + b[None, :, None]


class __NAME__:
    def __init__(self, weights=os.path.join(HERE, "__ID___weights.npz")):
        z = np.load(weights)
        self.w = {k: z[k] for k in z.files if k != "meta"}
        self.meta = json.loads(str(z["meta"])) if "meta" in z.files else {}

    def forward(self, x):
        """TODO: the network, layer by layer, matching the training framework."""
        raise NotImplementedError

    def read(self, raw):
        """TODO: preprocess exactly as in training, run the network, return outputs with
        calibrated error bars (apply the dev widening factors stored in ``self.meta``)."""
        raise NotImplementedError
