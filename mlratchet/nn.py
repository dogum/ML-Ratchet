"""Numpy building blocks for portable inference.

Every released model in the collection runs on numpy (and sometimes scipy) only: train in any
framework, fold BatchNorm into the convolutions, export the weights to ``.npz``, and check that
numpy matches the framework to about 1e-5. These are the layers those exports use.

Shapes follow PyTorch: 1-D input ``(batch, channels, length)``, 2-D ``(batch, channels, h, w)``,
weights ``(out, in, k)`` / ``(out, in, kh, kw)``.
"""
from __future__ import annotations

import numpy as np


def erf(x):
    """Numpy-only erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7) — no scipy needed."""
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return np.sign(x) * y


def gelu(x):
    """Exact GELU (PyTorch's default), via the numpy-only erf the collection's exports use."""
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476))


def gelu_tanh(x):
    """GELU, tanh approximation (PyTorch ``approximate='tanh'``)."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def linear(x, w, b=None):
    """``x @ w.T + b`` with PyTorch's ``(out, in)`` weight layout."""
    y = x @ w.T
    return y + b if b is not None else y


def conv1d(x, w, b=None, stride: int = 1, pad: int = 0, dilation: int = 1):
    """1-D cross-correlation (PyTorch ``Conv1d``) via a strided view and one einsum."""
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    n, c, L = x.shape
    o, ci, k = w.shape
    assert ci == c, f"channels {c} vs weight {ci}"
    span = dilation * (k - 1) + 1
    out_len = (L - span) // stride + 1
    s0, s1, s2 = x.strides
    cols = np.lib.stride_tricks.as_strided(x, (n, c, out_len, k), (s0, s1, s2 * stride, s2 * dilation), writeable=False)
    y = np.einsum("nclk,ock->nol", cols, w, optimize=True)
    return y + b[None, :, None] if b is not None else y


def conv2d(x, w, b=None, stride=1, pad=0):
    """2-D cross-correlation (PyTorch ``Conv2d``), integer or (h, w) stride/pad."""
    sh, sw = (stride, stride) if np.isscalar(stride) else stride
    ph, pw = (pad, pad) if np.isscalar(pad) else pad
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    n, c, H, W = x.shape
    o, ci, kh, kw = w.shape
    oh, ow = (H - kh) // sh + 1, (W - kw) // sw + 1
    s0, s1, s2, s3 = x.strides
    cols = np.lib.stride_tricks.as_strided(x, (n, c, oh, ow, kh, kw), (s0, s1, s2 * sh, s3 * sw, s2, s3), writeable=False)
    y = np.einsum("nchwij,ocij->nohw", cols, w, optimize=True)
    return y + b[None, :, None, None] if b is not None else y


def fold_batchnorm(w, b, gamma, beta, mean, var, eps: float = 1e-5):
    """Fold an inference-mode BatchNorm that follows a conv/linear layer into its weights.

    Returns ``(w', b')`` with ``bn(conv(x, w, b)) == conv(x, w', b')``.
    """
    scale = gamma / np.sqrt(var + eps)
    w2 = w * scale.reshape(-1, *([1] * (w.ndim - 1)))
    b0 = b if b is not None else np.zeros_like(mean)
    return w2, (b0 - mean) * scale + beta


def max_abs_diff(a, b) -> float:
    """Parity check helper: ``max |a − b|`` (aim for ~1e-5 between numpy and the training framework)."""
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
