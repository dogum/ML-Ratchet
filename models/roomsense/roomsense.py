"""Standalone NumPy/SciPy RoomSense inference."""
import json
from pathlib import Path

import numpy as np

from pipeline import FALLBACK, preprocess, read_wav


def _conv(x, weight, bias, stride, pad):
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    windows = np.lib.stride_tricks.sliding_window_view(x, weight.shape[-1], axis=-1)
    return np.einsum('bclk,ock->bol', windows[:, :, ::stride], weight, optimize=True) + bias[None, :, None]


def forward(weights, curves, context):
    """Return [log RT60, log volume, sigma(log RT60), sigma(log volume)]."""
    curves = np.asarray(curves, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    x = np.maximum(0, _conv(curves, weights['conv1.weight'], weights['conv1.bias'], 4, 3))
    x = np.maximum(0, _conv(x, weights['conv2.weight'], weights['conv2.bias'], 4, 2))
    x = np.concatenate([x.reshape(len(x), -1), context], axis=1)
    x = np.maximum(0, x @ weights['hidden.weight'].T + weights['hidden.bias'])
    raw = x @ weights['output.weight'].T + weights['output.bias']
    anchor = np.ma.median(np.ma.masked_where(context[:, 3:] == 0, context[:, :3]), axis=1).filled(np.log(FALLBACK))
    anchor = np.where(context[:, 3:].sum(1) >= 2, anchor, np.log(FALLBACK))
    raw[:, 0] += anchor
    raw[:, 1] += np.log(150)
    raw[:, 2:] = np.logaddexp(0, raw[:, 2:]) + 0.035
    return raw


class RoomSense:
    def __init__(self, weights_path=None):
        path = Path(weights_path) if weights_path else Path(__file__).with_name('roomsense_weights.npz')
        with np.load(path, allow_pickle=False) as bundle:
            self.meta = json.loads(str(bundle['meta']))
            self.weights = {key: bundle[key] for key in bundle.files if key != 'meta'}

    def read(self, samples, sample_rate):
        feature = preprocess(samples, sample_rate)
        reading = forward(self.weights, feature.curves[None], feature.context[None])[0]
        means = reading[:2].astype(float)
        sigma = reading[2:] * np.asarray(self.meta['widen'])
        estimates = np.exp(means)
        bounds = np.exp(means[:, None] + sigma[:, None] * [-1, 1])
        return {'rt60_s': float(estimates[0]), 'rt60_interval_s': bounds[0].tolist(),
                'classical_rt60_s': feature.classical_seconds, 'volume_m3': float(estimates[1]),
                'volume_interval_m3': bounds[1].tolist(), 'snr_db': feature.snr_db,
                'onset_s': feature.onset_seconds}

    def read_file(self, path):
        samples, rate = read_wav(path)
        return self.read(samples, rate)

