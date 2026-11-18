"""StormNet. Standalone NumPy inference for the next three complete hours' minimum Dst."""
import json
from pathlib import Path

import numpy as np
from stormphysics import HISTORY, drivers, projected


def features(wind, dst, mode, params, law):
    e, p = drivers(wind)
    cols = [dst, e[:, -1], p[:, -1], wind[:, -1, 1] / 1000, wind[:, -1, 0]]
    anchor = dst
    if mode == "history":
        path, fraction, total = projected(e, p, dst, params, law)
        anchor = path.min(axis=1)
        for n in (3, 12, 36):
            cols.extend((e[:, -n:].mean(axis=1), p[:, -n:].mean(axis=1)))
        cols.extend((fraction, total, anchor - dst, path[:, -1] - path[:, 0]))
    return np.stack(cols, axis=1), anchor


def design(x, w):
    z = np.clip((x - w["mean"]) / w["scale"], -8, 8)
    hidden = np.tanh(z @ w["kernel"] + w["bias"])
    return np.column_stack((np.ones(len(z)), z, hidden))


def forward(wind, dst, meta, w):
    x, anchor = features(wind, dst, meta["mode"], meta["physics"], meta["wind_law"])
    h = design(x, w)
    median = anchor + h @ w["location"]
    sigma = np.exp(np.clip(h @ w["spread"], -1, 5))
    return median, sigma


class StormNet:
    """Supply chronological completed hours in columns [Bz GSM, speed, pressure]."""
    def __init__(self, weights=None):
        path = Path(weights) if weights else Path(__file__).with_name("stormnet_weights.npz")
        with np.load(path, allow_pickle=False) as z:
            self.meta = json.loads(str(z["meta"]))
            self.weights = {k: z[k] for k in z if k != "meta"}

    def predict(self, wind, last_dst, *, calibrated=True):
        """Return minimum Dst and one 68% half-width in nT, as scalars or batch arrays."""
        wind = np.asarray(wind, dtype=float)
        single = wind.ndim == 2
        if single:
            wind = wind[None, ...]
        dst = np.atleast_1d(np.asarray(last_dst, dtype=float))
        if wind.ndim != 3 or wind.shape[1:] != (HISTORY, 3) or dst.shape != (len(wind),):
            raise ValueError("Expected wind (36, 3) and one Dst, or (N, 36, 3) and (N,) Dst")
        if (not np.isfinite(wind).all() or not np.isfinite(dst).all() or
                np.any(wind[..., 1:] <= 0)):
            raise ValueError("Supply finite observations with positive speed and dynamic pressure")
        val, sigma = forward(wind, dst, self.meta, self.weights)
        if calibrated:
            sigma = sigma * self.meta["widen_factor"]
        if single:
            return dict(min_dst=float(val[0]), sigma=float(sigma[0]))
        return dict(min_dst=val, sigma=sigma)
