"""StormNet. Standalone NumPy inference for the next three complete hours' minimum Dst."""
import json
from pathlib import Path

import numpy as np
from stormphysics import HISTORY, assimilated_fraction, drivers, projected


def features(wind, dst, mode, params, law, past_dst=None):
    e, p = drivers(wind)
    cols = [dst, e[:, -1], p[:, -1], wind[:, -1, 1] / 1000, wind[:, -1, 0]]
    anchor = dst
    if mode in ("history", "history_dst"):
        path, fraction, total = projected(e, p, dst, params, law)
        anchor = path.min(axis=1)
        for n in (3, 12, 36):
            cols.extend((e[:, -n:].mean(axis=1), p[:, -n:].mean(axis=1)))
        cols.extend((fraction, total, anchor - dst, path[:, -1] - path[:, 0]))
    if mode == "history_dst":
        # v1.1 (r5): the window's observed Dst, which v1.0 did not read
        assim = assimilated_fraction(e, p, past_dst, params)
        cols.extend((dst - past_dst[:, -4], dst - past_dst[:, -13], past_dst.mean(axis=1), past_dst.min(axis=1),
                     np.where(np.isfinite(assim), assim, fraction)))
    return np.stack(cols, axis=1), anchor


def design(x, w):
    z = np.clip((x - w["mean"]) / w["scale"], -8, 8)
    hidden = np.tanh(z @ w["kernel"] + w["bias"])
    return np.column_stack((np.ones(len(z)), z, hidden))


def forward(wind, dst, meta, w, past_dst=None):
    x, anchor = features(wind, dst, meta["mode"], meta["physics"], meta["wind_law"], past_dst)
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

    def predict(self, wind, last_dst, *, calibrated=True, past_dst=None):
        """Return minimum Dst and one 68% half-width in nT, as scalars or batch arrays.

        v1.1 reads the window's 36 hourly Dst values too (past_dst, oldest first, ending with last_dst)."""
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
        if self.meta["mode"] == "history_dst":
            if past_dst is None:
                raise ValueError("This release reads the 36 hourly Dst values of the wind window (past_dst)")
            past_dst = np.asarray(past_dst, dtype=float).reshape(len(wind), HISTORY)
            if not np.isfinite(past_dst).all():
                raise ValueError("Supply 36 finite hourly Dst values")
        val, sigma = forward(wind, dst, self.meta, self.weights, past_dst)
        if calibrated:
            by = self.meta.get("widen_by_dst")
            if by:
                # v1.0.1: one widen-only factor per bin of the last observed Dst, fitted on dev
                sigma = sigma * np.asarray(by["factors"])[np.searchsorted(by["edges"], dst, side="right")]
            else:
                sigma = sigma * self.meta["widen_factor"]
        if single:
            return dict(min_dst=float(val[0]), sigma=float(sigma[0]))
        return dict(min_dst=val, sigma=sigma)
