"""Fit a compact random-feature neural readout and conditional scale to simulated Dst only."""
import json

import numpy as np

from _paths import work
from noise import read_result
from simbank import load_bank
from stormphysics import PUBLISHED
from stormnet import design, features

RUNS = (
    ("r1", "instant", ("train1",), "Instantaneous wind and last Dst, single-state teacher"),
    ("r2", "history", ("train1",), "36-hour causal state estimates, same single-state teacher"),
    ("r3", "history", ("train2",), "Dev-calibrated fast and slow teacher with a matched physics anchor"),
    ("r4", "history", ("train1", "train2"), "Equal mixture of single and two-state simulations"),
)


def solve(h, y, weight):
    hw = h * weight[:, None]
    penalty = np.full(h.shape[1], 0.0002 * len(h))
    penalty[0] = 1e-8
    return np.linalg.solve(h.T @ hw + np.diag(penalty), hw.T @ y)


def train(rid, mode, banks, cal, seed=103, folder="ckpt"):
    arrays = [load_bank(b) for b in banks]
    data = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    params = cal["double"] if "train2" in banks else PUBLISHED
    x, anchor = features(data["wind"], data["dst"], mode, params, cal["wind_law"])
    y = data["truth"] - anchor
    # Identical features/initialisation across the three history rungs.
    rng = np.random.default_rng(seed)
    fold = np.arange(len(x)) % 5 != 0
    w = dict(mean=x[fold].mean(axis=0), scale=np.maximum(x[fold].std(axis=0), 1e-3),
             kernel=rng.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], 48)),
             bias=rng.uniform(-1, 1, 48))
    h = design(x, w)
    wt = np.ones(fold.sum())
    for _ in range(4):
        loc = solve(h[fold], y[fold], wt)
        wt = 1 / np.maximum(1., np.abs(y[fold] - h[fold] @ loc) / 3.)
    w["location"] = loc
    # A held-out fifth of the simulation bank estimates log absolute error. The dev factor
    # subsequently converts this conditional scale to a nominal 68% half-width.
    residual = np.abs(y[~fold] - h[~fold] @ loc)
    w["spread"] = solve(h[~fold], np.log(residual + 1.), np.ones(len(residual)))
    meta = dict(rung=rid, mode=mode, physics=list(params), wind_law=cal["wind_law"],
                widen_factor=1., seed=seed, simulated_examples=len(y))
    np.savez_compressed(work(folder, rid + ".npz", mkdir=True), meta=json.dumps(meta), **w)


if __name__ == "__main__":
    cal = read_result("calibration.json")
    for rid, mode, banks, _ in RUNS:
        train(rid, mode, banks, cal)
