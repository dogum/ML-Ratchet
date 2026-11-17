"""Build synthetic index observations along actual solar-wind trajectories."""
import sys
from pathlib import Path

import numpy as np
from _paths import MODEL

sys.path.insert(0, MODEL)
from physics import drivers, step

PHYSICS_VERSION = ("ring-1", "ring-2")


def simulate(wind, center, rng, *, scatter=True):
    """One independent stable ring-current system per window. Returns the complete synthetic Dst."""
    e, p = drivers(wind)
    n = len(wind)
    gain, tf, ts, share, b, c = center
    if scatter:
        gain = gain * rng.lognormal(0, 0.20, n)
        tf = tf * rng.lognormal(0, 0.18, n)
        ts = ts * rng.lognormal(0, 0.18, n)
        if share < 1:
            share = np.clip(rng.normal(share, 0.07, n), 0.05, 0.95)
        b = b * rng.lognormal(0, 0.12, n)
        c = c + rng.normal(0, 2., n)
    fast = np.zeros(n)
    slow = np.zeros(n)
    unresolved = np.zeros(n)
    trace = np.empty_like(e)
    for j, field in enumerate(e.T):
        fast = step(fast, field, gain, tf, share)
        slow = step(slow, field, gain, ts, 1 - share)
        if scatter:
            unresolved = 0.80 * unresolved + rng.normal(size=n) * (1.8 + 0.4 * field)
        trace[:, j] = fast + slow + b * p[:, j] - c + unresolved
    return trace
