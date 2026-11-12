"""Real background for the simulator: fetch and screen real instrument recordings, then lay simulated
signals into them, so the network never sees idealised noise.

Rules: use documented public endpoints only; respect the licence; never use a dev or exam unit's
recordings as training noise (unless they are quiet periods cut from dev, and then say so).
"""
import numpy as np

import _paths as _P  # noqa: F401


def load_pool() -> list[np.ndarray]:
    """TODO: return a list of real background windows (from _P.work('noise', ...))."""
    raise NotImplementedError


def inject(signal: np.ndarray, background: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Scale ``signal`` to the requested SNR against ``background`` and add them."""
    p_sig = np.mean(signal ** 2)
    p_bg = np.mean(background ** 2)
    k = np.sqrt(snr ** 2 * p_bg / max(p_sig, 1e-30))
    return background + k * signal
