"""The forward model for __NAME__: physics parameters in, a clean simulated observation out.

Write it from the governing equations, vectorised in numpy. Keep three things separate so the ratchet
can change one at a time:

  * ``simulate(params, rng)``  — the physics (versioned: bump PHYSICS_VERSION when it changes,
                                 and build a new frozen set)
  * ``draw_params(rng)``       — the prior over the world (match it to real data that predates the exam)
  * noise injection            — lives in noise.py: simulated signal + REAL background
"""
import numpy as np

PHYSICS_VERSION = 1


def draw_params(rng: np.random.Generator) -> dict:
    """Sample one set of physical parameters from the prior."""
    raise NotImplementedError("TODO: the prior over the physical world")


def simulate(params: dict, rng: np.random.Generator) -> np.ndarray:
    """Return the noiseless observation for ``params``."""
    raise NotImplementedError("TODO: the governing physics")


def targets(params: dict) -> np.ndarray:
    """What the network should predict, in the units it learns best (logs for scales, etc.)."""
    raise NotImplementedError
