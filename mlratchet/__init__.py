"""ML Ratchet: physics-first machine learning.

Every model in the collection learned only from physics simulations (laid into real instrument
noise where possible), moved up a rung only when it beat a frozen simulated test AND a small real
dev set, and was graded on real data that never chose anything.

    import mlratchet
    mlratchet.list_models()                         # every model in models/
    net = mlratchet.load("quakenet").QuakeNet()     # import a model's numpy inference code
    mlratchet.get("quakenet").card["exam"]          # its structured card

Building blocks for new models: :mod:`mlratchet.ratchet` (the keep/reject rule and its log),
:mod:`mlratchet.calibrate` (widen-only error bars, split & Mondrian conformal calibration),
:mod:`mlratchet.metrics` (skills and dumb baselines), :mod:`mlratchet.nn` (numpy layers, BatchNorm
folding), :mod:`mlratchet.gbm` (pure-numpy gradient-boosted trees and exact TreeSHAP),
:mod:`mlratchet.twintest` (real-vs-twin diagnostics, SHAP gap ranking, disagreement detection),
:mod:`mlratchet.splits` (unit-level dev/exam splits and fixed draws).
"""
__version__ = "0.1.0"

from .calibrate import MondrianConformal  # noqa: E402
from .gbm import TreeEnsemble  # noqa: E402
from .ratchet import Ratchet, Rung, gates  # noqa: E402
from .registry import Model, get, list_models, load, repo_root  # noqa: E402

__all__ = [
    "Model", "MondrianConformal", "Ratchet", "Rung", "TreeEnsemble",
    "gates", "get", "list_models", "load", "repo_root", "__version__",
]
