"""Export a trained rung to numpy: fold BatchNorm, save ``../__ID___weights.npz``, and check parity.

    python export.py r3

Parity target: numpy vs the training framework to about 1e-5 on a few real examples
(``mlratchet.nn.max_abs_diff``). Record the number in the model card.
"""
import json
import sys

import numpy as np

import _paths as _P
from mlratchet.nn import fold_batchnorm, max_abs_diff  # noqa: F401


def main(rung: str):
    weights: dict[str, np.ndarray] = {}
    meta = {"rung": rung, "sigma_widen": {}}  # widen-only factors fitted on dev
    # TODO: load the checkpoint, fold every BatchNorm into the conv before it, fill `weights`
    np.savez(f"{_P.MODEL}/__ID___weights.npz", meta=json.dumps(meta), **weights)
    # TODO: run __ID__.py and the framework on the same inputs; assert max_abs_diff(...) < 1e-4


if __name__ == "__main__":
    main(sys.argv[1])
