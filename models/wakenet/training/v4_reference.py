"""Physics v4 re-score (17 Nov 2026): the incumbent r6 and r9 on the new frozen set val4, plus physics-alone rows.

r6 is scored from the released weights (numpy, textbook-anchor features in val4.npz); r9 from its checkpoint
(tuned-anchor features in val4_tuned.npz). All val4 banks share seed 404, so they hold the same farms and targets.
Writes training/v4_reference.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
from v11_reference import _anchor_predict, _np_predict, dev_columns, sim_objective  # noqa: E402
from wakenet import WakeNet  # noqa: E402


def main():
    b = {k: np.load(_P.work("banks", f"{k}.npz")) for k in ("val4", "val4_tuned", "val4_tuned_v4")}
    same = all(np.array_equal(b["val4"]["eta_target"], b[k]["eta_target"]) for k in ("val4_tuned", "val4_tuned_v4"))
    dv4 = np.load(_P.work("banks", "dev_2019_graphs_tuned_v4.npz"))
    v10 = WakeNet(os.path.join(os.path.dirname(HERE), "wakenet_weights.npz"))
    import ratchet as R
    r9 = R.load_ckpt_model("r9")
    eta9, farm9 = R.predict_bank(r9, b["val4_tuned"])
    out = {
        "val4_banks_same_targets": bool(same),
        "r6_on_val4": sim_objective(*_np_predict(v10, b["val4"]), b["val4"]),
        "r9_on_val4": sim_objective(eta9, farm9, b["val4_tuned"]),
        "textbook_physics_alone_val4": sim_objective(*_anchor_predict(b["val4"]), b["val4"]),
        "tuned_v3_physics_alone_val4": sim_objective(*_anchor_predict(b["val4_tuned"]), b["val4_tuned"]),
        "tuned_v4_physics_alone": {"sim_val4": sim_objective(*_anchor_predict(b["val4_tuned_v4"]), b["val4_tuned_v4"]),
                                   "dev": dev_columns(*_anchor_predict(dv4), dv4)},
    }
    with open(os.path.join(HERE, "v4_reference.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
