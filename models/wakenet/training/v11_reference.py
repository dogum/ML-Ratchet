"""v1.1 groundwork (17 Nov 2026): reproduce v1.0 on the rebuilt banks, and score the tuned physics alone.

1. The released v1.0 weights (r6) on the rebuilt `val3` and `dev_2019_graphs` must reproduce ratchet.yaml
   (sim 0.5522; dev H1 / H2 / farm 9.38 / 9.56 / 2.88 %), which shows the banks were rebuilt exactly.
2. `val3_tuned` must hold the same farms and targets as `val3` (only the anchor features differ).
3. The reference row: the dev-tuned physics alone (the anchor of `val3_tuned` and `dev_2019_graphs_tuned`, no
   network), scored with the ratchet's own objective and dev columns. It is a reference, not a candidate.

Writes training/v11_reference.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
from wakenet import WakeNet  # noqa: E402


def _np_predict(net: WakeNet, z) -> tuple[np.ndarray, np.ndarray]:
    n = len(z["node_x"])
    eta = np.zeros(z["eta_anchor"].shape, np.float32)
    farm = np.zeros(n, np.float32)
    for i in range(n):
        o = net.forward_graph(z["node_x"][i], z["edge_x"][i], z["edge_mask"][i], z["online"][i], z["is_free"][i],
                              z["eta_anchor"][i])
        eta[i] = o["eta_pred"]
        farm[i] = o["farm_wake_pred"]
    return eta, farm


def _anchor_predict(z) -> tuple[np.ndarray, np.ndarray]:
    on = z["online"]
    eta = np.where(on, z["eta_anchor"].astype(np.float32), 0.0)
    farm = 1.0 - np.sum(eta * on, axis=1) / np.maximum(np.sum(on, axis=1), 1)
    return eta, farm.astype(np.float32)


def sim_objective(eta_p, farm_p, z) -> float:
    on, free = z["online"], z["is_free"]
    w = on & (~free)
    eta_t = z["eta_target"].astype(np.float32)
    farm_t = z["farm_wake_target"].astype(np.float32)
    sw = 1.0 - np.mean(np.abs(eta_p[w] - eta_t[w])) / max(np.mean(np.abs(1.0 - eta_t[w])), 1e-6)
    sf = 1.0 - np.mean(np.abs(farm_p - farm_t)) / max(np.mean(np.abs(farm_t)), 1e-6)
    return round(float(0.65 * sw + 0.35 * sf), 4)


def dev_columns(eta_p, farm_p, z) -> dict:
    on, free = z["online"], z["is_free"]
    w = on & (~free)
    h1 = np.array([str(s)[5:7] <= "06" for s in z["ts"]])
    e = np.abs(eta_p - z["eta_true"])
    return {
        "waked_mae": round(float(np.mean(e[w])), 4),
        "waked_mae_h1": round(float(np.mean(e[w & h1[:, None]])), 4),
        "waked_mae_h2": round(float(np.mean(e[w & ~h1[:, None]])), 4),
        "farm_mae": round(float(np.mean(np.abs(farm_p - z["farm_wake_true"]))), 4),
    }


def main():
    v10 = WakeNet(os.path.join(MODEL, "wakenet_weights.npz"))
    val = np.load(_P.work("banks", "val3.npz"))
    valt = np.load(_P.work("banks", "val3_tuned.npz"))
    dev = np.load(_P.work("banks", "dev_2019_graphs.npz"))
    devt = np.load(_P.work("banks", "dev_2019_graphs_tuned.npz"))

    same = {k: bool(np.array_equal(val[k], valt[k])) for k in ("online", "eta_target", "farm_wake_target")}
    out = {
        "val3_tuned_same_targets_as_val3": same,
        "v1.0_r6_reproduced": {"sim_val3": sim_objective(*_np_predict(v10, val), val),
                               "dev": dev_columns(*_np_predict(v10, dev), dev),
                               "ratchet_yaml": {"sim": 0.5522, "dev": {"waked_mae_h1": 0.0938, "waked_mae_h2": 0.0956, "farm_mae": 0.0288}}},
        "textbook_physics_alone": {"sim_val3": sim_objective(*_anchor_predict(val), val),
                                   "dev": dev_columns(*_anchor_predict(dev), dev)},
        "tuned_physics_alone": {"sim_val3": sim_objective(*_anchor_predict(valt), valt),
                                "dev": dev_columns(*_anchor_predict(devt), devt)},
    }
    with open(os.path.join(HERE, "v11_reference.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
