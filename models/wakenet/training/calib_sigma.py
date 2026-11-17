"""Widen-only 68 % error-bar factors per wind-speed bin, fitted on Kelmarsh 2019 dev (17 Nov 2026).

v1.0's network gives an error bar of about the same width (~11.7 %) at every wind speed, while its dev error triples
from near-rated to low wind: dev coverage was 47.5 % at 5-9 m/s, 63.6 % at 9-10.5 m/s and 88.0 % at 10.5-12 m/s
(68.7 % pooled, so the pooled factor was 1.00). This fits one widen-only factor per bin (never below 1) for waked
turbines, front-row turbines and farm wake loss, and writes them into the weights file's meta as sigma_factor.by_u.

    python calib_sigma.py ../wakenet_weights.npz
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import _paths as _P  # noqa: E402
from mlratchet.calibrate import coverage, widen_factor  # noqa: E402
from wakenet import WakeNet, U_EDGES  # noqa: E402

CACHE = {"textbook": "dev_2019_graphs", "tuned": "dev_2019_graphs_tuned", "tuned_v4": "dev_2019_graphs_tuned_v4"}


def main():
    path = os.path.abspath(sys.argv[1])
    net = WakeNet(path)
    z = np.load(_P.work("banks", CACHE[net.meta.get("anchor", "textbook")] + ".npz"))
    T = len(z["node_x"])
    eta = np.zeros(z["eta_anchor"].shape)
    sig = np.zeros_like(eta)
    farm = np.zeros(T)
    fsig = np.zeros(T)
    for i in range(T):
        o = net.forward_graph(z["node_x"][i], z["edge_x"][i], z["edge_mask"][i], z["online"][i], z["is_free"][i],
                              z["eta_anchor"][i])
        eta[i], sig[i] = o["eta_pred"], np.exp(o["log_sig_eta"])
        farm[i], fsig[i] = o["farm_wake_pred"], np.exp(o["log_sig_farm"])
    on, free = z["online"], z["is_free"]
    waked = on & ~free
    b = np.digitize(z["u_inf"], U_EDGES)
    out = {"edges": list(U_EDGES), "waked_eta": [], "free_eta": [], "farm_wake": [], "dev_coverage": []}
    for k in range(len(U_EDGES) + 1):
        m = (b == k)[:, None]
        rw, sw = (eta - z["eta_true"])[waked & m], sig[waked & m]
        rf, sf = (eta - z["eta_true"])[free & m], sig[free & m]
        rr, sr = (farm - z["farm_wake_true"])[b == k], fsig[b == k]
        fw, ff, fr = widen_factor(rw, sw), widen_factor(rf, sf), widen_factor(rr, sr)
        out["waked_eta"].append(round(fw, 4))
        out["free_eta"].append(round(ff, 4))
        out["farm_wake"].append(round(fr, 4))
        out["dev_coverage"].append({"n_waked": int(len(rw)), "before": round(100 * coverage(rw, sw), 1),
                                    "after": round(100 * coverage(rw, sw, fw), 1)})
    meta = dict(net.meta)
    meta["sigma_factor"] = dict(meta.get("sigma_factor", {}), by_u=out)
    zw = np.load(path)
    np.savez(path, meta=json.dumps(meta), **{k: zw[k] for k in zw.files if k != "meta"})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
