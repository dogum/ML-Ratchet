"""Dev twin test of what the network adds (17 Nov 2026).

On Kelmarsh 2019 dev, compare the network's correction (prediction minus its own anchor) with the real residual
(truth minus the same anchor), by stratum. If the network's corrections helped, they would point the same way as the
real residuals. Uses only dev. Writes training/twin_residuals.json.

    python twin_residuals.py r6            # textbook anchor (released v1.0)
    python twin_residuals.py r9            # tuned anchor
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
import physics as WP  # noqa: E402


def _predict(rid: str, z):
    if rid == "r6":
        from wakenet import WakeNet
        net = WakeNet(os.path.join(os.path.dirname(HERE), "wakenet_weights.npz"))
        eta = np.zeros(z["eta_anchor"].shape, np.float32)
        for i in range(len(eta)):
            eta[i] = net.forward_graph(z["node_x"][i], z["edge_x"][i], z["edge_mask"][i], z["online"][i],
                                       z["is_free"][i], z["eta_anchor"][i])["eta_pred"]
        return eta
    import ratchet as R
    net = R.load_ckpt_model(rid)
    return R.predict_bank(net, z)[0]


def main():
    rid = sys.argv[1]
    bank = "dev_2019_graphs_tuned" if rid.startswith("r9") else "dev_2019_graphs"
    z = np.load(_P.work("banks", f"{bank}.npz"))
    proc = np.load(_P.work("proc", "kelmarsh_dev_2019.npz"))
    eta = _predict(rid, z)
    anc = z["eta_anchor"].astype(np.float32)
    tru = z["eta_true"]
    on, free = z["online"], z["is_free"]
    w = on & ~free
    corr = eta - anc            # what the network adds
    resid = tru - anc           # what reality needed

    east, north = proc["east_m"], proc["north_m"]
    T, n = on.shape
    dxd = np.zeros((T, n)); depth = np.zeros((T, n))
    for t in range(T):
        x, y = WP.rotate_to_wind_frame(east, north, float(z["wd_inf"][t]))
        dx = (x[None, :] - x[:, None]) / 92.0
        dy = np.abs(y[None, :] - y[:, None]) / 92.0
        hit = on[t][:, None] & on[t][None, :] & (dx > 0.5) & (dx < 16.0) & (dy < 1.1 + 0.08 * np.maximum(dx, 0))
        np.fill_diagonal(hit, False)
        dmin = np.where(hit, dx, np.inf).min(axis=0)
        dxd[t] = np.where(np.isfinite(dmin), dmin, 99.0)
        depth[t] = 1.0 - anc[t]

    def stats(m):
        if m.sum() < 30:
            return None
        c, r = corr[m], resid[m]
        mae_a = float(np.mean(np.abs(r)))
        mae_n = float(np.mean(np.abs(tru[m] - eta[m])))
        return {"n": int(m.sum()), "mean_net_correction_pct": round(100 * float(c.mean()), 2),
                "mean_real_residual_pct": round(100 * float(r.mean()), 2),
                "median_real_residual_pct": round(100 * float(np.median(r)), 2),
                "corr_r": round(float(np.corrcoef(c, r)[0, 1]), 3),
                "anchor_mae_pct": round(100 * mae_a, 2), "net_mae_pct": round(100 * mae_n, 2)}

    u = z["u_inf"][:, None] * np.ones((1, n))
    swd = z["sigma_wd"][:, None] * np.ones((1, n))
    ti = z["ti_inf"][:, None] * np.ones((1, n))
    yawf = np.max(np.where(free, np.abs(z["yaw_deg"]), 0), axis=1)[:, None] * np.ones((1, n))
    out = {"rung": rid, "all_waked": stats(w)}
    for name, arr, edges in [("nearest_upstream_dx_D", dxd, [0, 4, 6, 8, 12, 99.1]),
                             ("anchor_depth", depth, [-1, 0.05, 0.15, 0.3, 0.5, 1.1]),
                             ("u_inf", u, [5, 7, 9, 10.5, 12.01]),
                             ("ti_inf", ti, [0, 0.06, 0.09, 0.12, 0.2, 1]),
                             ("vane_sigma_deg", swd, [0, 5, 7.5, 10, 30]),
                             ("front_row_max_yaw_deg", yawf, [0, 4, 8, 15, 90])]:
        out[name] = {f"{lo}-{hi}": stats(w & (arr >= lo) & (arr < hi)) for lo, hi in zip(edges[:-1], edges[1:])}
    out["per_turbine"] = {str(proc["codes"][i]): stats(w & (np.arange(n) == i)[None, :]) for i in range(n)}
    path = os.path.join(HERE, f"twin_residuals_{rid}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
