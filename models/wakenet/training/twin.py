"""Dev twin test for WakeNet: simulate Kelmarsh 2019's exact 6-turbine twin under physics v2 and v3
and compare how the network reads simulated twins versus real 2019 dev records.

Writes training/twin_report.json (never touches exam data).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
import physics as WP  # noqa: E402
import pipeline as PL  # noqa: E402
import simulator as SIM  # noqa: E402
from ratchet import load_ckpt_model, predict_bank  # noqa: E402


def build_kelmarsh_dev_twins(version: str, seed: int = 2019) -> dict:
    rng = np.random.default_rng(seed)
    z = np.load(_P.work("proc", "kelmarsh_dev_2019.npz"))
    backdrop = SIM.BackdropSampler(split="val")
    turb = WP.TURBINES["MM92"]
    east, north, elev, hub_h = z["east_m"], z["north_m"], z["elev_m"], z["hub_h_m"]
    T, n = z["online"].shape

    node_x = np.zeros((T, n, PL.N_NODE_FEAT), dtype=np.float32)
    edge_x = np.zeros((T, n, n, PL.N_EDGE_FEAT), dtype=np.float32)
    edge_mask = np.zeros((T, n, n), dtype=bool)
    eta_anchor = np.zeros((T, n), dtype=np.float32)
    eta_true = np.zeros((T, n), dtype=np.float32)
    farm_wake_true = np.zeros(T, dtype=np.float32)

    for t in range(T):
        on = z["online"][t]
        free = z["is_free"][t]
        u_inf = float(z["u_inf"][t])
        wd_inf = float(z["wd_inf"][t])
        ti_inf = float(z["ti_inf"][t])
        swd = float(z["sigma_wd"][t])
        rho = float(z["air_density"][t])
        yaw = z["yaw_deg"][t]
        h_ref = float(np.mean(hub_h[on]))

        if version == "v2":
            cfg = {
                "velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez",
                "wind_shear": 0.14, "terrain_follow": 1.0, "hill_speedup": 0.0, "sigma_wd_deg": 0.0,
                "air_density": rho, "ref_height": h_ref,
            }
        else:  # v3 calibrated
            cfg = {
                "velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez",
                "wind_shear": 0.14, "terrain_follow": 0.80, "hill_speedup": 0.18, "sigma_wd_deg": 0.25 * swd,
                "air_density": rho, "ref_height": h_ref,
            }

        sim = WP.solve_wake(east, north, hub_h, turb, u_inf, wd_inf, ti_inf, yaw_deg=yaw, online=on, elev_m=elev, params=cfg)
        eps_p = backdrop.eps_p[rng.integers(0, len(backdrop.eps_p), size=n)].astype(np.float64)
        p_noisy = np.where(on, np.maximum(sim["power_kw"] * (1.0 + eps_p), 1.0), 0.0)
        p_front = float(np.mean(p_noisy[free]))
        eta_t = np.where(on, p_noisy / max(p_front, 1.0), 0.0)

        g = PL.build_graph(
            east, north, hub_h, turb, u_inf=u_inf, wind_dir_deg=wd_inf, ti_inf=ti_inf,
            sigma_wd_deg=swd, yaw_deg=yaw, online=on, elev_m=elev, air_density=rho,
        )
        node_x[t] = g["node_x"]
        edge_x[t] = g["edge_x"]
        edge_mask[t] = g["edge_mask"]
        eta_anchor[t] = g["eta_anchor"]
        eta_true[t] = eta_t
        farm_wake_true[t] = float(1.0 - np.sum(eta_t[on]) / np.sum(on))

    return {
        "node_x": node_x, "edge_x": edge_x, "edge_mask": edge_mask,
        "online": z["online"], "is_free": z["is_free"],
        "eta_anchor": eta_anchor, "eta_true": eta_true, "farm_wake_true": farm_wake_true,
        "elev_m": elev, "hub_h_m": hub_h,
    }


def summarize_reading(net, ds: dict) -> dict:
    eta_p, farm_p = predict_bank(net, ds)
    on = ds["online"]
    free = ds["is_free"]
    waked = on & (~free)
    eta_t = ds["eta_true"]
    farm_t = ds["farm_wake_true"]

    # Low-elevation downstream turbines KWF5 & KWF6 (indices 4, 5: 142.9 m and 135.0 m vs 156.6 m at KWF2)
    low_elev_mask = np.zeros_like(waked)
    low_elev_mask[:, [4, 5]] = True
    low_w = waked & low_elev_mask

    # Short-tower turbines KWF3 & KWF6 (68.5 m vs 78.5 m)
    short_mask = np.zeros_like(waked)
    short_mask[:, [2, 5]] = True
    short_w = waked & short_mask

    return {
        "waked_mae": round(float(np.mean(np.abs(eta_p[waked] - eta_t[waked]))), 4),
        "waked_bias": round(float(np.mean(eta_p[waked] - eta_t[waked])), 4),
        "low_elev_kwf5_6_bias": round(float(np.mean(eta_p[low_w] - eta_t[low_w])), 4),
        "short_tower_kwf3_6_bias": round(float(np.mean(eta_p[short_w] - eta_t[short_w])), 4),
        "farm_wake_mae": round(float(np.mean(np.abs(farm_p - farm_t))), 4),
        "farm_wake_bias": round(float(np.mean(farm_p - farm_t)), 4),
    }


def main(rung_v2: str = "r3", rung_v3: str = "r5"):
    real_dev = dict(np.load(_P.work("banks", "dev_2019_graphs.npz")))
    tw_v2 = build_kelmarsh_dev_twins("v2", seed=201902)
    tw_v3 = build_kelmarsh_dev_twins("v3", seed=201903)

    net_v2 = load_ckpt_model(rung_v2)
    report = {
        "dev_snapshots": int(len(real_dev["node_x"])),
        f"{rung_v2}_on_v2_twins": summarize_reading(net_v2, tw_v2),
        f"{rung_v2}_on_real_dev": summarize_reading(net_v2, real_dev),
    }
    if os.path.exists(_P.work("ckpt", f"{rung_v3}.pt")):
        net_v3 = load_ckpt_model(rung_v3)
        report[f"{rung_v3}_on_v3_twins"] = summarize_reading(net_v3, tw_v3)
        report[f"{rung_v3}_on_real_dev"] = summarize_reading(net_v3, real_dev)

    out_path = os.path.join(HERE, "twin_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    r2 = sys.argv[1] if len(sys.argv) > 1 else "r3"
    r3 = sys.argv[2] if len(sys.argv) > 2 else "r5"
    main(r2, r3)
