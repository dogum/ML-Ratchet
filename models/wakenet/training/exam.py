"""One-shot held-out real exam for WakeNet.

Evaluates the shipped standalone numpy WakeNet (wakenet_weights.npz, chosen by ratchet.yaml before
any exam archive was opened) ONCE on:
  1. Kelmarsh 2020-2021 (15,059 10-min snapshots, 39,008 waked turbine readings; unseen years)
  2. Penmanshiel 2019-2020 (8,976 10-min snapshots, 78,327 waked turbine readings; zero-shot unseen 14-turbine MM82 farm)

Compares against:
  * Classical #1: NREL FLORIS 3-D Gaussian (Bastankhah-Porte-Agel + Gaussian yaw deflection + Crespo-Hernandez + SOSFS)
  * Classical #2: Classic Jensen Park top-hat + Jimenez yaw deflection (we=0.05, kd=0.05)
  * Dumb #1: Dev mean wake-loss guess (constant waked ratio & farm wake loss from Kelmarsh 2019 dev)
  * Dumb #2: Unwaked guess (zero wake loss everywhere)

Writes training/exam_results.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import _paths as _P  # noqa: E402
from mlratchet.calibrate import coverage  # noqa: E402
import physics as WP  # noqa: E402
from wakenet import WakeNet  # noqa: E402


def evaluate_farm_exam(net: WakeNet, proc_npz: str) -> dict:
    z = np.load(_P.work("proc", proc_npz))
    farm = str(z["farm"])
    turb_key = "MM92" if farm == "kelmarsh" else "MM82"
    turb = WP.TURBINES[turb_key]
    D = float(turb["rotor_diameter"])

    east = z["east_m"].astype(np.float64)
    north = z["north_m"].astype(np.float64)
    elev = z["elev_m"].astype(np.float64)
    hub_h = z["hub_h_m"].astype(np.float64)

    T, n = z["online"].shape
    dev_base = net.meta["dev_baselines"]
    dev_waked_eta = float(dev_base["mean_waked_eta"])
    dev_farm_wake = float(dev_base["mean_farm_wake_loss"])

    eta_wn = np.zeros((T, n), dtype=np.float32)
    sig_wn = np.zeros((T, n), dtype=np.float32)
    eta_floris = np.zeros((T, n), dtype=np.float32)
    eta_jensen = np.zeros((T, n), dtype=np.float32)

    farm_wn = np.zeros(T, dtype=np.float32)
    farm_sig_wn = np.zeros(T, dtype=np.float32)
    farm_floris = np.zeros(T, dtype=np.float32)
    farm_jensen = np.zeros(T, dtype=np.float32)

    direct_mask = np.zeros((T, n), dtype=bool)
    multi_wake_mask = np.zeros((T, n), dtype=bool)

    t0 = time.time()
    for t in range(T):
        on = z["online"][t]
        free = z["is_free"][t]
        u_inf = float(z["u_inf"][t])
        wd_inf = float(z["wd_inf"][t])
        ti_inf = float(z["ti_inf"][t])
        swd = float(z["sigma_wd"][t])
        rho = float(z["air_density"][t])
        yaw = z["yaw_deg"][t]
        p_front = float(z["p_front_kw"][t])

        # 1. WakeNet standalone numpy inference
        r_wn = net.read(
            east, north, hub_h, turb, u_inf=u_inf, wind_dir_deg=wd_inf,
            ti_inf=ti_inf, sigma_wd_deg=swd, yaw_deg=yaw, online=on,
            elev_m=elev, air_density=rho, p_front_kw=p_front, calibrated=True,
        )
        eta_wn[t] = r_wn["eta"]
        sig_wn[t] = r_wn["eta_sig"]
        farm_wn[t] = r_wn["farm_wake_loss"]
        farm_sig_wn[t] = r_wn["farm_wake_sig"]
        # Note: r_wn["eta_anchor"] IS exact NREL FLORIS 3-D Gaussian with wind_shear=0.14!
        eta_floris[t] = r_wn["eta_anchor"]
        farm_floris[t] = float(1.0 - np.sum(eta_floris[t, on]) / np.sum(on))

        # 2. Classic Jensen Park + Jimenez baseline
        r_jen = WP.solve_wake(
            east, north, np.mean(hub_h), turb, u_inf=u_inf, wind_dir_deg=wd_inf, ti_inf=ti_inf,
            yaw_deg=yaw, online=on,
            params={"velocity_model": "jensen", "deflection_model": "jimenez", "turbulence_model": "none", "wind_shear": 0.0, "air_density": rho},
        )
        den_j = float(np.mean(r_jen["power_kw"][free]))
        ej = np.where(on, r_jen["power_kw"] / max(den_j, 1.0), 0.0)
        eta_jensen[t] = ej
        farm_jensen[t] = float(1.0 - np.sum(ej[on]) / np.sum(on))

        # Pure geometric masks for stratum breakdowns (independent of targets!)
        xm, ym = WP.rotate_to_wind_frame(east, north, wd_inf)
        dx_d = (xm[None, :] - xm[:, None]) / D
        dy_d = np.abs(ym[None, :] - ym[:, None]) / D
        hit_dir = on[:, None] & on[None, :] & (dx_d > 0.5) & (dx_d < 8.0) & (dy_d < 0.65)
        np.fill_diagonal(hit_dir, False)
        direct_mask[t] = np.any(hit_dir, axis=0)

        hit_any = on[:, None] & on[None, :] & (dx_d > 0.5) & (dx_d < 16.0) & (dy_d < 1.1 + 0.08 * np.maximum(dx_d, 0.0))
        np.fill_diagonal(hit_any, False)
        multi_wake_mask[t] = np.sum(hit_any, axis=0) >= 2

    on_all = z["online"]
    free_all = z["is_free"]
    waked_all = on_all & (~free_all)

    eta_true = np.where(on_all, z["power_kw"] / z["p_front_kw"][:, None], 0.0)
    farm_true = 1.0 - np.sum(eta_true * on_all, axis=1) / np.maximum(np.sum(on_all, axis=1), 1)

    def _mae_row(mask_2d: np.ndarray) -> dict:
        t_v = eta_true[mask_2d]
        return {
            "n": int(np.sum(mask_2d)),
            "wakenet_pct": round(100.0 * float(np.mean(np.abs(eta_wn[mask_2d] - t_v))), 2),
            "wakenet_bias_pct": round(100.0 * float(np.mean(eta_wn[mask_2d] - t_v)), 2),
            "floris_gauss_pct": round(100.0 * float(np.mean(np.abs(eta_floris[mask_2d] - t_v))), 2),
            "jensen_pct": round(100.0 * float(np.mean(np.abs(eta_jensen[mask_2d] - t_v))), 2),
            "dev_mean_dumb_pct": round(100.0 * float(np.mean(np.abs(dev_waked_eta - t_v))), 2),
            "unwaked_dumb_pct": round(100.0 * float(np.mean(np.abs(1.0 - t_v))), 2),
        }

    # Upstream large yaw misalignment (>= 8 deg on at least one online front-row turbine)
    large_yaw_snap = np.any(free_all & (np.abs(z["yaw_deg"]) >= 8.0), axis=1)
    parked_snap = np.any(~on_all, axis=1)
    near_rated_snap = z["u_inf"] >= 10.5
    region2_snap = z["u_inf"] < 10.0

    # High-relief hilltop turbines (top quartile of elevation on the farm)
    z_q75 = np.quantile(elev, 0.75)
    hilltop_2d = waked_all & (elev[None, :] >= z_q75)

    res = {
        "farm": farm,
        "n_snapshots": int(T),
        "n_waked_readings": int(np.sum(waked_all)),
        "elapsed_s": round(time.time() - t0, 1),
        "all_waked": _mae_row(waked_all),
        "direct_wake": _mae_row(waked_all & direct_mask),
        "multi_wake_2plus": _mae_row(waked_all & multi_wake_mask),
        "large_yaw_8deg": _mae_row(waked_all & large_yaw_snap[:, None]),
        "partial_parked": _mae_row(waked_all & parked_snap[:, None]),
        "region2_below_10ms": _mae_row(waked_all & region2_snap[:, None]),
        "near_rated_10p5_12ms": _mae_row(waked_all & near_rated_snap[:, None]),
        "hilltop_turbines": _mae_row(hilltop_2d),
        "farm_wake_loss": {
            "n": int(T),
            "wakenet_pct": round(100.0 * float(np.mean(np.abs(farm_wn - farm_true))), 2),
            "wakenet_bias_pct": round(100.0 * float(np.mean(farm_wn - farm_true)), 2),
            "floris_gauss_pct": round(100.0 * float(np.mean(np.abs(farm_floris - farm_true))), 2),
            "jensen_pct": round(100.0 * float(np.mean(np.abs(farm_jensen - farm_true))), 2),
            "dev_mean_dumb_pct": round(100.0 * float(np.mean(np.abs(dev_farm_wake - farm_true))), 2),
            "unwaked_dumb_pct": round(100.0 * float(np.mean(np.abs(0.0 - farm_true))), 2),
        },
        "coverage_68_pct": {
            "waked_turbine_eta": round(100.0 * coverage(eta_wn[waked_all] - eta_true[waked_all], sig_wn[waked_all]), 1),
            "all_online_eta": round(100.0 * coverage(eta_wn[on_all] - eta_true[on_all], sig_wn[on_all]), 1),
            "farm_wake_loss": round(100.0 * coverage(farm_wn - farm_true, farm_sig_wn), 1),
        },
    }
    print(f"Finished {farm} exam ({T} snapshots) in {res['elapsed_s']}s", flush=True)
    return res


def main():
    net = WakeNet()
    kel = evaluate_farm_exam(net, "kelmarsh_exam_2020_2021.npz")
    pen = evaluate_farm_exam(net, "penmanshiel_exam_2019_2020.npz")
    out = {
        "released_rung": net.meta["released_rung"],
        "sigma_factor": net.meta["sigma_factor"],
        "kelmarsh_2020_2021": kel,
        "penmanshiel_2019_2020": pen,
    }
    out_path = os.path.join(HERE, "exam_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
