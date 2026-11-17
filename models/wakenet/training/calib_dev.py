"""Measure real-vs-textbook wake physics mismatches on Kelmarsh 2019 dev set ONLY.

Investigates two physical gaps between textbook steady-state FLORIS and real onshore SCADA:
  1. Nacelle wind-vane flutter vs true atmospheric meandering:
     Raw 10-min Wind direction StdDev on a nacelle behind spinning blades has median 7.7 deg.
     Feeding 100% of it as atmospheric wind-direction meandering over-smears direct wakes;
     measuring the true meandering fraction c_meander in [0, 1] on dev separates meso-scale
     meandering (~2.5-3.0 deg) from high-frequency rotor swirl over the nacelle vane.
  2. 3-D tower hub-height wind shear & gentle hill exposure speedup across Kelmarsh's
     mixed 68.5 m / 78.5 m towers and 21.5 m terrain relief.

Writes training/dev_calibration.json. Never touches exam archives.
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
import physics as WP  # noqa: E402


def eval_physics_config(z, cfg_fn, stride: int = 4) -> dict:
    east, north, elev, hub_h = z["east_m"], z["north_m"], z["elev_m"], z["hub_h_m"]
    turb = WP.TURBINES["MM92"]
    ts = z["ts"]
    N = len(ts)

    w_err_h1, w_err_h2, dir_err, dir_bias, sh_err, farm_err = [], [], [], [], [], []
    for t in range(0, N, stride):
        on = z["online"][t]
        free = z["is_free"][t]
        wmask = on & (~free)
        eta_true = z["power_kw"][t] / z["p_front_kw"][t]
        w_farm_true = 1.0 - float(np.sum(eta_true[on])) / float(np.sum(on))

        x, y = WP.rotate_to_wind_frame(east, north, float(z["wd_inf"][t]))
        dx = (x[None, :] - x[:, None]) / 92.0
        dy = np.abs(y[None, :] - y[:, None]) / 92.0
        hit = on[:, None] & on[None, :] & (dx > 0.5) & (dx < 8.0) & (dy < 0.65)
        np.fill_diagonal(hit, False)
        gdir = np.any(hit, axis=0) & wmask
        gsh = (~gdir) & wmask

        hh, ti_eff, params = cfg_fn(z, t, hub_h)
        r = WP.solve_wake(
            east, north, hh, turb, float(z["u_inf"][t]), float(z["wd_inf"][t]),
            ti_eff, yaw_deg=z["yaw_deg"][t], online=on, elev_m=elev, params=params,
        )
        denom = float(np.mean(r["power_kw"][free]))
        eta_pred = np.where(on, r["power_kw"] / max(denom, 1.0), 0.0)
        w_farm_pred = 1.0 - float(np.sum(eta_pred[on])) / float(np.sum(on))

        ae = np.abs(eta_pred[wmask] - eta_true[wmask]).tolist()
        if str(ts[t])[5:7] <= "06":
            w_err_h1.extend(ae)
        else:
            w_err_h2.extend(ae)
        if np.any(gdir):
            dir_err.extend(np.abs(eta_pred[gdir] - eta_true[gdir]).tolist())
            dir_bias.extend((eta_pred[gdir] - eta_true[gdir]).tolist())
        if np.any(gsh):
            sh_err.extend(np.abs(eta_pred[gsh] - eta_true[gsh]).tolist())
        farm_err.append(abs(w_farm_pred - w_farm_true))

    all_w = w_err_h1 + w_err_h2
    return {
        "waked_mae": round(float(np.mean(all_w)), 5),
        "waked_mae_h1": round(float(np.mean(w_err_h1)), 5),
        "waked_mae_h2": round(float(np.mean(w_err_h2)), 5),
        "direct_wake_mae": round(float(np.mean(dir_err)), 5),
        "direct_wake_bias": round(float(np.mean(dir_bias)), 5),
        "shoulder_mae": round(float(np.mean(sh_err)), 5),
        "farm_wake_mae": round(float(np.mean(farm_err)), 5),
    }


def main():
    z = np.load(_P.work("proc", "kelmarsh_dev_2019.npz"))

    # 1. Textbook 2-D Jensen
    m_jensen = eval_physics_config(
        z,
        lambda z, t, hh: (
            78.5,
            float(z["ti_inf"][t]),
            {"velocity_model": "jensen", "deflection_model": "jimenez", "turbulence_model": "none", "wind_shear": 0.0, "air_density": float(z["air_density"][t])},
        ),
    )

    # 2. Textbook steady 3-D FLORIS Gaussian (flat HH vs real HH shear=0.14)
    m_floris_flat = eval_physics_config(
        z,
        lambda z, t, hh: (
            78.5,
            float(z["ti_inf"][t]),
            {"velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez", "wind_shear": 0.0, "air_density": float(z["air_density"][t])},
        ),
    )
    m_floris_3d = eval_physics_config(
        z,
        lambda z, t, hh: (
            hh,
            float(z["ti_inf"][t]),
            {"velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez", "wind_shear": 0.14, "air_density": float(z["air_density"][t])},
        ),
    )

    # 3. Naive 100% raw nacelle vane meandering (shows over-smearing!)
    m_raw_vane = eval_physics_config(
        z,
        lambda z, t, hh: (
            hh,
            float(z["ti_inf"][t]),
            {"velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez", "wind_shear": 0.14, "sigma_wd_deg": float(z["sigma_wd"][t]), "air_density": float(z["air_density"][t])},
        ),
    )

    # 4. Coarse grid search on dev over (c_meander, hill_speedup, wind_shear)
    grid_rows = []
    best = None
    for c_m in [0.25, 0.35, 0.45]:
        for s_hill in [0.06, 0.12, 0.18]:
            for sh in [0.14, 0.18]:
                res = eval_physics_config(
                    z,
                    lambda z, t, hh, cm=c_m, shl=s_hill, wsh=sh: (
                        hh,
                        float(z["ti_inf"][t]),
                        {
                            "velocity_model": "gauss",
                            "deflection_model": "gauss",
                            "turbulence_model": "crespo_hernandez",
                            "wind_shear": wsh,
                            "sigma_wd_deg": cm * float(z["sigma_wd"][t]),
                            "hill_speedup": shl,
                            "terrain_follow": 0.80,
                            "air_density": float(z["air_density"][t]),
                        },
                    ),
                    stride=6,
                )
                row = {"c_meander": c_m, "hill_speedup": s_hill, "wind_shear": sh, **res}
                grid_rows.append(row)
                score = 0.65 * res["waked_mae"] + 0.35 * res["farm_wake_mae"]
                if best is None or score < best[0]:
                    best = (score, row)

    # Evaluate chosen dev calibration on dense stride=2 for exact comparison
    chosen = best[1]
    m_cal = eval_physics_config(
        z,
        lambda z, t, hh: (
            hh,
            float(z["ti_inf"][t]),
            {
                "velocity_model": "gauss",
                "deflection_model": "gauss",
                "turbulence_model": "crespo_hernandez",
                "wind_shear": chosen["wind_shear"],
                "sigma_wd_deg": chosen["c_meander"] * float(z["sigma_wd"][t]),
                "hill_speedup": chosen["hill_speedup"],
                "terrain_follow": 0.80,
                "air_density": float(z["air_density"][t]),
            },
        ),
        stride=2,
    )

    out = {
        "dev_set": "kelmarsh_dev_2019.npz (5,729 snapshots)",
        "jensen_flat_2d": m_jensen,
        "floris_gauss_flat_2d": m_floris_flat,
        "floris_gauss_3d_shear": m_floris_3d,
        "naive_raw_vane_meandering": m_raw_vane,
        "calibrated_v3": {
            "params": {
                "c_meander": chosen["c_meander"],
                "hill_speedup": chosen["hill_speedup"],
                "wind_shear": chosen["wind_shear"],
                "terrain_follow": 0.80,
            },
            "metrics": m_cal,
        },
        "grid_search": grid_rows,
    }
    out_path = os.path.join(HERE, "dev_calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "grid_search"}, indent=2))


if __name__ == "__main__":
    main()
