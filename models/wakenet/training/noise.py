"""Extract real unlabelled wind-farm backdrop fluctuations and inflow climate priors from Kelmarsh 2016-2018.

Uses ONLY geometrically unwaked front-row turbine readings from kelmarsh_backdrop_2016_2018.npz:
  * Real atmospheric inflow tuples (u_inf, wd_inf, ti_inf, sigma_wd, air_density)
  * Real spatial turbulent power & wind-speed spread across unwaked front-row turbines
  * Real nacelle yaw-tracking controller offsets on unwaked turbines.

Zero waked turbine power labels are read or stored.
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


def build_backdrop():
    z = np.load(_P.work("proc", "kelmarsh_backdrop_2016_2018.npz"))
    u_inf = z["u_inf"]
    wd_inf = z["wd_inf"]
    ti_inf = z["ti_inf"]
    sigma_wd = z["sigma_wd"]
    rho = z["air_density"]
    free = z["is_free"]
    on = z["online"]
    p_kw = z["power_kw"]
    u_scada = z["u_scada_ms"]
    yaw = z["yaw_deg"]
    hub_h = z["hub_h_m"]

    turb = WP.TURBINES["MM92"]
    ws_tab, p_tab = turb["wind_speed"], turb["power_kw"]
    h_ref = float(np.max(hub_h))

    # Extract real unwaked per-turbine multiplicative spatial fluctuations (ONLY from free[t, k] == True!)
    rel_p_fluct = []
    rel_u_fluct = []
    free_yaws = []

    for t in range(len(u_inf)):
        fmask = free[t]
        if int(np.sum(fmask)) < 2:
            continue
        # Expected clean power ratio across different hub heights in the front row
        u_exp = u_inf[t] * ((hub_h[fmask] / h_ref) ** 0.14) * ((rho[t] / 1.225) ** (1.0 / 3.0))
        cos_f = np.maximum(np.cos(np.deg2rad(yaw[t, fmask])), 0.0) ** (1.88 / 3.0)
        p_exp = np.interp(u_exp * cos_f, ws_tab, p_tab)
        if np.any(p_exp < 80.0):
            continue
        norm_ratio = p_kw[t, fmask] / p_exp
        mean_r = float(np.mean(norm_ratio))
        if mean_r <= 0.2:
            continue
        eps_p = (norm_ratio / mean_r) - 1.0  # zero-mean spatial turbulent power jitter across unwaked rotors!
        u_norm = u_scada[t, fmask] / ((hub_h[fmask] / h_ref) ** 0.14)
        eps_u = (u_norm / max(float(np.mean(u_norm)), 0.5)) - 1.0

        rel_p_fluct.extend(eps_p.tolist())
        rel_u_fluct.extend(eps_u.tolist())
        free_yaws.extend(yaw[t, fmask].tolist())

    rel_p_arr = np.clip(np.array(rel_p_fluct, dtype=np.float32), -0.35, 0.35)
    rel_u_arr = np.clip(np.array(rel_u_fluct, dtype=np.float32), -0.25, 0.25)
    free_yaw_arr = np.clip(np.array(free_yaws, dtype=np.float32), -30.0, 30.0)

    # Split backdrop pool 85% / 15% by chronological block so train bank and frozen val bank never share noise draws!
    n_inflow = len(u_inf)
    n_tr_i = int(0.85 * n_inflow)
    n_fluct = len(rel_p_arr)
    n_tr_f = int(0.85 * n_fluct)

    out_npz = _P.work("proc", "backdrop_pool.npz", mkdir=True)
    np.savez_compressed(
        out_npz,
        train_inflow=np.stack([u_inf[:n_tr_i], wd_inf[:n_tr_i], ti_inf[:n_tr_i], sigma_wd[:n_tr_i], rho[:n_tr_i]], axis=1).astype(np.float32),
        val_inflow=np.stack([u_inf[n_tr_i:], wd_inf[n_tr_i:], ti_inf[n_tr_i:], sigma_wd[n_tr_i:], rho[n_tr_i:]], axis=1).astype(np.float32),
        train_eps_p=rel_p_arr[:n_tr_f],
        val_eps_p=rel_p_arr[n_tr_f:],
        train_eps_u=rel_u_arr[:n_tr_f],
        val_eps_u=rel_u_arr[n_tr_f:],
        train_yaw=free_yaw_arr[:n_tr_f],
        val_yaw=free_yaw_arr[n_tr_f:],
        online_rate=np.float32(np.mean(on)),
    )

    summary = {
        "n_backdrop_snapshots": int(n_inflow),
        "n_unwaked_rotor_samples": int(n_fluct),
        "u_inf_quantiles_ms": [round(float(q), 3) for q in np.quantile(u_inf, [0.1, 0.5, 0.9])],
        "ti_inf_quantiles": [round(float(q), 4) for q in np.quantile(ti_inf, [0.1, 0.5, 0.9])],
        "sigma_wd_quantiles_deg": [round(float(q), 2) for q in np.quantile(sigma_wd, [0.1, 0.5, 0.9])],
        "unwaked_spatial_power_std": round(float(np.std(rel_p_arr)), 4),
        "unwaked_spatial_wind_std": round(float(np.std(rel_u_arr)), 4),
        "yaw_offset_std_deg": round(float(np.std(free_yaw_arr)), 2),
        "online_rate": round(float(np.mean(on)), 4),
    }
    print("Built real backdrop pool:", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    build_backdrop()
