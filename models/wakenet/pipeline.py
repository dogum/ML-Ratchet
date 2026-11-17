"""Feature extraction and physics-anchor graph construction for WakeNet.

Pure numpy only — shared identically by training, standalone inference (wakenet.py),
and the browser exporter (web/export_instrument.py).
"""
from __future__ import annotations

import os

import numpy as np

import physics as WP

# Textbook steady 3-D Bastankhah-Porte-Agel Gaussian anchor (matches NREL FLORIS defaults)
ANCHOR_PARAMS = {
    "velocity_model": "gauss",
    "deflection_model": "gauss",
    "turbulence_model": "crespo_hernandez",
    "combination_model": "sosfs",
    "wind_shear": 0.14,
    "terrain_follow": 1.0,
    "hill_speedup": 0.0,
    "sigma_wd_deg": 0.0,
    "n_grid": 3,
}

# v1.1 (17 Nov 2026): the same physics tuned on Kelmarsh 2019 dev (calibrated_v3 in training/dev_calibration.json):
# meso-scale meandering at 0.25 x the nacelle-vane sigma, linearised hill speedup 0.18, terrain-follow 0.80.
TUNED_ANCHOR = {"c_meander": 0.25, "hill_speedup": 0.18, "terrain_follow": 0.80, "wind_shear": 0.14}


def anchor_config(anchor: str | dict | None, sigma_wd_deg: float) -> dict:
    """Physics settings for the anchor: 'textbook' (v1.0), 'tuned' (v1.1) or an explicit override dict.

    None reads $WAKENET_ANCHOR (training scripts only; inference passes the release's own setting).
    """
    if anchor is None:
        anchor = os.environ.get("WAKENET_ANCHOR", "textbook")
    if anchor == "textbook":
        return dict(ANCHOR_PARAMS)
    over = TUNED_ANCHOR if anchor == "tuned" else dict(anchor)
    cfg = dict(ANCHOR_PARAMS)
    cfg["wind_shear"] = float(over["wind_shear"])
    cfg["hill_speedup"] = float(over["hill_speedup"])
    cfg["terrain_follow"] = float(over["terrain_follow"])
    cfg["sigma_wd_deg"] = float(over["c_meander"]) * float(sigma_wd_deg)
    return cfg


N_NODE_FEAT = 17
N_EDGE_FEAT = 12


def latlon_to_east_north(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local tangent-plane (East, North) [m] around the farm centroid."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    north = (lat - lat0) * 111_132.92
    east = (lon - lon0) * 111_412.84 * np.cos(np.deg2rad(lat0))
    return east, north


def build_graph(
    east_m: np.ndarray,
    north_m: np.ndarray,
    hub_h_m: np.ndarray | float,
    turbine: dict,
    u_inf: float,
    wind_dir_deg: float,
    ti_inf: float = 0.085,
    sigma_wd_deg: float = 4.5,
    yaw_deg: np.ndarray | float = 0.0,
    online: np.ndarray | None = None,
    elev_m: np.ndarray | float = 0.0,
    air_density: float = 1.225,
    anchor: str | dict | None = None,
) -> dict:
    """Build node features (N, 17), edge features (N, N, 12), edge mask (N, N), and physics anchor."""
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    n = len(east)
    D = float(turbine["rotor_diameter"])
    p_rated = float(turbine["rated_power_kw"])
    hub_h = np.broadcast_to(np.asarray(hub_h_m, dtype=np.float64), (n,)).copy()
    elev = np.broadcast_to(np.asarray(elev_m, dtype=np.float64), (n,)).copy()
    yaw = np.broadcast_to(np.asarray(yaw_deg, dtype=np.float64), (n,)).copy()
    on = np.ones(n, dtype=bool) if online is None else np.asarray(online, dtype=bool).copy()

    is_free = WP.freestream_mask(east, north, D, wind_dir_deg, online=on)

    anc_cfg = dict(anchor_config(anchor, sigma_wd_deg), air_density=float(air_density), ref_height=float(np.mean(hub_h[on])) if np.any(on) else float(np.mean(hub_h)))
    anc0 = WP.solve_wake(east, north, hub_h, turbine, u_inf, wind_dir_deg, ti_inf, yaw_deg=yaw, online=on, elev_m=elev, params=anc_cfg)

    # Probe directional sensitivity & meandering curvature at +- max(sigma_wd_deg, 2.0)
    d_th = float(np.clip(sigma_wd_deg, 1.5, 18.0))
    anc_p = WP.solve_wake(east, north, hub_h, turbine, u_inf, wind_dir_deg + d_th, ti_inf, yaw_deg=yaw + 0.35 * d_th, online=on, elev_m=elev, params=anc_cfg)
    anc_m = WP.solve_wake(east, north, hub_h, turbine, u_inf, wind_dir_deg - d_th, ti_inf, yaw_deg=yaw - 0.35 * d_th, online=on, elev_m=elev, params=anc_cfg)

    def _front_norm(p_arr: np.ndarray) -> np.ndarray:
        f_idx = is_free if np.any(is_free) else on
        denom = float(np.mean(p_arr[f_idx])) if np.any(f_idx) else 1.0
        if denom < 1.0:
            return np.where(on, 1.0, 0.0)
        return np.where(on, p_arr / denom, 0.0)

    eta_anc = _front_norm(anc0["power_kw"])
    eta_p = _front_norm(anc_p["power_kw"])
    eta_m = _front_norm(anc_m["power_kw"])

    d_eta_meander = 0.5 * (eta_p + eta_m) - eta_anc
    sens_wd = 0.5 * (np.abs(eta_p - eta_anc) + np.abs(eta_m - eta_anc))

    x_m, y_m = anc0["x_m"], anc0["y_m"]
    x_min = float(np.min(x_m[on])) if np.any(on) else float(np.min(x_m))
    h_mean = float(np.mean(hub_h[on])) if np.any(on) else float(np.mean(hub_h))
    z_mean = float(np.mean(elev[on])) if np.any(on) else float(np.mean(elev))

    ws_tab = np.asarray(turbine["wind_speed"], dtype=np.float64)
    p_tab = np.asarray(turbine["power_kw"], dtype=np.float64)
    ct_tab = np.asarray(turbine["thrust_coefficient"], dtype=np.float64)
    rho_scale = (float(air_density) / float(turbine.get("ref_air_density", 1.225))) ** (1.0 / 3.0)
    p_free_scalar = float(np.interp(u_inf * rho_scale, ws_tab, p_tab)) / max(p_rated, 1.0)
    ct_free_scalar = float(np.interp(u_inf, ws_tab, ct_tab))
    p_yaw_exp = float(turbine.get("cosine_loss_exponent_yaw", 1.88))

    cos_y = np.maximum(np.cos(np.deg2rad(yaw)), 0.0)
    cos_loss = 1.0 - (cos_y ** p_yaw_exp)
    def_anc = np.where(on, 1.0 - anc0["u_rotor_ms"] / np.maximum(anc0["u_free_ms"], 0.5), 0.0)
    ti_add = np.where(on, (anc0["ti_eff"] - ti_inf) / 0.10, 0.0)

    node_x = np.zeros((n, N_NODE_FEAT), dtype=np.float32)
    node_x[:, 0] = on.astype(np.float32)
    node_x[:, 1] = is_free.astype(np.float32)
    node_x[:, 2] = (u_inf - 8.5) / 3.0
    node_x[:, 3] = p_free_scalar
    node_x[:, 4] = ct_free_scalar
    node_x[:, 5] = (ti_inf - 0.09) / 0.05
    node_x[:, 6] = (sigma_wd_deg - 5.0) / 4.0
    node_x[:, 7] = np.clip(yaw / 25.0, -2.5, 2.5)
    node_x[:, 8] = np.clip(cos_loss, 0.0, 1.0)
    node_x[:, 9] = (hub_h - h_mean) / D
    node_x[:, 10] = (elev - z_mean) / D
    node_x[:, 11] = (x_m - x_min) / (10.0 * D)
    node_x[:, 12] = eta_anc
    node_x[:, 13] = def_anc
    node_x[:, 14] = ti_add
    node_x[:, 15] = d_eta_meander * 4.0
    node_x[:, 16] = sens_wd * 4.0

    # Pairwise directed edges j (upstream) -> i (downstream)
    dx_d = (x_m[None, :] - x_m[:, None]) / D
    dy_d = (y_m[None, :] - y_m[:, None]) / D
    dz_h_d = (hub_h[None, :] - hub_h[:, None]) / D
    dz_e_d = (elev[None, :] - elev[:, None]) / D

    adj = (
        on[:, None]
        & on[None, :]
        & (dx_d > 0.10)
        & (dx_d < 30.0)
        & (np.abs(dy_d) < 4.5 + 0.25 * np.maximum(dx_d, 0.0))
    )
    np.fill_diagonal(adj, False)

    dx_pos = np.where(adj, np.maximum(dx_d, 0.0), 0.0)
    pdef = anc0["pair_deficit"]
    povl = anc0["pair_overlap"]
    pdel = anc0["pair_deflection_d"]
    psy = np.maximum(anc0["pair_sigma_y_d"], 0.25)

    abs_dy_rel = np.abs(dy_d - pdel) / (psy + 0.35)
    meander_pi = np.deg2rad(sigma_wd_deg) * dx_pos

    edge_x = np.zeros((n, n, N_EDGE_FEAT), dtype=np.float32)
    edge_x[:, :, 0] = np.where(adj, 1.0 / (dx_pos + 2.0), 0.0)
    edge_x[:, :, 1] = np.where(adj, np.log1p(dx_pos) / 3.0, 0.0)
    edge_x[:, :, 2] = np.where(adj, np.clip(dy_d / 3.0, -2.5, 2.5), 0.0)
    edge_x[:, :, 3] = np.where(adj, np.clip(abs_dy_rel / 3.0, 0.0, 3.0), 0.0)
    edge_x[:, :, 4] = np.where(adj, np.clip(dz_h_d, -2.0, 2.0), 0.0)
    edge_x[:, :, 5] = np.where(adj, np.clip(dz_e_d, -3.0, 3.0), 0.0)
    edge_x[:, :, 6] = np.where(adj, pdef, 0.0)
    edge_x[:, :, 7] = np.where(adj, povl, 0.0)
    edge_x[:, :, 8] = np.where(adj, np.clip(pdel * 4.0, -2.5, 2.5), 0.0)
    edge_x[:, :, 9] = np.where(adj, np.clip(yaw[:, None] / 25.0, -2.5, 2.5), 0.0)
    edge_x[:, :, 10] = np.where(adj, anc0["ct"][:, None], 0.0)
    edge_x[:, :, 11] = np.where(adj, np.clip(meander_pi, 0.0, 3.0), 0.0)

    return {
        "node_x": node_x,
        "edge_x": edge_x,
        "edge_mask": adj,
        "online": on,
        "is_free": is_free,
        "eta_anchor": eta_anc.astype(np.float32),
        "u_anchor_ms": anc0["u_rotor_ms"].astype(np.float32),
        "u_free_ms": anc0["u_free_ms"].astype(np.float32),
        "x_m": x_m.astype(np.float32),
        "y_m": y_m.astype(np.float32),
    }
