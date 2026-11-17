"""Governing wind-farm wake physics in pure numpy.

Implements sequential 3-D upwind-to-downwind wake propagation on a rotor grid:
  * Jensen (1983) / Katic et al. (1986) top-hat wake expansion ("jensen")
  * Bastankhah & Porte-Agel (2014, 2016) near- and far-wake 3-D Gaussian deficit ("gauss")
  * Jimenez et al. (2010) and Bastankhah & Porte-Agel (2016) yaw wake deflection ("jimenez", "gauss")
  * Crespo & Hernandez (1996) wake-added turbulence with rotor-overlap weighting ("crespo_hernandez")
  * Sum-of-squares (SOSFS) and linear (FLS) deficit superposition
  * Vertical wind-shear profile, 3-D terrain elevation wake offset & hill exposure speedup,
    and 10-minute wind-direction meandering quadrature (Gaumond et al. 2014).

Standalone numpy only — no floris, torch, or mlratchet imports.
"""
from __future__ import annotations

import numpy as np

# Gauss-Hermite 3-point nodes & weights for N(0, 1): exact up to degree-5 polynomials
_GH3_X = np.array([-1.73205081, 0.0, 1.73205081], dtype=np.float64)
_GH3_W = np.array([1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0], dtype=np.float64)


def actuator_ct_from_cp(cp_elec: np.ndarray, eta_drive: float = 0.91, ct_max: float = 0.86) -> np.ndarray:
    """Derive 1-D momentum thrust coefficient Ct(U) from electrical Cp(U) = 4 a (1 - a)^2 * eta_drive."""
    cp_aero = np.clip(np.asarray(cp_elec, dtype=np.float64) / eta_drive, 0.0, 0.59259)
    # Solve 4 a (1 - a)^2 = cp_aero on the physical branch a in [0, 1/3] via Newton steps
    a = np.clip(0.25 * cp_aero, 0.0, 0.33)
    for _ in range(12):
        f = 4.0 * a * (1.0 - a) ** 2 - cp_aero
        df = 4.0 * (1.0 - a) * (1.0 - 3.0 * a)
        a = np.clip(a - f / np.maximum(df, 1e-6), 0.0, 1.0 / 3.0)
    ct = 4.0 * a * (1.0 - a)
    return np.clip(ct, 0.02, ct_max)


def _build_senvion_curves():
    ws = np.array(
        [0.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
         8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,
         14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 24.5, 30.0],
        dtype=np.float64,
    )
    # Senvion MM92 (2,050 kW, D = 92 m, IEC IIB) manufacturer curve
    p_mm92 = np.array(
        [0.0, 0.0, 20.0, 57.1, 94.0, 149.5, 205.0, 298.0, 391.0, 518.0, 645.0, 812.0,
         979.0, 1177.0, 1375.0, 1584.0, 1794.0, 1897.0, 2000.0, 2020.0, 2040.0, 2045.0, 2050.0,
         2050.0, 2050.0, 2050.0, 2050.0, 2050.0, 2050.0, 2050.0, 0.0, 0.0],
        dtype=np.float64,
    )
    area_92 = np.pi * (92.0 * 0.5) ** 2
    u_safe = np.maximum(ws, 1.0)
    cp_92 = np.where((ws >= 3.0) & (ws <= 24.0), (p_mm92 * 1e3) / (0.5 * 1.225 * area_92 * u_safe ** 3), 0.0)
    ct_92 = np.where((ws >= 3.0) & (ws <= 24.0), actuator_ct_from_cp(cp_92, eta_drive=0.915, ct_max=0.84), 0.02)
    # Below cut-inpeak, variable-speed rotor holds high Ct near cut-in
    ct_92 = np.where((ws >= 3.0) & (ws <= 7.0), np.maximum(ct_92, 0.80 - 0.01 * (7.0 - ws)), ct_92)

    # Senvion MM82 (2,050 kW, D = 82 m, IEC IIA) manufacturer curve
    p_mm82 = np.array(
        [0.0, 0.0, 0.0, 22.0, 59.0, 111.0, 163.0, 238.5, 314.0, 423.0, 532.0, 673.5,
         815.0, 988.5, 1162.0, 1345.0, 1528.0, 1676.5, 1825.0, 1902.5, 1980.0, 2007.5, 2035.0,
         2050.0, 2050.0, 2050.0, 2050.0, 2050.0, 2050.0, 2050.0, 2050.0, 0.0],
        dtype=np.float64,
    )
    area_82 = np.pi * (82.0 * 0.5) ** 2
    cp_82 = np.where((ws >= 3.5) & (ws <= 24.5), (p_mm82 * 1e3) / (0.5 * 1.225 * area_82 * u_safe ** 3), 0.0)
    ct_82 = np.where((ws >= 3.5) & (ws <= 24.5), actuator_ct_from_cp(cp_82, eta_drive=0.915, ct_max=0.84), 0.02)
    ct_82 = np.where((ws >= 3.5) & (ws <= 7.5), np.maximum(ct_82, 0.80 - 0.01 * (7.5 - ws)), ct_82)

    mm92 = {
        "name": "Senvion_MM92",
        "rotor_diameter": 92.0,
        "hub_height": 78.5,
        "rated_power_kw": 2050.0,
        "wind_speed": ws,
        "power_kw": p_mm92,
        "thrust_coefficient": ct_92,
        "ref_air_density": 1.225,
        "cosine_loss_exponent_yaw": 1.88,
        "cosine_loss_exponent_tilt": 1.88,
        "ref_tilt": 5.0,
        "tilt": 5.0,
    }
    mm82 = {
        "name": "Senvion_MM82",
        "rotor_diameter": 82.0,
        "hub_height": 59.0,
        "rated_power_kw": 2050.0,
        "wind_speed": ws,
        "power_kw": p_mm82,
        "thrust_coefficient": ct_82,
        "ref_air_density": 1.225,
        "cosine_loss_exponent_yaw": 1.88,
        "cosine_loss_exponent_tilt": 1.88,
        "ref_tilt": 5.0,
        "tilt": 5.0,
    }
    return {"MM92": mm92, "MM82": mm82}


TURBINES = _build_senvion_curves()


def make_parametric_turbine(
    rotor_diameter: float,
    rated_power_kw: float,
    hub_height: float,
    cp_max: float = 0.465,
    ct_region2: float = 0.81,
    u_cutin: float = 3.2,
    u_cutout: float = 25.0,
    cosine_loss_exponent_yaw: float = 1.88,
) -> dict:
    """Synthesise a physically consistent variable-speed pitch-regulated turbine curve."""
    ws = np.linspace(0.0, 30.0, 61)
    area = np.pi * (0.5 * rotor_diameter) ** 2
    p_aero_kw = 0.5 * 1.225 * area * cp_max * (ws ** 3) / 1e3
    # Smooth knee around rated wind speed
    u_cut_ramp = np.clip((ws - u_cutin) / 1.2, 0.0, 1.0)
    p_raw = p_aero_kw * (u_cut_ramp ** 1.5)
    # Soft pitch knee into rated power
    ratio = p_raw / max(rated_power_kw, 1.0)
    p_kw = rated_power_kw * ratio / (1.0 + ratio ** 5.0) ** (1.0 / 5.0)
    p_kw = np.where((ws >= u_cutin) & (ws <= u_cutout), p_kw, 0.0)

    u_safe = np.maximum(ws, 1.0)
    cp_elec = np.where(p_kw > 0, (p_kw * 1e3) / (0.5 * 1.225 * area * u_safe ** 3), 0.0)
    ct = actuator_ct_from_cp(cp_elec, eta_drive=0.915, ct_max=ct_region2 + 0.04)
    ct = np.where((ws >= u_cutin) & (ws <= 7.5), np.maximum(ct, ct_region2), ct)
    ct = np.where((ws >= u_cutin) & (ws <= u_cutout), ct, 0.02)
    return {
        "name": f"IEC_D{int(rotor_diameter)}_{int(rated_power_kw)}kW",
        "rotor_diameter": float(rotor_diameter),
        "hub_height": float(hub_height),
        "rated_power_kw": float(rated_power_kw),
        "wind_speed": ws,
        "power_kw": p_kw,
        "thrust_coefficient": ct,
        "ref_air_density": 1.225,
        "cosine_loss_exponent_yaw": float(cosine_loss_exponent_yaw),
        "cosine_loss_exponent_tilt": 1.88,
        "ref_tilt": 5.0,
        "tilt": 5.0,
    }


def wrap_deg(angle_deg):
    """Wrap angle(s) in degrees to (-180, 180]."""
    d = (np.asarray(angle_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return np.where(d == -180.0, 180.0, d)


def rotate_to_wind_frame(east_m: np.ndarray, north_m: np.ndarray, wind_dir_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (East, North) [m] into streamwise (+x downwind) and spanwise (+y left of downwind).

    Matches FLORIS's ``rotate_coordinates_rel_west``:
      wind_dir_deg = 270 (from West) -> +x = +East, +y = +North.
    """
    e = np.asarray(east_m, dtype=np.float64)
    n = np.asarray(north_m, dtype=np.float64)
    ec = 0.5 * (e.min() + e.max())
    nc = 0.5 * (n.min() + n.max())
    de, dn = e - ec, n - nc
    th = np.deg2rad(wind_dir_deg)
    s, c = np.sin(th), np.cos(th)
    x = -de * s - dn * c + ec
    y = +de * c - dn * s + nc
    return x, y


def freestream_mask(
    east_m: np.ndarray,
    north_m: np.ndarray,
    rotor_d_m: np.ndarray,
    wind_dir_deg: float,
    online: np.ndarray | None = None,
    wake_spread: float = 0.14,
    clearance_d: float = 1.6,
    max_dist_d: float = 25.0,
) -> np.ndarray:
    """Geometric front-row detector: True if no online turbine lies upwind inside the wake sector."""
    n = len(east_m)
    d = np.broadcast_to(np.asarray(rotor_d_m, dtype=np.float64), (n,))
    on = np.ones(n, dtype=bool) if online is None else np.asarray(online, dtype=bool)
    x, y = rotate_to_wind_frame(east_m, north_m, wind_dir_deg)
    # Pairwise dx[j, i] = x[i] - x[j] > 0 means j is upwind of i
    dx = x[None, :] - x[:, None]
    dy = np.abs(y[None, :] - y[:, None])
    d_up = d[:, None]
    waked_by_j = (
        on[:, None]
        & (dx > 0.2 * d_up)
        & (dx < max_dist_d * d_up)
        & (dy < clearance_d * d_up + wake_spread * dx)
    )
    np.fill_diagonal(waked_by_j, False)
    is_free = (~np.any(waked_by_j, axis=0)) & on
    # Fallback: if an array is >25D deep and every online unit got flagged, keep the most upwind online unit
    if not np.any(is_free) and np.any(on):
        idx_on = np.where(on)[0]
        is_free[idx_on[np.argmin(x[idx_on])]] = True
    return is_free


DEFAULT_WAKE_PARAMS = {
    "velocity_model": "gauss",          # "gauss" | "jensen"
    "deflection_model": "gauss",        # "gauss" | "jimenez" | "none"
    "turbulence_model": "crespo_hernandez",  # "crespo_hernandez" | "none"
    "combination_model": "sosfs",       # "sosfs" | "fls"
    # Bastankhah-Porte-Agel Gaussian parameters (FLORIS defaults)
    "alpha": 0.58,
    "beta": 0.077,
    "ka": 0.38,
    "kb": 0.004,
    "dm": 1.0,
    "ad": 0.0,
    "bd": 0.0,
    # Jensen & Jimenez parameters
    "we": 0.05,
    "kd": 0.05,
    # Crespo-Hernandez wake-added turbulence parameters (FLORIS defaults)
    "ch_initial": 0.1,
    "ch_constant": 0.5,
    "ch_ai": 0.8,
    "ch_downstream": -0.32,
    # Atmospheric boundary layer & real-farm extensions
    "wind_shear": 0.12,
    "wind_veer": 0.0,
    "air_density": 1.225,
    "n_grid": 3,                        # m x m rotor quadrature grid (FLORIS default = 3)
    "terrain_follow": 1.0,              # 1.0 = terrain-following streamlines (flat ground equivalent); <1 = partial 3-D terrain wake offset
    "hill_speedup": 0.0,                # linearised topographic speedup gain s_hill * (z_elev - mean_z) / H
    "sigma_wd_deg": 0.0,                # >0 integrates over 10-min wind-direction meandering N(wd, sigma_wd^2)
    "ct_scale": 1.0,                    # v4 (17 Nov 2026): multiplies the manufacturer thrust curve (1.0 = FLORIS)
}


def _solve_single_direction(
    east_m: np.ndarray,
    north_m: np.ndarray,
    elev_m: np.ndarray,
    hub_h_m: np.ndarray,
    turbine: dict,
    u_inf: float,
    wind_dir_deg: float,
    ti_inf: float,
    yaw_deg: np.ndarray,
    online: np.ndarray,
    cfg: dict,
) -> dict:
    """Sequential 3-D rotor-grid wake solve at one wind direction."""
    n = len(east_m)
    D = np.broadcast_to(np.asarray(turbine["rotor_diameter"], dtype=np.float64), (n,)).copy()
    H = np.broadcast_to(np.asarray(hub_h_m, dtype=np.float64), (n,)).copy()
    elev = np.broadcast_to(np.asarray(elev_m, dtype=np.float64), (n,)).copy()
    yaw_in = np.broadcast_to(np.asarray(yaw_deg, dtype=np.float64), (n,)).copy()
    on = np.broadcast_to(np.asarray(online, dtype=bool), (n,)).copy()

    ws_tab = np.asarray(turbine["wind_speed"], dtype=np.float64)
    p_tab = np.asarray(turbine["power_kw"], dtype=np.float64)
    ct_tab = np.asarray(turbine["thrust_coefficient"], dtype=np.float64)
    rho_ref = float(turbine.get("ref_air_density", 1.225))
    rho = float(cfg["air_density"])
    p_yaw = float(turbine.get("cosine_loss_exponent_yaw", 1.88))
    z_ref = float(cfg.get("ref_height", np.mean(H)))
    shear = float(cfg["wind_shear"])
    veer = float(cfg["wind_veer"])
    m = int(cfg["n_grid"])
    t_follow = float(cfg["terrain_follow"])
    s_hill = float(cfg["hill_speedup"])

    x_c, y_c = rotate_to_wind_frame(east_m, north_m, wind_dir_deg)
    order = np.argsort(x_c)
    inv_order = np.argsort(order)

    x_s = x_c[order]
    y_s = y_c[order]
    elev_s = elev[order]
    H_s = H[order]
    D_s = D[order]
    yaw_s = yaw_in[order]
    on_s = on[order]

    # Rotor m x m grid offsets in [-0.25*D, +0.25*D] (exact FLORIS TurbineGrid disc_area_radius = 0.5*R)
    if m <= 1:
        rel = np.array([0.0], dtype=np.float64)
    else:
        rel = np.linspace(-0.25, 0.25, m, dtype=np.float64)
    gy_rel, gz_rel = np.meshgrid(rel, rel, indexing="ij")  # (m, m)

    # Grid coordinates (n, m, m)
    xg = np.broadcast_to(x_s[:, None, None], (n, m, m)).copy()
    yg = y_s[:, None, None] + D_s[:, None, None] * gy_rel[None, :, :]
    zg_agl = np.maximum(H_s[:, None, None] + D_s[:, None, None] * gz_rel[None, :, :], 1.0)

    # Topographic hill speedup factor (1.0 when hill_speedup == 0.0)
    if s_hill != 0.0 and n > 1:
        z_mean = np.mean(elev_s)
        hill_fac = np.clip(1.0 + s_hill * (elev_s - z_mean) / np.maximum(H_s, 30.0), 0.80, 1.25)
    else:
        hill_fac = np.ones(n, dtype=np.float64)

    # Initial undisturbed velocity field on every rotor grid (n, m, m)
    u_init = u_inf * hill_fac[:, None, None] * (zg_agl / max(z_ref, 1.0)) ** shear
    wake_accum = np.zeros_like(u_init)  # accumulated sum of squares (sosfs) or linear sum (fls)
    ti_turb = np.broadcast_to(np.full((n, m, m), float(ti_inf), dtype=np.float64), (n, m, m)).copy()

    u_rotor_s = np.zeros(n, dtype=np.float64)
    ct_s = np.zeros(n, dtype=np.float64)
    p_kw_s = np.zeros(n, dtype=np.float64)
    p_free_s = np.zeros(n, dtype=np.float64)

    # Pairwise directed wake arrays in sorted order (j -> i), then un-permuted at end
    pair_def_s = np.zeros((n, n), dtype=np.float64)
    pair_ovl_s = np.zeros((n, n), dtype=np.float64)
    pair_del_s = np.zeros((n, n), dtype=np.float64)
    pair_sy_s = np.zeros((n, n), dtype=np.float64)

    vel_model = cfg["velocity_model"]
    def_model = cfg["deflection_model"]
    turb_model = cfg["turbulence_model"]
    comb_model = cfg["combination_model"]

    alpha, beta = float(cfg["alpha"]), float(cfg["beta"])
    ka, kb = float(cfg["ka"]), float(cfg["kb"])
    dm, ad, bd = float(cfg["dm"]), float(cfg["ad"]), float(cfg["bd"])
    we, kd = float(cfg["we"]), float(cfg["kd"])
    veer_rad = np.deg2rad(veer)
    cos_veer = np.cos(veer_rad)
    sin_veer = np.sin(veer_rad)
    sin2_veer = np.sin(2.0 * veer_rad)

    rho_scale = (rho / rho_ref) ** (1.0 / 3.0)

    # Freestream (unwaked) power for every turbine at its own hub shear, elevation & yaw=0
    u_free_rotor = np.cbrt(np.mean(u_init ** 3, axis=(1, 2)))
    p_free_s = np.interp(u_free_rotor * rho_scale, ws_tab, p_tab, left=0.0, right=0.0)
    p_free_s = np.where(on_s, p_free_s, 0.0)

    for j in range(n):
        if comb_model == "sosfs":
            u_j_grid = np.maximum(u_init[j] - np.sqrt(wake_accum[j]), 0.0)
        else:
            u_j_grid = np.maximum(u_init[j] - wake_accum[j], 0.0)

        u_avg_j = float(np.cbrt(np.mean(u_j_grid ** 3)))
        u_rotor_s[j] = u_avg_j

        if not on_s[j] or u_avg_j <= 0.1:
            ct_s[j] = 0.0
            p_kw_s[j] = 0.0
            continue

        cos_yaw = np.cos(np.deg2rad(yaw_s[j]))
        u_eff_j = u_avg_j * rho_scale * (max(cos_yaw, 0.0) ** (p_yaw / 3.0))
        p_kw_s[j] = float(np.interp(u_eff_j, ws_tab, p_tab, left=0.0, right=0.0))

        ct_raw = float(np.interp(u_avg_j, ws_tab, ct_tab, left=0.0001, right=0.0001)) * float(cfg.get("ct_scale", 1.0))
        ct_raw = float(np.clip(ct_raw, 0.0001, 0.9999))
        # FLORIS CosineLossTurbine: Ct = Ct_raw * cos(yaw); axial induction a = 0.5 / cos(yaw) * (1 - sqrt(1 - Ct_raw * cos^2(yaw)))
        ct_j = ct_raw * max(cos_yaw, 0.0)
        ct_s[j] = ct_j
        if cos_yaw > 1e-6:
            ai_j = (0.5 / cos_yaw) * (1.0 - np.sqrt(max(1.0 - ct_j * cos_yaw, 0.0)))
        else:
            ai_j = 0.0

        if j == n - 1:
            continue

        # Compute wake shed by turbine j onto downstream turbines idx = j+1 .. n-1
        sl = slice(j + 1, n)
        dx = xg[sl] - x_s[j]  # (n_down, m, m)
        down_mask = dx > 1e-4
        if not np.any(down_mask):
            continue

        Dj = D_s[j]
        Hj = H_s[j]
        ti_j = ti_turb[j]  # (m, m) rotor turbulence grid
        u_init_down = u_init[sl]

        # Effective wake centerline height at downstream positions:
        # terrain_follow=1 -> wake stays at Hj AGL; terrain_follow=0 -> wake stays at absolute altitude elev[j] + Hj
        dz_terrain = (1.0 - t_follow) * (elev_s[j] - elev_s[sl])[:, None, None]
        z_wake_agl = Hj + dz_terrain
        dz = zg_agl[sl] - z_wake_agl

        # 1. Yaw wake deflection delta(dx)
        if def_model == "none" or (abs(yaw_s[j]) < 1e-7 and ad == 0.0 and bd == 0.0):
            defl = np.zeros_like(dx)
        elif def_model == "jimenez":
            xi_init = cos_yaw * np.sin(np.deg2rad(yaw_s[j])) * ct_j / 2.0
            term = 2.0 * kd * dx / Dj + 1.0
            num_a = 15.0 * (term ** 4.0) + xi_init ** 2.0
            den_b = (30.0 * kd / Dj) * (term ** 5.0)
            c_term = xi_init * Dj * (15.0 + xi_init ** 2.0) / (30.0 * kd)
            defl = (xi_init * num_a / den_b - c_term) + (ad + bd * dx)
            defl = np.where(down_mask, defl, 0.0)
        else:  # Bastankhah & Porte-Agel (2016) Gaussian deflection
            yaw_opp = -yaw_s[j]
            cos_yo = np.cos(np.deg2rad(yaw_opp))
            ct_cos = np.clip(ct_j * cos_yo, 0.0, 0.9999)
            sqrt_1m_ct = np.sqrt(max(1.0 - ct_j, 1e-8))
            sqrt_1m_ctcos = np.sqrt(max(1.0 - ct_cos, 1e-8))
            uR_rel = ct_cos / max(2.0 * (1.0 - sqrt_1m_ctcos), 1e-8)
            u0_rel = sqrt_1m_ct
            x0_len = Dj * (cos_yo * (1.0 + sqrt_1m_ctcos)) / (
                np.sqrt(2.0) * np.maximum(4.0 * alpha * ti_j + 2.0 * beta * (1.0 - sqrt_1m_ct), 1e-6)
            )[None, :, :]
            ky = (ka * ti_j + kb)[None, :, :]
            kz = (ka * ti_j + kb)[None, :, :]
            C0 = 1.0 - u0_rel
            M0 = max(C0 * (2.0 - C0), 1e-8)
            E0 = C0 ** 2 - 3.0 * np.exp(1.0 / 12.0) * C0 + 3.0 * np.exp(1.0 / 3.0)
            sz0 = 0.5 * Dj * np.sqrt(max(uR_rel / (1.0 + u0_rel), 1e-8))
            sy0 = sz0 * max(cos_yo * cos_veer, 1e-6)
            theta_c0 = dm * (0.3 * np.deg2rad(yaw_opp) / max(cos_yo, 1e-6)) * (1.0 - sqrt_1m_ctcos)
            delta0 = np.tan(theta_c0) * x0_len

            dx_pos = np.maximum(dx, 0.0)
            near_d = (dx_pos / np.maximum(x0_len, 1e-6)) * delta0 + (ad + bd * dx_pos)
            sy_f = ky * np.maximum(dx_pos - x0_len, 0.0) + sy0
            sz_f = kz * np.maximum(dx_pos - x0_len, 0.0) + sz0
            m0_sq = np.sqrt(M0)
            mid_ratio = np.sqrt((sy_f * sz_f) / max(sy0 * sz0, 1e-8))
            ln_num = np.maximum((1.6 + m0_sq) * (1.6 * mid_ratio - m0_sq), 1e-12)
            ln_den = np.maximum((1.6 - m0_sq) * (1.6 * mid_ratio + m0_sq), 1e-12)
            far_d = delta0 + (theta_c0 * E0 / 5.2) * np.sqrt((sy0 * sz0) / np.maximum(ky * kz * M0, 1e-12)) * np.log(ln_num / ln_den) + (ad + bd * dx_pos)
            defl = np.where((dx >= 0.0) & (dx <= x0_len), near_d, np.where(dx > x0_len, far_d, 0.0))

        dy = yg[sl] - y_s[j] - defl

        # 2. Wake velocity deficit du (m/s)
        if vel_model == "jensen":
            r_rad = 0.5 * Dj
            r_wake = r_rad + we * np.maximum(dx, 0.0)
            inside = (dx > 0.001) & (np.sqrt(dy ** 2 + dz ** 2) < r_wake)
            c_jen = (r_rad / (r_wake + 0.001)) ** 2
            du = np.where(inside, u_init_down * (2.0 * ai_j * c_jen), 0.0)
            sy_rep = r_wake[:, m // 2, m // 2]
        else:  # Bastankhah & Porte-Agel (2014, 2016) near- + far-wake Gaussian
            yaw_opp = -yaw_s[j]
            cos_yo = np.cos(np.deg2rad(yaw_opp))
            sqrt_1m_ct = np.sqrt(max(1.0 - ct_j, 1e-8))
            uR_rel = ct_j / max(2.0 * (1.0 - sqrt_1m_ct), 1e-8)
            u0_rel = sqrt_1m_ct
            sz0 = 0.5 * Dj * np.sqrt(max(uR_rel / (1.0 + u0_rel), 1e-8))
            sy0 = sz0 * max(cos_yo * cos_veer, 1e-6)
            x0_len = Dj * cos_yo * (1.0 + sqrt_1m_ct) / (
                np.sqrt(2.0) * np.maximum(4.0 * alpha * ti_j + 2.0 * beta * (1.0 - sqrt_1m_ct), 1e-6)
            )[None, :, :]
            ky = (ka * ti_j + kb)[None, :, :]
            kz = (ka * ti_j + kb)[None, :, :]

            dx_pos = np.maximum(dx, 0.0)
            is_near = (dx > 0.1) & (dx < x0_len)
            is_far = dx >= x0_len
            in_wake = is_near | is_far

            ramp_up = dx_pos / np.maximum(x0_len, 1e-6)
            ramp_dn = 1.0 - ramp_up
            s_core = 0.501 * Dj * np.sqrt(ct_j / 2.0)
            sy_near = np.maximum(ramp_dn * s_core + ramp_up * sy0, 1e-3)
            sz_near = np.maximum(ramp_dn * s_core + ramp_up * sz0, 1e-3)

            sy_far = np.maximum(ky * (dx_pos - x0_len) + sy0, 1e-3)
            sz_far = np.maximum(kz * (dx_pos - x0_len) + sz0, 1e-3)

            sy = np.where(is_near, sy_near, sy_far)
            sz = np.where(is_near, sz_near, sz_far)

            a_el = (cos_veer ** 2) / (2.0 * sy ** 2) + (sin_veer ** 2) / (2.0 * sz ** 2)
            b_el = -sin2_veer / (4.0 * sy ** 2) + sin2_veer / (4.0 * sz ** 2)
            c_el = (sin_veer ** 2) / (2.0 * sy ** 2) + (cos_veer ** 2) / (2.0 * sz ** 2)
            r2 = a_el * (dy ** 2) - 2.0 * b_el * dy * dz + c_el * (dz ** 2)

            radicand = np.clip(1.0 - (ct_j * cos_yo) / (8.0 * sy * sz / (Dj ** 2)), 0.0, 1.0)
            C_amp = 1.0 - np.sqrt(radicand)
            du = np.where(in_wake, u_init_down * C_amp * np.exp(-r2), 0.0)
            sy_rep = sy[:, m // 2, m // 2]

        # Accumulate deficit onto downstream rotors
        if comb_model == "sosfs":
            wake_accum[sl] += du ** 2
        else:
            wake_accum[sl] += du

        # Record pairwise j -> i wake diagnostics at rotor hub/mean
        du_frac = np.mean(du / np.maximum(u_init_down, 0.1), axis=(1, 2))
        area_overlap = np.mean(du > 0.05, axis=(1, 2))
        pair_def_s[j, sl] = du_frac
        pair_ovl_s[j, sl] = area_overlap
        pair_del_s[j, sl] = defl[:, m // 2, m // 2] / Dj
        pair_sy_s[j, sl] = sy_rep / Dj

        # 3. Crespo-Hernandez wake-added turbulence (rotor-grid max-quadrature within 15D x +-2D)
        if turb_model == "crespo_hernandez" and ai_j > 1e-6:
            dx_safe = np.where(dx > 0.1, dx, 1.0)
            wat = (
                float(cfg["ch_constant"])
                * (ai_j ** float(cfg["ch_ai"]))
                * (max(float(ti_inf), 1e-4) ** float(cfg["ch_initial"]))
                * ((dx_safe / Dj) ** float(cfg["ch_downstream"]))
            )
            wat = np.where(dx > 0.1, wat, 0.0)
            in_wat_box = (dx > 0.0) & (np.abs(y_s[j] - yg[sl]) < 2.0 * Dj) & (dx <= 15.0 * Dj)
            ti_add = area_overlap[:, None, None] * np.nan_to_num(wat, posinf=0.0) * in_wat_box
            ti_turb[sl] = np.maximum(np.sqrt(ti_add ** 2 + float(ti_inf) ** 2), ti_turb[sl])

    # Un-permute back to original turbine index order 0 .. n-1
    u_rotor = u_rotor_s[inv_order]
    ct_out = ct_s[inv_order]
    p_kw = p_kw_s[inv_order]
    p_free = p_free_s[inv_order]
    ti_out = np.mean(ti_turb, axis=(1, 2))[inv_order]

    pair_def = pair_def_s[np.ix_(inv_order, inv_order)]
    pair_ovl = pair_ovl_s[np.ix_(inv_order, inv_order)]
    pair_del = pair_del_s[np.ix_(inv_order, inv_order)]
    pair_sy = pair_sy_s[np.ix_(inv_order, inv_order)]

    eta = np.where(p_free > 1.0, p_kw / np.maximum(p_free, 1.0), np.where(on, 1.0, 0.0))
    return {
        "x_m": x_c,
        "y_m": y_c,
        "u_rotor_ms": u_rotor,
        "u_free_ms": u_free_rotor[inv_order],
        "ti_eff": ti_out,
        "ct": ct_out,
        "power_kw": p_kw,
        "power_free_kw": p_free,
        "eta": eta,
        "wake_loss": np.where(on, 1.0 - eta, 0.0),
        "farm_power_kw": float(np.sum(p_kw)),
        "farm_free_kw": float(np.sum(p_free)),
        "farm_wake_loss": float(1.0 - np.sum(p_kw) / max(np.sum(p_free), 1.0)),
        "pair_deficit": pair_def,
        "pair_overlap": pair_ovl,
        "pair_deflection_d": pair_del,
        "pair_sigma_y_d": pair_sy,
    }


def solve_wake(
    east_m: np.ndarray,
    north_m: np.ndarray,
    hub_h_m: np.ndarray | float,
    turbine: dict,
    u_inf: float,
    wind_dir_deg: float,
    ti_inf: float = 0.08,
    yaw_deg: np.ndarray | float = 0.0,
    online: np.ndarray | None = None,
    elev_m: np.ndarray | float = 0.0,
    params: dict | None = None,
) -> dict:
    """Solve 3-D wind-farm wakes, optionally integrating over 10-minute wind-direction meandering."""
    cfg = dict(DEFAULT_WAKE_PARAMS)
    if params:
        cfg.update(params)
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    n = len(east)
    hub_h = np.broadcast_to(np.asarray(hub_h_m, dtype=np.float64), (n,))
    elev = np.broadcast_to(np.asarray(elev_m, dtype=np.float64), (n,))
    yaw = np.broadcast_to(np.asarray(yaw_deg, dtype=np.float64), (n,))
    on = np.ones(n, dtype=bool) if online is None else np.asarray(online, dtype=bool)

    sigma_wd = float(cfg.get("sigma_wd_deg", 0.0))
    if sigma_wd <= 0.15:
        return _solve_single_direction(east, north, elev, hub_h, turbine, u_inf, wind_dir_deg, ti_inf, yaw, on, cfg)

    # 3-point Gauss-Hermite quadrature over wind direction meandering N(wind_dir_deg, sigma_wd^2), reusing center at xq=0
    center = _solve_single_direction(east, north, elev, hub_h, turbine, u_inf, wind_dir_deg, ti_inf, yaw, on, cfg)
    w0 = _GH3_W[1]
    p_acc = w0 * center["power_kw"]
    pf_acc = w0 * center["power_free_kw"]
    u_acc = w0 * center["u_rotor_ms"]
    ti_acc = w0 * center["ti_eff"]
    pdef_acc = w0 * center["pair_deficit"]
    povl_acc = w0 * center["pair_overlap"]

    for xq, wq in ((_GH3_X[0], _GH3_W[0]), (_GH3_X[2], _GH3_W[2])):
        d_wd = xq * sigma_wd
        res = _solve_single_direction(
            east, north, elev, hub_h, turbine, u_inf,
            wind_dir_deg + d_wd, ti_inf, yaw + 0.35 * d_wd, on, cfg,
        )
        p_acc += wq * res["power_kw"]
        pf_acc += wq * res["power_free_kw"]
        u_acc += wq * res["u_rotor_ms"]
        ti_acc += wq * res["ti_eff"]
        pdef_acc += wq * res["pair_deficit"]
        povl_acc += wq * res["pair_overlap"]

    eta = np.where(pf_acc > 1.0, p_acc / np.maximum(pf_acc, 1.0), np.where(on, 1.0, 0.0))
    out = dict(center)
    out.update({
        "u_rotor_ms": u_acc,
        "ti_eff": ti_acc,
        "power_kw": p_acc,
        "power_free_kw": pf_acc,
        "eta": eta,
        "wake_loss": np.where(on, 1.0 - eta, 0.0),
        "farm_power_kw": float(np.sum(p_acc)),
        "farm_free_kw": float(np.sum(pf_acc)),
        "farm_wake_loss": float(1.0 - np.sum(p_acc) / max(np.sum(pf_acc), 1.0)),
        "pair_deficit": pdef_acc,
        "pair_overlap": povl_acc,
    })
    return out
