"""Procedural wind-farm wake simulator for WakeNet across physics versions v1, v2, v2_raw_vane, and v3.

Separates:
  1. Procedural wind-farm layout & rotor prior (N = 5..16 turbines, never copying real farm coordinates)
  2. Versioned governing wake & atmospheric boundary-layer physics (v1 -> v2 -> v2_raw_vane -> v3)
  3. Real unwaked backdrop injection from Kelmarsh 2017-2018 (backdrop_pool.npz, split train vs val).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
import physics as WP  # noqa: E402
import pipeline as PL  # noqa: E402

PHYSICS_VERSION = "v4"
N_MAX = 16


class BackdropSampler:
    """Draws real inflow tuples and real unwaked per-turbine fluctuations (disjoint train vs val)."""

    def __init__(self, split: str = "train"):
        z = np.load(_P.work("proc", "backdrop_pool.npz"))
        prefix = "train" if split == "train" else "val"
        self.inflow = z[f"{prefix}_inflow"]  # (M, 5): u_inf, wd_inf, ti_inf, sigma_wd, rho
        self.eps_p = z[f"{prefix}_eps_p"]
        self.eps_u = z[f"{prefix}_eps_u"]
        self.yaw = z[f"{prefix}_yaw"]


def sample_turbine(rng: np.random.Generator) -> dict:
    u = rng.random()
    if u < 0.35:
        return WP.TURBINES["MM92"]
    if u < 0.60:
        return WP.TURBINES["MM82"]
    D = float(rng.uniform(74.0, 132.0))
    # Specific power ~ 300-420 W/m^2 (realistic onshore IEC Class I-III)
    spec_w_m2 = float(rng.uniform(290.0, 410.0))
    area = np.pi * (0.5 * D) ** 2
    p_rated = round(spec_w_m2 * area / 1e3, 1)
    hh = float(D * rng.uniform(0.68, 0.95))
    return WP.make_parametric_turbine(
        rotor_diameter=D,
        rated_power_kw=p_rated,
        hub_height=hh,
        cp_max=float(rng.uniform(0.445, 0.480)),
        ct_region2=float(rng.uniform(0.78, 0.84)),
        u_cutin=float(rng.uniform(3.0, 3.6)),
        cosine_loss_exponent_yaw=float(rng.uniform(1.70, 2.05)),
    )


def sample_layout(rng: np.random.Generator, D: float, base_hh: float, with_3d: bool = True) -> dict:
    """Sample a procedural N-turbine wind farm layout (N in 5..16) with minimum spacing >= 2.7 D."""
    n = int(rng.integers(5, N_MAX + 1))
    kind = int(rng.integers(0, 4))

    if kind == 0:
        # Staggered multi-row array (2 to 4 rows)
        n_rows = int(rng.integers(2, min(5, n)))
        n_cols = int(np.ceil(n / n_rows))
        sx = float(rng.uniform(3.4, 7.5)) * D
        sy = float(rng.uniform(3.0, 5.8)) * D
        stag = float(rng.uniform(-0.5, 0.5)) * sy
        pts = []
        for r in range(n_rows):
            for c in range(n_cols):
                if len(pts) < n:
                    jx = rng.normal(0.0, 0.25 * D)
                    jy = rng.normal(0.0, 0.25 * D)
                    pts.append((r * sx + jx, c * sy + (r % 2) * stag + jy))
        xy = np.array(pts, dtype=np.float64)
    elif kind == 1:
        # Irregular onshore cluster via rejection dart-throwing (like UK land-constrained farms!)
        min_sep = float(rng.uniform(2.75, 4.4)) * D
        span = min_sep * np.sqrt(n) * float(rng.uniform(1.15, 1.65))
        pts = [(0.0, 0.0)]
        tries = 0
        while len(pts) < n and tries < 600:
            tries += 1
            cand = (float(rng.uniform(-0.5 * span, 0.5 * span)), float(rng.uniform(-0.5 * span, 0.5 * span)))
            if all(np.hypot(cand[0] - p[0], cand[1] - p[1]) >= min_sep for p in pts):
                pts.append(cand)
        while len(pts) < n:
            pts.append((len(pts) * 3.5 * D, rng.normal(0.0, 1.5 * D)))
        xy = np.array(pts, dtype=np.float64)
    elif kind == 2:
        # 1 or 2 curved ridge lines
        n_lines = 1 if n <= 7 and rng.random() < 0.5 else 2
        pts = []
        for l_idx in range(n_lines):
            m_l = n // n_lines + (1 if l_idx < (n % n_lines) else 0)
            s_step = float(rng.uniform(2.9, 4.8)) * D
            curv = float(rng.normal(0.0, 0.04 / D))
            for k in range(m_l):
                s = (k - 0.5 * (m_l - 1)) * s_step
                pts.append((l_idx * float(rng.uniform(4.2, 7.0)) * D + curv * s ** 2 + rng.normal(0, 0.15 * D), s))
        xy = np.array(pts[:n], dtype=np.float64)
    else:
        # Perturbed rectangular grid
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))
        sx = float(rng.uniform(3.2, 6.5)) * D
        sy = float(rng.uniform(3.2, 6.5)) * D
        pts = []
        for r in range(n_rows):
            for c in range(n_cols):
                if len(pts) < n:
                    pts.append((r * sx + rng.normal(0, 0.4 * D), c * sy + rng.normal(0, 0.4 * D)))
        xy = np.array(pts, dtype=np.float64)

    # Random compass rotation of the entire farm
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    ca, sa = np.cos(ang), np.sin(ang)
    east = ca * xy[:, 0] - sa * xy[:, 1]
    north = sa * xy[:, 0] + ca * xy[:, 1]
    east -= np.mean(east)
    north -= np.mean(north)

    hub_h = np.full(n, base_hh, dtype=np.float64)
    elev = np.zeros(n, dtype=np.float64)

    if with_3d:
        # 35% of farms have mixed tower hub heights (e.g., planning height restrictions like Kelmarsh's -10 m towers)
        if rng.random() < 0.35:
            n_short = int(rng.integers(1, max(2, n // 2)))
            idx_s = rng.choice(n, size=n_short, replace=False)
            hub_h[idx_s] = np.maximum(base_hh + rng.choice([-12.0, -10.0, -8.0, +8.0, +10.0]), 45.0)
        # Smooth rolling terrain relief (Gaussian hill + gentle linear ridge slope, 0 to 48 m span)
        if rng.random() < 0.80:
            amp_hill = float(rng.uniform(4.0, 32.0))
            cx = float(rng.uniform(-400.0, 400.0))
            cy = float(rng.uniform(-400.0, 400.0))
            width = float(rng.uniform(350.0, 1100.0))
            slope_e = float(rng.normal(0.0, 0.012))
            slope_n = float(rng.normal(0.0, 0.012))
            elev = amp_hill * np.exp(-((east - cx) ** 2 + (north - cy) ** 2) / (2.0 * width ** 2)) + slope_e * east + slope_n * north
            elev -= np.min(elev)

    return {"n": n, "east_m": east, "north_m": north, "elev_m": elev, "hub_h_m": hub_h}


def simulate_one(rng: np.random.Generator, version: str, backdrop: BackdropSampler) -> dict | None:
    """Simulate one wind-farm snapshot and return padded (N_MAX=16) graph inputs + noisy real-like targets."""
    turb = sample_turbine(rng)
    D = float(turb["rotor_diameter"])
    base_hh = float(turb["hub_height"])
    with_3d = version != "v1"
    lay = sample_layout(rng, D, base_hh, with_3d=with_3d)
    n = lay["n"]
    east, north, elev, hub_h = lay["east_m"], lay["north_m"], lay["elev_m"], lay["hub_h_m"]

    # Online status (20% of snapshots have 1 or 2 parked turbines)
    online = np.ones(n, dtype=bool)
    if version != "v1" and n >= 6 and rng.random() < 0.22:
        n_off = 1 if (n < 10 or rng.random() < 0.75) else 2
        online[rng.choice(n, size=n_off, replace=False)] = False

    # Inflow conditions
    if version == "v1":
        u_inf = float(rng.uniform(5.0, 12.0))
        wd_inf = float(rng.uniform(0.0, 360.0))
        ti_inf = float(rng.uniform(0.05, 0.16))
        sigma_wd = float(rng.uniform(2.0, 10.0))
        rho = 1.225
    else:
        # 75% empirical UK inflow tuples (with random wind direction to cover all farm alignments), 25% wide uniform
        idx = int(rng.integers(0, len(backdrop.inflow)))
        u_b, wd_b, ti_b, swd_b, rho_b = backdrop.inflow[idx]
        if rng.random() < 0.75:
            u_inf = float(np.clip(u_b + rng.normal(0.0, 0.35), 5.0, 12.0))
            wd_inf = float((wd_b + rng.uniform(-180.0, 180.0)) % 360.0)
            ti_inf = float(np.clip(ti_b * rng.uniform(0.85, 1.15), 0.03, 0.26))
            sigma_wd = float(np.clip(swd_b * rng.uniform(0.85, 1.15), 1.0, 16.0))
            rho = float(np.clip(rho_b + rng.normal(0.0, 0.01), 1.14, 1.30))
        else:
            u_inf = float(rng.uniform(5.0, 12.0))
            wd_inf = float(rng.uniform(0.0, 360.0))
            ti_inf = float(rng.uniform(0.04, 0.22))
            sigma_wd = float(rng.uniform(1.5, 14.0))
            rho = float(rng.uniform(1.16, 1.28))

    is_free = WP.freestream_mask(east, north, D, wd_inf, online=online)
    if int(np.sum(is_free)) < 2 or int(np.sum(online & (~is_free))) < 1:
        return None

    # Per-turbine mechanical yaw offsets: real tracking jitter + 30% active wake-steering campaigns!
    if version == "v1":
        yaw = rng.normal(0.0, 3.5, size=n)
    else:
        yaw = backdrop.yaw[rng.integers(0, len(backdrop.yaw), size=n)].astype(np.float64)
        if rng.random() < 0.30:
            # Active yaw-steering offset (+-6 to +-22 deg) on upstream turbines
            steer = rng.uniform(6.0, 22.0) * rng.choice([-1.0, 1.0])
            yaw = np.where(is_free, np.clip(yaw + steer, -28.0, 28.0), yaw)
    yaw = np.where(online, np.clip(yaw, -30.0, 30.0), 0.0)

    # True forward physics solve for this physics version
    h_ref = float(np.mean(hub_h[online]))
    if version == "v1":
        p_cfg = {
            "velocity_model": "jensen",
            "deflection_model": "jimenez",
            "turbulence_model": "none",
            "combination_model": "sosfs",
            "we": float(rng.uniform(0.04, 0.06)),
            "wind_shear": 0.0,
            "terrain_follow": 1.0,
            "hill_speedup": 0.0,
            "sigma_wd_deg": 0.0,
            "air_density": rho,
            "ref_height": h_ref,
        }
    elif version == "v2":
        p_cfg = {
            "velocity_model": "gauss",
            "deflection_model": "gauss",
            "turbulence_model": "crespo_hernandez",
            "combination_model": "sosfs",
            "ka": float(np.clip(rng.normal(0.38, 0.03), 0.28, 0.48)),
            "wind_shear": float(rng.uniform(0.10, 0.20)),
            "terrain_follow": 1.0,
            "hill_speedup": 0.0,
            "sigma_wd_deg": 0.0,
            "air_density": rho,
            "ref_height": h_ref,
        }
    elif version == "v2_raw_vane":
        p_cfg = {
            "velocity_model": "gauss",
            "deflection_model": "gauss",
            "turbulence_model": "crespo_hernandez",
            "combination_model": "sosfs",
            "ka": float(np.clip(rng.normal(0.38, 0.03), 0.28, 0.48)),
            "wind_shear": float(rng.uniform(0.10, 0.20)),
            "terrain_follow": 1.0,
            "hill_speedup": 0.0,
            "sigma_wd_deg": sigma_wd,  # Naive 100% raw vane StdDev smeared as atmospheric meandering!
            "air_density": rho,
            "ref_height": h_ref,
        }
    elif version == "v3":
        # Calibrated on Kelmarsh 2019 dev: separated meso-scale meandering (~0.25*sigma_wd) + 3D terrain hill exposure
        c_meander = float(rng.uniform(0.16, 0.36))
        s_hill = 0.0 if rng.random() < 0.20 else float(rng.uniform(0.05, 0.24))
        p_cfg = {
            "velocity_model": "gauss",
            "deflection_model": "gauss",
            "turbulence_model": "crespo_hernandez",
            "combination_model": "sosfs",
            "ka": float(np.clip(rng.normal(0.37, 0.045), 0.24, 0.50)),
            "wind_shear": float(rng.uniform(0.09, 0.21)),
            "terrain_follow": float(rng.uniform(0.68, 0.95)),
            "hill_speedup": s_hill,
            "sigma_wd_deg": c_meander * sigma_wd,
            "air_density": rho,
            "ref_height": h_ref,
        }
    elif version == "v4":
        # v3 plus a thrust scale drawn around the 2019 dev value (0.90; dev_calibration_v4.json). ka keeps v3's spread,
        # which already covers the dev choice (0.38).
        c_meander = float(rng.uniform(0.16, 0.36))
        s_hill = 0.0 if rng.random() < 0.20 else float(rng.uniform(0.05, 0.24))
        p_cfg = {
            "velocity_model": "gauss",
            "deflection_model": "gauss",
            "turbulence_model": "crespo_hernandez",
            "combination_model": "sosfs",
            "ka": float(np.clip(rng.normal(0.37, 0.045), 0.24, 0.50)),
            "wind_shear": float(rng.uniform(0.09, 0.21)),
            "terrain_follow": float(rng.uniform(0.68, 0.95)),
            "hill_speedup": s_hill,
            "sigma_wd_deg": c_meander * sigma_wd,
            "ct_scale": float(rng.uniform(0.82, 0.98)),
            "air_density": rho,
            "ref_height": h_ref,
        }
    else:
        raise ValueError(f"Unknown physics version: {version}")

    sim = WP.solve_wake(
        east, north, hub_h, turb, u_inf, wd_inf, ti_inf,
        yaw_deg=yaw, online=online, elev_m=elev, params=p_cfg,
    )
    p_clean = sim["power_kw"]
    u_clean = sim["u_rotor_ms"]
    if float(np.mean(p_clean[is_free])) < 60.0:
        return None

    # Inject REAL unwaked spatial turbulence fluctuations onto every turbine AND onto the front-row observation!
    if version == "v1":
        eps_p = rng.normal(0.0, 0.06, size=n)
        eps_u = rng.normal(0.0, 0.03, size=n)
        u_obs = float(np.clip(u_inf * (1.0 + rng.normal(0.0, 0.02)), 4.5, 13.0))
        wd_obs = float((wd_inf + rng.normal(0.0, 1.5)) % 360.0)
    else:
        ridx = rng.integers(0, len(backdrop.eps_p), size=n)
        eps_p = backdrop.eps_p[ridx].astype(np.float64)
        eps_u = backdrop.eps_u[ridx].astype(np.float64)
        # Front-row estimate of u_inf and wd_inf has real finite-front-row averaging error!
        u_obs = float(np.clip(u_inf * (1.0 + float(np.mean(eps_u[is_free]))), 4.5, 13.0))
        wd_obs = float((wd_inf + rng.normal(0.0, 1.6)) % 360.0)

    p_noisy = np.where(online, np.maximum(p_clean * (1.0 + eps_p), 1.0), 0.0)
    u_noisy = np.where(online, np.maximum(u_clean * (1.0 + eps_u), 0.2), 0.0)

    # Observed graph built ONLY from noisy front-row inflow (u_obs, wd_obs), exactly as on real SCADA!
    g = PL.build_graph(
        east, north, hub_h, turb,
        u_inf=u_obs, wind_dir_deg=wd_obs, ti_inf=ti_inf, sigma_wd_deg=sigma_wd,
        yaw_deg=yaw, online=online, elev_m=elev, air_density=rho,
    )
    obs_free = g["is_free"]
    if int(np.sum(obs_free)) < 1 or int(np.sum(online & (~obs_free))) < 1:
        return None

    p_front_obs = float(np.mean(p_noisy[obs_free]))
    if p_front_obs < 50.0:
        return None

    eta_target = np.where(online, np.clip(p_noisy / p_front_obs, 0.0, 1.45), 0.0)
    udef_target = np.where(online, np.clip(1.0 - u_noisy / np.maximum(g["u_free_ms"], 0.5), -0.25, 0.95), 0.0)
    farm_wake_target = float(1.0 - np.sum(eta_target[online]) / np.sum(online))

    # Pad to N_MAX = 16
    node_pad = np.zeros((N_MAX, PL.N_NODE_FEAT), dtype=np.float16)
    edge_pad = np.zeros((N_MAX, N_MAX, PL.N_EDGE_FEAT), dtype=np.float16)
    emask_pad = np.zeros((N_MAX, N_MAX), dtype=bool)
    on_pad = np.zeros(N_MAX, dtype=bool)
    free_pad = np.zeros(N_MAX, dtype=bool)
    anc_pad = np.zeros(N_MAX, dtype=np.float16)
    eta_pad = np.zeros(N_MAX, dtype=np.float16)
    udef_pad = np.zeros(N_MAX, dtype=np.float16)

    node_pad[:n] = g["node_x"].astype(np.float16)
    edge_pad[:n, :n] = g["edge_x"].astype(np.float16)
    emask_pad[:n, :n] = g["edge_mask"]
    on_pad[:n] = online
    free_pad[:n] = obs_free
    anc_pad[:n] = g["eta_anchor"].astype(np.float16)
    eta_pad[:n] = eta_target.astype(np.float16)
    udef_pad[:n] = udef_target.astype(np.float16)

    return {
        "node_x": node_pad,
        "edge_x": edge_pad,
        "edge_mask": emask_pad,
        "online": on_pad,
        "is_free": free_pad,
        "eta_anchor": anc_pad,
        "eta_target": eta_pad,
        "udef_target": udef_pad,
        "farm_wake_target": np.float32(farm_wake_target),
    }
