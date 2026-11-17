"""Independent verification of models/wakenet/physics.py against NREL FLORIS 4.7 and exact physical invariants.

Writes models/wakenet/training/physics_check.json (committed per docs/adding-a-model.md).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import yaml
from floris import FlorisModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)

import _paths as _P  # noqa: F401, E402
import physics as WP  # noqa: E402


def _load_floris_nrel5mw() -> dict:
    import floris

    fdir = pathlib.Path(floris.__file__).parent
    d = yaml.safe_load((fdir / "turbine_library" / "nrel_5MW.yaml").read_text())
    pt = d["power_thrust_table"]
    return {
        "name": "nrel_5MW",
        "rotor_diameter": float(d["rotor_diameter"]),
        "hub_height": float(d["hub_height"]),
        "rated_power_kw": float(np.max(pt["power"])),
        "wind_speed": np.asarray(pt["wind_speed"], dtype=np.float64),
        "power_kw": np.asarray(pt["power"], dtype=np.float64),
        "thrust_coefficient": np.asarray(pt["thrust_coefficient"], dtype=np.float64),
        "ref_air_density": float(pt.get("ref_air_density", 1.225)),
        "cosine_loss_exponent_yaw": float(pt.get("cosine_loss_exponent_yaw", 1.88)),
        "cosine_loss_exponent_tilt": float(pt.get("cosine_loss_exponent_tilt", 1.88)),
        "ref_tilt": float(pt.get("ref_tilt", 5.0)),
        "tilt": float(pt.get("ref_tilt", 5.0)),
    }


def _make_floris(vel_model: str, def_model: str, turb_model: str = "crespo_hernandez", shear: float = 0.12) -> FlorisModel:
    fm = FlorisModel("defaults")
    d = fm.core.as_dict()
    d["wake"]["model_strings"]["velocity_model"] = vel_model
    d["wake"]["model_strings"]["deflection_model"] = def_model
    d["wake"]["model_strings"]["turbulence_model"] = turb_model
    d["wake"]["model_strings"]["combination_model"] = "sosfs"
    # Test the pure Bastankhah-Porte-Agel (2014, 2016) / Jensen / Jimenez / Crespo-Hernandez baseline equations
    d["wake"]["enable_secondary_steering"] = False
    d["wake"]["enable_yaw_added_recovery"] = False
    d["wake"]["enable_transverse_velocities"] = False
    d["flow_field"]["wind_shear"] = shear
    return FlorisModel(d)


def _latlon_to_en(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0, lon0 = np.mean(lats), np.mean(lons)
    north = (lats - lat0) * 111_132.92
    east = (lons - lon0) * 111_412.84 * np.cos(np.deg2rad(lat0))
    return east, north


def main():
    turb = _load_floris_nrel5mw()
    D = turb["rotor_diameter"]
    H = turb["hub_height"]
    checks = []

    # 1. Bastankhah-Porte-Agel 3-D Gaussian wake across downstream spacings (3D..15D) and lateral offsets (-1.5D..1.5D)
    fm_g = _make_floris("gauss", "gauss", "crespo_hernandez", shear=0.12)
    u_diffs, p_rel_diffs = [], []
    for dx_d in [3.0, 5.0, 7.0, 10.0, 15.0]:
        for dy_d in [-1.5, -0.75, -0.35, 0.0, 0.35, 0.75, 1.5]:
            lx = [0.0, dx_d * D]
            ly = [0.0, dy_d * D]
            fm_g.set(layout_x=lx, layout_y=ly, wind_directions=[270.0], wind_speeds=[8.5], turbulence_intensities=[0.075])
            fm_g.run()
            u_fl = fm_g.turbine_average_velocities[0]
            p_fl = fm_g.get_turbine_powers()[0] / 1e3

            res = WP.solve_wake(
                lx, ly, H, turb, u_inf=8.5, wind_dir_deg=270.0, ti_inf=0.075,
                params={"velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez", "wind_shear": 0.12, "ref_height": H},
            )
            u_diffs.append(np.max(np.abs(res["u_rotor_ms"] - u_fl)))
            p_rel_diffs.append(np.max(np.abs(res["power_kw"] - p_fl) / np.maximum(p_fl, 1.0)))

    max_u_gauss = float(np.max(u_diffs))
    max_p_gauss = float(np.max(p_rel_diffs))
    checks.append({
        "name": "Bastankhah-Porte-Agel 3-D Gaussian single-wake grid (3D-15D x -1.5D..1.5D) vs NREL FLORIS 4.7",
        "reference": f"NREL FLORIS {yaml.__name__ and '4.7'} (gauss velocity + gauss deflection)",
        "max_velocity_abs_diff_ms": max_u_gauss,
        "max_power_rel_diff": max_p_gauss,
        "pass": max_u_gauss < 1e-9 and max_p_gauss < 1e-9,
    })

    # 2. Classic Jensen top-hat + Jimenez yaw deflection vs NREL FLORIS 4.7
    fm_j = _make_floris("jensen", "jimenez", "none", shear=0.12)
    lx_j = [0.0, 5.0 * D, 5.0 * D]
    ly_j = [0.0, 0.35 * D, -0.35 * D]
    j_u_diffs, j_p_diffs = [], []
    for yaw0 in [-20.0, -10.0, 0.0, 10.0, 20.0]:
        fm_j.set(
            layout_x=lx_j, layout_y=ly_j, wind_directions=[270.0], wind_speeds=[8.0],
            turbulence_intensities=[0.08], yaw_angles=np.array([[yaw0, 0.0, 0.0]]),
        )
        fm_j.run()
        u_fl = fm_j.turbine_average_velocities[0]
        p_fl = fm_j.get_turbine_powers()[0] / 1e3
        res = WP.solve_wake(
            lx_j, ly_j, H, turb, u_inf=8.0, wind_dir_deg=270.0, ti_inf=0.08, yaw_deg=[yaw0, 0.0, 0.0],
            params={"velocity_model": "jensen", "deflection_model": "jimenez", "turbulence_model": "none", "wind_shear": 0.12, "ref_height": H},
        )
        j_u_diffs.append(np.max(np.abs(res["u_rotor_ms"] - u_fl)))
        j_p_diffs.append(np.max(np.abs(res["power_kw"] - p_fl) / np.maximum(p_fl, 1.0)))

    max_u_jensen = float(np.max(j_u_diffs))
    max_p_jensen = float(np.max(j_p_diffs))
    checks.append({
        "name": "Jensen top-hat + Jimenez yaw deflection (-20..+20 deg) vs NREL FLORIS 4.7",
        "reference": "NREL FLORIS 4.7 (jensen + jimenez)",
        "max_velocity_abs_diff_ms": max_u_jensen,
        "max_power_rel_diff": max_p_jensen,
        "pass": max_u_jensen < 1e-9 and max_p_jensen < 1e-9,
    })

    # 3. Yaw-steering sweep (-25..+25 deg) with Gaussian deflection + CosineLossTurbine on a 3-turbine row vs FLORIS 4.7
    lx_y = [0.0, 6.0 * D, 12.0 * D]
    ly_y = [0.0, 0.20 * D, 0.40 * D]
    yaws = np.linspace(-25.0, 25.0, 21)
    p_fl_tot, p_wp_tot, yaw_p_diffs = [], [], []
    for y0 in yaws:
        yvec = [float(y0), float(0.5 * y0), 0.0]
        fm_g.set(
            layout_x=lx_y, layout_y=ly_y, wind_directions=[270.0], wind_speeds=[8.0],
            turbulence_intensities=[0.07], yaw_angles=np.array([yvec]),
        )
        fm_g.run()
        pfl = fm_g.get_turbine_powers()[0] / 1e3
        res = WP.solve_wake(
            lx_y, ly_y, H, turb, u_inf=8.0, wind_dir_deg=270.0, ti_inf=0.07, yaw_deg=yvec,
            params={"velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez", "wind_shear": 0.12, "ref_height": H},
        )
        p_fl_tot.append(float(np.sum(pfl)))
        p_wp_tot.append(res["farm_power_kw"])
        yaw_p_diffs.append(np.max(np.abs(res["power_kw"] - pfl) / np.maximum(pfl, 1.0)))

    max_yaw_p_diff = float(np.max(yaw_p_diffs))
    opt_yaw_fl = float(yaws[int(np.argmax(p_fl_tot))])
    opt_yaw_wp = float(yaws[int(np.argmax(p_wp_tot))])
    checks.append({
        "name": "3-turbine Gaussian yaw-steering sweep (-25..+25 deg) + Crespo-Hernandez TI + SOSFS vs NREL FLORIS 4.7",
        "reference": "NREL FLORIS 4.7 (gauss + gauss + crespo_hernandez + sosfs)",
        "max_power_rel_diff": max_yaw_p_diff,
        "optimal_yaw_deg_floris": opt_yaw_fl,
        "optimal_yaw_deg_wakenet": opt_yaw_wp,
        "steering_gain_pct": float(100.0 * (max(p_wp_tot) / p_wp_tot[len(yaws) // 2] - 1.0)),
        "pass": max_yaw_p_diff < 1e-9 and opt_yaw_fl == opt_yaw_wp,
    })

    # 4. Full 360-deg wind-rose sweep on real 14-turbine Penmanshiel layout vs NREL FLORIS 4.7
    pen_lats = np.array([55.902502, 55.900008, 55.905943, 55.903294, 55.900951, 55.898741, 55.907915,
                         55.904990, 55.903032, 55.900852, 55.908703, 55.907026, 55.905050, 55.902463])
    pen_lons = np.array([-2.306389, -2.301268, -2.302690, -2.298367, -2.293967, -2.289856, -2.297314,
                         -2.291806, -2.287585, -2.282371, -2.290986, -2.285887, -2.281650, -2.277329])
    pen_e, pen_n = _latlon_to_en(pen_lats, pen_lons)
    wds = np.arange(0.0, 360.0, 5.0)
    fm_g.set(
        layout_x=pen_e.tolist(), layout_y=pen_n.tolist(),
        wind_directions=wds.tolist(),
        wind_speeds=[9.0] * len(wds),
        turbulence_intensities=[0.085] * len(wds),
        yaw_angles=np.zeros((len(wds), len(pen_e))),
    )
    fm_g.run()
    pfl_all = fm_g.get_turbine_powers() / 1e3  # (72, 14)
    pwp_all = np.zeros_like(pfl_all)
    for k, wd in enumerate(wds):
        r = WP.solve_wake(
            pen_e, pen_n, H, turb, u_inf=9.0, wind_dir_deg=float(wd), ti_inf=0.085,
            params={"velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez", "wind_shear": 0.12, "ref_height": H},
        )
        pwp_all[k] = r["power_kw"]

    corr_360 = float(np.corrcoef(pfl_all.ravel(), pwp_all.ravel())[0, 1])
    max_rel_360 = float(np.max(np.abs(pwp_all - pfl_all) / np.maximum(pfl_all, 1.0)))
    checks.append({
        "name": "14-turbine Penmanshiel layout x 72 wind directions (0..355 deg) vs NREL FLORIS 4.7",
        "reference": "NREL FLORIS 4.7 (1,008 turbine-direction states)",
        "correlation": corr_360,
        "max_power_rel_diff": max_rel_360,
        "pass": max_rel_360 < 1e-8 and corr_360 > 0.999999,
    })

    # 5. Physical symmetries: exact 2-D rotational equivariance and offline-turbine removal equivalence
    phi = np.deg2rad(37.5)
    c, s = np.cos(phi), np.sin(phi)
    # Rotate farm coordinates clockwise by 37.5 deg and shift wind_dir clockwise by +37.5 deg
    e_rot = c * pen_e + s * pen_n
    n_rot = -s * pen_e + c * pen_n
    r_orig = WP.solve_wake(pen_e, pen_n, 59.0, WP.TURBINES["MM82"], u_inf=8.5, wind_dir_deg=215.0, ti_inf=0.09)
    r_rot = WP.solve_wake(e_rot, n_rot, 59.0, WP.TURBINES["MM82"], u_inf=8.5, wind_dir_deg=215.0 + 37.5, ti_inf=0.09)
    rot_err = float(np.max(np.abs(r_orig["power_kw"] - r_rot["power_kw"])))

    on_mask = np.ones(14, dtype=bool)
    on_mask[[0, 4]] = False
    r_masked = WP.solve_wake(pen_e, pen_n, 59.0, WP.TURBINES["MM82"], u_inf=8.5, wind_dir_deg=235.0, ti_inf=0.09, online=on_mask)
    r_sub = WP.solve_wake(pen_e[on_mask], pen_n[on_mask], 59.0, WP.TURBINES["MM82"], u_inf=8.5, wind_dir_deg=235.0, ti_inf=0.09)
    offline_err = float(np.max(np.abs(r_masked["power_kw"][on_mask] - r_sub["power_kw"])))

    checks.append({
        "name": "Rotational equivariance (+37.5 deg) and parked-turbine (online=False vs removed) exactness",
        "reference": "Exact analytical group & actuator-disk invariants",
        "rotation_max_kw_diff": rot_err,
        "offline_removal_max_kw_diff": offline_err,
        "pass": rot_err < 1e-11 and offline_err < 1e-11,
    })

    out = {
        "physics_version": "1.0",
        "all_passed": all(bool(x["pass"]) for x in checks),
        "checks": checks,
    }
    out_path = os.path.join(HERE, "physics_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    assert out["all_passed"], "One or more physics checks failed!"


if __name__ == "__main__":
    main()
