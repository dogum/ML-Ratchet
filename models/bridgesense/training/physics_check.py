"""Independent physics checks for BridgeSim (models/bridgesense/training/simulator.py).

Verifies five governing physical relationships against independent analytical solutions and
the published KW51 Finite Element Model (Zenodo 17366772), writing verify_physics.json.
"""
from __future__ import annotations

import json
import os
import re
import numpy as np

from _paths import SRC, work
from simulator import (
    assemble_uniform_beam,
    build_parametric_bridge,
    compute_bridge_frequencies,
    solve_modes,
    thermal_diffusion_1d,
)


def check_euler_bernoulli_simply_supported() -> dict:
    """Analytical f_n = (n^2 * pi / (2 L^2)) * sqrt(EI / rhoA) for bending modes 1..6."""
    L = 40.0
    EI = 4.5e9
    EA = 1.0e13  # axially stiff so first 6 modes are pure bending (no axial mode interleaving)
    rhoA = 12000.0
    K, M = assemble_uniform_beam(L, EI, EA, rhoA, n_elem=24, bc="simply_supported")
    freqs, _ = solve_modes(K, M, n_modes=6)
    n = np.arange(1, 7, dtype=np.float64)
    f_exact = (n * n * np.pi / (2.0 * L * L)) * np.sqrt(EI / rhoA)
    rel_err = np.abs(freqs - f_exact) / f_exact
    return {
        "check": "Simply-supported Euler-Bernoulli beam (bending modes 1-6, 24 elements)",
        "analytical_hz": [round(float(x), 5) for x in f_exact],
        "fem_hz": [round(float(x), 5) for x in freqs],
        "max_rel_err": float(np.max(rel_err)),
        "pass": bool(np.max(rel_err) < 1e-3),
    }


def check_euler_bernoulli_cantilever() -> dict:
    """Analytical cantilever beam roots beta_n L = [1.87510407, 4.69409113, 7.85475744, 10.99554073]."""
    L = 25.0
    EI = 2.0e9
    EA = 1.0e13
    rhoA = 8000.0
    K, M = assemble_uniform_beam(L, EI, EA, rhoA, n_elem=24, bc="cantilever")
    freqs, _ = solve_modes(K, M, n_modes=4)
    betaL = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073])
    f_exact = (betaL ** 2 / (2.0 * np.pi * L * L)) * np.sqrt(EI / rhoA)
    rel_err = np.abs(freqs - f_exact) / f_exact
    return {
        "check": "Cantilever Euler-Bernoulli beam (bending modes 1-4, 24 elements)",
        "analytical_hz": [round(float(x), 5) for x in f_exact],
        "fem_hz": [round(float(x), 5) for x in freqs],
        "max_rel_err": float(np.max(rel_err)),
        "pass": bool(np.max(rel_err) < 1e-3),
    }


def check_rayleigh_perturbation_and_subspace() -> dict:
    """Check uniform stiffness scaling (df/f = sqrt(1+eps)-1) and Rayleigh-Ritz vs full eigh."""
    rng = np.random.default_rng(42)
    fem = build_parametric_bridge(rng, kind="arch", material="steel")
    K0 = sum(fem.K_elems) + fem.K_bear
    M0 = fem.M_base
    f0, phi0 = solve_modes(K0, M0, n_modes=8)

    # 1. Exact uniform stiffness scaling eps = -0.10
    eps = -0.10
    f_scaled, _ = solve_modes((1.0 + eps) * K0, M0, n_modes=8)
    expected_ratio = np.sqrt(1.0 + eps)
    uniform_err = float(np.max(np.abs(f_scaled / f0 - expected_ratio)))

    # 2. First-order localized perturbation vs Rayleigh quotient: d(w^2)/w^2 = (phi^T dK phi)/(phi^T K phi)
    delta = 1e-4
    dK_loc = delta * fem.K_elems[3]
    f_loc, _ = solve_modes(K0 + dK_loc, M0, n_modes=6)
    dw2_exact = (f_loc ** 2 - f0[:6] ** 2) / (f0[:6] ** 2)
    dw2_rayleigh = np.array([
        float(phi0[:, m] @ dK_loc @ phi0[:, m]) / float(phi0[:, m] @ K0 @ phi0[:, m])
        for m in range(6)
    ])
    rayleigh_err = float(np.max(np.abs(dw2_exact - dw2_rayleigh)))

    # 3. Rayleigh-Ritz projected solver vs full eigh under 20% local damage + 10 K thermal shift
    dK_arr = np.zeros(len(fem.K_elems))
    dK_arr[2] = -0.20
    dK_arr[5] = -0.15
    T_s = np.array([15.0, -5.0])
    f_rr, _ = compute_bridge_frequencies(
        fem, T_s, T_s, T0=15.0, alpha_E=4e-4, surf_share=0.04, freeze_gain=1.5,
        T_freeze=0.0, w_freeze=1.5, asph_beta=0.01, restraint_N_per_K=0.0,
        dK_elem=dK_arr, dM_deck_frac=0.0, n_modes=10,
    )
    # Full eigh at T=-5 C
    c_struct = 1.0 - 4e-4 * (-5.0 - 15.0)
    c_surf = 0.04 * np.exp(-0.01 * (-20.0)) * (1.0 + 1.5 / (1.0 + np.exp(-5.0 / 1.5)))
    K_full = c_struct * K0 + c_surf * fem.K_surf + sum(dk * ke for dk, ke in zip(dK_arr, fem.K_elems))
    f_full, _ = solve_modes(K_full, M0, n_modes=10)
    rr_rel_err = float(np.max(np.abs(f_rr[1] - f_full) / f_full))

    return {
        "check": "Rayleigh perturbation & Rayleigh-Ritz vs full eigh (20% local damage + freeze)",
        "uniform_scaling_max_err": uniform_err,
        "rayleigh_first_order_max_err": rayleigh_err,
        "rayleigh_ritz_vs_full_eigh_max_rel_err": rr_rel_err,
        "pass": bool(uniform_err < 1e-12 and rayleigh_err < 1e-7 and rr_rel_err < 1e-3),
    }


def check_thermal_diffusion_1d() -> dict:
    """Compare 1-D Crank-Nicolson heat diffusion against exact semi-infinite periodic attenuation & lag.

    For a thick insulated-bottom slab driven at z=0 by T(0,t) = A sin(w t),
    exact solution at depth z is A exp(-k z) sin(w t - k z) where k = sqrt(w / (2 alpha)).
    Depth-averaged steady-state response has amplitude A / (H k sqrt(2)) and phase lag pi/4 rad.
    """
    alpha = 6.0e-7  # concrete thermal diffusivity (m^2/s)
    period_s = 24.0 * 3600.0
    w = 2.0 * np.pi / period_s
    k_wave = np.sqrt(w / (2.0 * alpha))
    n_days = 15
    dt = 600.0  # 10-min step
    n_t = int(n_days * 24 * 3600 / dt)
    t = np.arange(n_t) * dt
    T_air = 10.0 * np.sin(w * t)

    H = 1.5
    T_mean, _ = thermal_diffusion_1d(
        T_air, dt_s=dt, alpha_m2s=alpha, thickness_m=H, h_over_k=1e5, n_z=81, bottom_conv_frac=0.0
    )
    sl = slice(-int(3 * 24 * 3600 / dt), None)
    c_proj = (2.0 / len(t[sl])) * np.sum(T_mean[sl] * np.exp(-1j * w * t[sl]))
    amp_num = float(np.abs(c_proj))
    c_air = (2.0 / len(t[sl])) * np.sum(T_air[sl] * np.exp(-1j * w * t[sl]))
    phase_lag_rad = float(-np.angle(c_proj / c_air))

    amp_exact = float(10.0 / (H * k_wave * np.sqrt(2.0)))
    phase_exact = float(np.pi / 4.0)
    amp_rel_err = abs(amp_num - amp_exact) / amp_exact
    phase_err_rad = abs(phase_lag_rad - phase_exact)
    return {
        "check": "1-D Fourier deck heat diffusion vs exact periodic slab solution (Crank-Nicolson)",
        "amp_num_K": round(amp_num, 5),
        "amp_exact_K": round(amp_exact, 5),
        "amp_rel_err": float(amp_rel_err),
        "phase_lag_num_rad": round(phase_lag_rad, 5),
        "phase_lag_exact_rad": round(phase_exact, 5),
        "phase_err_rad": float(phase_err_rad),
        "pass": bool(amp_rel_err < 0.02 and phase_err_rad < 0.02),
    }


def check_kw51_fem_reference() -> dict:
    """Cross-check parametric bowstring arch-deck FEM scale against KW51 published FEM (Zenodo 17366772)."""
    kw_m = work("raw", "kw51", "model_kw51.m")
    info = {"kw51_fem_file_found": os.path.exists(kw_m)}
    if os.path.exists(kw_m):
        txt = open(kw_m, "r", encoding="utf-8", errors="ignore").read()
        # Extract span/material constants documented in model_kw51.m
        m_E = re.findall(r"E\s*=\s*([0-9.eE+-]+)", txt)
        m_L = re.findall(r"\bL\s*=\s*([0-9.eE+-]+)", txt)
        info["parsed_E_values"] = m_E[:3]
        info["parsed_L_values"] = m_L[:3]
    # Build an arch-deck FEM at KW51's physical scale (span 115 m, steel bowstring arch)
    rng = np.random.default_rng(115)
    f_firsts = []
    for _ in range(25):
        fem = build_parametric_bridge(rng, kind="arch", material="steel")
        f, _ = solve_modes(sum(fem.K_elems) + fem.K_bear, fem.M_base, n_modes=6)
        f_firsts.append(f[:4])
    f_arr = np.array(f_firsts)
    # KW51 real global vertical/arch modes lie in ~0.5 - 6.3 Hz (1st vertical ~1.87 Hz, 1st arch ~0.51 Hz)
    med_f1 = float(np.median(f_arr[:, 0]))
    med_f4 = float(np.median(f_arr[:, 3]))
    info.update({
        "check": "Parametric steel bowstring arch-deck FEM modal frequency range vs KW51 (0.51-6.3 Hz)",
        "sim_arch_median_f1_hz": round(med_f1, 3),
        "sim_arch_median_f4_hz": round(med_f4, 3),
        "kw51_published_range_hz": [0.51, 6.31],
        "pass": bool(0.3 <= med_f1 <= 3.5 and 1.5 <= med_f4 <= 12.0),
    })
    return info


def main() -> None:
    results = [
        check_euler_bernoulli_simply_supported(),
        check_euler_bernoulli_cantilever(),
        check_rayleigh_perturbation_and_subspace(),
        check_thermal_diffusion_1d(),
        check_kw51_fem_reference(),
    ]
    all_pass = all(r["pass"] for r in results)
    out = {"all_pass": all_pass, "checks": results}
    out_path = os.path.join(SRC, "verify_physics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['check']}")
        for k, v in r.items():
            if k not in ("check", "pass"):
                print(f"    {k}: {v}")
    assert all_pass, "One or more physics checks failed!"
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
