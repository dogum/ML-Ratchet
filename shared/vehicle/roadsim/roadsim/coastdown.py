"""Coast-down test emulation (SAE J2263 / J1263 style) using roadsim's road-load model.

The vehicle coasts in neutral from 80 to 15 mph on a flat track, in both directions, with a steady wind of random
speed/direction, speed sampled at 10 Hz with noise. Road-load force is recovered from the deceleration of the averaged
opposite-direction runs, F = m_eff * (-dv/dt), and fitted with F = A + B v + C v^2 (least squares), as in J2263.
"""
import math
import numpy as np
from .constants import G, MPH, LBF


def road_force(veh, v, rho, w_head, w_cross, T_amb=20.0):
    """road-load force (N) on flat ground for speed v (m/s) -- same formulas as sim.py."""
    rv = v + w_head
    if w_cross > 0:
        psi = math.degrees(math.atan2(w_cross, max(rv, 0.5)))
        cda = veh.CdA * (1 + veh.yaw_k * min(psi, 30.0) ** 2)
    else:
        cda = veh.CdA
    Kt = veh.tire_K_t
    fT = (1 + Kt * (20 - 25)) / max(1 + Kt * (T_amb - 25), 0.4) if veh.tire_temp_effects else 1.0
    return 0.5 * rho * cda * rv * abs(rv) + veh.mass * G * veh.Crr * fT * (1 + veh.Crr_v * v) + veh.parasitic_N


def coast(veh, v0=80 * MPH, v_end=15 * MPH, rho=1.1767, wind=0.0, wind_dir_rel=0.0, dt=0.1, T_amb=20.0):
    """integrate one coast-down run; returns (t, v). wind_dir_rel: wind FROM direction relative to heading (deg)."""
    wh = wind * math.cos(math.radians(wind_dir_rel)); wc = abs(wind * math.sin(math.radians(wind_dir_rel)))
    t = [0.0]; v = [v0]
    while v[-1] > v_end and len(v) < 100000:
        # RK2 on m_eff dv/dt = -F(v)
        a1 = -road_force(veh, v[-1], rho, wh, wc, T_amb) / veh.m_eff
        vm = v[-1] + 0.5 * dt * a1
        a2 = -road_force(veh, vm, rho, wh, wc, T_amb) / veh.m_eff
        v.append(v[-1] + dt * a2); t.append(t[-1] + dt)
    return np.array(t), np.array(v)


def fit_abc(veh, runs, noise_mph=0.05, rng=None, bins=np.arange(20, 76, 5) * MPH):
    """J2263-like analysis: per run, deceleration in speed bins from noisy 10 Hz data (local quadratic fit);
    average the opposite-direction pair; least-squares A + B v + C v^2. Returns A (N), B (N/(m/s)), C (N/(m/s)^2)."""
    rng = rng or np.random.default_rng(0)
    F_all = []
    for t, v in runs:
        vn = v + rng.normal(0, noise_mph * MPH, len(v))
        Fb = []
        for vb in bins:
            i = int(np.argmin(np.abs(v - vb)))
            sl = slice(max(i - 25, 0), min(i + 26, len(v)))
            p = np.polyfit(t[sl] - t[i], vn[sl], 2)
            Fb.append(-veh.m_eff * p[1])
        F_all.append(Fb)
    F = np.mean(np.array(F_all), axis=0)
    X = np.stack([np.ones_like(bins), bins, bins ** 2], 1)
    A, B, C = np.linalg.lstsq(X, F, rcond=None)[0]
    return A, B, C


def to_lbf(A, B, C):
    return A / LBF, B / LBF * MPH, C / LBF * MPH ** 2
