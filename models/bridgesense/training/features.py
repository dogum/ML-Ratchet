"""Deterministic physics-anchor feature extractor for BridgeSense.

Converts raw reference & query windows (T_ref [N,3], T_q [N,3], df_ref [N,M], df_q [N,M],
mask_ref [N,M], mask_q [N,M]) into:
  - x_ref  [N_ref, 10] : baseline temporal/thermal tokens
  - x_q    [N_q, 10]   : query temporal/thermal tokens
  - x_mode [M_max, 14] : per-mode physical & anchor tokens
  - m_mask [M_max]     : active mode mask (0/1)
  - g_feat [16]        : global physics-anchor & thermal extrapolation summary
  - anchor [3]         : (anchor_dk_pct, anchor_env_pct, mean_active_resid_pct)
  - mode_anchor [M_max]: per-mode linear-anchor implied dK/K (%) = 200 * mode_mean_resid

Pure numpy; mirrored identically in models/bridgesense/bridgesense.py and
site/src/instruments/bridgesense/net.ts.
"""
from __future__ import annotations

import numpy as np

from simulator import compute_physics_anchor


def _safe_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    ok = (mask > 0.5) & np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 4:
        return 0.0
    xv = x[ok] - np.mean(x[ok])
    yv = y[ok] - np.mean(y[ok])
    den = float(np.sqrt(np.dot(xv, xv) * np.dot(yv, yv)))
    if den < 1e-8:
        return 0.0
    return float(np.clip(np.dot(xv, yv) / den, -1.0, 1.0))


def extract_features(
    T_ref: np.ndarray,
    T_q: np.ndarray,
    df_ref: np.ndarray,
    df_q: np.ndarray,
    mask_ref: np.ndarray,
    mask_q: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute fixed-shape feature tensors and physics anchors for one episode."""
    T_ref = np.asarray(T_ref, dtype=np.float64)
    T_q = np.asarray(T_q, dtype=np.float64)
    df_ref = np.asarray(df_ref, dtype=np.float64)
    df_q = np.asarray(df_q, dtype=np.float64)
    mask_ref = np.asarray(mask_ref, dtype=np.float64)
    mask_q = np.asarray(mask_q, dtype=np.float64)

    N_ref, M = df_ref.shape
    N_q = df_q.shape[0]
    anchor = compute_physics_anchor(T_ref[:, 0], df_ref, mask_ref, T_q[:, 0], df_q, mask_q)

    resid_ref = anchor["resid_ref"]
    resid_q = anchor["resid_q"]
    slopes = anchor["slopes"]
    ref_std = anchor["ref_resid_std"]
    mode_mean_resid = anchor["mode_mean_resid"]
    mode_mean_raw = anchor["mode_mean_raw"]

    # Active modes: at least 3 valid points in ref and 2 in query
    active = (np.sum(mask_ref > 0.5, axis=0) >= 3) & (np.sum(mask_q > 0.5, axis=0) >= 2)
    m_mask = active.astype(np.float32)
    n_act = max(int(np.sum(active)), 1)

    def _time_tokens(T3: np.ndarray, df: np.ndarray, resid: np.ndarray, msk: np.ndarray) -> np.ndarray:
        N = len(T3)
        out = np.zeros((N, 10), dtype=np.float32)
        act_msk = msk * active[None, :]
        cnt = np.sum(act_msk > 0.5, axis=1)
        has = cnt > 0
        denom = np.maximum(cnt, 1.0)
        mean_df = np.sum(df * act_msk, axis=1) / denom
        mean_res = np.sum(resid * act_msk, axis=1) / denom
        # Cross-mode std at each timestamp
        sq = np.sum(((df - mean_df[:, None]) ** 2) * act_msk, axis=1) / denom
        std_df = np.sqrt(np.maximum(sq, 0.0))

        t0 = T3[:, 0]
        t24 = T3[:, 1]
        t72 = T3[:, 2]
        out[:, 0] = (t0 - 10.0) / 15.0
        out[:, 1] = (t24 - 10.0) / 15.0
        out[:, 2] = (t72 - 10.0) / 15.0
        out[:, 3] = np.clip((t0 - t24) / 10.0, -3.0, 3.0)
        out[:, 4] = np.maximum(0.0, 2.0 - t0) / 10.0
        out[:, 5] = np.maximum(0.0, 2.0 - t24) / 10.0
        out[:, 6] = np.clip(np.where(has, mean_df * 20.0, 0.0), -5.0, 5.0)
        out[:, 7] = np.clip(np.where(has, std_df * 20.0, 0.0), 0.0, 5.0)
        out[:, 8] = np.clip(np.where(has, mean_res * 20.0, 0.0), -5.0, 5.0)
        out[:, 9] = (cnt / float(n_act)).astype(np.float32)
        return out

    x_ref = _time_tokens(T_ref, df_ref, resid_ref, mask_ref)
    x_q = _time_tokens(T_q, df_q, resid_q, mask_q)

    # Per-mode tokens [M, 14]
    x_mode = np.zeros((M, 14), dtype=np.float32)
    bridge_mean_resid = float(np.mean(mode_mean_resid[active])) if np.any(active) else 0.0
    bridge_mean_raw = float(np.mean(mode_mean_raw[active])) if np.any(active) else 0.0

    cold_q_mask = T_q[:, 0] < 1.0
    warm_q_mask = ~cold_q_mask
    q_cold_corrs = []

    for m in range(M):
        if not active[m]:
            continue
        ok_q = mask_q[:, m] > 0.5
        ok_r = mask_ref[:, m] > 0.5
        q_std = float(np.std(resid_q[ok_q, m])) if np.sum(ok_q) >= 2 else 0.0
        ok_qc = ok_q & cold_q_mask
        ok_qw = ok_q & warm_q_mask
        cold_res = float(np.mean(resid_q[ok_qc, m])) if np.sum(ok_qc) >= 1 else 0.0
        warm_res = float(np.mean(resid_q[ok_qw, m])) if np.sum(ok_qw) >= 1 else mode_mean_resid[m]
        c_ref = _safe_corr(T_ref[:, 0], df_ref[:, m], mask_ref[:, m])
        c_qres = _safe_corr(T_q[:, 0], resid_q[:, m], mask_q[:, m])
        q_cold_corrs.append(c_qres)

        z_m = float(np.clip(mode_mean_resid[m] / ref_std[m], -30.0, 30.0))
        x_mode[m, 0] = float(np.clip(slopes[m] * 200.0, -4.0, 4.0))
        x_mode[m, 1] = float(np.clip(ref_std[m] * 100.0, 0.0, 5.0))
        x_mode[m, 2] = float(np.clip(mode_mean_raw[m] * 20.0, -5.0, 5.0))
        x_mode[m, 3] = float(np.clip(mode_mean_resid[m] * 20.0, -5.0, 5.0))
        x_mode[m, 4] = z_m / 5.0
        x_mode[m, 5] = float(np.clip(q_std * 100.0, 0.0, 5.0))
        x_mode[m, 6] = float(np.clip(cold_res * 20.0, -5.0, 5.0))
        x_mode[m, 7] = float(np.clip(warm_res * 20.0, -5.0, 5.0))
        x_mode[m, 8] = float(np.mean(ok_r))
        x_mode[m, 9] = float(np.mean(ok_q))
        x_mode[m, 10] = c_ref
        x_mode[m, 11] = c_qres
        x_mode[m, 12] = float(np.clip((mode_mean_resid[m] - bridge_mean_resid) * 20.0, -5.0, 5.0))
        x_mode[m, 13] = float(np.clip((mode_mean_raw[m] - bridge_mean_raw) * 20.0, -5.0, 5.0))

    # Global physics-anchor summary [16]
    g_feat = np.array([
        np.clip(anchor["anchor_dk_pct"] / 10.0, -5.0, 5.0),
        np.clip(anchor["anchor_env_pct"] / 5.0, -5.0, 5.0),
        np.clip(anchor["max_abs_z"] / 5.0, 0.0, 6.0),
        np.clip(anchor["mean_abs_z"] / 5.0, 0.0, 6.0),
        np.clip(anchor["resid_std_modes"] * 50.0, 0.0, 5.0),
        (anchor["T_ref_mean"] - 10.0) / 15.0,
        (anchor["T_ref_min"] - 10.0) / 15.0,
        (anchor["T_ref_max"] - 10.0) / 15.0,
        (anchor["T_q_mean"] - 10.0) / 15.0,
        (anchor["T_q_min"] - 10.0) / 15.0,
        np.clip(anchor["cold_extrap"] / 10.0, 0.0, 4.0),
        np.clip(anchor["warm_extrap"] / 10.0, 0.0, 4.0),
        anchor["freeze_frac_ref"],
        anchor["freeze_frac_q"],
        np.log(float(n_act)) / 3.0,
        float(np.mean(q_cold_corrs)) if q_cold_corrs else 0.0,
    ], dtype=np.float32)

    anchor_vec = np.array([
        anchor["anchor_dk_pct"],
        anchor["anchor_env_pct"],
        100.0 * bridge_mean_resid,
    ], dtype=np.float32)
    mode_anchor = (200.0 * mode_mean_resid).astype(np.float32)

    return {
        "x_ref": x_ref,
        "x_q": x_q,
        "x_mode": x_mode,
        "m_mask": m_mask,
        "g_feat": g_feat,
        "anchor": anchor_vec,
        "mode_anchor": mode_anchor,
    }
