"""BridgeSense — numpy-only inference for physics-first bridge structural health monitoring.

Reads a bridge's tracked modal frequencies and air temperatures over a healthy reference
window and a query monitoring window, and estimates:
  - p_change: probability of a real structural stiffness/mass change (vs thermal/freezing only)
  - delta_k_pct ± delta_k_sigma: effective global stiffness shift dK/K (%)
  - thermal_shift_pct ± thermal_shift_sigma: environmental frequency shift (%)
  - mode_dk_pct: per-mode implied stiffness shift (%) for active modes

Pure numpy only (no torch, no mlratchet). Loads weights relative to this file.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

WEIGHTS_PATH = Path(__file__).resolve().parent / "bridgesense_weights.npz"
MAX_MODES = 14


def _erf_as(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 erf — matches site/src/lib/nn/ops.ts to ~1e-7."""
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return np.where(x < 0.0, -y, y)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf_as(x * 0.7071067811865476))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * w + b


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


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


def compute_anchor_and_features(
    T_ref: np.ndarray,
    T_q: np.ndarray,
    df_ref: np.ndarray,
    df_q: np.ndarray,
    mask_ref: np.ndarray,
    mask_q: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build deterministic physics-anchor features identical to training/features.py."""
    T_ref = np.asarray(T_ref, dtype=np.float64)
    T_q = np.asarray(T_q, dtype=np.float64)
    df_ref = np.asarray(df_ref, dtype=np.float64)
    df_q = np.asarray(df_q, dtype=np.float64)
    mask_ref = np.asarray(mask_ref, dtype=np.float64)
    mask_q = np.asarray(mask_q, dtype=np.float64)

    N_ref, M = df_ref.shape
    N_q = df_q.shape[0]
    if M < MAX_MODES:
        pad = MAX_MODES - M
        df_ref = np.pad(df_ref, ((0, 0), (0, pad)))
        df_q = np.pad(df_q, ((0, 0), (0, pad)))
        mask_ref = np.pad(mask_ref, ((0, 0), (0, pad)))
        mask_q = np.pad(mask_q, ((0, 0), (0, pad)))
        M = MAX_MODES

    T_r0 = T_ref[:, 0]
    T_q0 = T_q[:, 0]
    slopes = np.zeros(M, dtype=np.float64)
    intercepts = np.zeros(M, dtype=np.float64)
    ref_std = np.full(M, 0.005, dtype=np.float64)
    T_ref_mean = float(np.nanmean(T_r0))
    T_ref_min = float(np.nanmin(T_r0))
    T_ref_max = float(np.nanmax(T_r0))

    for m in range(M):
        ok = (mask_ref[:, m] > 0.5) & np.isfinite(df_ref[:, m]) & np.isfinite(T_r0)
        if np.sum(ok) >= 4:
            # OLS with intercept, centred on this mode's valid readings
            x = T_r0[ok] - T_ref_mean
            y = df_ref[ok, m]
            xm, ym = float(np.mean(x)), float(np.mean(y))
            xc, yc = x - xm, y - ym
            vx = float(np.dot(xc, xc))
            a = float(np.clip(np.dot(xc, yc) / (vx + 1e-2), -0.015, 0.005)) if vx > 1e-4 else 0.0
            b = float(ym - a * xm)
            slopes[m], intercepts[m] = a, b
            ref_std[m] = float(max(np.std(y - (a * x + b)), 5e-4))
        elif np.sum(ok) >= 1:
            intercepts[m] = float(np.mean(df_ref[ok, m]))

    pred_ref = intercepts[None, :] + slopes[None, :] * (T_r0[:, None] - T_ref_mean)
    pred_q = intercepts[None, :] + slopes[None, :] * (T_q0[:, None] - T_ref_mean)
    resid_ref = np.where(mask_ref > 0.5, df_ref - pred_ref, 0.0)
    resid_q = np.where(mask_q > 0.5, df_q - pred_q, 0.0)

    q_counts = np.sum(mask_q > 0.5, axis=0)
    mode_mean_resid = np.zeros(M, dtype=np.float64)
    mode_mean_raw = np.zeros(M, dtype=np.float64)
    for m in range(M):
        if q_counts[m] > 0:
            ok = mask_q[:, m] > 0.5
            mode_mean_resid[m] = float(np.mean(resid_q[ok, m]))
            mode_mean_raw[m] = float(np.mean(df_q[ok, m]))

    # Same eligibility as the mode mask: >= 3 valid reference and >= 2 valid query readings
    active = (np.sum(mask_ref > 0.5, axis=0) >= 3) & (q_counts >= 2)
    if np.any(active):
        mean_resid = float(np.mean(mode_mean_resid[active]))
        anchor_dk_pct = 200.0 * mean_resid
        anchor_env_pct = 100.0 * float(np.mean(mode_mean_raw[active]) - mean_resid)
        mode_z = mode_mean_resid / ref_std
        max_abs_z = float(np.max(np.abs(mode_z[active])))
        mean_abs_z = float(np.mean(np.abs(mode_z[active])))
        resid_std_modes = float(np.std(mode_mean_resid[active])) if np.sum(active) >= 2 else 0.0
    else:
        anchor_dk_pct = anchor_env_pct = max_abs_z = mean_abs_z = resid_std_modes = 0.0

    T_q_min = float(np.nanmin(T_q0))
    T_q_mean = float(np.nanmean(T_q0))
    cold_extrap = max(0.0, T_ref_min - T_q_min)
    warm_extrap = max(0.0, float(np.nanmax(T_q0)) - T_ref_max)
    freeze_frac_q = float(np.mean(T_q0 < 1.0))
    freeze_frac_ref = float(np.mean(T_r0 < 1.0))

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
        sq = np.sum(((df - mean_df[:, None]) ** 2) * act_msk, axis=1) / denom
        std_df = np.sqrt(np.maximum(sq, 0.0))
        t0, t24, t72 = T3[:, 0], T3[:, 1], T3[:, 2]
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

    x_mode = np.zeros((M, 14), dtype=np.float32)
    bridge_mean_resid = float(np.mean(mode_mean_resid[active])) if np.any(active) else 0.0
    bridge_mean_raw = float(np.mean(mode_mean_raw[active])) if np.any(active) else 0.0
    cold_q_mask = T_q0 < 1.0
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
        c_ref = _safe_corr(T_r0, df_ref[:, m], mask_ref[:, m])
        c_qres = _safe_corr(T_q0, resid_q[:, m], mask_q[:, m])
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

    g_feat = np.array([
        np.clip(anchor_dk_pct / 10.0, -5.0, 5.0),
        np.clip(anchor_env_pct / 5.0, -5.0, 5.0),
        np.clip(max_abs_z / 5.0, 0.0, 6.0),
        np.clip(mean_abs_z / 5.0, 0.0, 6.0),
        np.clip(resid_std_modes * 50.0, 0.0, 5.0),
        (T_ref_mean - 10.0) / 15.0,
        (T_ref_min - 10.0) / 15.0,
        (T_ref_max - 10.0) / 15.0,
        (T_q_mean - 10.0) / 15.0,
        (T_q_min - 10.0) / 15.0,
        np.clip(cold_extrap / 10.0, 0.0, 4.0),
        np.clip(warm_extrap / 10.0, 0.0, 4.0),
        freeze_frac_ref,
        freeze_frac_q,
        np.log(float(n_act)) / 3.0,
        float(np.mean(q_cold_corrs)) if q_cold_corrs else 0.0,
    ], dtype=np.float32)

    return {
        "x_ref": x_ref,
        "x_q": x_q,
        "x_mode": x_mode,
        "m_mask": m_mask,
        "g_feat": g_feat,
        "anchor": np.array([anchor_dk_pct, anchor_env_pct, 100.0 * bridge_mean_resid], dtype=np.float32),
        "mode_anchor": (200.0 * mode_mean_resid).astype(np.float32),
    }


def _mha(q_in: np.ndarray, kv_in: np.ndarray, W: dict, prefix: str,
         n_heads: int, kv_mask: np.ndarray | None = None) -> np.ndarray:
    Lq, D = q_in.shape
    Lk = kv_in.shape[0]
    dh = D // n_heads
    Q = _linear(q_in, W[f"{prefix}.q_proj.weight"], W[f"{prefix}.q_proj.bias"]).reshape(Lq, n_heads, dh).transpose(1, 0, 2)
    K = _linear(kv_in, W[f"{prefix}.k_proj.weight"], W[f"{prefix}.k_proj.bias"]).reshape(Lk, n_heads, dh).transpose(1, 0, 2)
    V = _linear(kv_in, W[f"{prefix}.v_proj.weight"], W[f"{prefix}.v_proj.bias"]).reshape(Lk, n_heads, dh).transpose(1, 0, 2)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(float(dh))
    if kv_mask is not None:
        scores = np.where(kv_mask[None, None, :] < 0.5, -1e9, scores)
    attn = _softmax(scores, axis=-1)
    out = np.matmul(attn, V).transpose(1, 0, 2).reshape(Lq, D)
    return _linear(out, W[f"{prefix}.o_proj.weight"], W[f"{prefix}.o_proj.bias"])


class BridgeSense:
    """Physics-first Transformer for separating bridge structural stiffness changes from weather."""

    def __init__(self, weights_path: str | Path | None = None):
        p = Path(weights_path) if weights_path else WEIGHTS_PATH
        raw = np.load(p, allow_pickle=False)
        self.meta = json.loads(str(raw["meta"].item()))
        self.n_members = int(self.meta.get("n_members", 1))
        # One config per ensemble member (v1.1). v1.0 files carry a single shared "config".
        cfgs = self.meta.get("configs") or [self.meta["config"]] * self.n_members
        if len(cfgs) != self.n_members:
            raise ValueError(f"weights carry {len(cfgs)} configs for {self.n_members} members")
        self.cfgs: list[dict] = list(cfgs)
        self.widen_dk = float(self.meta.get("widen_dk", 1.0))
        self.widen_env = float(self.meta.get("widen_env", 1.0))
        self.prob_temp = float(self.meta.get("prob_temp", 1.0))

        self.members: list[dict[str, np.ndarray]] = []
        for k in range(self.n_members):
            pref = f"m{k}."
            W = {name[len(pref):]: raw[name].astype(np.float32) for name in raw.files if name.startswith(pref)}
            self.members.append(W)

    def _forward_one(self, W: dict[str, np.ndarray], cfg: dict, feat: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
        n_heads = int(cfg["n_heads"])
        n_t = int(cfg["n_t_layers"])
        n_m = int(cfg["n_m_layers"])
        use_cross = bool(cfg.get("use_cross_attn", True))
        use_anchor = bool(cfg.get("use_anchor", True))

        x_ref = feat["x_ref"].astype(np.float32)
        x_q = feat["x_q"].astype(np.float32)
        x_mode = feat["x_mode"].astype(np.float32)
        m_mask = feat["m_mask"].astype(np.float32)
        g_feat = feat["g_feat"].astype(np.float32)
        anchor = feat["anchor"].astype(np.float32)
        mode_anchor = feat["mode_anchor"].astype(np.float32)

        # 1. Temporal cross-attention
        h_ref = _gelu(_linear(x_ref, W["t_in.weight"], W["t_in.bias"]))
        h_q = _gelu(_linear(x_q, W["t_in.weight"], W["t_in.bias"])) + W["t_seg_q"].reshape(1, -1)
        for i in range(n_t):
            p = f"t_blocks.{i}"
            qn = _layernorm(h_q, W[f"{p}.ln_q.weight"], W[f"{p}.ln_q.bias"])
            kvn = _layernorm(h_ref if use_cross else h_q, W[f"{p}.ln_kv.weight"], W[f"{p}.ln_kv.bias"])
            h_q = h_q + _mha(qn, kvn, W, f"{p}.attn", n_heads)
            fn = _layernorm(h_q, W[f"{p}.ln_ff.weight"], W[f"{p}.ln_ff.bias"])
            ff = _linear(_gelu(_linear(fn, W[f"{p}.ff1.weight"], W[f"{p}.ff1.bias"])), W[f"{p}.ff2.weight"], W[f"{p}.ff2.bias"])
            h_q = h_q + ff
        h_q = _layernorm(h_q, W["t_ln.weight"], W["t_ln.bias"])
        w_t = _softmax(_linear(h_q, W["t_pool.weight"], W["t_pool.bias"]), axis=0)
        z_t = np.sum(w_t * h_q, axis=0)

        # 2. Mode-set masked self-attention
        h_m = _gelu(_linear(x_mode, W["m_in.weight"], W["m_in.bias"])) * m_mask[:, None]
        for i in range(n_m):
            p = f"m_blocks.{i}"
            xn = _layernorm(h_m, W[f"{p}.ln1.weight"], W[f"{p}.ln1.bias"])
            h_m = h_m + _mha(xn, xn, W, f"{p}.attn", n_heads, kv_mask=m_mask) * m_mask[:, None]
            fn = _layernorm(h_m, W[f"{p}.ln_ff.weight"], W[f"{p}.ln_ff.bias"])
            ff = _linear(_gelu(_linear(fn, W[f"{p}.ff1.weight"], W[f"{p}.ff1.bias"])), W[f"{p}.ff2.weight"], W[f"{p}.ff2.bias"])
            h_m = h_m + ff * m_mask[:, None]
        h_m = _layernorm(h_m, W["m_ln.weight"], W["m_ln.bias"]) * m_mask[:, None]

        scores_m = _linear(h_m, W["m_pool.weight"], W["m_pool.bias"]).reshape(-1)
        scores_m = np.where(m_mask < 0.5, -1e9, scores_m)
        w_m = _softmax(scores_m, axis=0)
        z_m_attn = np.sum(w_m[:, None] * h_m, axis=0)
        z_m_mean = np.sum(h_m, axis=0) / max(float(np.sum(m_mask)), 1.0)

        mode_corr = _linear(h_m, W["mode_head.weight"], W["mode_head.bias"]).reshape(-1)
        mode_dk = ((mode_anchor + mode_corr) if use_anchor else mode_corr) * m_mask

        # 3. Fusion MLP
        z_g = _gelu(_linear(g_feat, W["g_in.weight"], W["g_in.bias"]))
        z = np.concatenate([z_t, z_m_attn, z_m_mean, z_g], axis=-1)
        z = _gelu(_layernorm(_linear(z, W["fuse1.weight"], W["fuse1.bias"]), W["fuse_ln.weight"], W["fuse_ln.bias"]))
        z = _gelu(_linear(z, W["fuse2.weight"], W["fuse2.bias"]))
        out = _linear(z, W["head.weight"], W["head.bias"])

        logit = float(out[0])
        dk_mu = float((anchor[0] + out[1]) if use_anchor else out[1])
        dk_sig = float(np.exp(np.clip(out[2], -2.5, 3.5)))
        env_mu = float((anchor[1] + out[3]) if use_anchor else out[3])
        env_sig = float(np.exp(np.clip(out[4], -3.0, 3.0)))
        return {
            "logit": logit, "dk_mu": dk_mu, "dk_sig": dk_sig,
            "env_mu": env_mu, "env_sig": env_sig, "mode_dk": mode_dk,
        }

    def predict_features(self, feat: dict[str, np.ndarray], raw_bars: bool = False) -> dict:
        outs = [self._forward_one(W, cfg, feat) for W, cfg in zip(self.members, self.cfgs)]
        logit = float(np.mean([o["logit"] for o in outs]))
        dks = np.array([o["dk_mu"] for o in outs], dtype=np.float64)
        dk_vars = np.array([o["dk_sig"] ** 2 for o in outs], dtype=np.float64)
        envs = np.array([o["env_mu"] for o in outs], dtype=np.float64)
        env_vars = np.array([o["env_sig"] ** 2 for o in outs], dtype=np.float64)
        mode_dk = np.mean([o["mode_dk"] for o in outs], axis=0)

        dk_mu = float(np.mean(dks))
        dk_sig_raw = float(np.sqrt(np.mean(dk_vars) + np.var(dks)))
        env_mu = float(np.mean(envs))
        env_sig_raw = float(np.sqrt(np.mean(env_vars) + np.var(envs)))

        t_scale = 1.0 if raw_bars else self.prob_temp
        p_change = float(1.0 / (1.0 + np.exp(-np.clip(logit / t_scale, -30.0, 30.0))))
        dk_sig = dk_sig_raw if raw_bars else dk_sig_raw * self.widen_dk
        env_sig = env_sig_raw if raw_bars else env_sig_raw * self.widen_env

        return {
            "p_change": p_change,
            "logit_raw": logit,
            "delta_k_pct": dk_mu,
            "delta_k_sigma": dk_sig,
            "delta_k_sigma_raw": dk_sig_raw,
            "thermal_shift_pct": env_mu,
            "thermal_shift_sigma": env_sig,
            "thermal_shift_sigma_raw": env_sig_raw,
            "mode_dk_pct": mode_dk,
            "anchor_dk_pct": float(feat["anchor"][0]),
            "anchor_env_pct": float(feat["anchor"][1]),
            "active_modes": feat["m_mask"].astype(bool),
        }

    def predict(
        self,
        T_ref: np.ndarray,
        T_q: np.ndarray,
        df_ref: np.ndarray,
        df_q: np.ndarray,
        mask_ref: np.ndarray | None = None,
        mask_q: np.ndarray | None = None,
        raw_bars: bool = False,
    ) -> dict:
        """Read reference and query temperature/frequency windows and return structural health reading."""
        if mask_ref is None:
            mask_ref = np.isfinite(df_ref).astype(np.float32)
        if mask_q is None:
            mask_q = np.isfinite(df_q).astype(np.float32)
        df_ref = np.where(mask_ref > 0.5, df_ref, 0.0)
        df_q = np.where(mask_q > 0.5, df_q, 0.0)
        feat = compute_anchor_and_features(T_ref, T_q, df_ref, df_q, mask_ref, mask_q)
        return self.predict_features(feat, raw_bars=raw_bars)
