"""Classical SHM and dumb baselines for BridgeSense.

Baselines:
  1. Dumb (prevalence / zero shift): P(change) = dev_prevalence; dK/K = 0% (or raw 200*df).
  2. Classical 1 — Linear Thermal OLS + Control Chart:
     Per-mode linear regression df_m(T) = a_m(T - T0) + b_m fitted on the bridge's own
     reference baseline window. Detects change via RMS residual z-score (threshold/scale
     calibrated on KW51 Dev).
  3. Classical 2 — Dev-Tuned Bilinear Freezing + Mahalanobis Control Chart (Peeters & De Roeck 2001 style):
     Adds a piecewise-linear freezing hinge gamma * max(0, T_f - T) fitted strictly on
     KW51 Dev's healthy blocks (including freezing blocks), plus Mahalanobis / variance-weighted
     residual score and dev-fitted logistic probability map. This gives the classical physics
     baseline the exact same dev access as BridgeSense (the WakeNet/StormNet/SnowSense rule).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from simulator import compute_physics_anchor


class LinearOLSBaseline:
    """Standard per-mode linear thermal OLS baseline."""

    def __init__(self) -> None:
        self.a_prob = 1.0
        self.b_prob = -2.0

    def _raw_score(self, ep: dict) -> tuple[float, float, float]:
        anc = compute_physics_anchor(
            ep["T_ref"][:, 0], ep["df_ref"], ep["mask_ref"],
            ep["T_q"][:, 0], ep["df_q"], ep["mask_q"],
        )
        # RMS z-score across active modes
        act = np.sum(ep["mask_q"] > 0.5, axis=0) >= 2
        if np.any(act):
            z = anc["mode_mean_resid"][act] / anc["ref_resid_std"][act]
            rms_z = float(np.sqrt(np.mean(z * z)))
        else:
            rms_z = 0.0
        return rms_z, float(anc["anchor_dk_pct"]), float(anc["anchor_env_pct"])

    def fit_dev(self, dev_blocks: list[dict]) -> dict:
        scores = np.array([self._raw_score(b)[0] for b in dev_blocks], dtype=float)
        y = np.array([b["is_changed"] for b in dev_blocks], dtype=float)
        # Fit 2-param monotone logistic map P = sigmoid(a * score + b), a >= 0, on dev
        def nll(p):
            a = np.exp(p[0])
            b = p[1]
            z = np.clip(a * scores + b, -20.0, 20.0)
            pr = 1.0 / (1.0 + np.exp(-z))
            return float(-np.mean(y * np.log(pr + 1e-9) + (1 - y) * np.log(1 - pr + 1e-9)) + 1e-2 * (a * a))
        res = minimize(nll, [0.0, -2.0], method="Nelder-Mead")
        self.a_prob = float(np.exp(res.x[0]))
        self.b_prob = float(res.x[1])
        return {"a_prob": round(self.a_prob, 4), "b_prob": round(self.b_prob, 4)}

    def predict(self, ep: dict) -> dict[str, float]:
        rms_z, dk, env = self._raw_score(ep)
        z = np.clip(self.a_prob * rms_z + self.b_prob, -20.0, 20.0)
        return {
            "score": rms_z,
            "p_change": float(1.0 / (1.0 + np.exp(-z))),
            "dk_pct": dk,
            "env_pct": env,
        }


class DevTunedBilinearBaseline:
    """Dev-tuned bilinear freezing thermal regression + RMS z-score (fair classical rival)."""

    def __init__(self) -> None:
        self.T_freeze = 0.0
        self.gamma_freeze = 0.0   # fractional freq boost per degree below T_freeze
        self.a_prob = 1.0
        self.b_prob = -2.0

    def _eval_ep(self, ep: dict, T_freeze: float, gamma: float) -> tuple[float, float, float]:
        T_r = ep["T_ref"][:, 0]
        T_q = ep["T_q"][:, 0]
        # Subtract dev-tuned freezing hinge from both ref and query before OLS linear fit
        frz_r = gamma * np.maximum(0.0, T_freeze - T_r)[:, None]
        frz_q = gamma * np.maximum(0.0, T_freeze - T_q)[:, None]
        df_r_corr = np.where(ep["mask_ref"] > 0.5, ep["df_ref"] - frz_r, 0.0)
        df_q_corr = np.where(ep["mask_q"] > 0.5, ep["df_q"] - frz_q, 0.0)
        anc = compute_physics_anchor(T_r, df_r_corr, ep["mask_ref"], T_q, df_q_corr, ep["mask_q"])
        act = np.sum(ep["mask_q"] > 0.5, axis=0) >= 2
        if np.any(act):
            z = anc["mode_mean_resid"][act] / anc["ref_resid_std"][act]
            rms_z = float(np.sqrt(np.mean(z * z)))
            raw_mean = float(np.mean([
                np.mean(ep["df_q"][ep["mask_q"][:, m] > 0.5, m]) for m in np.where(act)[0]
            ]))
            env_pct = 100.0 * (raw_mean - anc["anchor_dk_pct"] / 200.0)
        else:
            rms_z = 0.0
            env_pct = 0.0
        return rms_z, float(anc["anchor_dk_pct"]), float(env_pct)

    def fit_dev(self, dev_blocks: list[dict]) -> dict:
        healthy = [b for b in dev_blocks if b["is_changed"] == 0]
        best_loss = 1e18
        best_params = (0.0, 0.0)
        # Grid search T_freeze in [-1, 3] C and gamma in [0, 0.02] / K on healthy dev blocks only!
        for tf in np.linspace(-1.0, 3.0, 9):
            for gm in np.linspace(0.0, 0.015, 31):
                errs = [abs(self._eval_ep(b, float(tf), float(gm))[1]) for b in healthy]
                loss = float(np.mean(errs))
                if loss < best_loss:
                    best_loss = loss
                    best_params = (float(tf), float(gm))
        self.T_freeze, self.gamma_freeze = best_params

        scores = np.array([self._eval_ep(b, self.T_freeze, self.gamma_freeze)[0] for b in dev_blocks], dtype=float)
        y = np.array([b["is_changed"] for b in dev_blocks], dtype=float)

        def nll(p):
            a = np.exp(p[0])
            b = p[1]
            z = np.clip(a * scores + b, -20.0, 20.0)
            pr = 1.0 / (1.0 + np.exp(-z))
            return float(-np.mean(y * np.log(pr + 1e-9) + (1 - y) * np.log(1 - pr + 1e-9)) + 1e-2 * (a * a))

        res = minimize(nll, [0.0, -2.0], method="Nelder-Mead")
        self.a_prob = float(np.exp(res.x[0]))
        self.b_prob = float(res.x[1])
        return {
            "T_freeze_C": round(self.T_freeze, 2),
            "gamma_freeze_per_K": round(self.gamma_freeze, 5),
            "healthy_dev_dk_mae": round(best_loss, 3),
            "a_prob": round(self.a_prob, 4),
            "b_prob": round(self.b_prob, 4),
        }

    def predict(self, ep: dict) -> dict[str, float]:
        rms_z, dk, env = self._eval_ep(ep, self.T_freeze, self.gamma_freeze)
        z = np.clip(self.a_prob * rms_z + self.b_prob, -20.0, 20.0)
        return {
            "score": rms_z,
            "p_change": float(1.0 / (1.0 + np.exp(-z))),
            "dk_pct": dk,
            "env_pct": env,
        }


def dumb_predict(ep: dict, dev_prevalence: float = 0.4) -> dict[str, float]:
    """Dumb baseline: constant dev prevalence for detection; raw 200*df or 0% for dK/K."""
    act = np.sum(ep["mask_q"] > 0.5, axis=0) >= 2
    if np.any(act):
        raw_mean = float(np.mean([
            np.mean(ep["df_q"][ep["mask_q"][:, m] > 0.5, m]) for m in np.where(act)[0]
        ]))
        raw_dk = 200.0 * raw_mean
    else:
        raw_dk = 0.0
    return {
        "score": abs(raw_dk),
        "p_change": float(dev_prevalence),
        "dk_zero_pct": 0.0,
        "dk_raw_pct": raw_dk,
    }
