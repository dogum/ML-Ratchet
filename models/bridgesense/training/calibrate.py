"""Fit widen-only error-bar factors and probability temperature on KW51 Dev ONLY.

Uses mlratchet.calibrate.widen_factor (never shrinks bars: factor >= 1.0) and
widen-only temperature scaling (T >= 1.0) on KW51 Dev blocks.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from scipy.optimize import minimize_scalar

from mlratchet.calibrate import coverage, widen_factor
from _paths import work
from dev_eval import load_kw51_dev_blocks, load_model_ckpt, predict_episode
from realdata import load_kw51_hourly


def estimate_kw51_post_retrofit_true_dk(dev_blocks: list[dict]) -> float:
    """Compute matched-temperature post-retrofit effective dK/K (%) strictly on KW51 Dev blocks.

    For post-retrofit dev blocks, compares frequencies against healthy baseline at matched temperatures
    (since baseline is Oct-Nov 2018 and post-retrofit is Oct 2019-Jan 2020, overlapping temperatures
    in [3, 18] C provide an exact model-free ground truth for the retrofit stiffness shift).
    """
    post_dks = []
    for b in dev_blocks:
        if b["stratum"] != "post_retrofit":
            continue
        # Only use non-freezing query hours (T >= 3 C) where linear thermal overlap with Oct-Nov baseline is exact
        Tr = b["T_ref"][:, 0]
        Tq = b["T_q"][:, 0]
        dks = []
        for m in range(b["df_q"].shape[1]):
            ok_r = (b["mask_ref"][:, m] > 0.5) & (Tr >= 3.0)
            ok_q = (b["mask_q"][:, m] > 0.5) & (Tq >= 3.0)
            if np.sum(ok_r) >= 4 and np.sum(ok_q) >= 3:
                # Fit linear T on ref
                t0 = float(np.mean(Tr[ok_r]))
                x = Tr[ok_r] - t0
                y = b["df_ref"][ok_r, m]
                a = float(np.dot(x, y) / (np.dot(x, x) + 1e-4))
                c = float(np.mean(y) - a * np.mean(x))
                pred_q = c + a * (Tq[ok_q] - t0)
                df_shift = float(np.mean(b["df_q"][ok_q, m] - pred_q))
                # dK/K = ((1 + df_shift)^2 - 1) * 100
                dks.append(100.0 * ((1.0 + df_shift) ** 2 - 1.0))
        if dks:
            post_dks.append(float(np.mean(dks)))
    return float(np.median(post_dks)) if post_dks else 2.5


def calibrate_on_dev(ckpt_names: list[str]) -> dict:
    models = load_model_ckpt(ckpt_names)
    dev_blocks = load_kw51_dev_blocks()
    post_true_dk = estimate_kw51_post_retrofit_true_dk(dev_blocks)

    logits, y_chg = [], []
    dk_res, dk_sig = [], []
    for b in dev_blocks:
        r = predict_episode(models, b)
        logits.append(r["logit"])
        y_chg.append(b["is_changed"])
        true_dk = 0.0 if b["is_changed"] == 0 else post_true_dk
        dk_res.append(r["dk_pct"] - true_dk)
        dk_sig.append(r["dk_sig"])

    logits = np.array(logits, dtype=float)
    y_chg = np.array(y_chg, dtype=float)
    dk_res = np.array(dk_res, dtype=float)
    dk_sig = np.array(dk_sig, dtype=float)

    w_dk = float(widen_factor(dk_res, dk_sig))
    # Widen-only temperature scaling T >= 1.0 minimizing dev NLL
    def nll_T(T):
        z = np.clip(logits / T, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        return float(-np.mean(y_chg * np.log(p + 1e-9) + (1.0 - y_chg) * np.log(1.0 - p + 1e-9)))
    opt_T = minimize_scalar(nll_T, bounds=(1.0, 8.0), method="bounded")
    prob_temp = float(max(1.0, opt_T.x))

    # Also check env_sig widening on healthy dev blocks (where true_env ~= mean_raw_df * 100)
    env_res, env_sig = [], []
    for b in dev_blocks:
        if b["is_changed"] == 0:
            r = predict_episode(models, b)
            act = np.sum(b["mask_q"] > 0.5, axis=0) >= 2
            raw_env = 100.0 * float(np.mean([
                np.mean(b["df_q"][b["mask_q"][:, m] > 0.5, m]) for m in np.where(act)[0]
            ]))
            env_res.append(r["env_pct"] - raw_env)
            env_sig.append(r["env_sig"])
    w_env = float(widen_factor(env_res, env_sig))

    cal = {
        "ckpts": ckpt_names,
        "widen_dk": round(w_dk, 4),
        "widen_env": round(w_env, 4),
        "prob_temp": round(prob_temp, 4),
        "dev_cov_dk_raw": round(float(coverage(dk_res, dk_sig, 1.0)), 4),
        "dev_cov_dk_widened": round(float(coverage(dk_res, dk_sig, w_dk)), 4),
        "dev_cov_env_raw": round(float(coverage(env_res, env_sig, 1.0)), 4),
        "dev_cov_env_widened": round(float(coverage(env_res, env_sig, w_env)), 4),
        "kw51_dev_post_retrofit_true_dk_pct": round(post_true_dk, 3),
    }
    out_path = work("calibration.json", mkdir=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print("Saved calibration:", json.dumps(cal, indent=2))
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()
    calibrate_on_dev(args.ckpts)


if __name__ == "__main__":
    main()
