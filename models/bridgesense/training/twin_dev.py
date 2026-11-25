"""Twin test on KW51 Dev blocks ONLY — measure simulator-vs-reality mismatch and produce cal_p3.json.

Strictly uses ONLY the KW51 warm reference baseline + the 10 frozen KW51 Dev blocks
(splits.json['kw51_dev']). Never reads Exam A, Exam B, or Exam C.
"""
from __future__ import annotations

import json
import os
import numpy as np

from _paths import work
from dev_eval import eval_on_kw51_dev, load_kw51_dev_blocks, load_model_ckpt, predict_episode
from simulator import compute_physics_anchor, simulate_episode


def main() -> None:
    dev_blocks = load_kw51_dev_blocks()
    # 1. Measure empirical physics statistics on KW51 Dev ONLY
    mild_resids, freeze_resids, post_resids = [], [], []
    freeze_extra_jumps = []
    for b in dev_blocks:
        anc = compute_physics_anchor(
            b["T_ref"][:, 0], b["df_ref"], b["mask_ref"],
            b["T_q"][:, 0], b["df_q"], b["mask_q"],
        )
        if b["stratum"] == "healthy_mild":
            mild_resids.append(anc["anchor_dk_pct"])
        elif b["stratum"] == "healthy_freezing":
            freeze_resids.append(anc["anchor_dk_pct"])
            # Per-mode jump on freezing days relative to linear extrapolation
            act = np.sum(b["mask_q"] > 0.5, axis=0) >= 2
            freeze_extra_jumps.append(float(np.max(np.abs(anc["mode_mean_resid"][act]))))
        elif b["stratum"] == "post_retrofit":
            post_resids.append(anc["anchor_dk_pct"])

    print("=== KW51 Dev empirical anchor stats (DEV ONLY) ===")
    print(f"  healthy_mild anchor dK/K (%):     {[round(x, 2) for x in mild_resids]}")
    print(f"  healthy_freezing anchor dK/K (%): {[round(x, 2) for x in freeze_resids]} (max mode jump: {[round(x*100, 2) for x in freeze_extra_jumps]}%)")
    print(f"  post_retrofit anchor dK/K (%):    {[round(x, 2) for x in post_resids]}")

    # 2. If r2 checkpoint exists, compare r2 on real KW51 Dev vs simulated twins of the same regime
    out = {
        "kw51_dev_anchor_mild_dk_pct": [round(x, 3) for x in mild_resids],
        "kw51_dev_anchor_freeze_dk_pct": [round(x, 3) for x in freeze_resids],
        "kw51_dev_anchor_post_dk_pct": [round(x, 3) for x in post_resids],
    }
    r2_ckpt = work("checkpoints", "r2.pt")
    if os.path.exists(r2_ckpt):
        real_res = eval_on_kw51_dev("r2", dev_blocks)
        print("\n=== r2 on REAL KW51 Dev ===")
        for row in real_res["rows"]:
            print(f"  {row['block_id']} | {row['stratum']:16s} | Tmin={row['T_min']:5.1f}C | "
                  f"P={row['p_change']:.3f} | dK={row['dk_pct']:+5.2f}% (anc={row['anchor_dk_pct']:+5.2f}%)")
        print(f"  REAL DEV summary: AUC={real_res['dev_auc']} | Brier={real_res['dev_brier']} | "
              f"healthy_dk_mae={real_res['dev_healthy_dk_mae']}% | freeze_P={real_res['dev_freeze_p_mean']}")

        # Simulate 60 arch-bridge p2 twins (20 mild, 20 freezing healthy, 20 retrofit) to compare!
        models = load_model_ckpt("r2")
        rng = np.random.default_rng(777)
        twin_mild_p, twin_frz_p, twin_chg_p = [], [], []
        for _ in range(120):
            ep = simulate_episode(rng, physics_version="p2", n_ref=32, n_q=32, max_modes=14)
            pr = predict_episode(models, ep)
            if ep["is_changed"] == 0:
                if float(np.min(ep["T_q"][:, 0])) < 0.0:
                    twin_frz_p.append(pr["p_change"])
                else:
                    twin_mild_p.append(pr["p_change"])
            else:
                twin_chg_p.append(pr["p_change"])
        out["r2_real_dev"] = real_res
        out["r2_twin_p_means"] = {
            "healthy_mild": round(float(np.mean(twin_mild_p)), 4) if twin_mild_p else None,
            "healthy_freezing": round(float(np.mean(twin_frz_p)), 4) if twin_frz_p else None,
            "changed": round(float(np.mean(twin_chg_p)), 4) if twin_chg_p else None,
        }
        print(f"  TWIN p2 summary: mild_P={out['r2_twin_p_means']['healthy_mild']} | "
              f"freeze_P={out['r2_twin_p_means']['healthy_freezing']} | changed_P={out['r2_twin_p_means']['changed']}")

    # 3. Derive dev-calibrated p3 prior from KW51 Dev mismatch
    # Measure empirical slow inter-window drift across KW51's 6 healthy dev blocks
    healthy_g_shifts = []
    healthy_m_stds = []
    for b in dev_blocks:
        if b["is_changed"] == 0:
            anc = compute_physics_anchor(
                b["T_ref"][:, 0], b["df_ref"], b["mask_ref"],
                b["T_q"][:, 0], b["df_q"], b["mask_q"],
            )
            act = np.sum(b["mask_q"] > 0.5, axis=0) >= 2
            res = anc["mode_mean_resid"][act]
            healthy_g_shifts.append(abs(float(np.mean(res))))
            healthy_m_stds.append(float(np.std(res)))

    max_frz_dk = max([abs(x) for x in freeze_resids], default=4.0)
    freeze_scale = float(np.clip(max_frz_dk / 5.0, 1.0, 2.2))
    mode_freeze_extra = float(np.clip(max(freeze_extra_jumps, default=0.05) * 1.1, 0.04, 0.12))
    global_drift_std = float(np.clip(np.mean(healthy_g_shifts) * 1.35, 0.0008, 0.0030))
    mode_drift_std = float(np.clip(np.mean(healthy_m_stds) * 1.25, 0.0012, 0.0035))
    cal_p3 = {
        "source": "KW51 Dev twin test (10 blocks, splits.json['kw51_dev'])",
        "freeze_scale": round(freeze_scale, 3),
        "mode_freeze_extra": round(mode_freeze_extra, 4),
        "global_drift_std": round(global_drift_std, 5),
        "mode_drift_std": round(mode_drift_std, 5),
        "T_freeze_lo": -3.5,
        "T_freeze_hi": 2.0,
        "noise_scale": 1.0,
    }
    with open(work("cal_p3.json", mkdir=True), "w", encoding="utf-8") as f:
        json.dump(cal_p3, f, indent=2)
    with open(work("twin_dev.json", mkdir=True), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved cal_p3.json:", cal_p3)


if __name__ == "__main__":
    main()
