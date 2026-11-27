"""First-pass 10-point forensic audit probe for FirePilot v1.0 (r1).

Run:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/firepilot_v1_0_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/firepilot"))
sys.path.insert(0, str(REPO / "models/firepilot/training"))

import firepilot as fp
import train_value as tv


def load_example_ctx(npz_path, rate=300.0):
    z = np.load(npz_path, allow_pickle=True)
    rec = json.loads(npz_path.with_suffix(".json").read_text())
    rec["rate"] = rate
    return fp.Context.from_store(z, rec), rec


def main():
    print("=== FIREPILOT v1.0 (r1) FORENSIC AUDIT PROBE ===")
    w_path = REPO / "models/firepilot/firepilot_weights.npz"
    w = dict(np.load(w_path))
    mu = w["mu"]
    sd = w["sd"]
    W0 = w["W0"]
    print(f"Weight keys: {sorted(w.keys())}, feature dim={len(mu)}")
    assert all(np.all(np.isfinite(v)) for v in w.values()), "Non-finite weights!"

    # 1. Inspect zero-variance feature feat[6] (W[-1] / 36.0); W0 is stored (31, 64)
    row6_norm = float(np.linalg.norm(W0[6]))
    print(f"feat[6]: mu={mu[6]:.10f}, sd={sd[6]:.2e}, ||W0[6]||_2={row6_norm:.3e}")
    assert sd[6] < 2e-6 and row6_norm > 1e-5

    # Check dead hidden/output units (weight matrices stored as (in_dim, out_dim))
    dead_units = 0
    for k in ("W0", "W1", "W2"):
        norms = np.linalg.norm(w[k], axis=0)
        dead = int((norms < 1e-4).sum())
        dead_units += dead
        print(f"  {k}: shape={w[k].shape}, min_out_norm={norms.min():.4f}, dead={dead}")
    assert dead_units == 0
    print(f"  sig_scale={float(w['sig_scale']):.4f}")

    # 2. Torch (train_value.Net) vs NumPy (ValueNet) parity
    vnet = fp.ValueNet(str(w_path))
    net_th = tv.Net(len(mu), W0.shape[1])
    net_th.l0.weight.data.copy_(torch.from_numpy(w["W0"].T))
    net_th.l0.bias.data.copy_(torch.from_numpy(w["b0"]))
    net_th.l1.weight.data.copy_(torch.from_numpy(w["W1"].T))
    net_th.l1.bias.data.copy_(torch.from_numpy(w["b1"]))
    net_th.l2.weight.data.copy_(torch.from_numpy(w["W2"].T))
    net_th.l2.bias.data.copy_(torch.from_numpy(w["b2"]))
    net_th.eval()

    rng = np.random.default_rng(0)
    X_rand = mu + rng.standard_normal((64, len(mu))).astype(np.float32) * sd
    Z_rand = (X_rand - mu) / sd
    with torch.no_grad():
        th_out = net_th(torch.from_numpy(Z_rand.astype(np.float32))).numpy()
    th_mean = 1.0 / (1.0 + np.exp(-th_out[:, 0]))
    th_sig = np.exp(np.clip(th_out[:, 1], -5.0, 2.0)) * float(w["sig_scale"])
    np_mean, np_sig = vnet(X_rand)
    d_mean = float(np.max(np.abs(th_mean - np_mean)))
    d_sig = float(np.max(np.abs(th_sig - np_sig)))
    print(f"Torch vs NumPy parity: max|d_mean|={d_mean:.2e}, max|d_sig|={d_sig:.2e}")
    assert d_mean < 1e-6 and d_sig < 1e-6

    # 3. Verify work_clock(start, lon=-110) across 24 start hours
    wh_vals = []
    for h in range(24):
        ts = pd.Timestamp(f"2023-08-15 {h:02d}:00:00")
        _, W_arr = fp.work_clock(ts, lon=-110.0, horizon_h=72.0)
        wh_vals.append(float(W_arr[-1]))
    print(f"work_clock 72h range across 24 start hours: min={min(wh_vals):.2f}h, max={max(wh_vals):.2f}h")
    assert max(wh_vals) >= 35.9 and min(wh_vals) <= 34.1

    # 4. Counterfactual feat[6] perturbation on all 3 shipped example fires
    ex_files = sorted((REPO / "models/firepilot/examples").glob("*.npz"))
    print("\nShipped examples feat[6] explosion check (rate=300 m/h):")
    for npz_p in ex_files:
        ctx, rec = load_example_ctx(npz_p, rate=300.0)
        fid = npz_p.stem
        base_f6 = float(ctx._scn[6])
        z_base = float((base_f6 - mu[6]) / sd[6])

        # Plan with nominal afternoon W[-1]=34h
        pl_nom, in_nom = fp.plan_value(ctx, vnet, np.random.default_rng(0))
        pr_nom = ctx.plan_eval(pl_nom)[0]["s_proxy"]

        # Shift ONLY feat[6] to morning/overnight detection W[-1]=36h (+2h -> z = +55,556)
        ctx._scn[6] = 36.0 / 36.0
        pl_36, in_36 = fp.plan_value(ctx, vnet, np.random.default_rng(0))
        pr_36 = ctx.plan_eval(pl_36)[0]["s_proxy"]

        # Shift ONLY feat[6] to 48h horizon W[-1]=22h (-12h -> z = -333,333)
        ctx._scn[6] = 22.0 / 36.0
        pl_22, in_22 = fp.plan_value(ctx, vnet, np.random.default_rng(0))
        pr_22 = ctx.plan_eval(pl_22)[0]["s_proxy"]
        ctx._scn[6] = base_f6

        print(
            f"  {fid}: base_f6={base_f6:.6f} (z={z_base:+.2f}) | "
            f"nom pred={in_nom['pred']:.4f} proxy={pr_nom:.4f} d_h={pl_nom['d_h']:.0f}m | "
            f"W=36h pred={in_36['pred']:.4f} proxy={pr_36:.4f} d_h={pl_36['d_h']:.0f}m | "
            f"W=22h pred={in_22['pred']:.4f} proxy={pr_22:.4f} d_h={pl_22['d_h']:.0f}m"
        )
        assert abs(z_base) < 1e-3
        assert in_36["pred"] > in_nom["pred"] + 0.08
        assert in_22["pred"] < 0.15

    # 5. Ratchet promotion-rule arithmetic replay (5/5 rungs)
    rj = json.loads((REPO / "models/firepilot/training/ratchet.json").read_text())
    inc_val = None
    inc_gate = None
    for row in rj["rungs"]:
        v = row["val"]["saved_mean"]
        g = row["gate"]["saved_mean"]
        if inc_val is None:
            calc = "kept"
        else:
            calc = "kept" if (v > inc_val + 0.01 and g >= inc_gate * 0.95) else "rejected"
        assert calc == row["decision"], f"Mismatch on {row['rung']}: {calc} vs {row['decision']}"
        if row["decision"] == "kept":
            inc_val, inc_gate = v, g
    print(f"\nRatchet replay: {len(rj['rungs'])}/{len(rj['rungs'])} rungs match published rule.")
    print("FIREPILOT PROBE COMPLETE.")


if __name__ == "__main__":
    main()
