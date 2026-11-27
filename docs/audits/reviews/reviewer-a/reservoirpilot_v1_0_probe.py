"""Forensic audit probe for ReservoirPilot v1.0 (Reviewer A, first pass).

Checks:
  1. Weight health, input standardization bounds, and dead units.
  2. Exact reproduction of all 6 shipped ResOpsUS exam dams in examples/exam_dams.json.
  3. Torch (training/policy.py rollout_torch) vs NumPy (reservoirpilot.py Pilot.run) parity.
  4. Causal boundary: future inflow perturbation (t+1 onward), 1-day lagged inflow replay,
     and evaluation-period capacity cap sensitivity (-10% cap counterfactual).
  5. Counterfactual feature ablation: removing calibration-half inflow climatology (Icurve=0).
  6. Replay of all 10 logged rows in training/ratchet.json under the g4 promotion rule.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/reservoirpilot_v1_0_probe.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / "models/reservoirpilot"),
    str(ROOT / "models/reservoirpilot/training"),
]
import reservoirpilot as rp
import policy as rp_tr_pol


def main():
    w_path = ROOT / "models/reservoirpilot/reservoirpilot_weights.npz"
    w = np.load(w_path)
    pol = rp.Pilot.load(str(w_path))

    # 1. Weight health & normalization stats
    dead = []
    for k in w.files:
        arr = w[k]
        if not np.issubdtype(arr.dtype, np.number):
            continue
        assert np.isfinite(arr).all(), f"non-finite {k}"
        if arr.ndim == 2:
            dr = int((np.max(np.abs(arr), axis=1) < 1e-8).sum())
            dc = int((np.max(np.abs(arr), axis=0) < 1e-8).sum())
            if dr or dc:
                dead.append((k, arr.shape, dr, dc))
    assert not dead, f"dead units: {dead}"
    print(f"weights: finite, 0 dead rows/cols | sd range [{w['sd'].min():.4f}, {w['sd'].max():.4f}]")

    # Build Torch mirror
    net_t = rp_tr_pol.torch_policy(15, 64, 2, "hd").double()
    lin = [m for m in net_t if hasattr(m, "weight")]
    with torch.no_grad():
        for i, m in enumerate(lin):
            m.weight.copy_(torch.tensor(w[f"W{i}"].T, dtype=torch.float64))
            m.bias.copy_(torch.tensor(w[f"b{i}"], dtype=torch.float64))
    mu_t = torch.tensor(w["mu"], dtype=torch.float64)
    sd_t = torch.tensor(w["sd"], dtype=torch.float64)

    # 2–5. Packaged 6 exam dams
    dams = json.loads((ROOT / "models/reservoirpilot/examples/exam_dams.json").read_text())
    max_torch_diff = 0.0
    max_fut_diff = 0.0
    cap_eq_obs = []
    print("\nDam-by-dam checks (6 shipped exam dams):")
    for d in dams:
        dates = pd.date_range(d["start"], periods=len(d["I"]), freq="D")
        doy = dates.day_of_year.values
        Dc = np.array(d["Dcurve"], float)
        Ic = np.array(d["Icurve"], float)
        tk = {
            "I": np.array(d["I"], float),
            "D": Dc[doy - 1],
            "Dcurve": Dc,
            "Icurve": Ic,
            "Qmax": float(d["Qmax"]),
            "cap": float(d["cap"]),
            "S0": float(d["S0"]),
            "dates": dates,
        }
        R_np, S_np = pol.run(tk)
        c_np = rp.cost_components(R_np, S_np, tk)
        assert abs(c_np["J"] - d["exam_J"]["pilot"]) < 1e-6, f"exam_J mismatch on {d['res_id']}"

        # Torch parity
        ca = rp_tr_pol.task_arrays(tk, len(tk["I"]))
        batch = {k: torch.tensor(v[None, :], dtype=torch.float64) for k, v in ca.items()}
        batch.update(
            {
                "I": torch.tensor(tk["I"][None, :], dtype=torch.float64),
                "q": torch.tensor([tk["Qmax"]], dtype=torch.float64),
                "cap": torch.tensor([tk["cap"]], dtype=torch.float64),
                "S0": torch.tensor([tk["S0"]], dtype=torch.float64),
                "ema0": torch.tensor([ca["Ic30"][0]], dtype=torch.float64),
            }
        )
        with torch.no_grad():
            J_t, _ = rp_tr_pol.rollout_torch(net_t, mu_t, sd_t, batch, "clim", terminal=0.0, tail=0, kind="hd")
        t_diff = abs(float(J_t[0]) - c_np["J"])
        max_torch_diff = max(max_torch_diff, t_diff)

        # Future-inflow causality check (5x inflow strictly after midpoint day)
        mid = len(tk["I"]) // 2
        I_fut = tk["I"].copy()
        I_fut[mid + 1 :] *= 5.0
        R_fut, _ = pol.run(dict(tk, I=I_fut))
        fut_diff = float(np.max(np.abs(R_fut[: mid + 1] - R_np[: mid + 1])))
        max_fut_diff = max(max_fut_diff, fut_diff)

        # 1-day lagged inflow replay (policy sees I[t-1] while true mass balance evolves under I[t])
        I_true = tk["I"]
        I_seen = np.r_[tk["Icurve"][doy[0] - 1], I_true[:-1]]
        R_l, S_l = pol.run(tk, seen=I_seen)
        J_lag = rp.cost_components(R_l, S_l, tk)["J"]
        J_rule = rp.cost_components(*rp.tuned_rule(tk), tk)["J"]

        # Eval-period peak storage ratio & -10% cap counterfactual
        s_ratio = float(np.max(d["S_obs"]) / d["cap"])
        if abs(s_ratio - 1.0) < 1e-4:
            cap_eq_obs.append(d["res_id"])
        tk_lo = dict(tk, cap=tk["cap"] * 0.90, S0=min(tk["S0"], tk["cap"] * 0.90))
        J_cap_pil = rp.cost_components(*pol.run(tk_lo), tk_lo)["J"]
        J_cap_rule = rp.cost_components(*rp.tuned_rule(tk_lo), tk_lo)["J"]

        # Climatology ablation (Icurve = 0)
        J_noclim = rp.cost_components(*pol.run(dict(tk, Icurve=np.zeros_like(Ic))), tk)["J"]

        print(
            f"  {d['res_id']:10s}: J={c_np['J']:.6f} (rule={J_rule:.6f}) | lag1d={J_lag:.6f} | "
            f"S_obs_max/cap={s_ratio:.4f} | cap*0.9 pil={J_cap_pil:.5f} vs rule={J_cap_rule:.5f} | "
            f"noclim={J_noclim:.5f} | meanS={np.mean(S_np)/tk['cap']:.3f}"
        )

    assert max_fut_diff == 0.0, f"future inflow leak: {max_fut_diff}"
    print(f"\nmax Torch-vs-NumPy J diff: {max_torch_diff:.3e}")
    print(f"max future-inflow leakage diff: {max_fut_diff:.3e}")
    print(f"dams with max(S_obs)/cap == 1.0000: {cap_eq_obs} ({len(cap_eq_obs)}/6)")

    # 6. Ratchet replay
    rj = json.loads((ROOT / "models/reservoirpilot/training/ratchet.json").read_text())
    inc_sj, inc_dev = 0.0, None
    n_checked = 0
    for row in rj["rungs"]:
        if not row["sim"]:
            assert row["kept"] is False
            n_checked += 1
            continue
        sj = row["sim"]["g4"]["S_J"]
        dev_j = row["dev"]["J_median"]
        dev_sop = row["dev"]["J_sop_median"]
        if inc_dev is None:
            kept = (sj > 0.01) and (dev_j <= dev_sop)
        else:
            kept = (sj > inc_sj + 0.01) and (dev_j <= 1.05 * inc_dev)
        assert kept == row["kept"], f"ratchet mismatch on {row['id']}"
        if kept:
            inc_sj, inc_dev = sj, dev_j
        n_checked += 1
    print(f"ratchet.json replay: {n_checked}/{len(rj['rungs'])} rows verified -> final={rj['final']}")


if __name__ == "__main__":
    main()
