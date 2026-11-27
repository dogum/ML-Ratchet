"""First-pass 10-point forensic audit probe for FireSize v1.0 (r7 + widen).

Run:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/firesize_v1_0_probe.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/firesize"))
import firesize as fs


def main():
    print("=== FIRESIZE v1.0 (r7 + widen) FORENSIC AUDIT PROBE ===")
    m = fs.FireSize()
    w = m.W
    feats = m.features
    mu, sd, fill = w["mu"], w["sd"], w["fill"]
    n_ens = int(w["n_ens"])

    # 1. Weights, dead rows, ensemble diversity, extreme sd audit
    dead_total = 0
    min_norm = 1e9
    for e in range(n_ens):
        for k in (f"e{e}_L0_w", f"e{e}_L1_w", f"e{e}_L2_w"):
            arr = w[k]
            assert np.all(np.isfinite(arr))
            norms = np.linalg.norm(arr, axis=1)
            dead_total += int((norms < 1e-4).sum())
            min_norm = min(min_norm, float(norms.min()))
    cos = []
    for i in range(n_ens):
        for j in range(i + 1, n_ens):
            a, b = w[f"e{i}_L0_w"].ravel(), w[f"e{j}_L0_w"].ravel()
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    print(f"n_ens={n_ens}, temp={m.temp:.4f}, dead_rows={dead_total}, min_norm={min_norm:.4f}, L0_cos={[round(c, 4) for c in cos]}")
    assert dead_total == 0 and max(cos) < 0.10 and m.temp >= 1.0

    # CRITICAL: pt_SB (index 15) has sd = 1.00e-6 with large non-zero weights (0.76-0.79) and unclipped standardization
    idx_sb = feats.index("pt_SB")
    col15_norms = [float(np.linalg.norm(w[f"e{e}_L0_w"][:, idx_sb])) for e in range(n_ens)]
    print(
        f"Slash-Blowdown scales: pt_SB sd={sd[idx_sb]:.2e} (col norms={[round(v, 3) for v in col15_norms]}), "
        f"f5_SB sd={sd[feats.index('f5_SB')]:.4f}, f13_SB sd={sd[feats.index('f13_SB')]:.4f}"
    )
    assert sd[idx_sb] <= 1.01e-6 and min(col15_norms) > 0.7

    # 2. Shipped 8 demo examples, exam parity, and SB ignition collapse
    with open(REPO / "models/firesize/examples/demo_examples.csv") as fh:
        rows = list(csv.DictReader(fh))
    d_base = {f: np.array([float(r[f]) for r in rows]) for f in feats}
    o_base = m.predict(d_base)
    exam_p = np.array([float(r["exam_p_escape"]) for r in rows])
    assert np.allclose(o_base["p_escape"], exam_p, atol=1e-4)

    # Set ignition cell fuel to Slash-Blowdown (pt_SB=1.0, clearing previous pt_* one-hot)
    d_sb = {k: v.copy() for k, v in d_base.items()}
    for k in feats:
        if k.startswith("pt_"):
            d_sb[k][:] = 0.0
    d_sb["pt_SB"][:] = 1.0
    o_sb = m.predict(d_sb)
    print(
        f"Ignition on SB cell (pt_SB=1.0, z=+1e6): all 8 fires collapse from p_escape "
        f"[{o_base['p_escape'].min():.4f}, {o_base['p_escape'].max():.4f}] -> "
        f"max p_escape={o_sb['p_escape'].max():.6f}, identical median={o_sb['median_acres'][0]:.4f} ac"
    )
    assert np.all(o_sb["p_escape"] == 0.0) and np.allclose(o_sb["median_acres"], o_sb["median_acres"][0])

    # Nearby SB patch without pt_SB=1 (f5_SB=0.25, f13_SB=0.10)
    d_near = {k: v.copy() for k, v in d_base.items()}
    d_near["f5_SB"][:] = 0.25
    d_near["f13_SB"][:] = 0.10
    o_near = m.predict(d_near)
    print(f"Nearby SB patch (f5_SB=0.25, f13_SB=0.10): max p_escape drops {o_base['p_escape'].max():.4f} -> {o_near['p_escape'].max():.4f}")
    assert o_near["p_escape"].max() < 0.005

    # 3. 3-member PyTorch vs NumPy export parity
    X_real = m.matrix(d_base)
    rng = np.random.default_rng(0)
    X_test = np.vstack([X_real, mu + rng.standard_normal((32, len(feats))) * sd])
    Z_test = ((np.where(np.isfinite(X_test), X_test, fill[None, :]) - mu) / sd).astype(np.float32)
    out_th = np.zeros((len(X_test), 12), dtype=np.float64)
    for e in range(n_ens):
        h = torch.from_numpy(Z_test)
        k = 0
        while f"e{e}_L{k}_w" in w:
            h = torch.nn.functional.linear(
                h, torch.from_numpy(w[f"e{e}_L{k}_w"].astype(np.float32)), torch.from_numpy(w[f"e{e}_L{k}_b"].astype(np.float32))
            )
            if f"e{e}_L{k + 1}_w" in w:
                h = torch.relu(h)
            k += 1
        out_th += torch.softmax(h / m.temp, dim=1).numpy()
    out_th /= n_ens
    diff = float(np.max(np.abs(out_th - m.probs(X_test))))
    print(f"3-member Torch(f32) vs NumPy probs parity: {diff:.2e}")
    assert diff < 1e-6

    # 4. All-NaN guardrail (silent median imputation)
    o_nan = m.predict(np.full((1, len(feats)), np.nan))
    print(f"all_nan guardrail: silently returned p_escape={o_nan['p_escape'][0]:.4f}, median_ac={o_nan['median_acres'][0]:.2f}")
    assert np.isfinite(o_nan["p_escape"][0])

    # 5. Ratchet arithmetic replay (9/9 rungs on latest frozen set v1|v1rep2)
    rj = json.loads((REPO / "models/firesize/training/ratchet.json").read_text())
    inc_rpss = inc_auc = inc_mae = None
    for r in rj["rungs"]:
        rpss = float(r["sim_latest"]["rpss"])
        auc = float(r["dev_gate"]["esc_auc"])
        mae = float(r["dev_gate"]["mae"])
        if inc_rpss is None:
            calc = True
        else:
            calc = (rpss > inc_rpss + 0.01) and (auc >= inc_auc - 0.01) and (mae <= inc_mae * 1.05)
        assert calc == r["kept"], f"Mismatch on {r['rung']}: {calc} vs {r['kept']}"
        if r["kept"]:
            inc_rpss, inc_auc, inc_mae = rpss, auc, mae
    print(f"Ratchet replay: {len(rj['rungs'])}/{len(rj['rungs'])} rungs match published rule (r7 gate AUC margin = +0.0004).")
    print("FIRESIZE PROBE COMPLETE.")


if __name__ == "__main__":
    main()
