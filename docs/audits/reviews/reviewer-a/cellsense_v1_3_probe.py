"""First-pass 10-point forensic audit probe for CellSense v1.3 (LFP + NMC811).

Run:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/cellsense_v1_3_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/cellsense"))
import cellsense as cs


def run_torch_folded(npz_path, X_np, ex_np=None):
    z = np.load(npz_path)
    m = json.loads(str(z["meta"]))
    x = torch.from_numpy(X_np)
    x = torch.nn.functional.gelu(
        torch.nn.functional.conv1d(x, torch.from_numpy(z["stem_w"]), torch.from_numpy(z["stem_b"]), padding=3)
    )
    for i, (s, p1, p2) in enumerate(m["blocks"]):
        y = torch.nn.functional.gelu(
            torch.nn.functional.conv1d(x, torch.from_numpy(z[f"b{i}c1_w"]), torch.from_numpy(z[f"b{i}c1_b"]), stride=s, padding=p1)
        )
        y = torch.nn.functional.conv1d(y, torch.from_numpy(z[f"b{i}c2_w"]), torch.from_numpy(z[f"b{i}c2_b"]), padding=p2)
        sk = torch.nn.functional.conv1d(x, torch.from_numpy(z[f"b{i}sk_w"]), torch.from_numpy(z[f"b{i}sk_b"]), stride=s)
        x = torch.nn.functional.gelu(y + sk)
    h = torch.cat([x.mean(dim=-1), x.max(dim=-1).values], dim=-1)
    if ex_np is not None:
        h = torch.cat([h, torch.from_numpy(ex_np)], dim=-1)
    h = torch.nn.functional.gelu(torch.nn.functional.linear(h, torch.from_numpy(z["h0_w"]), torch.from_numpy(z["h0_b"])))
    h = torch.nn.functional.linear(h, torch.from_numpy(z["h1_w"]), torch.from_numpy(z["h1_b"]))
    return h.numpy()


def main():
    print("=== CELLSENSE v1.3 (LFP + NMC811) FORENSIC AUDIT PROBE ===")
    mdir = REPO / "models/cellsense"
    npz_specs = [
        ("cellsense_weights.npz", 8, 0),
        ("cellsense_life_weights.npz", 8, 3),
        ("cellsense_partial_weights.npz", 8, 0),
        ("cellsense_nmc811_weights.npz", 6, 0),
        ("cellsense_nmc811_m1.npz", 6, 0),
        ("cellsense_nmc811_m2.npz", 6, 0),
        ("cellsense_nmc811_m3.npz", 6, 0),
    ]

    # 1. Finiteness, dead filters, and folded Torch vs NumPy parity
    rng = np.random.default_rng(0)
    max_parity = 0.0
    for nm, cin, nex in npz_specs:
        p = mdir / nm
        z = np.load(p)
        dead = 0
        for k in z.files:
            if k == "meta":
                continue
            arr = z[k]
            assert np.all(np.isfinite(arr)), f"Non-finite in {nm}:{k}"
            if k.endswith("_w") and arr.ndim >= 2:
                norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
                dead += int((norms < 1e-4).sum())
        assert dead == 0, f"Dead units in {nm}: {dead}"
        X = rng.standard_normal((4, cin, 250)).astype(np.float32)
        ex = rng.standard_normal((4, nex)).astype(np.float32) if nex else None
        out_np = cs.CellSenseNet(str(p)).forward(X, ex)
        out_th = run_torch_folded(str(p), X, ex)
        diff = float(np.max(np.abs(out_np - out_th)))
        max_parity = max(max_parity, diff)
        print(f"  {nm:32s}: finite=OK, dead=0, folded Torch<->NumPy={diff:.2e}")
    assert max_parity < 1.5e-5

    # 2. NMC811 4-member ensemble collinearity (shared --seed 0 initialization)
    nmc_files = [mdir / f for f in ("cellsense_nmc811_weights.npz", "cellsense_nmc811_m1.npz", "cellsense_nmc811_m2.npz", "cellsense_nmc811_m3.npz")]
    nmc_w = [np.load(f) for f in nmc_files]
    print("\nNMC811 4-member ensemble pairwise cosine similarity (6 pairs):")
    for layer in ("stem_w", "b0c1_w", "b2c1_w", "h0_w", "h1_w"):
        cos = []
        for i in range(4):
            for j in range(i + 1, 4):
                a = nmc_w[i][layer].ravel()
                b = nmc_w[j][layer].ravel()
                cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        print(f"  {layer:8s}: min={min(cos):.4f}, max={max(cos):.4f}")
        if layer == "stem_w":
            assert min(cos) > 0.85

    # 3. Empty / all-NaN input guardrails (LFP and NMC811)
    lfp = cs.CellSense(chemistry="lfp")
    nmc = cs.CellSense(chemistry="nmc811")
    nan250 = np.full(250, np.nan)
    r_lfp_nan = lfp.diagnose(nan250, nan250)
    r_lfp_zero = lfp.diagnose(np.zeros(250), np.full(250, 30.0))
    r_nmc_nan = nmc.diagnose(nan250)
    print(
        f"\nEmpty/all-NaN silent outputs:\n"
        f"  LFP all-NaN : q_total={r_lfp_nan['q_total']:.4f} +/- {r_lfp_nan['q_total_sig']:.4f} Ah, lli={r_lfp_nan['lli']:.4f}\n"
        f"  LFP zero-Q  : q_total={r_lfp_zero['q_total']:.4f} +/- {r_lfp_zero['q_total_sig']:.4f} Ah\n"
        f"  NMC all-NaN : q_total={r_nmc_nan['q_total']:.4f} +/- {r_nmc_nan['q_total_sig']:.4f} Ah, lli={r_nmc_nan['lli']:.4f}"
    )
    assert np.isfinite(r_lfp_nan["q_total"]) and np.isfinite(r_nmc_nan["q_total"])

    # 4. NMC811 endrule boundary discontinuity at 2.50 V (index 249)
    ex_nmc = np.load(mdir / "examples/m50t_W9.npz")
    Q_all = ex_nmc["Q"]
    q_full = Q_all[5].copy()
    q_ref = Q_all[0].copy()
    r_full = nmc.diagnose(q_full, q_ref)
    q_miss_last = q_full.copy()
    q_miss_last[-1] = np.nan
    r_miss = nmc.diagnose(q_miss_last, q_ref)
    dq_mAh = (r_miss["q_total"] - r_full["q_total"]) * 1000.0
    sig_ratio = r_miss["q_total_sig"] / r_full["q_total_sig"]
    print(
        f"\nNMC811 endrule boundary (m50t_W9[5]):\n"
        f"  full curve     : q_total={r_full['q_total']:.4f} +/- {r_full['q_total_sig']:.4f} Ah\n"
        f"  drop last 2.50V: q_total={r_miss['q_total']:.4f} +/- {r_miss['q_total_sig']:.4f} Ah "
        f"(dq={dq_mAh:+.2f} mAh, sd ratio={sig_ratio:.2f}x)"
    )
    assert sig_ratio > 5.0

    # 5. Ratchet promotion-rule arithmetic replay (LFP 9/9 + NMC811 13/13)
    r_lfp = json.loads((mdir / "training/ratchet.json").read_text())
    inc_obj = None
    inc_dev = None
    inc_life = None
    for idx, r in enumerate(r_lfp):
        obj = r["objective"]
        dev_m = r["dev"]["q_total_mae_mAh"]
        dl = r["dev_life"]["mpe_all"] if r.get("dev_life") else None
        if inc_obj is None:
            calc = True
        else:
            ok_obj = (obj - inc_obj) > 0.01
            ok_dev = dev_m <= inc_dev + 1.0
            ok_life = True if (idx < 5 or inc_life is None or dl is None) else (dl <= inc_life + 2.0)
            calc = ok_obj and ok_dev and ok_life
        assert calc == r["accepted"], f"LFP rung {idx+1} mismatch: {calc} vs {r['accepted']}"
        if r["accepted"]:
            inc_obj, inc_dev = obj, dev_m
            if dl is not None:
                inc_life = dl
    print(f"\nLFP ratchet replay: {len(r_lfp)}/{len(r_lfp)} rungs match.")

    r_nmc = json.loads((mdir / "training/nmc_ratchet.json").read_text())["rungs"]
    inc_s = inc_dp = inc_df = None
    for r in r_nmc:
        s = r["sim"]["ocpd"]
        dp = r["dev"]["q_partial_mAh"]
        df = r["dev"]["q_full_mAh"]
        if inc_s is None:
            calc = "kept"
        else:
            calc = "kept" if ((s - inc_s) > 0.01 and dp <= inc_dp * 1.05 and df <= inc_df + 2.0) else "rejected"
        assert calc == r["decision"], f"NMC rung {r['id']} mismatch: {calc} vs {r['decision']}"
        if r["decision"] == "kept":
            inc_s, inc_dp, inc_df = s, dp, df
    print(f"NMC811 ratchet replay: {len(r_nmc)}/{len(r_nmc)} rungs match (n12 margin=+0.0102 vs +0.01).")
    print("CELLSENSE PROBE COMPLETE.")


if __name__ == "__main__":
    main()
