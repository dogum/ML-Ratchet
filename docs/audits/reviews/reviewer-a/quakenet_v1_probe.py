"""First-pass 10-point forensic audit probe for QuakeNet v1 (q7_zrt).

Run:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/quakenet_v1_probe.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/quakenet"))
import pipeline as qpl
import quakenet as qn


def main():
    print("=== QUAKENET v1 (q7_zrt) FORENSIC AUDIT PROBE ===")
    w = np.load(REPO / "models/quakenet/quakenet_weights.npz", allow_pickle=True)
    meta = json.loads(str(w["meta"]))

    # 1. Weights, finiteness, dead filters, widen-only sigma factors
    dead_total = 0
    min_norm = 1e9
    for k in w.files:
        if k == "meta":
            continue
        arr = w[k]
        assert np.all(np.isfinite(arr))
        if k.endswith("_w") and arr.ndim >= 2:
            norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
            dead_total += int((norms < 1e-4).sum())
            min_norm = min(min_norm, float(norms.min()))
    sf = meta["sigma_factor"]
    print(f"Dead filters={dead_total}, min_norm={min_norm:.4f}, sigma_factor={sf}")
    assert dead_total == 0 and all(v >= 1.0 for v in sf.values())

    # 2. Shipped 4 examples, 90-degree horizontal rotation equivariance, and folded Torch parity
    net = qn.QuakeNet()
    ex_files = sorted(glob.glob(str(REPO / "models/quakenet/examples/*.npz")))
    W_list, S_list = [], []
    max_rot_res = 0.0
    for f in ex_files:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        X, pg = z["zne"].astype(float), m["p_guess"]
        r0 = net.read(X, pg)
        W, ls, _, peak = qpl.read(X, pg)
        W_list.append(W)
        S_list.append([(ls + 6.0) / 2.0, np.log10(max(float(peak), 1.0))])
        # 90-deg CW horizontal rotation (N, E) -> (-E, N): expected baz shift +90 deg
        Xr = X.copy()
        Xr[1], Xr[2] = -X[2], X[1]
        rr = net.read(Xr, pg)
        res90 = ((rr["baz"] - (r0["baz"] + 90.0) + 180.0) % 360.0) - 180.0
        max_rot_res = max(max_rot_res, abs(res90))
        r60 = net.read(X, pg + 60)
        print(
            f"  {Path(f).stem:22s}: Mw={r0['mw']:.2f}({m['usgs_mw']:.1f}) dist={r0['dist']:.1f}({m['dist_deg']:.1f}) "
            f"baz={r0['baz']:.1f} | rot90_res={res90:+.1f}deg | p_guess+60s Mw={r60['mw']:.2f} dist={r60['dist']:.1f}"
        )
    assert max_rot_res < 5.0

    W_arr = np.stack(W_list)
    S_arr = np.array(S_list, np.float32)

    def run_torch_qn(x_np, s_np):
        env = np.log10(
            np.stack([np.convolve(np.pad(np.abs(c), 4), np.ones(9) / 9, "valid") for c in np.abs(x_np).reshape(-1, x_np.shape[-1])])
            .reshape(x_np.shape)
            + 1e-3
        )
        h = torch.from_numpy(np.concatenate([x_np, env], 1).astype(np.float32))
        for i in range(meta["n_blocks"]):
            y = torch.nn.functional.gelu(
                torch.nn.functional.conv1d(
                    h, torch.from_numpy(w[f"b{i}_c1_w"]), torch.from_numpy(w[f"b{i}_c1_b"]), stride=2, padding=w[f"b{i}_c1_w"].shape[2] // 2
                )
            )
            y = torch.nn.functional.conv1d(
                y, torch.from_numpy(w[f"b{i}_c2_w"]), torch.from_numpy(w[f"b{i}_c2_b"]), padding=w[f"b{i}_c2_w"].shape[2] // 2
            )
            sk = torch.nn.functional.conv1d(h, torch.from_numpy(w[f"b{i}_sk_w"]), torch.from_numpy(w[f"b{i}_sk_b"]), stride=2)
            h = torch.nn.functional.gelu(y + sk)
        z = torch.cat([h.mean(-1), h.max(-1).values, torch.from_numpy(s_np.astype(np.float32))], 1)
        for j in range(meta["n_head"]):
            z = torch.nn.functional.linear(z, torch.from_numpy(w[f"h{j}_w"]), torch.from_numpy(w[f"h{j}_b"]))
            if j < meta["n_head"] - 1:
                z = torch.nn.functional.gelu(z)
        return z.numpy()

    d_real = float(np.max(np.abs(run_torch_qn(W_arr, S_arr) - net.forward(W_arr, S_arr))))
    rng = np.random.default_rng(0)
    Wr = rng.standard_normal((4, 3, 2460)).astype(np.float32)
    Sr = rng.standard_normal((4, 2)).astype(np.float32)
    d_rand = float(np.max(np.abs(run_torch_qn(Wr, Sr) - net.forward(Wr, Sr))))
    print(f"Folded Torch vs NumPy parity: 4 real={d_real:.2e}, 4 rand={d_rand:.2e}")
    assert d_real < 5e-5 and d_rand < 1e-5

    # 3. API guardrails: all-zero, white noise, single-NaN, all-NaN
    z0 = np.load(ex_files[0])
    m0 = json.loads(str(z0["meta"]))
    X0, pg0 = z0["zne"].astype(float), m0["p_guess"]
    r_zero = net.read(np.zeros_like(X0), pg0)
    r_wn = net.read(np.random.default_rng(7).standard_normal(X0.shape) * 1e-6, pg0)
    X_1nan = X0.copy()
    X_1nan[0, 100] = np.nan
    r_1nan = net.read(X_1nan, pg0)
    print(
        f"Guardrails:\n"
        f"  all_zero   : Mw={r_zero['mw']:.2f} +/- {r_zero['mw_sig']:.2f}, dist={r_zero['dist']:.1f} deg\n"
        f"  white_noise: Mw={r_wn['mw']:.2f} +/- {r_wn['mw_sig']:.2f}, dist={r_wn['dist']:.1f} deg, z={r_wn['depth']:.0f} km\n"
        f"  one_nan    : Mw={r_1nan['mw']}, dist={r_1nan['dist']}, baz={r_1nan['baz']}"
    )
    assert r_zero["mw"] < 0 and r_zero["dist"] < 0
    assert r_wn["mw"] > 5.5 and np.isnan(r_1nan["mw"])

    # 4. Trainer seeding & resume check
    tr_txt = (REPO / "models/quakenet/training/train.py").read_text()
    assert "torch.manual_seed(a.seed)" in tr_txt and "rng.bit_generator.state = r['rng']" in tr_txt
    print("Seeding check: training/train.py seeds Torch + NumPy and restores RNG state on resume.")
    print("QUAKENET v1 PROBE COMPLETE.")


if __name__ == "__main__":
    main()
