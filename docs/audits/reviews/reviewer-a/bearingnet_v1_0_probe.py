"""First-pass 10-point forensic audit probe for BearingNet v1.0 (r2).

Run:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/bearingnet_v1_0_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/bearingnet"))
import bearingnet as bn
import pipeline as PL


def main():
    print("=== BEARINGNET v1.0 (r2) FORENSIC AUDIT PROBE ===")
    w_path = REPO / "models/bearingnet/bearingnet_weights.npz"
    w = np.load(w_path, allow_pickle=True)
    meta = json.loads(str(w["meta"]))

    # 1. Weights, finiteness, dead output filters, calibration factors
    dead_total = 0
    min_norm = 1e9
    for k in w.files:
        if k == "meta":
            continue
        arr = w[k]
        assert np.all(np.isfinite(arr)), f"Non-finite in {k}"
        if k.endswith("_w"):
            norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
            dead_total += int((norms < 1e-4).sum())
            min_norm = min(min_norm, float(norms.min()))
    cal = meta["calib"]
    print(f"Dead filters={dead_total}, min_norm={min_norm:.4f}, calib={cal}")
    assert dead_total == 0 and cal["speed"] >= 1.0 and cal["size"] >= 1.0 and cal["temp"] >= 1.0

    # 2. Shipped 14 examples & folded PyTorch vs NumPy parity
    net = bn.BearingNet()
    ex_files = sorted((REPO / "models/bearingnet/examples").glob("*.npz"))
    SP_list, SC_list = [], []
    correct = 0
    healthy_as_or = 0
    for f in ex_files:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        x, fs = z["x"], float(z["fs"])
        r = net.read(x, fs=fs, bearing=m["bearing"], rpm_nominal=m["rpm_nominal"])
        g = bn.BEARINGS[m["bearing"]] if isinstance(m["bearing"], str) else m["bearing"]
        xp = PL.pad_short(PL.to_fs(x, fs))
        s0 = PL.windows(xp, 1)[0]
        f0, sc0, _ = PL.window_features(xp[s0 : s0 + PL.NWIN], g, m["rpm_nominal"] / 60.0)
        SP_list.append(f0)
        SC_list.append(sc0)
        correct += int(r["condition"] == m["truth"])
        if m["truth"] == "N" and r["condition"] == "OR":
            healthy_as_or += 1
    SP = np.stack(SP_list)
    SC = np.stack(SC_list)
    print(f"14 shipped examples: {correct}/14 right; healthy-truth misclassified as OR: {healthy_as_or}/4")
    assert healthy_as_or == 4

    def run_torch_bn(sp_np, sc_np):
        x = torch.from_numpy(sp_np.astype(np.float32))
        x = torch.nn.functional.gelu(
            torch.nn.functional.conv1d(x, torch.from_numpy(w["stem_w"]), torch.from_numpy(w["stem_b"]), stride=2, padding=3)
        )
        for i in range(meta["n_blocks"]):
            y = torch.nn.functional.gelu(
                torch.nn.functional.conv1d(x, torch.from_numpy(w[f"b{i}_c1_w"]), torch.from_numpy(w[f"b{i}_c1_b"]), stride=2, padding=2)
            )
            y = torch.nn.functional.conv1d(y, torch.from_numpy(w[f"b{i}_c2_w"]), torch.from_numpy(w[f"b{i}_c2_b"]), padding=2)
            sk = torch.nn.functional.conv1d(x, torch.from_numpy(w[f"b{i}_sk_w"]), torch.from_numpy(w[f"b{i}_sk_b"]), stride=2)
            x = torch.nn.functional.gelu(y + sk)
        h = torch.cat([x.mean(dim=-1), x.max(dim=-1).values, torch.from_numpy(sc_np.astype(np.float32))], dim=-1)
        h = torch.nn.functional.gelu(torch.nn.functional.linear(h, torch.from_numpy(w["h0_w"]), torch.from_numpy(w["h0_b"])))
        h = torch.nn.functional.gelu(torch.nn.functional.linear(h, torch.from_numpy(w["h1_w"]), torch.from_numpy(w["h1_b"])))
        return torch.nn.functional.linear(h, torch.from_numpy(w["h2_w"]), torch.from_numpy(w["h2_b"])).numpy()

    d_real = float(np.max(np.abs(run_torch_bn(SP, SC) - net.forward(SP, SC))))
    rng = np.random.default_rng(0)
    SPr = rng.standard_normal((8, SP.shape[1], SP.shape[2])).astype(np.float32)
    SCr = rng.standard_normal((8, SC.shape[1])).astype(np.float32)
    d_rand = float(np.max(np.abs(run_torch_bn(SPr, SCr) - net.forward(SPr, SCr))))
    print(f"Folded Torch vs NumPy parity: 14 real={d_real:.2e}, 8 rand={d_rand:.2e}")
    assert max(d_real, d_rand) < 1e-5

    # 3. Single-NaN silent 'N' (Healthy) bug, dead-sensor zero -> OR(0.60), white noise -> OR, short-16 -> N
    z0 = np.load(REPO / "models/bearingnet/examples/03_CWRU_135_OR.npz")
    m0 = json.loads(str(z0["meta"]))
    x0, fs0 = z0["x"].astype(float), float(z0["fs"])
    r_base = net.read(x0, fs=fs0, bearing=m0["bearing"], rpm_nominal=m0["rpm_nominal"])

    x_1nan = x0.copy()
    x_1nan[100] = np.nan
    r_1nan = net.read(x_1nan, fs=fs0, bearing=m0["bearing"], rpm_nominal=m0["rpm_nominal"])
    r_allnan = net.read(np.full_like(x0, np.nan), fs=fs0, bearing=m0["bearing"], rpm_nominal=m0["rpm_nominal"])
    r_zero = net.read(np.zeros_like(x0), fs=fs0, bearing=m0["bearing"], rpm_nominal=m0["rpm_nominal"])
    r_noise = net.read(np.random.default_rng(7).standard_normal(len(x0)), fs=fs0, bearing=m0["bearing"], rpm_nominal=m0["rpm_nominal"])
    r_short16 = net.read(x0[:16], fs=fs0, bearing=m0["bearing"], rpm_nominal=m0["rpm_nominal"])

    print(
        f"\n03_CWRU_135_OR (n={len(x0)} samples) guardrails:\n"
        f"  base      : {r_base['condition']} (p={r_base['p'][r_base['condition']]:.3f}, rpm={r_base['rpm']:.1f})\n"
        f"  one_nan   : {r_1nan['condition']} (p={r_1nan['p']}, rpm={r_1nan['rpm']})\n"
        f"  all_nan   : {r_allnan['condition']} (p={r_allnan['p']}, rpm={r_allnan['rpm']})\n"
        f"  all_zero  : {r_zero['condition']} (p_OR={r_zero['p']['OR']:.3f}, rpm={r_zero['rpm']:.1f})\n"
        f"  white_noi : {r_noise['condition']} (p_OR={r_noise['p']['OR']:.3f})\n"
        f"  short_16  : {r_short16['condition']} (p_N={r_short16['p']['N']:.3f})"
    )
    assert r_base["condition"] == "OR"
    assert r_1nan["condition"] == "N" and np.isnan(r_1nan["p"]["N"])
    assert r_zero["condition"] == "OR" and r_zero["p"]["OR"] > 0.55

    # 4. Ratchet replay (4/4 rungs) & seeding check
    rj = json.loads((REPO / "models/bearingnet/training/ratchet.json").read_text())
    rungs = rj if isinstance(rj, list) else rj["rungs"]
    inc_obj = inc_dev = None
    for row in rungs:
        obj = float(row["objective"])
        dev_acc = float(row["dev"]["bal_acc"]) if isinstance(row.get("dev"), dict) and "bal_acc" in row["dev"] else None
        if inc_obj is None:
            calc = True
        else:
            ok_obj = (obj - inc_obj) > 0.01
            ok_dev = True if (dev_acc is None or inc_dev is None) else (dev_acc >= inc_dev - 0.02)
            calc = ok_obj and ok_dev
        assert calc == row["accepted"], f"Mismatch on {row['ckpt']}: {calc} vs {row['accepted']}"
        if row["accepted"]:
            inc_obj = obj
            if dev_acc is not None:
                inc_dev = dev_acc
    bt_txt = (REPO / "models/bearingnet/training/bn_train.py").read_text()
    assert "torch.manual_seed(a.seed)" in bt_txt and "rng.bit_generator.state = r['rng']" in bt_txt
    print(f"\nRatchet replay: {len(rungs)}/{len(rungs)} rungs match; bn_train.py seeds Torch+NumPy and restores RNG on resume.")
    print("BEARINGNET PROBE COMPLETE.")


if __name__ == "__main__":
    main()
