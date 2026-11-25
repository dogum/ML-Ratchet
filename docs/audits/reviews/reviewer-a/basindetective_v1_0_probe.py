"""First-pass 10-point forensic audit probe for BasinDetective v1.0 (R5).

Run:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/basindetective_v1_0_probe.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/basindetective"))
sys.path.insert(0, str(REPO / "models/basindetective/training"))

import basindetective as bd
import train as bd_train


def main():
    print("=== BASINDETECTIVE v1.0 (R5) FORENSIC AUDIT PROBE ===")
    det = bd.load()
    n_m = len(det.members)
    fnames = det.names
    nt = len(det.targets)

    # 1. Weights, dead rows, normalization stats, ensemble diversity
    dead_total = 0
    min_norm = 1e9
    iqr_min, iqr_max = 1e9, 0.0
    for i, m in enumerate(det.members):
        for k, v in m.W.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                assert np.all(np.isfinite(v)), f"Non-finite in m{i}_{k}"
        for k in ("w0", "w1", "wo"):
            norms = np.linalg.norm(m.W[k], axis=1)
            dead_total += int((norms < 1e-4).sum())
            min_norm = min(min_norm, float(norms.min()))
        iqr_min = min(iqr_min, float(m.W["x_iqr"].min()))
        iqr_max = max(iqr_max, float(m.W["x_iqr"].max()))
    cosines = []
    for i in range(n_m):
        for j in range(i + 1, n_m):
            a = det.members[i].W["w0"].ravel()
            b = det.members[j].W["w0"].ravel()
            cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    print(
        f"Members={n_m}, dead_rows={dead_total}, min_row_norm={min_norm:.4f}, "
        f"x_iqr=[{iqr_min:.5f}, {iqr_max:.5f}], w0_cos=[{min(cosines):.4f}, {max(cosines):.4f}]"
    )
    print(f"sig_scale={det.sig_scale.tolist()}, cls_temp={det.cls_temp}")
    assert dead_total == 0 and max(cosines) < 0.10 and np.all(det.sig_scale >= 1.0)

    # 2. Shipped examples vs expected.json & real Z spread
    exp = json.loads((REPO / "models/basindetective/examples/expected.json").read_text())
    cal_files = sorted(glob.glob(str(REPO / "models/basindetective/examples/*_cal.csv")))
    X_real = []
    worst_exp = 0.0
    examples = {}
    for f in cal_files:
        b = Path(f).name[:-8]
        cal = pd.read_csv(f, index_col=0, parse_dates=True)
        tgt = pd.read_csv(f.replace("_cal.csv", "_tgt.csv"), index_col=0, parse_dates=True)
        r = bd.predict(det, cal[["P", "T_mean", "PET"]], cal["Q_mm"].values, tgt)
        x = bd.featurize(cal[["P", "T_mean", "PET"]], cal["Q_mm"].values, tgt)[0]
        X_real.append(x)
        examples[b] = (cal, tgt, r)
        for t in r["changes"]:
            worst_exp = max(worst_exp, abs(r["changes"][t]["mean"] - exp[b][t]))
    X_real = np.stack(X_real, 0)
    Z_real = det.members[0].norm(X_real)
    clipped = int((np.abs(Z_real) >= 6.0).sum())
    print(f"Examples vs expected.json: max|diff|={worst_exp:.2e}; real Z=[{Z_real.min():.3f}, {Z_real.max():.3f}], clipped={clipped}/{Z_real.size}")
    assert worst_exp < 1e-6 and clipped == 0

    # 3. 5-member PyTorch vs NumPy parity (float64 and float32)
    rng = np.random.default_rng(42)
    X_rand = det.members[0].W["x_med"] + rng.standard_normal((16, len(fnames))) * det.members[0].W["x_iqr"]
    X_test = np.vstack([X_real, X_rand])
    max_f64 = 0.0
    max_f32 = 0.0
    for m in det.members:
        H = m.W["w0"].shape[0]
        nl = int(m.W["n_layers"])
        Z = m.norm(X_test)
        on = m.raw(X_test)
        for dtype, is64 in ((torch.float64, True), (torch.float32, False)):
            net = bd_train.MLP(len(fnames), nt, H, nl, 0.0).to(dtype)
            for l in range(nl):
                net.hs[l].weight.data.copy_(torch.from_numpy(m.W[f"w{l}"]).to(dtype))
                net.hs[l].bias.data.copy_(torch.from_numpy(m.W[f"b{l}"]).to(dtype))
            net.out.weight.data.copy_(torch.from_numpy(m.W["wo"]).to(dtype))
            net.out.bias.data.copy_(torch.from_numpy(m.W["bo"]).to(dtype))
            net.eval()
            with torch.no_grad():
                ot = net(torch.from_numpy(Z).to(dtype)).numpy()
            d = float(np.max(np.abs(ot - on)))
            if is64:
                max_f64 = max(max_f64, d)
            else:
                max_f32 = max(max_f32, d)
    print(f"5-member Torch<->NumPy raw parity: float64={max_f64:.2e}, float32={max_f32:.2e}")
    assert max_f64 < 1e-12 and max_f32 < 1e-5

    # 4. Identical-climate counterfactual (tgt == cal): measures baked-in ERA5 backward drift prior
    print("\nIdentical-climate (tgt == cal) drift check:")
    for b, (cal, _, _) in examples.items():
        rid = bd.predict(det, cal[["P", "T_mean", "PET"]], cal["Q_mm"].values, cal[["P", "T_mean", "PET"]])
        dlq = rid["changes"]["d_logq"]["mean"]
        drr = rid["changes"]["d_rr"]["mean"]
        dlq5 = rid["changes"]["d_logq5"]["mean"]
        dhfd = rid["changes"]["d_hfd"]["mean"]
        print(f"  {b}: d_logq={dlq:+.4f}, d_rr={drr:+.4f}, d_logq5={dlq5:+.4f}, d_hfd={dhfd:+.2f}d")
        assert dlq < 0.0 and drr < 0.0

    # 5. Physical direction & API guardrail checks on camels_04185000
    cal0, tgt0, r0 = examples["camels_04185000"]
    tgt_p20 = tgt0.copy()
    tgt_p20["P"] = tgt_p20["P"] * 1.20
    r_p20 = bd.predict(det, cal0[["P", "T_mean", "PET"]], cal0["Q_mm"].values, tgt_p20)
    r_const = bd.predict(det, cal0[["P", "T_mean", "PET"]], np.full(len(cal0), np.nanmean(cal0["Q_mm"].values)), tgt0)
    print(
        f"\nPhysical response (camels_04185000): +20% P d_logq {r0['changes']['d_logq']['mean']:+.4f} -> "
        f"{r_p20['changes']['d_logq']['mean']:+.4f}; constant Q p_reg {r0['regulation']['p_regulated']:.4f} -> "
        f"{r_const['regulation']['p_regulated']:.4f}"
    )
    assert r_p20["changes"]["d_logq"]["mean"] > r0["changes"]["d_logq"]["mean"] + 0.3
    assert r_const["regulation"]["p_regulated"] > 0.4

    # API guardrails: all-NaN Q and <365d crash with KeyError('bfi_lh'); all-zero Q returns silently
    for label, q_in, n_rows in [
        ("all_nan_Q", np.full(len(cal0), np.nan), len(cal0)),
        ("short_30d", cal0["Q_mm"].values[:30], 30),
        ("short_180d", cal0["Q_mm"].values[:180], 180),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                bd.predict(det, cal0[["P", "T_mean", "PET"]].iloc[:n_rows], q_in, tgt0.iloc[:n_rows])
                raise AssertionError(f"Expected KeyError for {label}")
            except KeyError as e:
                print(f"  guardrail {label:12s}: unhandled KeyError({e})")
                assert str(e) == "'bfi_lh'"

    r_zero = bd.predict(det, cal0[["P", "T_mean", "PET"]], np.zeros(len(cal0)), tgt0)
    print(
        f"  guardrail all_zero_Q  : silently returned d_logq={r_zero['changes']['d_logq']['mean']:+.4f} "
        f"+/- {r_zero['changes']['d_logq']['sd']:.4f}, p_reg={r_zero['regulation']['p_regulated']:.4f}"
    )
    assert np.isfinite(r_zero["changes"]["d_logq"]["mean"])

    # 6. Ratchet arithmetic replay (7/7 rungs)
    rj = json.loads((REPO / "models/basindetective/training/ratchet.json").read_text())
    # Re-scores on updated physics are logged in the rung reasons; verify all 7 decisions
    rescore_inc = {"R1b": 0.5464, "R1e": 0.5464, "R2": 0.4966, "R3": 0.5263, "R4": 0.5082, "R5": 0.5082}
    inc_dev = None
    for row in rj["rungs"]:
        nm = row["name"]
        if inc_dev is None:
            calc = True
        else:
            sim_ok = (row["sim_skill"] - rescore_inc[nm]) > 0.01
            dev_ok = row["dev_composite"] >= inc_dev - 0.02
            calc = sim_ok and dev_ok
        assert calc == row["kept"], f"Ratchet mismatch on {nm}: {calc} vs {row['kept']}"
        if row["kept"]:
            inc_dev = row["dev_composite"]
    print(f"\nRatchet replay: {len(rj['rungs'])}/{len(rj['rungs'])} rungs match published rule.")
    print("BASINDETECTIVE PROBE COMPLETE.")


if __name__ == "__main__":
    main()
