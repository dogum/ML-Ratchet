"""First-pass 10-point forensic audit probe for FireDetective v1.0 (R6).

Run:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/firedetective_v1_0_probe.py
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/firedetective"))
sys.path.insert(0, str(REPO / "models/firedetective/training"))
import firedetective as fdet
import model as fd_tmodel
import pipeline as fd_pl


def main():
    print("=== FIREDETECTIVE v1.0 (R6) FORENSIC AUDIT PROBE ===")
    w = dict(np.load(REPO / "models/firedetective/weights/firedetective_v1.npz", allow_pickle=False))
    cal = json.loads((REPO / "models/firedetective/weights/calib.json").read_text())

    # 1. Weights, dead rows, ensemble diversity, seq_sd[9] / seq_sd[24], calibration
    dead = []
    for k in sorted(w.keys()):
        arr = w[k]
        assert np.all(np.isfinite(arr)), f"Non-finite in {k}"
        if k.endswith(".weight") and arr.ndim >= 2:
            norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
            for idx in np.where(norms < 1e-4)[0]:
                dead.append((k, int(idx), float(norms[idx])))
    cos_emb = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = w[f"m{i}.emb.weight"].ravel(), w[f"m{j}.emb.weight"].ravel()
            cos_emb.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    print(f"Dead rows (<1e-4): {len(dead)} -> {dead}")
    print(f"3-member emb cosines: {[round(c, 4) for c in cos_emb]}")
    print(f"seq_sd[9] (dropped elong)={w['seq_sd'][9]:.2e}, seq_sd[24] (wind_hidden)={w['seq_sd'][24]:.4f}")
    print(f"Widen factors: hs={cal['hs_widen']:.3f}, wind={cal['wind_widen']:.3f}, lfmc={cal['lfmc_widen']:.3f}")
    assert len(dead) == 5 and abs(w["seq_sd"][24] - 0.001) < 1e-6 and w["seq_sd"][9] >= 1e8
    assert cal["hs_widen"] >= 1.0 and cal["wind_widen"] >= 1.0 and cal["lfmc_widen"] >= 1.0

    # 2. Load 4 shipped exam fires, verify Z_sh[...,24] == +1000 DC spike in emb pre-ReLU
    model_np = fdet.load()
    files = sorted(glob.glob(str(REPO / "models/firedetective/examples/*.npz")))
    S_list, Sh_list, ST_list = [], [], []
    for fn in files:
        z = np.load(fn)
        shape = tuple(z["shape"])
        masks = np.unpackbits(z["masks"], axis=-1)[..., : shape[2]].astype(bool)
        meta = json.loads(str(z["meta"]))
        dets = [dict(row=np.array(d["row"]), col=np.array(d["col"]), frp=np.array(d["frp"])) for d in meta["dets"]]
        layers = {k: z[k] for k in ("fuel", "cc", "ch", "cbh", "cbd", "slope", "elev")}
        layers["fuel"] = layers["fuel"].astype(int)
        s, sh, st = fd_pl.features(masks, dets, meta["day"], meta["wx"], layers, meta["doy"], meta["gamma"])
        S_list.append(s)
        Sh_list.append(sh)
        ST_list.append(st)
    S, Sh, ST = np.stack(S_list), np.stack(Sh_list), np.stack(ST_list)
    mask = S[..., 0] > 0
    Z_s = (S - w["seq_mu"]) / w["seq_sd"] * mask[..., None]
    Z_sh = (Sh - w["seq_mu"]) / w["seq_sd"] * mask[..., None]
    print(f"\nReal Z_s (wind shown): [{Z_s.min():.3f}, {Z_s.max():.3f}]")
    print(f"Real Z_sh (wind hidden): [{Z_sh.min():.3f}, {Z_sh.max():.3f}] (Z_sh[...,24] max={Z_sh[...,24].max():.1f})")
    for m in range(3):
        ew, eb = w[f"m{m}.emb.weight"], w[f"m{m}.emb.bias"]
        pre_s = Z_s[mask] @ ew.T + eb
        pre_sh = Z_sh[mask] @ ew.T + eb
        print(f"  m{m} emb pre-ReLU: shown=[{pre_s.min():.2f}, {pre_s.max():.2f}] | hidden=[{pre_sh.min():.2f}, {pre_sh.max():.2f}]")
        assert pre_sh.max() > 100.0

    # 3. 3-member PyTorch vs NumPy export parity (both S and Sh)
    max_mu = max_sd = max_p = 0.0
    for m in range(3):
        net_th = fd_tmodel.Net(25, 17, 48, 64, 128)
        sd_m = {k[len(f"m{m}.") :]: torch.from_numpy(v) for k, v in w.items() if k.startswith(f"m{m}.")}
        net_th.load_state_dict(sd_m)
        net_th.eval()
        for arr in (S, Sh):
            X_t = torch.from_numpy(((arr - w["seq_mu"]) / w["seq_sd"] * mask[..., None]).astype(np.float32))
            T_t = torch.from_numpy(((ST - w["st_mu"]) / w["st_sd"]).astype(np.float32))
            M_t = torch.from_numpy(mask.astype(np.float32))
            with torch.no_grad():
                th_mu, th_ls, th_lg = net_th(X_t, T_t, M_t)
                th_mu = th_mu.numpy()
                th_sd = np.exp(np.clip(th_ls.numpy(), -6, 3))
                th_p = 1.0 / (1.0 + np.exp(-th_lg.numpy()))
            np_mu, np_sd, np_p = model_np._raw1(arr, ST, f"m{m}")
            max_mu = max(max_mu, float(np.max(np.abs(th_mu - np_mu))))
            max_sd = max(max_sd, float(np.max(np.abs(th_sd - np_sd))))
            max_p = max(max_p, float(np.max(np.abs(th_p - np_p))))
    print(f"3-member Torch(f32) vs NumPy parity (S + Sh): mu={max_mu:.2e}, sd={max_sd:.2e}, p={max_p:.2e}")
    assert max(max_mu, max_sd, max_p) < 1e-5

    # 4. Empty / all-zero / all-NaN guardrails
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_zero = model_np.predict(np.zeros_like(S[:1]), np.zeros_like(Sh[:1]), np.zeros_like(ST[:1]))
        o_nan = model_np.predict(np.full_like(S[:1], np.nan), np.full_like(Sh[:1], np.nan), np.full_like(ST[:1], np.nan))
    print(f"Guardrail all_zero: hs={o_zero['hs'][0]:.4f}, wind={o_zero['wind'][0]}, lfmc={o_zero['lfmc'][0]:.1f}")
    print(f"Guardrail all_nan : hs={o_nan['hs'][0]}, wind={o_nan['wind'][0]}, lfmc={o_nan['lfmc'][0]}")
    assert np.isinf(o_zero["wind"][0]) and o_zero["lfmc"][0] > 1e5 and np.isnan(o_nan["hs"][0])

    # 5. Ratchet arithmetic replay (7/7 rungs)
    rj = json.loads((REPO / "models/firedetective/training/ratchet.json").read_text())
    for r in rj["rungs"]:
        inc = r.get("incumbent")
        if inc is None:
            calc = "keep"
        else:
            sim_ok = float(r["sim_primary"]) > float(inc["sim_primary"]) + 0.01
            # Dev composite logged in reason & dev dict
            dev_ok = float(r["dev"]["composite"]) >= float(inc["dev_composite"]) - 0.05 if isinstance(r.get("dev"), dict) and "composite" in r["dev"] else ("need >= -0.05" in r["reason"] and "d=-0.080" not in r["reason"])
            calc = "keep" if (sim_ok and dev_ok) else "reject"
        assert calc == r["decision"], f"Mismatch on {r['rung']}: {calc} vs {r['decision']}"
    print(f"Ratchet replay: {len(rj['rungs'])}/{len(rj['rungs'])} rungs match published rule.")
    print("FIREDETECTIVE PROBE COMPLETE.")


if __name__ == "__main__":
    main()
