"""Export the released WakeNet checkpoint to standalone numpy weights (wakenet_weights.npz),
fit widen-only 68% error-bar factors on Kelmarsh 2019 dev, draw fixed-seed stratified real
exam examples for demo.py, and verify numpy == PyTorch to < 1e-5.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import _paths as _P  # noqa: E402
from mlratchet.calibrate import coverage, widen_factor  # noqa: E402
from mlratchet.nn import max_abs_diff  # noqa: E402
from model import EdgeWakeGNN  # noqa: E402
import wakenet as WN  # noqa: E402


def export_weights():
    lad = yaml.safe_load((Path(MODEL) / "ratchet.yaml").read_text(encoding="utf-8"))["ladders"][0]
    rel = next(r for r in lad["rungs"] if r["decision"] == "released")
    rid = rel["id"]

    ckpt = torch.load(_P.work("ckpt", f"{rid}.pt"), map_location="cpu", weights_only=False)
    net = EdgeWakeGNN(**ckpt["arch"])
    net.load_state_dict(ckpt["ema_state"])
    net.eval()

    # Evaluate on Kelmarsh 2019 dev to fit widen-only 68% factors and dev dumb-baseline means
    anchor = ckpt.get("anchor", "textbook")
    dev = np.load(_P.work("banks", "dev_2019_graphs_tuned.npz" if anchor == "tuned" else "dev_2019_graphs.npz"))
    n_dev = len(dev["node_x"])
    eta_p = np.zeros_like(dev["eta_true"])
    sig_eta_raw = np.zeros_like(dev["eta_true"])
    farm_p = np.zeros(n_dev, dtype=np.float32)
    sig_farm_raw = np.zeros(n_dev, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_dev, 256):
            sl = slice(i, min(i + 256, n_dev))
            out = net(
                torch.from_numpy(dev["node_x"][sl]),
                torch.from_numpy(dev["edge_x"][sl]),
                torch.from_numpy(dev["edge_mask"][sl]),
                torch.from_numpy(dev["online"][sl]),
                torch.from_numpy(dev["is_free"][sl]),
                torch.from_numpy(dev["eta_anchor"][sl]),
            )
            eta_p[sl] = out["eta_pred"].numpy()
            sig_eta_raw[sl] = torch.exp(out["log_sig_eta"]).numpy()
            farm_p[sl] = out["farm_wake_pred"].numpy()
            sig_farm_raw[sl] = torch.exp(out["log_sig_farm"]).numpy()

    on = dev["online"]
    free = dev["is_free"]
    waked = on & (~free)

    res_waked = (eta_p[waked] - dev["eta_true"][waked]).astype(np.float64)
    sig_waked = sig_eta_raw[waked].astype(np.float64)
    f_waked = widen_factor(res_waked, sig_waked)

    res_free = (eta_p[free] - dev["eta_true"][free]).astype(np.float64)
    sig_free = sig_eta_raw[free].astype(np.float64)
    f_free = widen_factor(res_free, sig_free)

    res_farm = (farm_p - dev["farm_wake_true"]).astype(np.float64)
    f_farm = widen_factor(res_farm, sig_farm_raw.astype(np.float64))

    dev_mean_waked_eta = float(np.mean(dev["eta_true"][waked]))
    dev_mean_farm_wake = float(np.mean(dev["farm_wake_true"]))

    meta = {
        "model": "wakenet",
        "version": os.environ.get("WAKENET_VERSION", "1.0"),
        "released_rung": rid,
        "anchor": anchor,
        "n_params": int(ckpt["n_params"]),
        "arch": ckpt["arch"],
        "physics_version": "v3",
        "sigma_factor": {
            "waked_eta": round(f_waked, 4),
            "free_eta": round(f_free, 4),
            "farm_wake": round(f_farm, 4),
        },
        "dev_coverage_raw": {
            "waked_eta": round(100.0 * coverage(res_waked, sig_waked, 1.0), 2),
            "farm_wake": round(100.0 * coverage(res_farm, sig_farm_raw, 1.0), 2),
        },
        "dev_coverage_calibrated": {
            "waked_eta": round(100.0 * coverage(res_waked, sig_waked, f_waked), 2),
            "farm_wake": round(100.0 * coverage(res_farm, sig_farm_raw, f_farm), 2),
        },
        "dev_baselines": {
            "mean_waked_eta": round(dev_mean_waked_eta, 5),
            "mean_farm_wake_loss": round(dev_mean_farm_wake, 5),
        },
    }

    tensors = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in ckpt["ema_state"].items()}
    weights_path = os.path.join(MODEL, "wakenet_weights.npz")
    np.savez(weights_path, meta=json.dumps(meta), **tensors)
    print(f"Exported {weights_path} ({os.path.getsize(weights_path)/1e6:.2f} MB), meta={json.dumps(meta, indent=2)}", flush=True)

    # Parity check: standalone numpy WakeNet vs PyTorch EdgeWakeGNN on 50 dev graphs
    np_net = WN.WakeNet(weights_path)
    worst_eta, worst_sig, worst_farm = 0.0, 0.0, 0.0
    with torch.no_grad():
        for i in range(50):
            nx_i = dev["node_x"][i]
            ex_i = dev["edge_x"][i]
            em_i = dev["edge_mask"][i]
            on_i = dev["online"][i]
            fr_i = dev["is_free"][i]
            an_i = dev["eta_anchor"][i]

            t_out = net(
                torch.from_numpy(nx_i[None]), torch.from_numpy(ex_i[None]),
                torch.from_numpy(em_i[None]), torch.from_numpy(on_i[None]),
                torch.from_numpy(fr_i[None]), torch.from_numpy(an_i[None]),
            )
            n_out = np_net.forward_graph(nx_i, ex_i, em_i, on_i, fr_i, an_i)
            worst_eta = max(worst_eta, max_abs_diff(t_out["eta_pred"][0].numpy(), n_out["eta_pred"]))
            worst_sig = max(worst_sig, max_abs_diff(t_out["log_sig_eta"][0].numpy(), n_out["log_sig_eta"]))
            worst_farm = max(worst_farm, abs(float(t_out["farm_wake_pred"][0].item()) - float(n_out["farm_wake_pred"])))

    parity = {
        "n_graphs_checked": 50,
        "max_abs_diff_eta": worst_eta,
        "max_abs_diff_log_sig_eta": worst_sig,
        "max_abs_diff_farm_wake": worst_farm,
        "pass": max(worst_eta, worst_sig, worst_farm) < 1e-5,
    }
    with open(os.path.join(HERE, "parity_numpy_vs_torch.json"), "w", encoding="utf-8") as f:
        json.dump(parity, f, indent=2)
    print("Parity check (numpy vs PyTorch):", json.dumps(parity, indent=2), flush=True)
    assert parity["pass"], f"Numpy parity failed: {parity}"


def export_stratified_examples():
    """Draw 6 real exam snapshots by fixed seed within input-defined strata (never selected by error)."""
    rng = np.random.default_rng(20261116)
    ex_dir = Path(MODEL) / "examples"
    for old in ex_dir.glob("*.npz"):
        old.unlink()
    for old in ex_dir.glob("*.json"):
        old.unlink()

    zk = np.load(_P.work("proc", "kelmarsh_exam_2020_2021.npz"))
    zp = np.load(_P.work("proc", "penmanshiel_exam_2019_2020.npz"))

    def _pick(z, mask, label: str, tag: str) -> dict:
        cand = np.where(mask)[0]
        assert len(cand) > 0, f"Empty stratum {tag}"
        idx = int(rng.choice(cand))
        farm = str(z["farm"])
        turb_key = "MM92" if farm == "kelmarsh" else "MM82"
        fname = f"{tag}.npz"
        meta = {
            "id": tag,
            "stratum": label,
            "farm": farm,
            "turbine_model": turb_key,
            "timestamp_utc": str(z["ts"][idx]),
            "codes": [str(c) for c in z["codes"].tolist()],
            "u_inf_ms": round(float(z["u_inf"][idx]), 3),
            "wind_dir_deg": round(float(z["wd_inf"][idx]), 2),
            "ti_inf": round(float(z["ti_inf"][idx]), 4),
            "sigma_wd_deg": round(float(z["sigma_wd"][idx]), 2),
            "air_density": round(float(z["air_density"][idx]), 4),
            "p_front_kw": round(float(z["p_front_kw"][idx]), 2),
        }
        np.savez(
            ex_dir / fname,
            meta=json.dumps(meta),
            east_m=z["east_m"].astype(np.float32),
            north_m=z["north_m"].astype(np.float32),
            elev_m=z["elev_m"].astype(np.float32),
            hub_h_m=z["hub_h_m"].astype(np.float32),
            rotor_d_m=z["rotor_d_m"].astype(np.float32),
            online=z["online"][idx],
            is_free=z["is_free"][idx],
            yaw_deg=z["yaw_deg"][idx].astype(np.float32),
            power_true_kw=z["power_kw"][idx].astype(np.float32),
            u_scada_ms=z["u_scada_ms"][idx].astype(np.float32),
        )
        return {"file": fname, **meta}

    k_all_on = np.all(zk["online"], axis=1) & (zk["u_inf"] >= 7.0) & (zk["u_inf"] <= 10.0)
    k_max_free_yaw = np.max(np.where(zk["is_free"], np.abs(zk["yaw_deg"]), 0.0), axis=1)

    p_all_on = np.all(zp["online"], axis=1) & (zp["u_inf"] >= 7.0) & (zp["u_inf"] <= 10.0)
    p_max_free_yaw = np.max(np.where(zp["is_free"], np.abs(zp["yaw_deg"]), 0.0), axis=1)

    manifest = [
        _pick(zk, k_all_on & (zk["wd_inf"] >= 238.0) & (zk["wd_inf"] <= 252.0), "Kelmarsh SW prevailing pair wakes", "01_kelmarsh_sw_wakes"),
        _pick(zk, k_all_on & (zk["wd_inf"] >= 299.0) & (zk["wd_inf"] <= 307.0), "Kelmarsh NW 3-turbine row (KWF3->KWF5->KWF6)", "02_kelmarsh_nw_3row"),
        _pick(zk, k_all_on & (k_max_free_yaw >= 10.0), "Kelmarsh upstream yaw misalignment (>=10 deg)", "03_kelmarsh_yaw_offset"),
        _pick(zk, (np.sum(~zk["online"], axis=1) == 1) & (zk["u_inf"] >= 7.0) & (zk["u_inf"] <= 10.0) & (zk["wd_inf"] >= 235.0) & (zk["wd_inf"] <= 315.0), "Kelmarsh 1 turbine parked/offline", "04_kelmarsh_parked_unit"),
        _pick(zp, p_all_on & (zp["wd_inf"] >= 220.0) & (zp["wd_inf"] <= 245.0), "Penmanshiel (unseen 14-turbine farm) SW deep array", "05_penmanshiel_sw_14wt"),
        _pick(zp, p_all_on & (zp["wd_inf"] >= 295.0) & (zp["wd_inf"] <= 325.0) & (p_max_free_yaw >= 8.0), "Penmanshiel (unseen 14-turbine farm) NW ridge + yaw offset", "06_penmanshiel_nw_yaw"),
    ]
    with open(ex_dir / "examples.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} fixed-seed stratified real exam examples to {ex_dir}", flush=True)


if __name__ == "__main__":
    export_weights()
    export_stratified_examples()
