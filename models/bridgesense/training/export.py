"""Export released BridgeSense checkpoint(s) to pure-numpy bridgesense_weights.npz and verify parity vs PyTorch."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import numpy as np
import torch

from _paths import MODEL, work
from dev_eval import load_kw51_dev_blocks, load_model_ckpt, predict_episode

sys.path.insert(0, MODEL)
from bridgesense import BridgeSense  # noqa: E402


def export_weights(ckpt_names: list[str], cal_path: str | None = None, version: str = "1.1") -> dict:
    cal = json.load(open(cal_path)) if (cal_path and os.path.exists(cal_path)) else {}
    arrays: dict[str, np.ndarray] = {}
    configs: list[dict] = []
    total_params = 0
    for k, name in enumerate(ckpt_names):
        path = name if os.path.exists(name) else work("checkpoints", f"{name}.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # Keep every member's own config (v1.0 kept only the last one, which broke mixed ensembles)
        configs.append(dict(ck["config"]))
        sd = ck["state_dict"]
        for pname, tensor in sd.items():
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            arrays[f"m{k}.{pname}"] = arr
            total_params += arr.size

    meta = {
        "model": "bridgesense",
        "version": version,
        "n_members": len(ckpt_names),
        "params_per_member": total_params // len(ckpt_names),
        "total_params": total_params,
        "config": configs[0],
        "configs": configs,
        "members": [os.path.basename(n) for n in ckpt_names],
        "widen_dk": float(cal.get("widen_dk", 1.0)),
        "widen_env": float(cal.get("widen_env", 1.0)),
        "prob_temp": float(cal.get("prob_temp", 1.0)),
    }
    arrays["meta"] = np.array(json.dumps(meta))
    dst = os.path.join(MODEL, "bridgesense_weights.npz")
    # Write to a temporary file and check parity BEFORE replacing the released weights.
    tmp = work("bridgesense_weights.candidate.npz", mkdir=True)
    np.savez_compressed(tmp, **arrays)

    # Verify numpy == PyTorch on all 10 KW51 Dev blocks
    pt_models = load_model_ckpt(ckpt_names)
    np_model = BridgeSense(tmp)
    dev_blocks = load_kw51_dev_blocks()
    max_diff = 0.0
    for b in dev_blocks:
        r_pt = predict_episode(pt_models, b)
        r_np = np_model.predict(b["T_ref"], b["T_q"], b["df_ref"], b["df_q"], b["mask_ref"], b["mask_q"], raw_bars=True)
        diffs = [
            abs(r_pt["logit"] - r_np["logit_raw"]),
            abs(r_pt["dk_pct"] - r_np["delta_k_pct"]),
            abs(r_pt["dk_sig"] - r_np["delta_k_sigma_raw"]),
            abs(r_pt["env_pct"] - r_np["thermal_shift_pct"]),
            abs(r_pt["env_sig"] - r_np["thermal_shift_sigma_raw"]),
            float(np.max(np.abs(r_pt["mode_dk"] - r_np["mode_dk_pct"]))),
        ]
        max_diff = max(max_diff, max(diffs))

    print(f"Numpy vs PyTorch max abs diff over {len(dev_blocks)} dev blocks: {max_diff:.3e}")
    if not max_diff < 1e-4:
        raise RuntimeError(f"Parity failed ({max_diff:.3e}); {dst} left unchanged, candidate at {tmp}")
    shutil.copyfile(tmp, dst)
    os.remove(tmp)
    size_kb = os.path.getsize(dst) / 1024.0
    print(f"Exported {dst} ({size_kb:.1f} KB, {total_params:,} params across {len(ckpt_names)} member(s))")
    return {"path": dst, "size_kb": round(size_kb, 1), "total_params": total_params, "max_abs_diff": max_diff}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*", help="Checkpoint names (default: read from calibration.json)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    cal_path = work("calibration.json")
    ckpts = args.ckpts
    if not ckpts:
        cal = json.load(open(cal_path))
        ckpts = cal["ckpts"]
    export_weights(ckpts, cal_path=cal_path)


if __name__ == "__main__":
    main()
