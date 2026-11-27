"""Draw a fixed-seed stratified set of 8 real CC BY 4.0 exam examples for models/bridgesense/examples/.

All bundled examples come strictly from CC BY 4.0 datasets (Concrete Box-Girder Zenodo 20522191 +
NASA POWER T2M; Hell Bridge Test Arena Zenodo 14632942) — zero KW51 CC BY-NC-SA 4.0 data is
redistributed in git. Fixed seed 20261124 within strata (never cherry-picked by score).
"""
from __future__ import annotations

import json
import os
import numpy as np

from _paths import MODEL
from exam import load_hbta_exam_episodes
from realdata import load_boxgirder_episodes


def main() -> None:
    rng = np.random.default_rng(20261124)
    bg_eps = load_boxgirder_episodes(block_days=5)["episodes"]
    hbta_eps = load_hbta_exam_episodes()

    def _pick(pool: list[dict], k: int) -> list[dict]:
        idx = sorted(rng.choice(len(pool), size=min(k, len(pool)), replace=False).tolist())
        return [pool[i] for i in idx]

    selected = []
    # Concrete Box-Girder (2 modes, concrete + asphalt freeze-thaw & static load)
    selected += _pick([e for e in bg_eps if e["stratum"] == "healthy_freezing"], 2)
    selected += _pick([e for e in bg_eps if e["stratum"] == "static_load"], 2)
    # Hell Bridge Test Arena (8 global truss modes, undamaged & imposed physical damage)
    selected += _pick([e for e in hbta_eps if e["stratum"] == "undamaged"], 1)
    selected += _pick([e for e in hbta_eps if e["stratum"] == "local_damage"], 1)
    selected += _pick([e for e in hbta_eps if e["stratum"] == "global_damage"], 2)

    ex_dir = os.path.join(MODEL, "examples")
    os.makedirs(ex_dir, exist_ok=True)

    meta_list = []
    arrays = {}
    for i, ep in enumerate(selected):
        arrays[f"e{i}_T_ref"] = ep["T_ref"].astype(np.float32)
        arrays[f"e{i}_T_q"] = ep["T_q"].astype(np.float32)
        arrays[f"e{i}_df_ref"] = ep["df_ref"].astype(np.float32)
        arrays[f"e{i}_df_q"] = ep["df_q"].astype(np.float32)
        arrays[f"e{i}_mask_ref"] = ep["mask_ref"].astype(np.float32)
        arrays[f"e{i}_mask_q"] = ep["mask_q"].astype(np.float32)

        if ep["bridge"] == "boxgirder":
            bridge_label = "Concrete Box-Girder (Hamburg, 2 modes)"
            if ep["is_changed"] == 0:
                truth_label = f"Healthy winter freeze (Tmin {ep['T_min']:.1f} C, dK=0%)"
                true_dk = 0.0
            else:
                truth_label = f"Static load +{ep['added_mass_kg']} kg ({ep['start']})"
                true_dk = None
        else:
            bridge_label = "Hell Bridge Test Arena (Norway, 8 modes)"
            st = ep["state"]
            if st == "UDS":
                truth_label = "Undamaged control (UDS, dK=0%)"
            elif ep["stratum"] == "local_damage":
                truth_label = f"Imposed local damage {st} (dK={ep['true_dk_pct']:+.2f}%)"
            else:
                truth_label = f"Imposed global damage {st} (dK={ep['true_dk_pct']:+.2f}%)"
            true_dk = float(ep["true_dk_pct"])

        meta_list.append({
            "idx": i,
            "id": ep["block_id"],
            "bridge": ep["bridge"],
            "bridge_label": bridge_label,
            "stratum": ep["stratum"],
            "is_changed": int(ep["is_changed"]),
            "true_dk_pct": true_dk,
            "n_modes": int(ep["n_modes"]),
            "T_min": float(ep["T_min"]),
            "T_mean": float(ep["T_mean"]),
            "truth_label": truth_label,
            "licence": "CC BY 4.0",
        })

    np.savez_compressed(os.path.join(ex_dir, "examples.npz"), **arrays)
    with open(os.path.join(ex_dir, "examples.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": 20261124, "examples": meta_list}, f, indent=2)

    readme = """# BridgeSense real exam examples

Eight real exam windows drawn with fixed seed `20261124` within strata from two unseen full-scale bridges (zero training overlap, zero dev overlap):

- **Concrete Box-Girder Bridge (4 blocks, 2 modes):** Koehncke et al. (2026), Zenodo [20522191](https://zenodo.org/records/20522191), **CC BY 4.0**, paired with NASA POWER hourly air temperature (2024).
- **Hell Bridge Test Arena (4 recordings, 8 global truss modes):** Svendsen et al. (2025), Zenodo [14632942](https://zenodo.org/records/14632942), **CC BY 4.0**.

KW51 railway bridge (`CC BY-NC-SA 4.0`, Zenodo 3745914) is not redistributed here; `training/fetch_data.py` downloads it into git-ignored `work/` to reproduce Exam A.
"""
    with open(os.path.join(ex_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"Saved {len(meta_list)} CC BY 4.0 exam examples to {ex_dir}")
    for m in meta_list:
        print(f"  [{m['idx']}] {m['id']:22s} | {m['stratum']:16s} | {m['truth_label']}")


if __name__ == "__main__":
    main()
