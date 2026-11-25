"""Run BridgeSense on eight real exam cases from two unseen full-scale bridges (CC BY 4.0).

Pure numpy; runs offline in under a second.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from bridgesense import BridgeSense

HERE = Path(__file__).resolve().parent


def main() -> None:
    model = BridgeSense()
    meta = json.loads((HERE / "examples" / "examples.json").read_text(encoding="utf-8"))
    data = np.load(HERE / "examples" / "examples.npz")

    print("BridgeSense v1.0 — zero-shot real exam cases (Concrete Box-Girder + Hell Bridge Test Arena)")
    print(f"{'Case':22s} | {'Stratum':16s} | {'P(chg)':>6s} | {'BridgeSense dK/K':>16s} | {'OLS anc':>8s} | Truth")
    print("-" * 108)
    for ex in meta["examples"]:
        i = ex["idx"]
        r = model.predict(
            data[f"e{i}_T_ref"],
            data[f"e{i}_T_q"],
            data[f"e{i}_df_ref"],
            data[f"e{i}_df_q"],
            data[f"e{i}_mask_ref"],
            data[f"e{i}_mask_q"],
        )
        dk_str = f"{r['delta_k_pct']:+6.2f}% +/- {r['delta_k_sigma']:4.2f}%"
        anc_str = f"{r['anchor_dk_pct']:+6.2f}%"
        print(f"{ex['id']:22s} | {ex['stratum']:16s} | {r['p_change']:6.3f} | {dk_str:>16s} | {anc_str:>8s} | {ex['truth_label']}")


if __name__ == "__main__":
    main()
