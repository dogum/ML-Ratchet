"""Export BridgeSense's live browser instrument: float16 weights, 8 real CC BY 4.0 exam cases,
and reference outputs (raw + cold_snap + stiff_drop variants) for site/tests/parity.test.ts.

    python models/bridgesense/web/export_instrument.py    # writes site/public/instruments/bridgesense/
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

from bridgesense import BridgeSense  # noqa: E402
from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("bridgesense")


def main() -> None:
    net = BridgeSense()
    # Export member 0 weights as float16 (browser sees f16; reference outputs use identical f16 roundtrip)
    W_raw = {k: v for k, v in net.members[0].items()}
    W_f16 = roundtrip(W_raw, "f16")
    net.members[0] = W_f16
    write_pack(OUT / "weights", W_raw, dtype="f16", meta=net.meta)

    ex_meta = json.loads((open(os.path.join(MODEL, "examples", "examples.json"), encoding="utf-8")).read())["examples"]
    ex_npz = np.load(os.path.join(MODEL, "examples", "examples.npz"))

    tensors = {}
    meta_out = []
    cases = []

    for i, m in enumerate(ex_meta):
        T_ref = ex_npz[f"e{i}_T_ref"].astype(np.float32)
        T_q = ex_npz[f"e{i}_T_q"].astype(np.float32)
        df_ref = ex_npz[f"e{i}_df_ref"].astype(np.float32)
        df_q = ex_npz[f"e{i}_df_q"].astype(np.float32)
        mask_ref = ex_npz[f"e{i}_mask_ref"].astype(np.float32)
        mask_q = ex_npz[f"e{i}_mask_q"].astype(np.float32)

        tensors[f"e{i}_T_ref"] = T_ref
        tensors[f"e{i}_T_q"] = T_q
        tensors[f"e{i}_df_ref"] = df_ref
        tensors[f"e{i}_df_q"] = df_q
        tensors[f"e{i}_mask_ref"] = mask_ref
        tensors[f"e{i}_mask_q"] = mask_q

        # Deterministic parity variants: raw, cold_snap (-6 C on T_q), stiff_drop (-6% dK/K on active modes)
        variants = [
            ("raw", T_q, df_q),
            ("cold_snap", (T_q - 6.0).astype(np.float32), df_q),
            ("stiff_drop", T_q, np.where(mask_q > 0.5, (1.0 + df_q) * np.sqrt(0.94) - 1.0, 0.0).astype(np.float32)),
        ]
        for vname, tq_v, dfq_v in variants:
            r = net.predict(T_ref, tq_v, df_ref, dfq_v, mask_ref, mask_q)
            cases.append({
                "example": i,
                "variant": vname,
                "outputs": [
                    float(r["logit_raw"]),
                    float(r["delta_k_pct"]),
                    float(r["delta_k_sigma"]),
                    float(r["thermal_shift_pct"]),
                    float(r["thermal_shift_sigma"]),
                ],
            })

        r0 = net.predict(T_ref, T_q, df_ref, df_q, mask_ref, mask_q)
        meta_out.append({
            **m,
            "python": {
                "p_change": round(float(r0["p_change"]), 5),
                "delta_k_pct": round(float(r0["delta_k_pct"]), 4),
                "delta_k_sigma": round(float(r0["delta_k_sigma"]), 4),
                "thermal_shift_pct": round(float(r0["thermal_shift_pct"]), 4),
                "thermal_shift_sigma": round(float(r0["thermal_shift_sigma"]), 4),
                "anchor_dk_pct": round(float(r0["anchor_dk_pct"]), 4),
            },
        })

    write_pack(OUT / "examples", tensors, dtype="f32", meta=meta_out)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    w_kb = os.path.getsize(OUT / "weights.bin") / 1024.0
    e_kb = os.path.getsize(OUT / "examples.bin") / 1024.0
    print(f"Wrote {OUT}: {len(meta_out)} examples ({e_kb:.1f} KB), weights ({w_kb:.1f} KB), {len(cases)} parity cases")


if __name__ == "__main__":
    main()
