"""Export WakeNet's live browser instrument to site/public/instruments/wakenet/:
  * weights.{json,bin}   float16 weights (~0.5 MB) + meta + MM92/MM82 turbine curves
  * examples.{json,bin}  6 stratified real exam snapshots (Kelmarsh 2020-21 & Penmanshiel 2019-20)
  * reference.json       24 parity cases (6 examples x 4 variants: raw, steer15, rot12, park_front)
                         computed with float16-roundtripped weights so TS matches Python to ~1e-6.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, ROOT)

import physics as WP  # noqa: E402
from wakenet import WakeNet  # noqa: E402
from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("wakenet")


def _apply_variant(ex_meta: dict, z: dict, variant: str) -> dict:
    wd = float(ex_meta["wind_dir_deg"])
    swd = float(ex_meta["sigma_wd_deg"])
    yaw = z["yaw_deg"].astype(np.float64).copy()
    on = z["online"].astype(bool).copy()
    free = z["is_free"].astype(bool)

    if variant == "raw":
        pass
    elif variant == "steer15":
        yaw = np.where(free & on, np.clip(yaw + 15.0, -30.0, 30.0), yaw)
    elif variant == "rot12":
        wd = (wd + 12.0) % 360.0
    elif variant == "park_front":
        f_idx = np.where(free & on)[0]
        if len(f_idx) > 0:
            on[int(f_idx[0])] = False
    else:
        raise ValueError(variant)

    return {"wind_dir_deg": wd, "sigma_wd_deg": swd, "yaw_deg": yaw, "online": on}


def _parity_vec(r: dict) -> list[float]:
    return (
        r["eta_anchor"].astype(float).tolist()
        + r["eta"].astype(float).tolist()
        + r["eta_sig"].astype(float).tolist()
        + r["u_rotor_ms"].astype(float).tolist()
        + [float(r["farm_wake_loss"]), float(r["farm_wake_sig"])]
    )


def main():
    net = WakeNet()
    W_orig = {k: v for k, v in net.W.items()}
    net.W = roundtrip(W_orig, "f16")

    turb_meta = {}
    for k, t in WP.TURBINES.items():
        turb_meta[k] = {
            "name": t["name"],
            "rotor_diameter": float(t["rotor_diameter"]),
            "hub_height": float(t["hub_height"]),
            "rated_power_kw": float(t["rated_power_kw"]),
            "wind_speed": [float(x) for x in t["wind_speed"]],
            "power_kw": [float(x) for x in t["power_kw"]],
            "thrust_coefficient": [float(x) for x in t["thrust_coefficient"]],
            "ref_air_density": float(t["ref_air_density"]),
            "cosine_loss_exponent_yaw": float(t["cosine_loss_exponent_yaw"]),
        }

    w_meta = dict(net.meta)
    w_meta["turbines"] = turb_meta
    write_pack(OUT / "weights", W_orig, dtype="f16", meta=w_meta)

    xs, ex_list, cases = {}, [], []
    files = sorted(glob.glob(os.path.join(MODEL, "examples", "*.npz")))
    for i, fp in enumerate(files):
        z = np.load(fp)
        m = json.loads(str(z["meta"]))
        xs[f"east_{i}"] = z["east_m"].astype(np.float32)
        xs[f"north_{i}"] = z["north_m"].astype(np.float32)
        xs[f"elev_{i}"] = z["elev_m"].astype(np.float32)
        xs[f"hub_h_{i}"] = z["hub_h_m"].astype(np.float32)
        xs[f"yaw_{i}"] = z["yaw_deg"].astype(np.float32)
        xs[f"online_{i}"] = z["online"].astype(np.float32)
        xs[f"free_{i}"] = z["is_free"].astype(np.float32)
        xs[f"ptrue_{i}"] = z["power_true_kw"].astype(np.float32)

        for variant in ("raw", "steer15", "rot12", "park_front"):
            st = _apply_variant(m, z, variant)
            r = net.read(
                east_m=z["east_m"],
                north_m=z["north_m"],
                hub_h_m=z["hub_h_m"],
                turbine=m["turbine_model"],
                u_inf=m["u_inf_ms"],
                wind_dir_deg=st["wind_dir_deg"],
                ti_inf=m["ti_inf"],
                sigma_wd_deg=st["sigma_wd_deg"],
                yaw_deg=st["yaw_deg"],
                online=st["online"],
                elev_m=z["elev_m"],
                air_density=m["air_density"],
                p_front_kw=m["p_front_kw"],
                calibrated=True,
            )
            cases.append({
                "example": i,
                "variant": variant,
                "outputs": _parity_vec(r),
            })

        ex_list.append(m)

    write_pack(OUT / "examples", xs, dtype="f32", meta=ex_list)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    w_mb = os.path.getsize(OUT / "weights.bin") / 1e6
    e_kb = os.path.getsize(OUT / "examples.bin") / 1e3
    print(f"Wrote {OUT}: {len(ex_list)} examples, {len(cases)} parity cases (weights {w_mb:.2f} MB, examples {e_kb:.1f} KB)")


if __name__ == "__main__":
    main()
