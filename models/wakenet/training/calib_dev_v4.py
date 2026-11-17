"""Physics v4 dev calibration (17 Nov 2026): wake-growth sensitivity to turbulence (ka) and thrust scale (ct_scale).

Holds the v3 settings from dev_calibration.json (meandering, hill speedup, shear, terrain-follow) and searches
ka in {0.38, 0.5, 0.65, 0.8} x ct_scale in {1.0, 0.9, 0.8} on Kelmarsh 2019 dev, scored like calib_dev.py
(0.65 waked MAE + 0.35 farm MAE; stride 6 for the grid, stride 2 for the chosen setting).
Writes training/dev_calibration_v4.json. Dev only.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402
from calib_dev import eval_physics_config  # noqa: E402

V3 = json.load(open(os.path.join(HERE, "dev_calibration.json"), encoding="utf-8"))["calibrated_v3"]["params"]


def _cfg(ka: float, cs: float):
    def f(z, t, hh):
        return (hh, float(z["ti_inf"][t]), {
            "velocity_model": "gauss", "deflection_model": "gauss", "turbulence_model": "crespo_hernandez",
            "wind_shear": V3["wind_shear"], "sigma_wd_deg": V3["c_meander"] * float(z["sigma_wd"][t]),
            "hill_speedup": V3["hill_speedup"], "terrain_follow": V3["terrain_follow"],
            "air_density": float(z["air_density"][t]), "ka": ka, "ct_scale": cs,
        })
    return f


def _run(args):
    ka, cs, stride = args
    z = np.load(_P.work("proc", "kelmarsh_dev_2019.npz"))
    return {"ka": ka, "ct_scale": cs, **eval_physics_config(z, _cfg(ka, cs), stride=stride)}


def main():
    grid = [(ka, cs, 6) for ka in (0.38, 0.5, 0.65, 0.8) for cs in (1.0, 0.9, 0.8)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(_run, grid))
    best = min(rows, key=lambda r: 0.65 * r["waked_mae"] + 0.35 * r["farm_wake_mae"])
    chosen = _run((best["ka"], best["ct_scale"], 2))
    v3_dense = _run((0.38, 1.0, 2))
    out = {"dev_set": "kelmarsh_dev_2019.npz (5,729 snapshots)", "held_from_v3": V3,
           "calibrated_v4": {"params": {**V3, "ka": best["ka"], "ct_scale": best["ct_scale"]}, "metrics": chosen},
           "v3_same_stride": v3_dense, "grid_search": rows}
    with open(os.path.join(HERE, "dev_calibration_v4.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
    for r in sorted(rows, key=lambda r: 0.65 * r["waked_mae"] + 0.35 * r["farm_wake_mae"]):
        print(f"ka {r['ka']:.2f} ct {r['ct_scale']:.2f}: waked {100*r['waked_mae']:.2f}  farm {100*r['farm_wake_mae']:.2f}  direct bias {100*r['direct_wake_bias']:+.2f}")
    print("chosen (stride 2):", {k: v for k, v in chosen.items()})
    print("v3 (stride 2):    ", v3_dense)


if __name__ == "__main__":
    main()
