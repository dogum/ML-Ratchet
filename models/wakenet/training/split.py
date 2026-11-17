"""Freeze and verify the WakeNet real-data split, writing split.json (committed to git).

Split design (zero overlap across roles):
  * backdrop (unlabelled inflow & unwaked noise): Kelmarsh 2016-2018
  * dev (calibration, twin test, per-half-year gate, widen-only sigma): Kelmarsh 2019 (H1 Jan-Jun, H2 Jul-Dec)
  * exam_kelmarsh (held-out time exam, same 6-turbine MM92 farm): Kelmarsh 2020-2021
  * exam_penmanshiel (zero-shot unseen farm exam, 14-turbine MM82 farm): Penmanshiel 2019-2020
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _paths as _P  # noqa: E402

SEED = 20261116


def _sha256_strings(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    for s in arr.tolist():
        h.update((str(s) + "\n").encode("utf-8"))
    return h.hexdigest()


def _stats(npz_name: str) -> dict:
    z = np.load(_P.work("proc", npz_name))
    ts = z["ts"]
    on = z["online"]
    free = z["is_free"]
    waked = on & (~free)
    yaw = z["yaw_deg"]
    # Upstream yaw >= 8 deg on active waked snapshots
    has_large_yaw = np.any(on & (np.abs(yaw) >= 8.0), axis=1)
    has_parked = np.any(~on, axis=1)
    d = {
        "file": npz_name,
        "farm": str(z["farm"]),
        "n_turbines": int(on.shape[1]),
        "n_snapshots": int(len(ts)),
        "n_waked_turbine_readings": int(np.sum(waked)),
        "n_large_yaw_snapshots": int(np.sum(has_large_yaw)),
        "n_partial_parked_snapshots": int(np.sum(has_parked)),
        "first_ts": str(ts[0]),
        "last_ts": str(ts[-1]),
        "ts_sha256": _sha256_strings(ts),
    }
    return d


def main():
    back = _stats("kelmarsh_backdrop_2016_2018.npz")
    dev = _stats("kelmarsh_dev_2019.npz")
    ex_k = _stats("kelmarsh_exam_2020_2021.npz")
    ex_p = _stats("penmanshiel_exam_2019_2020.npz")

    # Check strict temporal disjointness on Kelmarsh
    z_b = set(np.load(_P.work("proc", "kelmarsh_backdrop_2016_2018.npz"))["ts"].tolist())
    z_d_arr = np.load(_P.work("proc", "kelmarsh_dev_2019.npz"))["ts"]
    z_d = set(z_d_arr.tolist())
    z_k = set(np.load(_P.work("proc", "kelmarsh_exam_2020_2021.npz"))["ts"].tolist())
    assert not (z_b & z_d), "Overlap between backdrop and dev!"
    assert not (z_b & z_k), "Overlap between backdrop and Kelmarsh exam!"
    assert not (z_d & z_k), "Overlap between dev and Kelmarsh exam!"

    h1_mask = np.array([str(s)[5:7] <= "06" for s in z_d_arr])
    dev["strata"] = {
        "H1_2019_Jan_Jun": int(np.sum(h1_mask)),
        "H2_2019_Jul_Dec": int(np.sum(~h1_mask)),
    }

    doc = {
        "seed": SEED,
        "wind_speed_window_ms": [5.0, 12.0],
        "roles": {
            "backdrop": back,
            "dev": dev,
            "exam_kelmarsh": ex_k,
            "exam_penmanshiel": ex_p,
        },
    }
    doc["split_sha256"] = hashlib.sha256(
        json.dumps(doc["roles"], sort_keys=True).encode("utf-8")
    ).hexdigest()

    out_path = os.path.join(HERE, "split.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote frozen split to {out_path} (sha256={doc['split_sha256'][:16]}...)")
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
