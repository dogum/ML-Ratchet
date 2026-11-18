"""Exam 2: the sealed 2022-2025 reserve (solar maximum), opened once on 17 Nov 2026 after the v1.0.1 decisions were
committed (a36824b). Same rows, whole-block bootstrap, seeds and verdict rules as the primary exam (exam.py), for
v1.0 (single widening factor) and v1.0.1 (bars binned by the last observed Dst; identical point forecasts) against
the same four competitors. Writes exam_reserve.json, sources_reserve.json and replay storms to work/exam_reserve/.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from _paths import MODEL, work
from baselines import forecast
from exam import ROWS, score_row
from noise import HERE, PROTOCOL, write_json
from split import period
from stormnet import StormNet


def main():
    release = Path(MODEL) / "stormnet_weights.npz"
    v101 = StormNet(release)
    cal = json.loads((HERE / "calibration.json").read_text())
    cases, meta, counts, hashes = period(PROTOCOL["sealed_reserve_years"], reserve=True)
    if len(np.unique(cases["time"])) != len(cases["time"]):
        raise ValueError("Duplicate reserve issue keys")
    read = v101.predict(cases["wind"], cases["dst"])
    single = v101.predict(cases["wind"], cases["dst"], calibrated=False)["sigma"] * v101.meta["widen_factor"]
    predictions = {"StormNet": read["min_dst"], **forecast(cases, cal)}
    out = dict(opened="2026-11-17", after_commit="a36824b", released=v101.meta["rung"], version=v101.meta["version"],
               weights_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),
               first_issue_utc=str(cases["time"][0]), last_issue_utc=str(cases["time"][-1]), eligibility=counts,
               distinct_storm_blocks=sum(m["kind"] == "storm" for m in meta),
               deepest_block=min((m for m in meta if m["minimum"] is not None), key=lambda m: m["minimum"]))
    for tag, sigma in (("v1.0.1", read["sigma"]), ("v1.0", single)):
        rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
        out[tag] = {name: score_row(cases, meta, predictions, sigma, cut, rng) for name, cut in ROWS}
    write_json(HERE / "exam_reserve.json", out)
    write_json(HERE / "sources_reserve.json", hashes)
    dest = Path(work("exam_reserve"))
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "reserve_cases.npz", time=cases["time"].astype("datetime64[h]").astype("int64"),
                        wind=cases["wind"][:, -1].astype("float32"), dst=cases["dst"], truth=cases["truth"],
                        group=cases["group"], forecast=read["min_dst"], sigma=read["sigma"],
                        **{k.replace("'", "").replace(" ", "_").replace("–", "-"): v for k, v in predictions.items() if k != "StormNet"})
    for tag in ("v1.0.1", "v1.0"):
        for name, row in out[tag].items():
            if "scores" in row:
                print(f"{tag:7s} {name:24s} n={row['issues']:6d} blocks={row['blocks']:4d} StormNet {row['scores']['StormNet']:.3f}  "
                      f"best other {min(v for k, v in row['scores'].items() if k != 'StormNet'):.3f}  "
                      f"diff {row['difference']:+.3f} {row['difference_ci95']}  {row['verdict']}  cov {row['coverage_pct']:.1f}")


if __name__ == "__main__":
    main()
