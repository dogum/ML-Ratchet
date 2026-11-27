"""Replay the raw fuel-patch path on all eight release examples."""
import csv
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "models/firesize"
sys.path.insert(0, str(ROOT))
import firesize


def rebuild(row, patch, fuel):
    # Only the three landscape summaries are packaged. Replicate them spatially
    # while preserving the separate centre and 5x5 mean slope.
    slope = np.full((13, 13), (25 * row["slope5"] - row["slope_pt"]) / 24)
    slope[6, 6] = row["slope_pt"]
    day = date.fromisoformat(patch["discovery_date"]).timetuple().tm_yday
    hour = np.arctan2(row["hr_sin"], row["hr_cos"]) * 12 / np.pi % 24
    region = next(name for name in firesize.REGIONS if row["r_" + name] == 1)
    cause = next(name for key, name in zip(("c_hum", "c_nat", "c_miss"), firesize.CAUSES) if row[key])
    wx = {key: row[key] for key in firesize.WXF}
    prev = {key: row[key + "_m1"] for key in firesize.WXF}
    return firesize.make_features(fuel, np.full_like(slope, row["cc5"]),
                                 np.full_like(slope, row["ch5"]), slope, wx, prev,
                                 cause, row["fed"],
                                 None if row["hr_miss"] else hour, day, region)


def main():
    m = firesize.FireSize()
    patches = {int(p["FOD_ID"]): p for p in json.loads((ROOT / "examples/patches.json").read_text())}
    rare = [(f, float(sd)) for f, sd in zip(m.features, m.W["sd"]) if sd <= 0.001]
    print("features with sd <= 0.001", rare)
    with (ROOT / "examples/demo_examples.csv").open() as stream:
        for record in csv.DictReader(stream):
            row = {f: float(record[f]) for f in m.features}
            patch = patches[int(record["FOD_ID"])]
            fuel = np.array(patch["fuel13"])
            base = rebuild(row, patch, fuel)
            np.testing.assert_allclose(m.matrix(base), m.matrix(row), atol=2e-5, rtol=1e-6)
            p0 = m.predict(base)["p_escape"].item()
            np.testing.assert_allclose(p0, float(record["exam_p_escape"]), atol=1e-5)
            # Same number of new SB cells, placed at the ignition or next door.
            adjacent = fuel.copy()
            adjacent[6, 7] = 201
            p_adj = m.predict(rebuild(row, patch, adjacent))["p_escape"].item()
            for code in range(201, 205):
                centre = fuel.copy()
                centre[6, 6] = code
                features = rebuild(row, patch, centre)
                out = m.predict(features)
                assert features["pt_SB"] == 1 and out["p_escape"].item() == 0
                np.testing.assert_allclose(out["median_acres"], 1.7263, atol=5e-5)
            # A small contiguous SB patch abutting the ignition with centre intact.
            nearby = fuel.copy()
            nearby[3:10, 7:9] = 201
            features = rebuild(row, patch, nearby)
            assert features["pt_SB"] == 0
            p_patch = m.predict(features)["p_escape"].item()
            assert p_adj > 0 and 0 < p_patch < p0
            print(record["FOD_ID"], "original / adjacent cell / adjacent block / centre",
                  f"{p0:.7f} {p_adj:.7f} {p_patch:.7g} 0",
                  "nearby SB fractions", features["f5_SB"], features["f13_SB"])
    assert len(patches) == 8 and [name for name, _ in rare] == ["pt_SB"]


if __name__ == "__main__":
    main()
