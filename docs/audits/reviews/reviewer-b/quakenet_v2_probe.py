"""Reproduce the metadata-only fit, split counts and promotion arithmetic on shipped files."""
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from scipy.optimize import least_squares

MODEL = Path(__file__).resolve().parents[4] / "models" / "quakenet-v2"
sys.path.insert(0, str(MODEL))
from quakenet2 import KM, gc_dist, tt


def fit_stations(stations):
    lat, lon, arrival = np.array([
        (s["lat"], s["lon"], s["t_start"] + s["p_guess"]) for s in stations
    ]).T
    arrival -= arrival.mean()

    def residual(source):
        x, y, depth = source
        times = tt("P", gc_dist(x, y, lat, lon), depth)
        return times - times.mean() - arrival

    grid = [(x, y, 50.) for x in range(-80, 81, 10) for y in range(-180, 180, 10)]
    starts = sorted(grid, key=lambda point: np.linalg.norm(residual(point)))[:4]
    fits = [least_squares(residual, start, bounds=([-90, -540, 2], [90, 540, 700]))
            for start in starts]
    return min(fits, key=lambda result: result.cost).x


def main():
    splits = json.loads((MODEL / "splits.json").read_text())
    dev, exam, multi = (set(splits[key]) for key in ("dev", "exam", "exam_multi"))
    print(f"Unique IDs: dev {len(dev)}, exam {len(exam)}, multi {len(multi)}, "
          f"dev/exam overlap {len(dev & exam)}, multi outside exam {len(multi - exam)}")
    rows = json.loads((MODEL / "training/ratchet.json").read_text())
    incumbent = {}
    decisions = 0
    for row in rows:
        bank = row["frozen"]
        score = mean(row["skill"].values())
        assert abs(score - row["objective"]) < 1e-10, (bank, row["rung"])
        if row["rung"] == "r0":
            incumbent[bank] = row
            continue
        prev = incumbent[bank]
        ratio = mean(v / prev["dev"][k] for k, v in row["dev"].items())
        keep = score - prev["objective"] > 0.01 and ratio <= 1.05
        assert keep == (row["decision"] in ("kept", "released")), (bank, row["rung"])
        if keep:
            incumbent[bank] = row
        decisions += 1
    print(f"Promotion decisions matching the log: {decisions}")
    print("Event ID, station count, epicentre error (m), fitted depth (km)")
    for path in sorted((MODEL / "examples").glob("*.npz")):
        with np.load(path, allow_pickle=False) as record:
            meta = json.loads(str(record["meta"]))
        assert path.stem in multi
        x, y, depth = fit_stations(meta["stations"])
        event = meta["event"]  # Catalogue coordinates enter only when scoring the fit.
        metres = 1000 * KM * gc_dist(x, y, event["lat"], event["lon"])
        print(f"{path.stem} {len(meta['stations'])} {metres:.2f} {depth:.1f}")


if __name__ == "__main__":
    main()
