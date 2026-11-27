"""Restore valid directions in a packaged historical NDBC year, then replay the model."""
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "models/swellcast"
sys.path.insert(0, str(ROOT))
import pipeline
import swellcast


def main():
    rows = json.loads((ROOT / "training/ratchet.json").read_text())
    inc = None
    for row in rows:
        score = np.mean(list(row["val"]["skill"].values()))
        assert np.isclose(score, row["objective"])
        keep = inc is None or (score > inc["objective"] + .01 and np.mean([
            row["dev"][k] / inc["dev"][k] for k in ("logD_mae", "t0_mae_h", "dir_mean_deg")]) <= 1.05)
        assert keep == row["accepted"], row["ckpt"]
        if keep:
            inc = row
    print(len(rows), "network decisions matched")
    final = json.loads((ROOT / "training/final.json").read_text())
    dev_storms = set(json.loads((ROOT / "training/dev_storms_v1.json").read_text())["dev_storms"])
    exam_rows = final["exam_pred"]
    exam_storms = {r["sid"] for r in exam_rows}
    hits = sum(r["pdet"] > final["det_thr"] for r in exam_rows)
    print("confirmed exam rows/storms, dev storms, overlap, hits",
          len(exam_rows), len(exam_storms), len(dev_storms), len(exam_storms & dev_storms), hits)
    assert not exam_storms & dev_storms
    folder = ROOT / "examples/raw"
    paths = [folder / f"51004{code}2018.txt.gz" for code in "wdj"]
    meta = json.loads((folder / "olivia.json").read_text())
    original = pipeline.read_ndbc(*paths)
    parser = pipeline.parse_raw

    def direction_aware(path):
        times, freqs, parsed = parser(path)
        if path == paths[1]:
            with gzip.open(path, "rt") as stream:
                raw = np.loadtxt(stream, comments="#")[:, -len(freqs):]
            assert raw.shape == parsed.shape
            valid = raw == 99.0  # Valid compass bearing inside [0,360). Leave all other fields alone.
            assert np.isnan(parsed[valid]).all()
            inside = (times >= centre - 120) & (times < centre + 120)
            print("99 degree cells year/window", valid.sum(), valid[inside].sum())
            parsed[valid] = raw[valid]
        return times, freqs, parsed

    w, centre = pipeline.window(original, meta["window_centre_hour"])
    with patch.object(pipeline, "parse_raw", direction_aware):
        corrected = pipeline.read_ndbc(*paths)
    w2, _ = pipeline.window(corrected, centre)
    np.testing.assert_equal(w[0], w2[0])
    model = swellcast.SwellCast()
    for label, win in (("released", w), ("restore directions", w2)):
        r = model.read(win)
        print(label, *(f"{k}={r[k]:.6f}" for k in ("D_km", "dir", "t0", "p_swell")))
    # Equal weekly spacing without looking at storm locations, plus the exam's 70% coverage rule.
    reads = []
    for h in range(original["h0"] + 120, original["h0"] + len(original["E"]) - 120, 7 * 24):
        win, _ = pipeline.window(original, h)
        if np.isfinite(win[0, :, 4]).mean() >= .7:
            reads.append(win)
    scan = model.predict(np.stack(reads), bs=4)
    print("weekly detections/eligible", scan["detected"].sum(), len(reads),
          "score p10/50/90", np.percentile(scan["p_swell"], [10, 50, 90]))
    for path in sorted((ROOT / "examples").glob("*.npz")):
        with np.load(path) as z:
            win = z["x"]
        rot = win.copy()
        rot[1], rot[2] = -win[2], win[1]
        a, b = model.read(win), model.read(rot)
        residual = (b["dir"] - a["dir"] - 90 + 180) % 360 - 180
        print(path.stem, "90-degree rotation residual deg", round(residual, 3),
              "detected before/after", int(a["detected"]), int(b["detected"]))


if __name__ == "__main__":
    main()
