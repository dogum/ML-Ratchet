"""Export the published weights and real frames. Reference cases exercise both sides of Bz=0."""
import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))
from mlratchet.webexport import instrument_dir, write_json, write_pack
from stormphysics import drivers, projected
from stormnet import StormNet

if __name__ == "__main__":
    out = instrument_dir("stormnet")
    net = StormNet()
    cal = json.loads((MODEL / "training/calibration.json").read_text())
    records = json.loads((MODEL / "examples/records.json").read_text())
    meta = dict(**net.meta, classical=cal["double"])
    # All 5 KiB of weights fit comfortably even at release precision.
    write_pack(out / "weights", net.weights, dtype="f32", meta=meta)
    arrays, reference = {}, []
    for i, record in enumerate(records):
        with np.load(MODEL / "examples" / record["file"]) as z:
            for name in ("wind", "last_dst", "past_dst", "future_dst"):
                arrays[f"{name}{i}"] = z[name]
            for j, (wind, dst) in enumerate(zip(z["wind"], z["last_dst"])):
                for shift in (-8., 0., 8.):
                    adjusted = wind.astype(float)
                    adjusted[-3:, 0] += shift
                    r = net.predict(adjusted, dst)
                    e, p = drivers(adjusted[None])
                    path, _, _ = projected(e, p, np.array([dst]), cal["double"], meta["wind_law"])
                    reference.append(dict(example=i, frame=j, variant=f"Bz{shift:+g}", shift=shift,
                                     outputs=[r["min_dst"], r["sigma"], float(path.min())]))
    write_pack(out / "examples", arrays, dtype="f32", meta={"records": records})
    write_json(out / "reference.json", dict(cases=reference, tolerance=2e-6))
    print(f"{len(records)} records, {len(reference)} parity cases")
