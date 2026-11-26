"""Replay public TransitNet vetting with equivalent epochs, reordered rows and absent signals."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "models/transitnet"
sys.path.insert(0, str(ROOT))
from transitnet import TransitNet


def main():
    api = TransitNet()
    for file in sorted((ROOT / "examples").glob("*.npz")):
        with np.load(file) as z:
            m = json.loads(str(z["meta"]))
            secs = [{key: z[f"{key}{i}"] for key in ("t", "f", "c1", "c2")} for i in range(m["n_sectors"])]
        def read(sectors, cycles=0):
            return api.read(sectors, m["period"], m["epoch"] + cycles * m["period"], m["duration_hours"], star=m["star"])

        cases = {"original": (secs, 0), "epoch + P": (secs, 1), "epoch + 2P": (secs, 2),
                 "reverse sectors": (secs[::-1], 0),
                 "flat flux + no centroid": ([{"t": s["t"], "f": np.ones_like(s["f"])} for s in secs], 0),
                 "reversed rows": ([{k: v[::-1] for k, v in s.items()} for s in secs], 0)}
        erase, remove_centroid = [], []
        for sector in secs:
            t = sector["t"]
            phase = (t - m["epoch"] + m["period"] / 2) % m["period"] - m["period"] / 2
            outside = np.abs(phase) > m["duration_hours"] / 24
            # Excise the same in-transit rows from timestamps, photometry and centroids.
            erase.append({k: v[outside] for k, v in sector.items()})
            remove_centroid.append({k: sector[k] for k in ("t", "f")})
        rng = np.random.default_rng(13)
        cases["flux permuted within sector"] = ([{"t": s["t"], "f": rng.permutation(s["f"])} for s in secs], 0)
        cases["transit region omitted"] = (erase, 0)
        cases["centroids omitted"] = (remove_centroid, 0)
        for label, (sectors, cycles) in cases.items():
            try:
                out = read(sectors, cycles)
            except ValueError as exc:
                if label != "reversed rows":
                    raise
                print(file.stem, label, type(exc).__name__, str(exc))
                continue
            print(file.stem, label, "planet/EB/blend/nothing, k, SNR",
                  *[round(float(out[k]), 6) for k in ("p_planet", "p_eb", "p_blend", "p_nothing", "k", "snr")])
            if label == "original":
                original = out
            if label == "flat flux + no centroid":
                assert all(np.isnan(out["p_" + key]) for key in ("planet", "eb", "blend", "nothing"))
            if label == "transit region omitted":
                assert out["snr"] == -np.inf and out["p_nothing"] > .8
            if label == "reverse sectors":
                np.testing.assert_allclose(out["p_planet"], original["p_planet"], atol=1e-5)


if __name__ == "__main__":
    main()
