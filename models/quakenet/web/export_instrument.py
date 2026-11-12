"""Export QuakeNet's live instrument for the website: float16 weights, the four real exam records as the network
sees them (the reading pipeline runs here, in Python), and reference outputs for the browser parity test.

    python models/quakenet/web/export_instrument.py      # writes site/public/instruments/quakenet/
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import pipeline as PL  # noqa: E402
from quakenet import QuakeNet  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("quakenet")


def main():
    net = QuakeNet()
    W = {k: v for k, v in net.W.items()}
    net.W = roundtrip(W, "f16")  # the browser sees float16 weights; reference outputs use the same values
    write_pack(OUT / "weights", W, dtype="f16", meta=net.meta)

    xs, ex, cases = {}, [], []
    for i, f in enumerate(sorted(glob.glob(os.path.join(MODEL, "examples", "*.npz")))):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        Wn, ls, p, peak = PL.read(z["zne"].astype(float), int(m["p_guess"]))
        xs[f"x{i}"] = Wn
        s = np.array([[(ls + 6) / 2, np.log10(max(peak, 1.0))]], np.float32)
        # parity cases: as recorded, and with the record cut after 900 s (zeros), both deterministic
        for variant, x in (("raw", Wn), ("cut900", np.where(np.arange(PL.L) < 900, Wn, 0).astype(np.float32))):
            o = net.forward(x[None], s)[0]
            cases.append({"example": i, "variant": variant, "s": s[0].tolist(), "outputs": o.tolist()})
        r = net.read(z["zne"], m["p_guess"])
        ex.append({
            "id": m["event_id"], "place": m["place"], "time": m["time"], "station": m["station"],
            "ls": float(ls), "peak": float(peak), "pre": PL.PRE,
            "truth": {"mw": m["usgs_mw"], "magType": m["usgs_magType"], "depth": m["usgs_depth_km"],
                      "dist": m["dist_deg"], "baz": m["baz_deg"]},
            "python": {k: r[k] for k in ("mw", "mw_sig", "dist", "dist_sig", "depth", "logz", "logz_sig", "baz", "kappa")},
        })
    write_pack(OUT / "examples", xs, dtype="f32", meta=ex)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    print(f"wrote {OUT}: {len(ex)} records, weights {os.path.getsize(OUT / 'weights.bin') / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
