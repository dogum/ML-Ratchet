"""Export the released weights, the whole-storm gallery and Python reference cases for the browser instrument.

Reference cases sample issue hours across every gallery storm, recorded and with what-if winds on the last three
hours (Bz shifted, speed and pressure scaled), and check StormNet, its 68% half-width, the three physics
competitors and the anchor's fast and slow reservoirs.
"""
import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))
from mlratchet.webexport import instrument_dir, write_json, write_pack  # noqa: E402
from stormnet import StormNet  # noqa: E402
from stormphysics import PUBLISHED, drivers, projected  # noqa: E402

VARIANTS = [(0., 1., 1.), (-8., 1., 1.), (8., 1., 1.), (-6., 1.3, 2.)]


def fast_slow(e, p, dst, params):
    """The anchor's reservoirs at issue time, as projected() splits the corrected Dst."""
    from stormphysics import reconstruct
    gain, tf, ts, share, b, c = params[:6]
    fast, slow = reconstruct(e, params)
    total = fast + slow
    frac = np.divide(fast, total, out=np.full_like(total, share), where=np.abs(total) > 1e-8)
    corrected = dst - b * p[:, -1] + c
    return frac * corrected, (1 - frac) * corrected


if __name__ == "__main__":
    out = instrument_dir("stormnet")
    net = StormNet()
    cal = json.loads((MODEL / "training/calibration.json").read_text())
    meta = dict(**net.meta, classical=cal["double"], single=cal["single"], published=list(PUBLISHED))
    write_pack(out / "weights", net.weights, dtype="f32", meta=meta)
    storms = json.loads((MODEL / "examples/storms.json").read_text(encoding="utf-8"))
    arrays = {}
    reference = []
    rng = np.random.default_rng(931)
    with np.load(MODEL / "examples/storms.npz") as z:
        for i, info in enumerate(storms):
            wind, dst = z[f"wind{i}"].astype(np.float64), z[f"dst{i}"].astype(np.float64)
            arrays[f"wind{i}"], arrays[f"dst{i}"] = z[f"wind{i}"], z[f"dst{i}"]
            ks = [k for k in range(35, len(dst) - 3)
                  if np.isfinite(wind[k - 35:k + 1]).all() and np.isfinite(dst[k:k + 4]).all()]
            for k in sorted(rng.choice(ks, size=min(5, len(ks)), replace=False)):
                for bz, sp, pr in VARIANTS:
                    w = wind[k - 35:k + 1].copy()
                    w[-3:, 0] += bz
                    w[-3:, 1] *= sp
                    w[-3:, 2] *= pr
                    r = net.predict(w, dst[k])
                    e, p = drivers(w[None])
                    comp = [float(projected(e, p, np.array([dst[k]]), prm, meta["wind_law"])[0].min())
                            for prm in (cal["double"], cal["single"], PUBLISHED)]
                    f, s = fast_slow(e, p, np.array([dst[k]]), net.meta["physics"])
                    reference.append(dict(example=i, variant=f"k{k} Bz{bz:+g} v{sp:g} p{pr:g}", storm=i, k=int(k),
                                          bz=bz, speed=sp, pressure=pr,
                                          outputs=[r["min_dst"], r["sigma"], *comp, float(f[0]), float(s[0])]))
    write_pack(out / "storms", arrays, dtype="f32", meta={"storms": storms})
    for stale in ("examples.json", "examples.bin"):
        (out / stale).unlink(missing_ok=True)
    write_json(out / "reference.json", dict(cases=reference, tolerance=2e-5))
    print(f"{len(storms)} storms, {len(reference)} parity cases")
