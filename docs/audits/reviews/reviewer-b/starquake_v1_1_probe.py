"""Keep the measured flux fixed and vary the empty cadence-grid boundaries."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "models/starquake"
sys.path.insert(0, str(ROOT))
import pipeline
from starquake import StarQuake


def main():
    model = StarQuake()
    n = int(90 * 86400 / pipeline.DT)
    for file in sorted((ROOT / "examples").glob("*.npz")):
        with np.load(file) as z:
            seg = pipeline.load_kepler(z)[20000:20000 + n]
        for name, x in (("90 days", seg), ("90 empty days appended", np.pad(seg, (0, n), constant_values=np.nan)),
                        ("90 empty days prepended", np.pad(seg, (n, 0), constant_values=np.nan)),
                        ("finite values permuted", shuffled(seg))):
            r = model.read_ppm(x, teff=4800.)
            print(file.stem, name, "days,p_osc,numax,dnu,mass,numax_err,dnu_err",
                  *[round(float(r[k]), 6) for k in ("days", "p_osc", "numax", "dnu", "mass", "numax_err", "dnu_err")])
            if name == "90 days":
                days, dnu = r["days"], r["dnu"]
            if "empty" in name:
                assert r["days"] == days and abs(r["dnu"] - dnu) > .005
        # Isolate the two paths by crossing the spectra and the duration/duty input to the network.
        a = pipeline.prepare(seg)
        b = pipeline.prepare(np.pad(seg, (0, n), constant_values=np.nan))
        for label, spec, scal in (("baseline", a, a), ("padded spectrum only", b, a),
                                  ("changed duty only", a, b), ("both", b, b)):
            o = model.net.forward(spec[0], spec[1], scal[2])
            print(file.stem, label, "net numax", round(np.exp(o["ln_numax"]), 5),
                  "net dnu", round(np.exp(o["ln_dnu"]), 5), "duty", round(float(scal[2][1]), 5))
        probs = []
        for seed in range(10):
            permuted = shuffled(seg, seed)
            probs.append(model.read_ppm(permuted, hybrid=False)["p_osc"])
        print(file.stem, "ten time permutations, detections/range", sum(p > .5 for p in probs), min(probs), max(probs))
    # Fix the cadence mask, temperature and measured RMS while varying only the noise draw.
    with np.load(ROOT / "examples/2695975.npz") as z:
        mask = np.isfinite(pipeline.load_kepler(z)[20000:20000 + n])
    for rms in (200., 500., 1000.):
        detections = []
        for seed in range(10):
            noise = np.full(n, np.nan)
            draw = np.random.default_rng(seed).standard_normal(mask.sum())
            noise[mask] = rms * (draw - draw.mean()) / draw.std()
            np.testing.assert_allclose(np.nanstd(noise), rms)
            detections.append(model.read_ppm(noise, teff=5800.)["p_osc"])
        count = sum(p > .5 for p in detections)
        if rms == 500:
            assert 0 < count < len(detections)
        print("fixed RMS in ppm, detections, probability range", rms,
              count, min(detections), max(detections))
    for value in (0., np.nan):
        r = model.read_ppm(np.full(n, value), teff=4800.)
        print("constant", value, {k: r[k] for k in ("days", "detected", "p_osc", "mass")})


def shuffled(x, seed=17):
    y = x.copy()
    ok = np.isfinite(y)
    y[ok] = np.random.default_rng(seed).permutation(y[ok])
    return y


if __name__ == "__main__":
    main()
