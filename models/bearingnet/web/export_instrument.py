"""Export BearingNet's live instrument for the website: float16 weights, the 14 real exam recordings (12 kHz, float16 —
the browser runs the whole front end itself: kurtogram band, five envelope spectra, fault-aligned strips) and reference
outputs for the browser parity test.

    python models/bearingnet/web/export_instrument.py      # writes site/public/instruments/bearingnet/
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
from bearingnet import BearingNet  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("bearingnet")


def kit_rng(seed):
    """site/src/instruments/kit.ts rng(): mulberry32 + Box-Muller, bit-for-bit, so a noisy parity case is reproducible."""
    a = seed & 0xFFFFFFFF
    M = 0xFFFFFFFF

    def u():
        nonlocal a
        a = (a + 0x6D2B79F5) & M
        t = ((a ^ (a >> 15)) * (1 | a)) & M
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M)) & M) ^ t
        return ((t ^ (t >> 14)) & M) / 4294967296

    def n():
        x = 0.0
        while x == 0:
            x = u()
        y = u()
        return np.sqrt(-2 * np.log(x)) * np.cos(2 * np.pi * y)

    return n


def noisy(x, sigma_rms, seed):
    """x + sigma_rms * rms(x) * N(0, 1), the draw the browser makes for the same seed"""
    if sigma_rms <= 0:
        return x
    sd = sigma_rms * float(np.sqrt(np.mean((x - x.mean()) ** 2)))
    n = kit_rng(seed)
    return x + sd * np.array([n() for _ in range(len(x))])


def raw_outputs(net, x, geom, f_nom):
    """the network's raw outputs for every window of the recording, concatenated (n_windows * 8)"""
    F, S = [], []
    for s in PL.windows(x, 8):
        f, sc, _ = PL.window_features(x[s:s + PL.NWIN], geom, f_nom)
        F.append(f)
        S.append(sc)
    return net.forward(np.stack(F), np.stack(S)).reshape(-1)


def main():
    net = BearingNet()
    W = {k: v for k, v in net.W.items()}
    net.W = roundtrip(W, "f16")  # the browser sees float16 weights; reference outputs use the same values
    write_pack(OUT / "weights", W, dtype="f16", meta=net.meta)

    xs, ex, cases = {}, [], []
    files = sorted(glob.glob(os.path.join(MODEL, "examples", "*.npz")))
    # perturbed parity cases: a speed error, and added noise drawn with the browser's own RNG
    speed_cases = {2: 6.0, 7: -5.0, 12: 6.0}
    noise_cases = {4: (2.0, 7 * 101 + 4), 10: (1.5, 8 * 101 + 10)}
    for i, f in enumerate(files):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        assert abs(float(z["fs"]) - PL.FS) < 1e-6
        x16 = z["x"].astype(np.float16)
        xs[f"x{i}"] = x16
        x = x16.astype(np.float64)  # what the browser reads
        g, rpm = m["bearing"], float(m["rpm_nominal"])
        cases.append({"example": i, "variant": "raw", "rpm": rpm, "sigma": 0, "seed": 0,
                      "outputs": raw_outputs(net, x, g, rpm / 60).tolist()})
        if i in speed_cases:
            r2 = rpm * (1 + speed_cases[i] / 100)
            cases.append({"example": i, "variant": f"speed{speed_cases[i]:+.0f}%", "rpm": r2, "sigma": 0, "seed": 0,
                          "outputs": raw_outputs(net, x, g, r2 / 60).tolist()})
        if i in noise_cases:
            sig, seed = noise_cases[i]
            cases.append({"example": i, "variant": f"noise{sig}", "rpm": rpm, "sigma": sig, "seed": seed,
                          "outputs": raw_outputs(net, noisy(x, sig, seed), g, rpm / 60).tolist()})
        r = net.read(x, PL.FS, g, rpm)
        name = m["name"]
        rig = "CWRU" if name.startswith("CWRU") else "Paderborn"
        ex.append({
            "name": name, "rig": rig, "file": name.split()[1], "cond": " ".join(name.split()[2:]),
            "truth": m["truth"], "bearing": g, "rpm_nominal": rpm, "rpm_true": float(m["rpm_true"]),
            "size_mm": float(m["size_mm"]), "n": int(len(x)),
            "orders": {k: float(v) for k, v in PL.orders(**g).items()},
            "python": {"condition": r["condition"], "p": r["p"], "rpm": r["rpm"], "rpm_sig": r["rpm_sig"], "size_mm": r["size_mm"]},
        })
        print(f"{name:34s} truth {m['truth']:2s} net {r['condition']:2s} p={r['p'][m['truth']]:.2f} rpm {r['rpm']:.0f}±{r['rpm_sig']:.0f}")
    write_pack(OUT / "examples", xs, dtype="f16", meta=ex)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    right = sum(e["python"]["condition"] == e["truth"] for e in ex)
    size = sum(os.path.getsize(OUT / n) for n in os.listdir(OUT))
    print(f"wrote {OUT}: {len(ex)} recordings ({right} read right), {len(cases)} parity cases, {size / 1e6:.2f} MB total")


if __name__ == "__main__":
    main()
