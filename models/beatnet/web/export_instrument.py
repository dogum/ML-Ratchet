"""Export BeatNet's live instrument for the website: float16 weights (+ the front-end filter coefficients), the four
real MIT-BIH DS2 exam excerpts (raw ECG as recorded, cardiologist beat positions and labels), and reference outputs
for the browser parity test. The browser redoes the whole front end (band-pass, decimation, windows, local QRS scale,
RR features) on the possibly perturbed signal, so the reference cases include perturbed variants made with the exact
same noise generator as the instrument (mulberry32 + Box-Muller, ported below).

    python models/beatnet/web/export_instrument.py      # writes site/public/instruments/beatnet/

NOT A MEDICAL DEVICE: research and teaching only.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import pipeline as PL  # noqa: E402
from beatnet import BeatNet  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("beatnet")
FS = 360
ADC = 200.0  # MIT-BIH: 200 ADC units per mV (11-bit converter) — the excerpts are exact multiples of 1/200 mV


def mulberry32(seed):
    """site/src/instruments/kit.ts rng(): the same 32-bit integer arithmetic, so 'add noise' is identical in Python."""
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
        return math.sqrt(-2 * math.log(x)) * math.cos(2 * math.pi * y)

    return n


def perturb(ecg, noise=0.0, wander=0.0, seed=1):
    """net.ts perturb(): white 'muscle' noise (sigma in mV, seeded) and a two-tone respiration-like baseline wander
    (amplitude in mV) added to the raw 360 Hz signal before the front end."""
    x = ecg.astype(np.float64)
    if noise > 0:
        n = mulberry32(seed)
        x = x + noise * np.array([n() for _ in range(len(x))])
    if wander > 0:
        t = np.arange(len(x)) / FS
        x = x + wander * (np.sin(2 * np.pi * 0.21 * t + 0.6) + 0.5 * np.sin(2 * np.pi * 0.07 * t + 2.1)) / 1.5
    return x


VARIANTS = {"raw": dict(), "muscle": dict(noise=0.15, seed=11), "wander": dict(wander=1.2), "both": dict(noise=0.3, wander=0.8, seed=5)}


def main():
    net = BeatNet()
    W = dict(net.W)
    net.W = roundtrip(W, "f16")  # the browser sees float16 weights; reference outputs use the same values
    meta = dict(net.meta)
    meta.update(sos=PL.SOS.tolist(), zi=ss.sosfilt_zi(PL.SOS).tolist(), half=PL.HALF, nb=PL.NB, fs0=PL.FS0, fs=PL.FS)
    write_pack(OUT / "weights", W, dtype="f16", meta=meta)

    xs, keep, ex, cases = {}, {}, [], []
    for i, f in enumerate(sorted(glob.glob(os.path.join(MODEL, "examples", "*.npz")))):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        ecg, beats, ref = z["ecg"], z["beats"].astype(np.int64), z["ref"]
        adc = np.round(ecg.astype(np.float64) * ADC)
        assert np.array_equal((adc / ADC).astype(np.float32), ecg) and np.abs(adc).max() < 2048  # exact in float16
        xs[f"ecg{i}"] = adc
        xs[f"beats{i}"] = beats
        keep[f"ecg{i}"], keep[f"beats{i}"] = "f16", "i32"
        show = (beats >= z["show"][0]) & (beats < z["show"][1])
        for variant, kw in VARIANTS.items():
            if i >= 2 and variant == "both":
                continue
            x = perturb(ecg, **kw)
            out = net.classify(x, FS, beats)
            cases.append({"example": i, "variant": variant, **{k: kw.get(k, 0) for k in ("noise", "wander", "seed")},
                          "outputs": out["raw"][show].ravel().tolist()})
            if variant == "raw":
                clean = out
        ex.append({
            "record": m["record"], "why": m["why"], "start_s": m["start_s"], "fs": FS, "show": z["show"].tolist(),
            "ref": "".join(ref.tolist()),
            "python": {"label": "".join(clean["label"].tolist()), "hr": float(clean["heart_rate_bpm"])},
        })
    write_pack(OUT / "examples", xs, dtype="f32", meta=ex, keep=keep)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    tot = sum(os.path.getsize(OUT / n) for n in os.listdir(OUT))
    print(f"wrote {OUT}: {len(ex)} records, {len(cases)} parity cases, {tot / 1e6:.2f} MB total")


if __name__ == "__main__":
    main()
