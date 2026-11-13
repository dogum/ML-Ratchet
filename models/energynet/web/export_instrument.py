"""Export EnergyNet's live instrument for the website: the 5-network ensemble in float16, the five real exam trips in
examples/ (VED, one per powertrain) as 1 Hz arrays, and reference outputs for the browser parity test.

    python models/energynet/web/export_instrument.py      # writes site/public/instruments/energynet/

The browser recomputes everything from the 1 Hz route: features, road-load physics and the ensemble. The controls change
the route or the vehicle record; perturb() below is the one definition of what each control does (ui.ts mirrors it).
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

import features as F  # noqa: E402
from energynet import EnergyNet  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("energynet")
FIELDS = ["v", "grade", "T_amb", "wind_speed", "wind_dir", "heading", "rho_air", "surface"]


def perturb(trip, veh, p):
    """The instrument's controls, applied to the model's inputs (float64, as F.prep sees them).
    p: dT (air temperature shift, C; air density follows at constant pressure), wind ([speed m/s, direction relative to
    heading, deg: 0 head, 90 cross, 180 tail] replaces station wind), grade (added, fraction), mass (multiplier on
    mass_kg), soc (start charge 0..1), thermal ('warm' | 'cold')."""
    t = {k: np.asarray(trip[k], np.float64).copy() for k in FIELDS}
    if p.get("dT"):
        T1 = t["T_amb"] + p["dT"]
        t["rho_air"] = t["rho_air"] * (t["T_amb"] + 273.15) / (T1 + 273.15)
        t["T_amb"] = T1
    if p.get("wind") is not None:
        ws, rel = p["wind"]
        t["wind_speed"] = np.full(len(t["v"]), float(ws))
        t["wind_dir"] = (t["heading"] + rel) % 360.0
    if p.get("grade"):
        t["grade"] = t["grade"] + p["grade"]
    v = dict(veh)
    if p.get("mass") is not None:
        v["mass_kg"] = veh["mass_kg"] * p["mass"]
    if p.get("soc") is not None:
        v["soc_start"] = p["soc"]
    if p.get("thermal") is not None:
        v["thermal"] = p["thermal"]
    return t, v


def outputs(net, trip, veh):
    """Parity vector: the 149 features, the 5 members' raw outputs (log E/E_phys, log sigma, 5 log components),
    then predict(): energy MJ, sigma (log, widened), physics estimate MJ, value, lo68, hi68 (display unit), 5 components MJ."""
    x, E, dist, dur = net.features(trip, veh)
    mem = []
    for m in net.net.m:
        a, s, c = m.forward(x[None])
        mem += [float(a[0]), float(np.log(s[0]))] + [float(q) for q in c[0]]
    r = net.predict(trip, veh)
    fin = [r["energy_MJ"], r["sigma_log"], r["physics_estimate_MJ"], r["value"], r["lo68"], r["hi68"]] + list(r["components_MJ"].values())
    return [float(q) for q in x] + mem + [float(q) for q in fin], r


def main():
    net = EnergyNet()
    paths = sorted(glob.glob(os.path.join(MODEL, "weights", "*.npz")))
    W, keep = {}, {}
    for j, p in enumerate(paths):
        z = np.load(p)
        for k in z.files:
            if k != "meta":
                W[f"m{j}_{k}"] = z[k]
        keep[f"m{j}_x_mu"] = keep[f"m{j}_x_sd"] = "f32"  # normalisation stays float32 (exact); layers go float16
    meta = {"members": [os.path.basename(p)[:-4] for p in paths], "mode": net.net.mode, "phys_k": F.PHYS_K,
            "sigma_factors": net.k, "n_params": int(sum(v.size for k, v in W.items() if "x_" not in k))}
    write_pack(OUT / "weights", W, dtype="f16", meta=meta, keep=keep)

    # the browser sees float16 layers: reference outputs use the same values
    R = roundtrip(W, "f16")
    for j, m in enumerate(net.net.m):
        g = lambda k: R[f"m{j}_{k}"].astype(np.float64)  # noqa: E731
        m.L = [(g(f"body_{i}_weight"), g(f"body_{i}_bias")) for i in (0, 2, 4, 6)]
        m.H = (g("head_weight"), g("head_bias"))
    full = EnergyNet()  # untouched float64 weights, to report what float16 costs

    exs = json.load(open(os.path.join(MODEL, "examples", "examples.json")))
    xs, meta_ex, cases, f16diff = {}, [], [], []
    for i, e in enumerate(exs):
        z = np.load(os.path.join(MODEL, "examples", e["trip_id"] + ".npz"))
        trip = {k: z[k] for k in FIELDS}
        veh = e["vehicle"]
        assert not any(np.isnan(np.asarray(trip[k], float)).any() for k in FIELDS), "examples have no gaps; JS skips F.prep's gap fill"
        assert float(z["valid"].mean()) == 1.0, "every second logged: no valid-share scaling needed (see demo.py)"
        for k in FIELDS:
            xs[f"t{i}_{k}"] = trip[k]
        plug = veh["architecture"] == "bev" or veh["architecture"].startswith("phev")
        variants = [("raw", {}), ("cold15", {"dT": -15.0}), ("mass1.3", {"mass": 1.3}), ("hot+head", {"dT": 12.0, "wind": [8.0, 0.0]}),
                    ("hill+coldstart", {"grade": 0.02, "thermal": "cold"}), ("down+tail", {"grade": -0.015, "wind": [6.0, 180.0], "mass": 0.85})]
        if plug:
            variants.append(("soc", {"soc": 0.95 if veh["architecture"] != "bev" else 0.3}))
        for name, p in variants:
            t, v = perturb(trip, veh, p)
            o, r = outputs(net, t, v)
            cases.append({"example": i, "variant": name, "p": p, "outputs": o})
            if name == "raw":
                py = r
                f16diff.append(abs(r["energy_MJ"] / full.predict(t, v)["energy_MJ"] - 1))
        P = F.prep({k: np.asarray(trip[k], float) for k in FIELDS})
        meta_ex.append({
            "id": e["trip_id"], "stratum": e["stratum"], "vehicle": veh, "measured_MJ": e["measured"],
            "baselines_MJ": e["baselines_MJ"], "exam_prediction_MJ": e["exam_prediction_MJ"], "n": int(P["N"]),
            "km": float(P["v"].sum() / 1000), "T_mean": float(P["T"].mean()), "climb_m": float(np.clip(P["v"] * P["grade"], 0, None).sum()),
            "python": {k: py[k] for k in ("energy_MJ", "value", "unit", "lo68", "hi68", "sigma_log", "physics_estimate_MJ")},
        })
    write_pack(OUT / "examples", xs, dtype="f32", meta=meta_ex, keep={f"t{i}_surface": "u8" for i in range(len(exs))})
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(exs)} trips, {len(cases)} parity cases, {tot / 1e6:.2f} MB total; "
          f"float16 vs float64 weights, energy on the raw trips: max {max(f16diff):.1e} relative")


if __name__ == "__main__":
    main()
