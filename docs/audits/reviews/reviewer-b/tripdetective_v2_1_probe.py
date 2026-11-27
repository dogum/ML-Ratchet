"""Exercise the public whole-mission reader on the packaged ZTBus day."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "models/tripdetective"
sys.path.insert(0, str(ROOT))
import tripdetective as td
from demo import load_example


def masses(reader, legs):
    estimates = reader.read_mission(legs)
    return np.array([r["mass_kg"] for r in estimates])


def main():
    reader = td.TripDetectiveBus()
    with np.load(ROOT / "examples/ZTBus_exam_mission.npz") as z:
        n = len(json.loads(str(z["info"]))["legs"])
        legs = [{key.removeprefix(f"leg{i}_"): z[key] for key in z.files if key.startswith(f"leg{i}_")}
                for i in range(n)]
    k = int(np.argmax([len(leg["v"]) for leg in legs]))
    rest = [j for j in range(n) if j != k]
    base = masses(reader, legs)
    results = {}
    for label, power in (("missing", np.nan), ("zero", 0.0)):
        changed = legs.copy()
        changed[k] = legs[k] | {"power_W": np.full_like(legs[k]["power_W"], power, dtype=float)}
        results[label] = masses(reader, changed)
    np.testing.assert_array_equal(results["missing"], results["zero"])
    masked = legs.copy()
    masked[k] = legs[k] | {"power_W": np.full(len(legs[k]["v"]), np.nan), "valid": np.zeros(len(legs[k]["v"]))}
    masked_mass = masses(reader, masked)
    print("explicit all-invalid leg still counted, kg", masked_mass[k],
          "unchanged leg difference from original", masked_mass[rest[0]] - base[rest[0]])
    print("original nonfinite power samples", [int((~np.isfinite(leg["power_W"])).sum()) for leg in legs])
    omit = masses(reader, [legs[j] for j in rest])
    print("legs, selected, samples", n, k, len(legs[k]["v"]))
    print("selected kg, measured/missing", base[k], results["missing"][k])
    for label, comp in (("original", base[rest]), ("omitted", omit)):
        print("unchanged legs missing minus", label, "kg range",
              np.min(results["missing"][rest] - comp), np.max(results["missing"][rest] - comp))
    np.testing.assert_allclose(masses(reader, legs[::-1])[::-1], base, atol=1e-7, rtol=0)

    car = td.TripDetective()
    for file in sorted((ROOT / "examples").glob("VED_*.npz")):
        info, trips = load_example(file)
        est = info["trips"][0]["fuel_source"] != "obd_fuel_rate_Lph"
        normal = car.read_car(trips, fuel_estimated=est)
        zero = [trip | {"grade": np.zeros_like(trip["v"])} for trip in trips]
        missing = [trip | {"grade": np.full_like(trip["v"], np.nan, dtype=float)} for trip in trips]
        flat = car.read_car(zero, fuel_estimated=est)["curb_kg"]
        unknown = car.read_car(missing, fuel_estimated=est)["curb_kg"]
        print(file.stem, "curb kg observed/missing/flat", normal["curb_kg"], unknown, flat)
        assert unknown == flat

    dev_errors = ("mass_mae_car", "disp_mae_car", "mass_mae_trip", "disp_mae_trip")
    for name in ("ratchet.json", "ratchet_v2.json", "ratchet_v21.json", "ratchet_bus.json"):
        log = json.loads((ROOT / "training" / name).read_text())
        inc = None
        for row in log:
            keep = True
            if inc is not None:
                a, b = row["dev"], inc["dev"]
                gain = row["objective"] - inc["objective"]
                if name == "ratchet_bus.json":
                    keep = gain > .01 and a["mae_lc"] <= 1.05 * b["mae_lc"] and a["slope"] >= b["slope"] - .1
                else:
                    ratio = np.mean([a[k] / b[k] for k in dev_errors])
                    acc = a["vedtype_acc_trip"] - b["vedtype_acc_trip"]
                    keep = gain > .01 and ratio <= 1.05 and acc >= -.03
                    if "rule_v21" in row:
                        print("output constraint, original rule", keep, "gain", gain, "dev accuracy gain", acc)
                        keep = gain >= 0 and ratio <= 1.05 and acc > .01
            assert keep == row["accepted"], (name, row["ckpt"])
            if keep:
                inc = row
        print(name, len(log), "decisions matched")
    rows = json.loads((ROOT / "training/ratchet_buscal.json").read_text())
    inc = rows[1]
    for row in rows[2:]:
        keep = row["skill"] > inc["skill"] + .01 and row["cv"]["r"] >= inc["cv"]["r"]
        assert keep == row["accepted"], row["rung"]
        if keep:
            inc = row
    print("mission calibration candidates replayed", len(rows) - 2, "winner", inc["rung"])


if __name__ == "__main__":
    main()
