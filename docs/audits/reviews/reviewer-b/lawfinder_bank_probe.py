"""Read-only simulation controls for LawFinder's paired road and building bank labels."""
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / p) for p in ("models/lawfinder", "models/lawfinder/training",
                                        "shared/vehicle/roadsim", "shared/building/buildingsim")]
from lawfinder import LawFinder



def road_control():
    import law_road
    from roadsim import load_vehicle, make_trip, simulate
    law = LawFinder()["road"]
    spec = load_vehicle(dict(architecture="bev", **{"class": "car"}))
    v = 16 + 6 * np.sin(np.arange(600) * 2 * np.pi / 90)
    m = spec["mass_kg"] + (spec.get("payload_kg") or 0)
    vehicle = law_road.vehicle_inputs(m, spec["CdA_m2"], spec["Crr"], "bev")
    estimates, battery = [], []
    for grade in (-.02, 0., .02):
        trip = make_trip(v, grade=grade, T_amb=20.)
        data = law_road.trip_inputs(v, trip["grade"], trip["T_amb"])
        o = simulate(spec, trip, options=law_road.PHYSICS["p1"], init_soc=.6, record=True, rng=7)
        E = np.nansum(o["batt_current"] * o["batt_voltage"]) / 1e6 / (data["dist_m"] / 1000)
        merged = data | vehicle
        estimates.append(float(law(**merged)))
        battery.append(E)
        print("grade, R, law, simulated MJ/km", grade, vehicle["R"], estimates[-1], E)
    assert np.all(np.diff(estimates) < 0) and np.all(np.diff(battery) > 0)


def building_control():
    import bldg_bank
    from buildingsim.weather import clear_cache, SIGMA
    times = pd.date_range("2019-01-01", "2020-01-01", freq="h", tz="UTC")
    i = np.arange(len(times))
    T = 8 + 12 * np.sin(2 * np.pi * i / 8760) + 4 * np.sin(2 * np.pi * i / 24)
    G = 350 * np.maximum(0, np.sin(np.pi * (i % 24 - 6) / 12))
    w = pd.DataFrame(dict(T_out=T, RH=70., GHI=G, DNI=G, DHI=.3 * G, wind_dir=180., pressure=101325.,
                          IR_horiz=SIGMA * (T + 273.15) ** 4 * .8, cloud=.5), index=times)
    meta = dict(lat=41.9, lon=-87.6, elev=200., tz="UTC")
    spec = dict(primary_use="residential_single", floor_area_m2=150., year_built=2000, climate_zone="5A",
                system="heat_pump_air", weather_hint="fixture", ventilation=dict(natural_ventilation=False))
    clear_cache()
    result = []
    for wind, fresh in ((2., False), (8., False), (8., True)):
        if fresh:
            clear_cache()
        cache = {"fixture": (w.assign(wind_speed=wind), meta)}
        r = bldg_bank.one(spec, np.random.default_rng(2), "1.4", cache)
        result.append(r["eui"])
        print("wind, fresh, delta T, annual EUI", wind, fresh, r["dT"], r["eui"])
    np.testing.assert_allclose(result[0], result[1], rtol=0, atol=1e-8)
    assert result[2] > result[1]
    clear_cache()


if __name__ == "__main__":
    # The training path initializer seeds a work directory on import.
    with TemporaryDirectory() as work:
        os.environ["ML_RATCHET_WORK"] = work
        road_control()
        building_control()
