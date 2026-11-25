"""Second-pass boundary tests on three released public APIs. Run from any directory."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parents[4] / "models"
for name in ("airfoilnet", "reservoirpilot", "snowsense"):
    sys.path.insert(0, str(MODELS / name))

import airfoilnet as af
import reservoirpilot as rp
import snowsense as ss


def airfoils():
    model = af.AirfoilNet()
    with TemporaryDirectory() as temp:
        path = Path(temp) / "geometry.dat"
        for file in sorted((MODELS / "airfoilnet/examples").glob("*.json")):
            case = json.loads(file.read_text())
            xy = af.normalize(case["coords"])
            i = np.argmin(xy[:, 0])
            upper, lower = xy[i::-1], xy[i:]
            raw = np.vstack((upper, lower))
            np.savetxt(path, raw, header=f"airfoil\n{len(upper)} {len(lower)}", comments="")
            angles = np.arange(-6., 7.)
            def lift(coords, a=angles):
                return model.polar(coords, Re=case["Re"], alphas=a)["cl"]
            baseline = lift(xy)
            parser_error = np.max(abs(lift(af.read_dat(path)) - baseline))
            assert parser_error < 1e-9
            print(file.stem, "max lift difference, parsed / raw / mirrored, -6..6 deg",
                  parser_error, max(abs(lift(raw) - baseline)),
                  max(abs(lift(xy * [1, -1], -angles) + baseline)))


def reservoir():
    # Known capacity, one full calibration year, identical first-day observations.
    dates = pd.date_range("2020-01-01", periods=396)
    phase = np.arange(len(dates)) * (2 * np.pi / 366)
    df = pd.DataFrame(dict(storage_hm3=70 + 5 * np.sin(phase), inflow_cms=15 + 8 * np.cos(phase),
                           release_cms=14 + 7 * np.sin(phase)), index=dates)
    cal, evaluation = df.iloc[:366], df.iloc[366:].copy()
    model = rp.Pilot.load()
    first_releases = []
    for last_storage in (80., 100., 120.):
        evaluation.iloc[-1, evaluation.columns.get_loc("storage_hm3")] = last_storage
        task = rp.make_task(cal, evaluation, 100.)
        release, _ = model.run(task)
        first_releases.append(release[0])
        print("last observed storage / task cap / first release (hm³, hm³, m³/s)",
              last_storage, task["cap"], release[0])
    assert first_releases[0] == first_releases[1] and first_releases[1] != first_releases[2]


def snow():
    model = ss.SnowSense()
    examples = MODELS / "snowsense/examples"
    for station, meta in json.loads((examples / "stations.json").read_text()).items():
        data = pd.read_csv(examples / f"{station}.csv", index_col="date", parse_dates=True)
        for day in ("2022-04-17", "2022-04-18"):
            daily = model.predict(data.loc[:day], meta["lat"], meta["elev_m"], wys=[2022])["daily"]
            after = pd.DatetimeIndex(daily.index) > pd.Timestamp(day)
            expected = (0, 0) if day.endswith("17") else (365, 165)
            assert (len(daily), sum(after)) == expected
            print(station, day, "returned / beyond cutoff", len(daily), sum(after),
                  "last date", daily.index[-1] if len(daily) else None)


if __name__ == "__main__":
    airfoils()
    reservoir()
    snow()
