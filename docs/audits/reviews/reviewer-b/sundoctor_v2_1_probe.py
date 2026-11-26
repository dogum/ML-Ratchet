"""Replay meter CSV order and midday power loss with identical site settings."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4] / "models/sundoctor"
sys.path.insert(0, str(ROOT))
import sundoctor as sd

CA = dict(lat=38.996, lon=-122.134, tilt=25, azimuth=180, dc_kw=37.2, tz="US/Pacific", n_strings=6)
CO = dict(lat=40.3864, lon=-104.5512, mount="tracker", max_angle=60, gcr=.35,
          backtrack=False, dc_kw=592.25, tz="US/Mountain")


def main():
    for name in ("ratchet.json", "ratchet2.json"):
        rows = json.loads((ROOT / "training" / name).read_text())
        best = -np.inf
        for row in rows:
            vals = row["val"]
            score = (vals["poa"]["objective"] + vals["home"]["objective"]) / 2 if "poa" in vals else vals["objective"]
            assert np.isclose(score, row["objective"])
            keep = score > best + .01
            assert keep == row["accepted"], row["ckpt"]
            if keep:
                best = score
        print(name, len(rows), "decisions matched")
    model = sd.SunDoctor()
    for file, site in zip(sorted((ROOT / "examples").glob("*.csv")), (CA, CO), strict=True):
        df = pd.read_csv(file)
        time = pd.to_datetime(df.iloc[:, 0])
        # One month keeps the probe inexpensive and includes a full 14-day inference window.
        sample = df[time < time.iloc[0] + pd.Timedelta(days=30)].copy()
        time = time.loc[sample.index]
        assert time.is_monotonic_increasing
        gap = sample.copy()
        gap.loc[(time.dt.hour >= 10) & (time.dt.hour < 15), "ac_power_kw"] = np.nan
        night = sample.copy()
        night.loc[time.dt.hour < 5, "ac_power_kw"] = np.nan
        reversed_rows = sample.iloc[::-1]
        out = {}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "meter.csv"
            for label, frame in (("original", sample), ("descending", reversed_rows),
                                 ("resorted", reversed_rows.sort_values(sample.columns[0])), ("10–15 gap", gap),
                                 ("00–05 gap", night)):
                frame.to_csv(path, index=False)
                summary = model.diagnose_csv(path, **site)["summary"]
                out[label] = (summary["days"], summary["kwh_made"])
                print(file.stem, label, "valid days, kWh", out[label])
        assert out["resorted"] == out["original"]
        assert out["descending"] == (0, 0.0)
        assert 0 < out["10–15 gap"][1] < out["original"][1]


if __name__ == "__main__":
    main()
