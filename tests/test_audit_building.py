"""Regression tests for the building-platform audit findings fixed on 27 Nov 2026 (docs/audits/)."""
from __future__ import annotations

import importlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(model: str, module: str):
    """The model's module through the registry, which keeps sibling modules (pipeline, common, features) private."""
    import mlratchet
    return mlratchet.get(model).module(module)


def test_buildingdetective_short_record_keeps_the_full_year_answer():
    bd = _load("buildingdetective", "buildingdetective")
    path = ROOT / "models/buildingdetective/examples"
    z = np.load(path / "examples.npz")
    rec = json.loads((path / "examples.json").read_text())["hourly"][0]
    key = rec["key"]
    local = pd.to_datetime(z[key + "_utc"], unit="s", utc=True).tz_convert(rec["tz"])
    det = bd.Detective()

    def read(days):
        sl = slice(0, days * 24)
        meters = {c: z[key + "_" + c][sl] for c in rec.get("channels", ["elec"])}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = det.read_hourly(meters, z[key + "_T"][sl], z[key + "_GHI"][sl], local[sl], rec["lat"])
        return float(r["floor_area_m2"][0]), max(r["use"], key=r["use"].get), sum("record covers" in str(x.message) for x in w)

    full = read(365)
    assert full[2] == 0
    for days in (330, 180):
        area, use, n_warn = read(days)
        assert n_warn == 1
        assert use == full[1]                                  # was healthcare at 330 and 180 days
        assert 0.5 < area / full[0] < 1.5                      # was 295.6 / 99.5 and 58.6 / 99.5


def test_buildingsim_weather_cache_key_includes_wind():
    sys.path.insert(0, str(ROOT / "shared/building/buildingsim"))
    from buildingsim import weather as W
    idx = pd.date_range("2021-01-01", periods=72, freq="h", tz="UTC")
    base = pd.DataFrame(dict(T_out=np.linspace(-5, 5, 72), RH=60.0, GHI=0.0, DNI=0.0, DHI=0.0, wind_speed=2.0,
                             wind_dir=180.0, pressure=101325.0), index=idx)
    meta = dict(lat=40.0, lon=-75.0, tz="America/New_York", elev=10.0)
    a = W.prepare(base, meta)
    b = W.prepare(base.assign(wind_speed=8.0), meta)
    assert not np.allclose(a["wind"], b["wind"])


@pytest.fixture(scope="module")
def flexlab():
    ff = _load("faultfinder", "faultfinder")
    df = pd.read_csv(ROOT / "models/faultfinder/examples/flexlab_lbnl_szvav_flexlab_2017-09-11.csv")
    return ff, ff.FaultFinder(), df


def test_faultfinder_other_days_in_the_file_do_not_change_the_score(flexlab):
    ff, model, df = flexlab
    day = df.copy(); day["T_sa"] = 18.0                     # a flat supply sensor on the day read
    nxt = day.copy()
    nxt["time_local"] = (pd.to_datetime(nxt["time_local"]) + pd.Timedelta(days=1)).astype(str)
    alone = model.read_day(day)
    for later in (18.0, 19.0):
        both = pd.concat([day, nxt.assign(T_sa=later)], ignore_index=True)
        assert model.read_day(both)["p_fault"] == alone["p_fault"]


def test_faultfinder_refuses_self_reference_and_empty_days(flexlab):
    ff, model, df = flexlab
    with pytest.raises(ValueError):
        model.read_day(df, ref_days=[df])
    empty = df.copy()
    for c in empty.columns:
        if c != "time_local":
            empty[c] = np.nan
    with pytest.raises(ValueError):
        model.read_day(empty)


def test_buildingsim_weather_cache_accepts_nullable_dtypes():
    sys.path.insert(0, str(ROOT / "shared/building/buildingsim"))
    from buildingsim import weather as W
    idx = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
    w = pd.DataFrame(dict(T_out=np.linspace(-5, 5, 48), RH=60.0, GHI=0.0, DNI=0.0, DHI=0.0, wind_speed=3.0,
                          wind_dir=180.0, pressure=101325.0), index=idx)
    w["RH"] = w["RH"].astype("Float64")
    w["wind_dir"] = w["wind_dir"].astype("Int64")
    W.prepare(w, dict(lat=40.0, lon=-75.0, tz="America/New_York", elev=10.0))
