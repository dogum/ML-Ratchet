"""Focused follow-ups to reviewer-a. Invoke a single model per process."""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parents[4] / "models"


def buildingdetective():
    import buildingdetective as bd
    import common
    import features

    path = MODELS / "buildingdetective"
    net = bd.Net(path / "weights/hourly.npz")

    def output(x):
        mu, _, logits = net.raw(x)
        return round(10 ** float(mu[0, 0]), 1), common.USES[
            np.argmax(logits[0, : len(common.USES)])
        ]

    for folder, name in (("examples", "examples"), ("examples/bdg2", "bdg2_hourly")):
        z = np.load(path / folder / (name + ".npz"))
        records = json.loads((path / folder / (name + ".json")).read_text())["hourly"]
        for rec in records[:1] if name == "examples" else records:
            key = rec["key"]
            local = pd.to_datetime(z[key + "_utc"], unit="s", utc=True).tz_convert(
                rec["tz"]
            )

            def vector(days):
                sl = slice(0, days * 24)
                meters = {
                    c: z[key + "_" + c][sl] for c in rec.get("channels", ["elec"])
                }
                return features.feat_hourly(
                    meters,
                    z[key + "_T"][sl],
                    z[key + "_GHI"][sl],
                    local[sl],
                    rec["lat"],
                )

            x = vector(365)
            if name == "examples":
                print(key, "365 d", output(x))
                for days in (330, 180, 90):
                    partial = vector(days)
                    restored = partial.copy()
                    restored[[6, 7]] = x[[6, 7]]
                    print(
                        days,
                        "partial / restore two span fields",
                        output(partial),
                        output(restored),
                    )
            else:
                clipped = np.clip((x - net.xm) / net.xs, -8, 8) * net.xs + net.xm
                only = x.copy()
                only[[904, 1197]] = clipped[[904, 1197]]
                print(
                    key,
                    "raw / two fields / all fields",
                    output(x),
                    output(only),
                    output(clipped),
                )


def demandcast():
    from demandcast import DemandCast
    from features import COMP, demand_x

    z = np.load(MODELS / "demandcast/examples/exam_days.npz")
    net = DemandCast()
    x = z["xd"].copy()
    assert np.all(x[:, 328] == 0)
    y = z["d_true"]

    def mape(v):
        return np.mean(np.abs(net.predict_demand(v, z["pd_total"])["mw"] - y) / y) * 100

    # Packed columns 329:497 contain the 168 clipped ratios. Only the first 79
    # columns and the missing fraction of the 329-input release depend on them.
    # Any positive constant historical prior reconstructs those same ratios.
    ph = {k: np.ones(168) for k in ("total", "heat", "cool", "T")}
    pt = {k: np.ones(24) for k in (*COMP, "total")}
    wt = {k: np.zeros(24) for k in ("T", "q", "ghi")}
    fields = np.r_[np.arange(79), 328]

    def encode(rho):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            return demand_x(
                1 + rho.astype(float),
                ph,
                pt,
                wt,
                dict(daytype=0, dst=0, doy=1),
                np.zeros(5),
                0,
            )

    baseline = x[:, :329].copy()
    print("baseline MAPE", mape(baseline))
    for start in (0, 72, 144):
        changed = baseline.copy()
        for i, row in enumerate(x):
            rho = row[329:].copy()
            np.testing.assert_allclose(encode(rho)[fields], row[fields], atol=1e-6)
            rho[start : start + 24] = np.nan
            changed[i, fields] = encode(rho)[fields]
        neutral = changed.copy()
        neutral[:, 328] = 0
        print(
            "gap start / all affected fields MAPE / suppress only flag",
            start,
            mape(changed),
            mape(neutral),
        )


def loadcast():
    import loadcast as lc

    path = MODELS / "loadcast/examples/bdg2"
    net = lc.LoadCast()
    for rec in json.loads((path / "examples.json").read_text()):
        if rec["meter"] == "electricity":
            continue
        weather = pd.read_csv(path / rec["weather_file"], index_col=0, parse_dates=True)
        if weather.index.tz is None:
            weather.index = weather.index.tz_localize("UTC")
        full = rec["meters_present"]
        hvac = lc.hvac_class_from_meters(full)
        for meters in (full, [k for k in full if k != "electricity"]):
            x = lc.static_features(
                rec["use"],
                rec["area_m2"],
                rec.get("year_built"),
                rec["meter"],
                meters,
                hvac,
                "unknown",
                lc.annual_climate([12.0, 16.0]),
                rec["lat"],
            )
            # Same season and explicit HVAC on both calls. Check hours moving in
            # the same direction so an annual mean cannot hide a few huge outliers.
            kw = net.forecast(
                weather,
                rec["tz"],
                rec["lat"],
                rec["use"],
                rec["area_m2"],
                rec["meter"],
                meters,
                "2017-01-01",
                "2018-01-01",
                year_built=rec.get("year_built"),
                hvac=hvac,
            )["kW"]
            if meters == full:
                original, static = kw, x
            else:
                assert np.flatnonzero(x != static).tolist() == [18]
                print(
                    rec["building"],
                    "hours higher / total",
                    (kw > original).sum(),
                    len(kw),
                    "median kW before/after",
                    original.median(),
                    kw.median(),
                )


if __name__ == "__main__":
    jobs = {f.__name__: f for f in (buildingdetective, demandcast, loadcast)}
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=jobs)
    name = parser.parse_args().model
    sys.path[:0] = [str(MODELS / name), str(MODELS / name / "training")]
    jobs[name]()
