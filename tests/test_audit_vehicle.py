"""Regression tests for the vehicle-platform audit findings fixed on 27 Nov 2026 (docs/audits/)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(model: str, module: str):
    """The model's module through the registry, which keeps sibling modules (pipeline, common, features) private."""
    import mlratchet
    return mlratchet.get(model).module(module)


@pytest.fixture(scope="module")
def ztbus_day():
    td = _load("tripdetective", "tripdetective")
    with np.load(ROOT / "models/tripdetective/examples/ZTBus_exam_mission.npz") as z:
        n = len(json.loads(str(z["info"]))["legs"])
        legs = [{k.removeprefix(f"leg{i}_"): z[k] for k in z.files if k.startswith(f"leg{i}_")} for i in range(n)]
    return td, td.TripDetectiveBus(), legs


def _masses(reader, legs):
    return np.array([r["mass_kg"] for r in reader.read_mission(legs)])


def test_tripdetective_bus_leg_without_power_leaves_its_day(ztbus_day):
    _, reader, legs = ztbus_day
    k = int(np.argmax([len(leg["v"]) for leg in legs]))
    rest = [j for j in range(len(legs)) if j != k]
    omitted = _masses(reader, [legs[j] for j in rest])
    for change in ({"power_W": np.full(len(legs[k]["v"]), np.nan)}, {"valid": np.zeros(len(legs[k]["v"]))}):
        day = legs.copy()
        day[k] = legs[k] | change
        out = _masses(reader, day)
        assert np.isnan(out[k])
        np.testing.assert_allclose(out[rest], omitted, rtol=0, atol=1e-9)


def test_tripdetective_bus_partial_dropout_is_bounded(ztbus_day):
    _, reader, legs = ztbus_day
    k = int(np.argmax([len(leg["v"]) for leg in legs]))
    rest = [j for j in range(len(legs)) if j != k]
    base = _masses(reader, legs)
    power = legs[k]["power_W"].astype(float).copy()
    power[: len(power) // 2] = np.nan
    day = legs.copy()
    day[k] = legs[k] | {"power_W": power}
    out = _masses(reader, day)
    # Before the fix: -789.56 kg on the leg and +142.12 kg on every other leg.
    assert abs(out[k] - base[k]) < 300
    assert np.max(np.abs(out[rest] - base[rest])) < 60


@pytest.mark.parametrize("speed", [np.nan, 0.0])
def test_tripdetective_car_refuses_a_log_without_motion(ztbus_day, speed):
    td, _, _ = ztbus_day
    with pytest.raises(ValueError):
        td.TripDetective().read_car([{"v": np.full(600, speed), "fuel_rate": np.full(600, 1.0)}])


def test_hybridpilot_preview_reads_nothing_past_its_window():
    hp = _load("hybridpilot", "hybridpilot")
    rng = np.random.default_rng(0)
    n = 400
    trip = {"v": np.abs(rng.normal(12, 4, n)), "grade": rng.normal(0, 0.02, n)}
    vp = dict(arch="hev_powersplit", hd=False, mass=1600.0, m_eff=1650.0, P_eng_max=90e3, P_mot_max=60e3, E_batt=4.5e6,
              CdA=0.65, Crr=0.009, soc_lo=0.4, soc_hi=0.8)
    k, L = 100, 30
    base = hp.TripContext(vp, trip).preview(k, L)
    changed = {key: val.copy() for key, val in trip.items()}
    changed["v"][k + L + 1:] += 10.0
    changed["grade"][k + L + 1:] += 0.05
    np.testing.assert_array_equal(hp.TripContext(vp, changed).preview(k, L), base)


def _energynet_norm_stats():
    """train.norm_stats, read from the source: importing train.py needs torch and puts EnergyNet's `dataset` and
    `features` modules under names other models' tests also use."""
    src = (ROOT / "models/energynet/training/train.py").read_text(encoding="utf-8")
    ns = {"np": np}
    exec(src[src.index("def norm_stats"):src.index("def export")], ns)
    return ns["norm_stats"]


def test_energynet_robust_norm_keeps_work_shares_alive():
    norm_stats = _energynet_norm_stats()
    rng = np.random.default_rng(1)
    X = rng.normal(0.3, 0.1, size=(20000, 3))
    X[:20, 1] = rng.uniform(1e5, 6e5, 20)          # 0.1 % moving-downhill trips with Ep ~ 0
    mu0, sd0 = norm_stats(X)                        # historical rungs
    mu1, sd1 = norm_stats(X, robust=True)
    assert sd0[1] > 1e3                              # the defect: the column's scale is set by the outliers
    assert sd1[1] < 0.2 and abs(mu1[1] - 0.3) < 0.02
    np.testing.assert_allclose(sd1[[0, 2]], sd0[[0, 2]], rtol=0.05)


def test_lawfinder_warns_outside_declared_ranges():
    lf = _load("lawfinder", "lawfinder").LawFinder()
    bldg = lf["bldg"]
    inside = dict(H=3.0, C=0.5, A=1.5, Y=2000)
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("error")
        bldg.predict(**inside)                       # silent inside the bank's ranges
    with pytest.warns(UserWarning, match="outside the declared range"):
        bldg.predict(**(inside | {"Y": 2025}))
    with pytest.raises(ValueError):
        bldg.predict(strict=True, **(inside | {"Y": 2025}))
    assert bldg.out_of_range(**(inside | {"Y": np.array([1990, 2024, 2030])})) == {"Y": 2}
