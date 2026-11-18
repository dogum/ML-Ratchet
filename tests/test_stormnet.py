"""Protect StormNet's data roles, forecast clock and release/scorecard agreement."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml
from mlratchet import load


MODEL = Path(__file__).resolve().parents[1] / "models" / "stormnet"


@pytest.fixture
def training(monkeypatch):
    # Training scripts are standalone programs. Isolate their sibling imports from other models.
    modules = {}
    for name in ("_paths", "noise", "split"):
        spec = importlib.util.spec_from_file_location(name, MODEL / "training" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def test_issue_cutoff_and_target(training, monkeypatch):
    split = training["split"]
    n = 90
    clock = np.datetime64("2008-01-01T00", "h") + np.arange(n)
    dst = np.arange(n, dtype=float)
    wind = np.tile([-3., 450., 2.], (n, 1))
    monkeypatch.setattr(split, "read_year", lambda *args, **kwargs: (clock, wind, dst, "fixture"))
    d, _, _, _ = split.period([2008])
    assert d["time"][0] == clock[36]
    assert d["dst"][0] == 35 and d["truth"][0] == 36
    np.testing.assert_array_equal(d["future"][0], [36, 37, 38])
    # A new future changes the answer key while the forecast remains bit-for-bit identical.
    wind[36:39, 0] = -42
    dst[36:39] = [-80, -60, -40]
    other, _, _, _ = split.period([2008])
    np.testing.assert_array_equal(d["wind"][0], other["wind"][0])
    assert other["truth"][0] == -80 and other["dst"][0] == 35


def test_gap_fill_and_sealed_roles(training):
    noise = training["noise"]
    a = np.tile([-2., 450., 1.], (7, 1))
    a[1:5, 0] = np.nan
    a[5, 0] = -20
    filled = noise.fill_wind(a)
    np.testing.assert_array_equal(filled[:5, 0], [-2., -2., -2., np.nan, np.nan])
    for y, labelled in [(2001, True), (2005, True), (2016, False), (2022, True), (2025, False)]:
        with pytest.raises(ValueError):
            noise.read_year(y, index=labelled)
    roles = [set(noise.PROTOCOL[k]) for k in ("training_wind_years", "frozen_wind_years",
             "dev_years", "primary_years", "sealed_reserve_years")]
    assert sum(map(len, roles)) == len(set.union(*roles))


def test_released_batch_and_uncertainty():
    net = load("stormnet").StormNet()
    with np.load(MODEL / "examples/storm.npz") as z:
        batch = net.predict(z["wind"], z["last_dst"])
        single = [net.predict(x, y) for x, y in zip(z["wind"], z["last_dst"])]
        for key in batch:
            np.testing.assert_allclose(batch[key], [r[key] for r in single], atol=1e-9)
        raw = net.predict(z["wind"], z["last_dst"], calibrated=False)
        np.testing.assert_allclose(batch["sigma"], raw["sigma"] * net.meta["widen_factor"])
        with pytest.raises(ValueError):
            net.predict(z["wind"][0, :-1], z["last_dst"][0])


def test_complete_exam_matches_release_and_card():
    result = json.loads((MODEL / "training/exam.json").read_text())
    card = yaml.safe_load((MODEL / "model.yaml").read_text())
    assert result["weights_sha256"] == hashlib.sha256((MODEL / "stormnet_weights.npz").read_bytes()).hexdigest()
    assert len(card["exam"]["results"]) == len(result["rows"]) == 5
    for advertised, row in zip(card["exam"]["results"], result["rows"].values()):
        assert advertised["verdict"] == row["verdict"]
        assert advertised["model"] == pytest.approx(row["scores"]["StormNet"], abs=.00051)
        assert len(advertised["baselines"]) == 4
    extreme = result["rows"]["Extreme minimum"]
    assert extreme["blocks"] == 1
    assert extreme["difference_ci95"] is None and extreme["coverage_ci95"] is None
