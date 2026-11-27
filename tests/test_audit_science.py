"""Regression tests for the standalone-model audit findings fixed on 27 Nov 2026 (docs/audits/)."""
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


def test_airfoilnet_lednicer_blocks_read_like_selig():
    af = _load("airfoilnet", "airfoilnet")
    net = af.AirfoilNet()
    d = json.loads((ROOT / "models/airfoilnet/examples/bw3_100000.json").read_text())
    X = np.asarray(d["coords"], float)
    k = int(np.argmin(X[:, 0]))
    lednicer = np.vstack([X[:k + 1][::-1], X[k:]])          # upper LE->TE, then lower LE->TE
    a, b = net.polar(X, Re=1e5), net.polar(lednicer, Re=1e5)
    np.testing.assert_allclose(a["cl"], b["cl"], atol=1e-9)  # was up to 5.44 apart


def test_cellsense_refuses_an_all_nan_discharge():
    cs = _load("cellsense", "cellsense")
    with pytest.raises(ValueError):
        cs.q_of_v(np.full(50, np.nan), np.full(50, np.nan))
    with pytest.raises(ValueError):
        cs.q_of_v_nmc(np.full(50, np.nan), np.full(50, np.nan))


def test_beatnet_fills_one_missing_sample_and_refuses_long_gaps():
    bn = _load("beatnet", "beatnet")
    model = bn.BeatNet()
    fs = 360.0
    t = np.arange(int(20 * fs)) / fs
    ecg = np.sin(2 * np.pi * 1.2 * t) ** 63 * 1.5             # a crude beat train, ~72 bpm
    ref = model.classify(ecg, fs)
    one = ecg.copy(); one[len(one) // 2] = np.nan
    out = model.classify(one, fs)
    assert np.isfinite(out["probs"]).all()                   # was NaN for every beat, each labelled 'N'
    np.testing.assert_allclose(out["probs"], ref["probs"], atol=0.05)
    gap = ecg.copy(); gap[1000:1100] = np.nan
    with pytest.raises(ValueError):
        model.classify(gap, fs)


def test_swellcast_refuses_an_all_nan_window():
    sc = _load("swellcast", "swellcast")
    with pytest.raises(ValueError):
        sc.SwellCast().read(np.full((3, 80, 34), np.nan))


def test_chirpnet_far_with_no_louder_background_is_the_poisson_limit():
    cn = _load("chirpnet", "chirpnet")
    model = cn.ChirpNet(str(ROOT / "models/chirpnet/chirpnet_weights.npz"))
    far, bound = model.far_per_hour(1e9)
    assert bound and far == pytest.approx(2.303 / (model.bg_live_s / 3600.0))
