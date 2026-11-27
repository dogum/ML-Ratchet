"""Regression tests for the grid-platform audit findings fixed on 27 Nov 2026 (docs/audits/)."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(model: str, module: str):
    """The model's module through the registry, which keeps sibling modules (pipeline, common, features) private."""
    import mlratchet
    return mlratchet.get(model).module(module)


@pytest.fixture(scope="module")
def gp():
    return _load("gridpulse", "gridpulse")


@pytest.fixture(scope="module")
def trace():
    d = pd.read_csv(ROOT / "models/gridpulse/examples/trace_example.csv")
    return d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)


def test_gridpulse_reading_at_n_ignores_later_samples(gp, trace):
    t, f = trace
    model = gp.GridPulse()
    i0 = 80
    for N in (5, 10, 30):
        base = model.read(t, f, N=N, onset_index=i0)
        g = f.copy()
        g[i0 + N - 1:i0 + N + 1] = np.nan          # a gap ending at the cutoff
        g[i0 + N + 1:] += 0.05                     # and a change after it
        moved = model.read(t, g, N=N, onset_index=i0)
        live = model.read(t[:i0 + N + 1], g[:i0 + N + 1], N=N, onset_index=i0)   # the record ends at +N s
        for k in ("loss_MW", "E_GWs", "nadir_Hz", "settle_Hz"):
            assert moved[k] == pytest.approx(live[k], rel=0, abs=1e-12), (N, k)
        assert np.isfinite(base["loss_MW"])


def test_gridpulse_complete_trace_unchanged_by_the_fix(gp, trace):
    t, f = trace
    model = gp.GridPulse()
    i0 = 80
    for N in (5, 10, 60):
        w_old = f[i0 - 20:i0 + 61] - np.nanmean(f[i0 - 60:i0])          # pre-fix window on a complete trace
        old = model.read_windows(w_old[None], N, "GB")
        new = model.read(t, f, N=N, onset_index=i0)
        for k in ("loss_MW", "E_GWs", "nadir_Hz", "settle_Hz"):
            assert new[k] == pytest.approx(float(old[k][0]), rel=0, abs=1e-9)


def test_plantdetective_missing_hour_inside_a_run_is_not_a_start():
    pdc = _load("plantdetective", "pdcore")
    x = np.r_[np.zeros(10), np.full(200, 150.0), np.zeros(30), np.full(200, 150.0), np.zeros(10)]
    holed = x.copy()
    holed[[60, 120, 300, 350]] = np.nan              # four isolated hours inside the two runs
    s, L, v = pdc.runs(pdc._bridge(np.where(np.isfinite(holed), holed, 0) > 0.5, ~np.isfinite(holed)))
    assert int(np.sum(v & (s > 0))) == 2                 # two real starts, not six
    s0, L0, v0 = pdc.runs(np.where(np.isfinite(holed), holed, 0) > 0.5)
    assert int(np.sum(v0 & (s0 > 0))) == 6               # what the old code counted
    gap = x.copy(); gap[205:215] = np.nan              # a stretch running -> missing -> off stays a stop
    s2, _, v2 = pdc.runs(pdc._bridge(np.where(np.isfinite(gap), gap, 0) > 0.5, ~np.isfinite(gap)))
    assert int(np.sum(v2 & (s2 > 0))) == 2


def test_batterypilot_hour_zero_does_not_read_its_own_rt_price():
    bp = _load("batterypilot", "batterypilot")
    n = 72
    rng = np.random.default_rng(3)
    path = dict(rt=rng.uniform(20, 60, n), da=rng.uniform(20, 60, n), df=np.full(n, 30000.0), T=np.full(n, 15.0),
                hour=np.arange(n) % 24, day=np.arange(n) // 24, dow=(np.arange(n) // 24) % 7, doy=np.arange(n) // 24)
    bat = bp.Battery(P=1.0, duration_h=2.0)
    t = np.arange(n)
    X0, _ = bp.build(path, t, bat.d, bat.c_lin(path["T"]))
    moved = dict(path, rt=path["rt"].copy())
    moved["rt"][0] += 100.0
    X1, _ = bp.build(moved, t, bat.d, bat.c_lin(moved["T"]))
    np.testing.assert_array_equal(X1[0], X0[0])          # hour 0 cannot see hour 0's RT price
    assert np.any(X1[1] != X0[1])                        # hour 1 can


def test_batterypilot_causal_wear_reads_only_past_temperatures():
    bp = _load("batterypilot", "batterypilot")
    bat = bp.Battery(P=1.0, duration_h=2.0)
    T = np.linspace(0, 30, 96)
    base = bp.causal_wear(bat, T)
    T2 = T.copy(); T2[60:] += 5.0
    np.testing.assert_array_equal(bp.causal_wear(bat, T2)[:108], base[:108])  # hour h uses T[h - 48]
    # a plan issued at hour s covers targets up to s + 34 h; their wear must come from hours before s
    for s0 in (14, 38, 62):
        for target in range(s0, min(s0 + 35, len(T))):
            moved = T.copy(); moved[s0:] += 9.0                                   # change everything from s0 on
            assert bp.causal_wear(bat, moved)[target] == base[target]
