"""The shared grid platform (shared/grid: GridSim + GridData) is intact and importable.

The data are not in git (`mlratchet data grid`); the tests that need them skip when they are absent."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "shared" / "grid"
sys.path.insert(0, str(GRID / "gridsim"))
sys.path.insert(0, str(GRID))
HAVE_DATA = (GRID / "data" / "proc" / "fleet" / "fleet_ERCO.parquet").exists()


def test_frozen_splits_unchanged():
    """Every grid exam depends on this file; it must never change (HANDOFF.md section 4)."""
    raw = (GRID / "data" / "proc" / "splits.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest().startswith("f8787bc2")
    sp = json.loads(raw)
    assert sp["sha256_of_content"].startswith("895b4e9a")
    assert sp["time"]["dev"] == [2019, 2021] and sp["time"]["exam"] == [2023, 2024, 2025]


def test_manifest_lists_every_shipped_file():
    man = json.loads((GRID / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["n_files"] == len(man["files"]) == 1064
    assert man["splits_sha256_of_content"].startswith("895b4e9a")
    assert man["files"]["data/proc/splits.json"][0].startswith("f8787bc2")


def test_no_sandbox_paths_in_platform_code():
    bad = [str(f.relative_to(GRID)) for f in GRID.rglob("*") if f.suffix in (".py", ".sh") and f.is_file()
           and ("/home/claude" in f.read_text(encoding="utf-8", errors="replace")
                or "/mnt/user-data" in f.read_text(encoding="utf-8", errors="replace"))]
    assert not bad, bad


def test_data_command_is_registered():
    from mlratchet.cli import DATA_TARGETS
    assert (ROOT.joinpath(*DATA_TARGETS["grid"])).exists()


def test_rocof_matches_the_swing_equation():
    """RoCoF = dP f0 / (2 H S) for pure inertia (checks/frequency_analytic.json: exact to ~1e-14)."""
    from gridsim.frequency import frequency_event, make_system_state
    for H, mva, dP in [(2, 20000, 500), (6, 50000, 1800)]:
        st = make_system_state([dict(tech="coal_steam", MW=0, pmax_MW=mva * 0.9, H_s=H, mva=mva,
                                     governor=dict(enabled=False))], load_MW=mva, region="GB", D=0, ufls=[])
        r = frequency_event(st, dP, t_end=2.0)
        assert r["rocof_Hz_s"] == pytest.approx(-dP * 50.0 / (2 * H * mva), rel=1e-9)


def test_dc_and_ac_powerflow_on_a_three_bus_case():
    """Flows add up and AC losses are small and positive on a tiny meshed network."""
    from gridsim import powerflow
    ppc = {"baseMVA": 100.0,
           "bus": np.array([[1, 3, 0, 0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
                            [2, 1, 150, 30, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
                            [3, 1, 100, 20, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9]], float),
           "gen": np.array([[1, 250, 0, 300, -300, 1.0, 100, 1, 400, 0]], float),
           "branch": np.array([[1, 2, 0.01, 0.10, 0.02, 250, 250, 250, 0, 0, 1, -360, 360],
                               [1, 3, 0.01, 0.10, 0.02, 250, 250, 250, 0, 0, 1, -360, 360],
                               [2, 3, 0.01, 0.10, 0.02, 250, 250, 250, 0, 0, 1, -360, 360]], float)}
    dc = powerflow(ppc, method="dc")
    f = np.asarray(dc["Pf_MW"])
    assert f[0] + f[1] == pytest.approx(250.0, abs=1e-6)          # slack covers 250 MW of load, lossless
    ac = powerflow(ppc, method="ac")
    assert ac["converged"]
    assert 0 < float(np.sum(ac["losses_MW"])) < 10


@pytest.mark.skipif(not HAVE_DATA, reason="grid data not present (mlratchet data grid)")
def test_griddata_reads_the_frozen_dev_sets():
    import griddata as gd
    assert len(gd.events("GB", split="dev")) == 4
    assert len(gd.events("ERCO", split="dev")) == 5
    d = gd.eia930("ERCO", split="dev")
    assert sorted(set(d.index.year)) == [2019, 2021]


def test_per_state_loss_lists_are_read_per_state():
    """HANDOFF P0.6: a list of per-state loss lists was read as one event list for every state."""
    from gridsim.frequency import frequency_event, make_system_state
    mk = lambda: make_system_state([dict(tech="coal_steam", MW=0, pmax_MW=27000, H_s=5, mva=30000,  # noqa: E731
                                         governor=dict(enabled=False))], load_MW=30000, region="GB", D=1.0, ufls=[])
    ev = [[(0.0, 500.0)], [(0.0, 800.0), (5.0, 200.0)]]
    a = frequency_event([mk(), mk()], ev, t_end=10, record=False)
    one = [frequency_event(mk(), e, t_end=10)["nadir_Hz"] for e in ev]
    assert [x["nadir_Hz"] for x in a] == pytest.approx(one, abs=1e-9)


def test_gb_label_corrections_apply_to_v1_rows_only():
    import pandas as pd
    import griddata as gd
    c = gd.label_corrections()
    r = c[c.event_id == "GB_20191203T200030"].iloc[0]
    assert (float(r.old), float(r.new)) == (1787.7, 854.7)
    d = pd.DataFrame(dict(event_id=["GB_20191203T200030", "GB_X"], loss_MW=[1787.7, 500.0],
                          loss_source=["v1", "v1"], loss_note=[None, None]))
    a = gd.apply_label_corrections(d)
    assert a.loss_MW.tolist() == [854.7, 500.0] and a.loss_MW_v1.tolist() == [1787.7, 500.0]
    assert a.loss_corrected.tolist() == [True, False]
    assert gd.apply_label_corrections(a).loss_MW.tolist() == [854.7, 500.0]     # idempotent
    with pytest.warns(UserWarning):
        gd.apply_label_corrections(d.assign(loss_MW=[1000.0, 500.0]))           # unexpected value: left alone


def test_physics_1_2_is_1_1_plus_rt_scarcity_only():
    """GridSim 1.2 (opt-in) changes nothing but the RT scarcity layer: same calibration, same outage defaults; the
    scarcity adder is zero under 1.0 and 1.1 and non-negative under 1.2."""
    import pandas as pd
    from gridsim import fleet, market

    assert fleet.calibration("1.2") == fleet.calibration("1.1")
    assert fleet.OUTAGE_DEFAULTS["1.2"] == fleet.OUTAGE_DEFAULTS["1.1"]
    times = pd.date_range("2019-07-01", periods=24 * 60, freq="h", tz="UTC")
    for v in ("1.0", "1.1"):
        assert not market.enabled(v)
        assert np.all(market.rt_scarcity(times, version=v, seed=3) == 0.0)
    if market.params("ERCO") is None:
        pytest.skip("RT scarcity parameter file not present")
    a = market.rt_scarcity(times, version="1.2", seed=3)
    assert market.enabled("1.2") and a.min() >= 0.0 and a.max() > 0.0
    assert np.array_equal(a, market.rt_scarcity(times, version="1.2", seed=3))   # own random stream: reproducible
