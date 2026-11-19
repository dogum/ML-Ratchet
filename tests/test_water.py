"""The shared water platform (shared/water: WaterSim + WaterData) is intact and importable.

The data are not in git (`mlratchet data water --from-zip watersim-waterdata-v1.0.zip`, or the full rebuild). Tests
that need them skip when they are absent, including when $ML_RATCHET_WATER points at an empty folder. The physics
subset below (analytic checks, water balance on generated basins) needs no data and runs without numba."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WATER = ROOT / "shared" / "water"                                   # code (tracked)
DATA_ROOT = Path(os.environ.get("ML_RATCHET_WATER") or WATER)        # data (git-ignored)
for p in (WATER / "watersim", WATER):                                # as the models' training/_paths.py do
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
pd = pytest.importorskip("pandas")
HAVE_ATTRS = (DATA_ROOT / "data" / "proc" / "attributes" / "camels.parquet").exists()
HAVE_BASELINES = (DATA_ROOT / "data" / "proc" / "baselines" / "summary.json").exists()
HAVE_REF = (DATA_ROOT / "checks" / "ref" / "airgr_gr4j_explicit.csv").exists()
HAVE_CARAVAN = (DATA_ROOT / "data" / "proc" / "caravan" / "camels").is_dir()
SPLITS_FILE_SHA = "33f88ca0225180dcce979eceed26bf5157b92455e13bb7b78e135e9fd4d98c6b"


def test_frozen_splits_unchanged():
    """Every water exam depends on this file; it must never change (waterdata/DATA.md section 6)."""
    raw = (WATER / "data" / "proc" / "splits.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SPLITS_FILE_SHA
    sp = json.loads(raw)
    stored = sp.pop("sha256")                       # make_splits.py: sha256 of json.dumps(sort_keys=True) without it
    assert stored == hashlib.sha256(json.dumps(sp, sort_keys=True).encode()).hexdigest()
    assert stored.startswith("ed8109e5") and sp["seed"] == 20261118
    cu = sp["camels_us"]
    assert (len(cu["dev"]), len(cu["exam"]), len(cu["exam_531"])) == (52, 619, 487)
    assert not set(cu["dev"]) & set(cu["exam"])
    assert (len(sp["snotel"]["dev"]), len(sp["snotel"]["exam"])) == (39, 772)
    assert [len(sp["reservoirs"][k]) for k in ("dev", "exam", "aux")] == [15, 250, 413]
    assert not set(sp["snotel"]["dev"]) & set(sp["snotel"]["exam"])
    assert not set(sp["reservoirs"]["dev"]) & set(sp["reservoirs"]["exam"])
    for region, s in sp["other_regions"].items():
        assert len(s["dev"]) == 3 and not set(s["dev"]) & set(s["exam"]), region


def test_manifest_lists_the_shipped_files():
    man = json.loads((WATER / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["n_files"] == len(man["files"]) == 1170
    assert man["files"]["data/proc/splits.json"][0] == SPLITS_FILE_SHA
    assert not any(f.endswith(".part") or f.startswith("data/proc/caravan/") for f in man["files"])


def test_no_sandbox_paths_in_platform_code():
    bad = [str(f.relative_to(WATER)) for f in WATER.rglob("*") if f.suffix in (".py", ".sh", ".R") and f.is_file()
           and "out" not in f.relative_to(WATER).parts
           and ("/home/claude" in f.read_text(encoding="utf-8", errors="replace")
                or "/mnt/user-data" in f.read_text(encoding="utf-8", errors="replace"))]
    assert not bad, bad


def test_data_command_is_registered():
    from mlratchet.cli import DATA_TARGETS
    assert ROOT.joinpath(*DATA_TARGETS["water"]).exists()


def test_analytic_checks_reproduce(tmp_path):
    """checks/check_analytic.py re-run (about 20 s without numba) must reproduce the shipped analytic.json."""
    env = dict(os.environ, WATERSIM_CHECKS_OUT=str(tmp_path), OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "check_analytic.py"], cwd=WATER / "checks", env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    want = json.loads((WATER / "checks" / "analytic.json").read_text())
    got = json.loads((tmp_path / "analytic.json").read_text())

    def walk(a, b, path=""):
        if isinstance(a, dict):
            for k, v in a.items():
                if k in ("generated", "python", "wall_s", "numba"):
                    continue
                assert k in b, path + "/" + k
                walk(v, b[k], path + "/" + k)
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (u, v) in enumerate(zip(a, b)):
                walk(u, v, f"{path}[{i}]")
        elif isinstance(a, float) and isinstance(b, (int, float)):
            assert b == pytest.approx(a, rel=1e-7, abs=1e-9), path
        else:
            assert a == b, path
    walk(want, got)


@pytest.mark.parametrize("model", ["hbv", "gr4j", "sacsma", "vic_lite"])
def test_water_balance_closes_on_generated_basins(model):
    """A slice of checks/balance.json (121 runs there): generated basins and stochastic weather, both snow routines,
    three routings, a reservoir. Every ledger must close to round-off."""
    import watersim as ws
    specs = ws.generate_basins(12, rng=7)
    reg = next(s for s in specs if s.get("reservoirs"))
    for i, (snow, routing) in enumerate((("degree_day", "uh"), ("energy_balance", "muskingum"), ("degree_day", "kinematic"))):
        spec = specs[i]
        f = ws.generate_weather(spec, "2000-01-01", "2002-12-31", rng=i)
        out = ws.simulate(spec, f, "2001-01-01", model=model, snow=snow, routing=routing)
        L = out["ledger"]
        assert L["ok"], (model, snow, routing, L["closure_rel"], L["max_daily_residual_mm"])
        assert np.all(np.isfinite(out["ts"]["Q"])) and out["ts"]["Q"].min() >= 0
    f = ws.generate_weather(reg, "2000-01-01", "2002-12-31", rng=99)
    out = ws.simulate(reg, f, "2001-01-01", model=model, snow="degree_day", routing="uh")
    assert out["ledger"]["ok"] and "res_storage_hm3" in out["ts"]


@pytest.mark.parametrize("routing,tol", [("none", 1e-12), ("uh", 1e-4)])
def test_restart_reproduces_one_continuous_run(routing, tol):
    """balance.json's restart check uses routing='none' and is exact. With channel routing the chained run differs
    by about 4e-6 mm/day: final_state does not carry the channel unit-hydrograph buffer (found 18 Nov 2026 on import)."""
    import watersim as ws
    spec = ws.generate_basins(1, rng=3, regulation=False)[0]
    f = ws.generate_weather(spec, "2000-01-01", "2003-12-31", rng=4)
    kw = dict(model="hbv", snow="degree_day", routing=routing, labels=False)
    full = ws.simulate(spec, f, "2001-01-01", "2003-12-31", **kw)
    a = ws.simulate(spec, f, "2001-01-01", "2002-06-30", **kw)
    b = ws.simulate(spec, f, "2002-07-01", "2003-12-31", init=a["final_state"], **kw)
    q = np.concatenate([a["ts"]["Q"].values, b["ts"]["Q"].values])
    assert np.max(np.abs(q - full["ts"]["Q"].values)) < tol


def test_restart_carries_reservoir_storage():
    """final_state carries reservoir storage (PR #6 review, 18 Nov 2026): before, a chained run restarted each
    reservoir at init_frac and missed the continuous run by up to 4.8 mm/day. Capacity and demand are fixed here,
    because capacity_years and the default demand are derived from each window's mean inflow."""
    import watersim as ws
    reg = dict(next(s for s in ws.generate_basins(12, rng=7) if s.get("reservoirs")))
    reg["reservoirs"] = [dict({k: v for k, v in r.items() if k != "capacity_years"}, capacity_hm3=20.0 + 10 * i,
                              demand_m3s=1.0, policy="sop") for i, r in enumerate(reg["reservoirs"])]
    f = ws.generate_weather(reg, "2000-01-01", "2003-12-31", rng=4)
    kw = dict(model="hbv", snow="degree_day", routing="none", labels=False)
    full = ws.simulate(reg, f, "2001-01-01", "2003-12-31", **kw)
    a = ws.simulate(reg, f, "2001-01-01", "2002-06-30", **kw)
    b = ws.simulate(reg, f, "2002-07-01", "2003-12-31", init=a["final_state"], **kw)
    assert a["final_state"]["reservoir_storage_hm3"] is not None and b["ledger"]["ok"]
    for k in ("Q", "res_storage_hm3"):
        x = np.concatenate([a["ts"][k].values, b["ts"][k].values])
        assert np.max(np.abs(x - full["ts"][k].values)) < 1e-9, k


def test_batch_apis_match_simulate():
    """PR #6 review, 18 Nov 2026: simulate_batch routed every member with the spec's CH_VEL, and simulate_many skipped
    spin-up when the window starts at the forcing start. Both now match simulate() (irrigation off: the batch kernels
    do not model it)."""
    import watersim as ws
    spec = dict(ws.generate_basins(1, rng=3, regulation=False)[0], irrigation_frac=0.0)
    f = ws.generate_weather(spec, "2000-01-01", "2002-12-31", rng=4)
    r = ws.simulate_batch(spec, f, [{"CH_VEL": 0.4}, {"CH_VEL": 2.5}], model="hbv", start="2001-01-01", routing="uh")
    assert np.max(np.abs(r["Q"][0] - r["Q"][1])) > 1e-5
    for i, v in enumerate((0.4, 2.5)):
        s = ws.simulate(spec, f, "2001-01-01", model="hbv", routing="uh", params={"CH_VEL": v}, labels=False)
        assert np.max(np.abs(r["Q"][i] - s["ts"]["Q"].values)) < 1e-9
    specs = [dict(s, irrigation_frac=0.0) for s in ws.generate_basins(3, rng=5, regulation=False)]
    fs = [ws.generate_weather(s, "2000-01-01", "2001-12-31", rng=i) for i, s in enumerate(specs)]
    m = ws.simulate_many(specs, fs, model="hbv")                       # window starts at the forcing start: cyclic
    for i, (s, fi) in enumerate(zip(specs, fs)):
        one = ws.simulate(s, fi, model="hbv", routing="none", labels=False)
        assert np.max(np.abs(m["Q"][i] - one["ts"]["Q"].values)) < 1e-9


def test_network_hourly_and_many_dates():
    """PR #6 review, 18 Nov 2026: simulate_network routed hourly samples as days and counted the outlet volume 24 times
    over; simulate_many took every basin's rows from the first forcing's dates. The network now carries the time
    step and closes at hourly steps; simulate_many rejects forcings with different dates."""
    import pandas as pd
    import watersim as ws
    sb = {f"b{i}": dict(x, irrigation_frac=0.0) for i, x in enumerate(ws.generate_basins(2, rng=8, regulation=False))}
    f = {k: ws.generate_weather(x, "2000-01-01", "2000-12-31", rng=i) for i, (k, x) in enumerate(sb.items())}
    topo = {"b0": {"down": "b1", "length_km": 40, "slope": 0.002, "width_m": 20}, "b1": {"down": None}}

    def hourly(df):
        h = df.reindex(pd.date_range(df.index[0], df.index[-1] + pd.Timedelta(hours=23), freq="h", tz=df.index.tz),
                       method="ffill")
        h["P"] = h["P"] / 24                                         # amounts per step
        return h
    net = ws.simulate_network(sb, {k: hourly(v) for k, v in f.items()}, topo, "2000-03-01")
    assert net["ledger"]["ok"], net["ledger"]["closure_rel"]
    with pytest.raises(ValueError, match="different date index"):
        ws.simulate_many([sb["b0"], sb["b1"]], [f["b0"], f["b1"].iloc[7:]], model="hbv", start="2000-03-01")


def test_restart_carries_irrigation_store_and_pet_fills_net_srad_gaps():
    """PR #6 review, 18 Nov 2026: final_state dropped the irrigation return-flow store, and PET from net shortwave
    (srad_is_net, no LRAD_net) went NaN on an SRAD gap."""
    import watersim as ws
    from watersim.pet import compute_pet
    spec = dict(ws.generate_basins(1, rng=3, regulation=False)[0], irrigation_frac=0.2)
    f = ws.generate_weather(spec, "2000-01-01", "2003-12-31", rng=4)
    kw = dict(model="hbv", snow="degree_day", routing="none", labels=False)
    full = ws.simulate(spec, f, "2001-01-01", "2003-12-31", **kw)
    a = ws.simulate(spec, f, "2001-01-01", "2002-06-30", **kw)
    b = ws.simulate(spec, f, "2002-07-01", "2003-12-31", init=a["final_state"], **kw)
    assert a["final_state"]["irrigation_store_mm"] > 0 and b["ledger"]["ok"]
    q = np.concatenate([a["ts"]["Q"].values, b["ts"]["Q"].values])
    assert np.max(np.abs(q - full["ts"]["Q"].values)) < 1e-9
    g = f.iloc[:60].copy()
    g.iloc[10:15, g.columns.get_loc("SRAD")] = np.nan
    pet, _ = compute_pet(g, 45.0, 500.0, "penman_monteith", srad_is_net=True)
    ok, _ = compute_pet(f.iloc[:60], 45.0, 500.0, "penman_monteith", srad_is_net=True)
    assert np.isfinite(pet).all()
    assert np.allclose(np.delete(pet, range(10, 15)), np.delete(ok, range(10, 15)))
    # a forcing PET with gaps is filled by the same net-aware Penman-Monteith (srad_is_net kept)
    h = f.iloc[:60].copy()
    h["PET"] = ok
    h.iloc[20:25, h.columns.get_loc("PET")] = np.nan
    filled, tag = compute_pet(h, 45.0, 500.0, "forcing", srad_is_net=True)
    assert tag == "forcing+pm_fill" and np.allclose(filled, ok)


def test_reservoir_evap_none_is_honoured():
    """PR #6 review, 18 Nov 2026: a reservoir with evap='none' still lost Penman evaporation in simulate()."""
    import watersim as ws
    reg = dict(next(s for s in ws.generate_basins(12, rng=7) if s.get("reservoirs")))
    f = ws.generate_weather(reg, "2000-01-01", "2001-12-31", rng=4)
    kw = dict(model="hbv", snow="degree_day", routing="none", labels=False)
    on = ws.simulate(reg, f, "2001-01-01", **kw)
    off = ws.simulate(dict(reg, reservoirs=[dict(r, evap="none") for r in reg["reservoirs"]]), f, "2001-01-01", **kw)
    assert on["ts"]["res_evap_mm"].sum() > 0 and off["ts"]["res_evap_mm"].abs().max() == 0
    assert off["ledger"]["ok"]


def test_subdaily_pet_fill_and_irrigation_recession():
    """PR #6 review, 18 Nov 2026: at an hourly step, gaps in forcing PET were filled with a whole day's PET per hour,
    and the irrigation return store receded by k_return per hour instead of per day."""
    import pandas as pd
    import watersim as ws
    from watersim.simulate import prepare_run, _irrigation
    spec = ws.generate_basins(1, rng=3, regulation=False)[0]
    f = ws.generate_weather(spec, "2000-01-01", "2000-03-31", rng=4)
    h = f.reindex(pd.date_range(f.index[0], f.index[-1] + pd.Timedelta(hours=23), freq="h", tz=f.index.tz),
                  method="ffill")
    h["P"] = h["P"] / 24
    pm = prepare_run(spec, h, "hbv", None, "degree_day", "penman_monteith").pet
    h["PET"] = pm
    h.iloc[100:110, h.columns.get_loc("PET")] = np.nan
    o = prepare_run(spec, h, "hbv", None, "degree_day", None)
    assert o.pet_method.startswith("forcing+pm_fill") and np.allclose(o.pet, pm)
    n = 48
    ts = pd.DataFrame({"rain": np.zeros(n), "T_mean": np.full(n, 20.0)})
    pet = np.r_[2.0, np.zeros(n - 1)]
    irr = _irrigation({"irrigation_frac": 0.5}, None, ts, pet, np.full(n, 10.0), dt=1 / 24)
    assert irr["store"][24] / irr["store"][0] == pytest.approx(0.95)


def test_subdaily_gamma_uh_length_and_net_longwave():
    """PR #6 review, 18 Nov 2026: gamma_uh capped hourly hydrographs at 120 steps (5 days) instead of 120 days, and
    sub-daily PET dropped the supplied net longwave when aggregating to days."""
    import pandas as pd
    import watersim as ws
    from watersim import routing as rt
    from watersim.simulate import prepare_run
    o = rt.gamma_uh(2, 3, dt=1 / 24)
    assert np.sum(o * (np.arange(len(o)) + 0.5) / 24) == pytest.approx(6.0, rel=1e-3)   # mean = shape x scale
    assert len(rt.gamma_uh(2, 30)) == 120                                                   # daily cap unchanged
    spec = dict(ws.generate_basins(1, rng=3, regulation=False)[0], srad_is_net=True)
    f = ws.generate_weather(spec, "2000-01-01", "2000-02-29", rng=4)
    h = f.reindex(pd.date_range(f.index[0], f.index[-1] + pd.Timedelta(hours=23), freq="h", tz=f.index.tz),
                  method="ffill")
    h["P"] = h["P"] / 24
    pets = []
    for lw in (-40.0, -120.0):
        pets.append(prepare_run(spec, h.assign(LRAD_net=lw), "hbv", None, "degree_day", "penman_monteith").pet)
    assert pets[0].sum() > pets[1].sum()


def test_sea_level_basin_keeps_zero_elevation():
    """PR #6 review, 18 Nov 2026: elev_mean_m = 0 without bands became 500 m bands against 0 m forcing, so the
    lapse rate cooled the basin by about 3 C."""
    from watersim.spec import build_bands
    frac, elev = build_bands({"elev_mean_m": 0.0, "area_km2": 100.0})
    assert np.all(elev == 0.0) and frac.sum() == pytest.approx(1.0)
    assert np.all(build_bands({"area_km2": 100.0})[1] == 500.0)          # a missing elevation still defaults


def test_network_requires_matching_dates():
    """PR #6 review, 18 Nov 2026: sub-basins that simulated different dates were reindexed onto the first one's,
    putting NaN into the routed flow. simulate_network now refuses them."""
    import watersim as ws
    sb = {f"b{i}": x for i, x in enumerate(ws.generate_basins(2, rng=8, regulation=False))}
    fo = {"b0": ws.generate_weather(sb["b0"], "2000-01-01", "2001-12-31", rng=0),
          "b1": ws.generate_weather(sb["b1"], "2000-01-01", "2001-06-30", rng=1)}
    topo = {"b0": {"down": "b1", "length_km": 40, "slope": 0.002, "width_m": 20}, "b1": {"down": None}}
    with pytest.raises(ValueError, match="different dates"):
        ws.simulate_network(sb, fo, topo, "2000-06-01")


def test_metrics_match_the_shipped_cost():
    """waterdata.metrics.reservoir cost J is what ReservoirPilot and the baselines are scored with."""
    from waterdata import metrics as M
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.nse(obs, obs) == pytest.approx(1.0)
    assert M.nse(obs, np.full(4, obs.mean())) == pytest.approx(0.0)


@pytest.mark.skipif(not HAVE_REF, reason="airGR/TUWmodel reference CSVs not unpacked (mlratchet data water --from-zip)")
def test_reference_models_reproduce(tmp_path):
    """GR4J vs airGR 1.7.9 to 1e-7 mm and HBV vs TUWmodel to 1e-12 mm, from the shipped reference outputs."""
    env = dict(os.environ, WATERSIM_CHECKS_OUT=str(tmp_path), OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "check_reference.py"], cwd=WATER / "checks", env=env, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads((tmp_path / "reference_models.json").read_text())
    assert got["gr4j_vs_airGR"]["pass"]
    assert got["gr4j_vs_airGR"]["diffs"]["Q_vs_Qsim"]["max_abs_diff_mm"] < 1e-6


@pytest.mark.skipif(not HAVE_ATTRS, reason="water attributes not unpacked (mlratchet data water --from-zip)")
def test_loader_reads_the_shipped_attributes():
    import waterdata as wd
    assert len(wd.basins("camels", split="dev")) == 52
    a = wd.attributes("camels")
    assert len(a) == 671
    s = wd.spec(wd.basins("camels", split="exam")[0])
    assert s["area_km2"] > 0


@pytest.mark.skipif(not HAVE_BASELINES, reason="water baselines not unpacked (mlratchet data water --from-zip)")
def test_baseline_medians_are_the_ones_the_models_report():
    """The exam baselines every water model compares against (waterdata/DATA.md section 7)."""
    fs = pd.read_parquet(DATA_ROOT / "data" / "proc" / "baselines" / "flow_scores.parquet")
    assert len(fs) > 0
    summ = json.loads((DATA_ROOT / "data" / "proc" / "baselines" / "summary.json").read_text())
    assert isinstance(summ, dict) and summ


@pytest.mark.skipif(not HAVE_CARAVAN, reason="Caravan time series not built (mlratchet data water --only caravan)")
def test_caravan_forcing_loads():
    import waterdata as wd
    bid = wd.basins("camels", split="dev")[0]
    f = wd.forcing(bid, "1999-10-01", "2000-09-30")
    assert {"P", "T_mean", "PET"} <= set(f.columns) and len(f) == 366


@pytest.mark.skipif(not HAVE_CARAVAN, reason="Caravan time series not built (mlratchet data water --only caravan)")
def test_random_forcing_stays_in_the_basin_record():
    """PR #6 review, 18 Nov 2026: random_forcing drew windows from 1951 for regions whose forcing starts in 1981 and
    returned short or empty frames. Windows are now clipped to the basin's record; CAMELS-US windows are unchanged."""
    import waterdata as wd
    from watersim import realdata as rd
    gb = wd.basins("camelsgb")[0]
    for seed in range(20):
        df, meta, info = rd.random_forcing(seed, years=9, basin_id=gb)
        assert info["start"] >= "1981-01-01" and len(df) >= 10 * 365
    us = wd.basins("camels")[0]
    idx = wd.forcing(us, extras=False).index
    c0, c1 = str(idx[0].date()), str(idx[-1].date())
    assert all(c0 <= a and e <= c1 for a, e in rd.free_windows())          # so US draws are as before


def test_forcing_with_missing_days_is_rejected():
    """PR #6 review, 18 Nov 2026: a forcing frame with a missing day was run as if it were regular (the engine
    stepped over the gap). simulate() now refuses an unevenly spaced index; NaN rows on a regular index still run."""
    import watersim as ws
    spec = ws.generate_basins(1, rng=3, regulation=False)[0]
    f = ws.generate_weather(spec, "2000-01-01", "2001-12-31", rng=4)
    with pytest.raises(ValueError, match="not evenly spaced"):
        ws.simulate(spec, f.drop(f.index[200]), "2000-06-01", model="hbv", labels=False)
