"""The shared building platform (shared/building: BuildingSim + BuildingData) is intact and importable.

The data are not in git (`mlratchet data building --from-zip ...`); the tests that need them skip when they are absent,
including when $ML_RATCHET_BUILDING points at an empty folder."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BLDG = ROOT / "shared" / "building"                                  # code (tracked)
DATA_ROOT = Path(os.environ.get("ML_RATCHET_BUILDING") or BLDG)        # data (git-ignored)
for p in (BLDG / "buildingsim", BLDG):                               # as the models' training/_paths.py do
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
HAVE_SURVEYS = (DATA_ROOT / "data" / "proc" / "eia" / "recs2020.parquet").exists() and \
    (DATA_ROOT / "data" / "proc" / "splits.json").exists()
HAVE_EPW = (DATA_ROOT / "buildingsim" / "refdata" / "weather" / "epw").is_dir()
HAVE_BDG2 = (DATA_ROOT / "data" / "proc" / "bdg2" / "buildings.parquet").exists() and HAVE_SURVEYS
HAVE_OTHER = (DATA_ROOT / "data" / "proc" / "other" / "catalogue.json").exists()
HAVE_PRIOR = (DATA_ROOT / "buildingsim" / "refdata" / "priors" / "resstock_prior.parquet").exists()


def test_frozen_splits_unchanged():
    """Every building exam depends on this file; it must never change (HANDOFF.md, DATA.md section 9)."""
    raw = (BLDG / "data" / "proc" / "splits.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "524aee157fffff7c45f3d119823bb3ccd81e73046bf9f15e8e8af523185fba75"
    sp = json.loads(raw)
    stored = sp.pop("sha256")                       # make_splits.py: sha256 of json.dumps(sort_keys=True) without it
    assert stored == hashlib.sha256(json.dumps(sp, sort_keys=True).encode()).hexdigest()
    assert stored.startswith("7ac15e9f") and sp["seed"] == 20261116
    assert (len(sp["bdg2"]["dev"]), len(sp["bdg2"]["exam"])) == (52, 1584)
    assert [len(sp["recs2020"][k]) for k in ("prior", "dev", "exam")] == [9247, 1848, 7401]
    assert [len(sp["cbecs2018"][k]) for k in ("prior", "dev", "exam")] == [3217, 661, 2558]
    assert (len(sp["faults"]["dev"]), len(sp["faults"]["exam"])) == (20, 217)
    assert not set(sp["bdg2"]["dev"]) & set(sp["bdg2"]["exam"])


def test_manifest_lists_the_shipped_files():
    man = json.loads((BLDG / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["n_files"] == len(man["files"])
    assert man["files"]["data/proc/splits.json"][0].startswith("524aee15")
    assert man["splits_sha256_field"].startswith("7ac15e9f")
    assert not any(f.endswith(".part") for f in man["files"])


def test_no_sandbox_paths_in_platform_code():
    bad = [str(f.relative_to(BLDG)) for f in BLDG.rglob("*") if f.suffix in (".py", ".sh") and f.is_file()
           and "out" not in f.relative_to(BLDG).parts
           and ("/home/claude" in f.read_text(encoding="utf-8", errors="replace")
                or "/mnt/user-data" in f.read_text(encoding="utf-8", errors="replace"))]
    assert not bad, bad


def test_data_command_is_registered():
    from mlratchet.cli import DATA_TARGETS
    assert ROOT.joinpath(*DATA_TARGETS["building"]).exists()


def test_steady_state_conduction_matches_the_5r1c_network():
    """checks/analytic.json: constant -10 C outdoors, 20 C inside -> heating = UA_network x 30 K to ~2e-9."""
    sys.path.insert(0, str(BLDG / "checks"))
    import check_analytic as ca
    r = ca.steady_state()
    for mass in ("light", "heavy"):
        assert r[mass]["rel_error_vs_network"] < 1e-7
        assert r[mass]["simulated_W"] == pytest.approx(11094.3292, rel=1e-6)
        assert r[mass]["ledger_closure"] < 1e-6


def test_a_short_simulation_closes_its_ledgers():
    """Two days of a heat-pump house on synthetic weather (no data files needed)."""
    import pandas as pd
    from buildingsim import simulate, ledger
    from buildingsim.weather import SIGMA
    idx = pd.date_range("2019-01-14", periods=48 * 4, freq="15min", tz="UTC")
    T = -5.0 + 5.0 * np.sin(np.arange(len(idx)) * 2 * np.pi / 96)
    w = pd.DataFrame(dict(T_out=T, RH=70.0, GHI=0.0, DNI=0.0, DHI=0.0, wind_speed=3.0, wind_dir=180.0,
                          pressure=101325.0, IR_horiz=SIGMA * (T + 273.15) ** 4 * 0.8, cloud=0.5), index=idx)
    spec = dict(primary_use="residential_single", floor_area_m2=150, year_built=2000, climate_zone="5A",
                system="heat_pump_air")
    out = simulate(spec, (w, dict(lat=41.9, lon=-87.6, elev=200.0, tz="UTC")), str(idx[0]), str(idx[-1]), dt="15min")
    assert ledger(out)["ok"]
    assert out["summary"]["kWh_by_meter"]["electricity"] > 0
    assert 18.0 < float(out["ts"]["T_zone"].iloc[-48:].mean()) < 22.0


@pytest.mark.skipif(not HAVE_EPW, reason="buildingsim/refdata not present (mlratchet data building --from-zip ...)")
def test_bundled_tmyx_weather_reads():
    from buildingsim import read_epw, epw_path
    w, meta = read_epw(epw_path("Chicago"))
    assert len(w) == 8760 and -30 < w["T_out"].min() < -10 and 30 < w["T_out"].max() < 40


@pytest.mark.skipif(not HAVE_SURVEYS, reason="building data not present (mlratchet data building --from-zip ...)")
def test_buildingdata_reads_the_frozen_splits():
    import buildingdata as bd
    assert len(bd.recs(split="dev")) == 1848 and len(bd.cbecs(split="exam")) == 2558
    f = bd.faults(split="exam")
    assert len(f) == 217 and set(f.split) == {"exam"}
    assert len(bd.tariffs()) == 19
    p = bd.tariff_prices("pge_e_tou_c", 2017)
    assert len(p) == 8760 and p["energy_$per_kWh"].between(0.0, 1.0).all()


@pytest.mark.skipif(not HAVE_BDG2, reason="BDG2 data not present (mlratchet data building --from-zip ...)")
def test_buildingdata_reads_bdg2_dev():
    import buildingdata as bd
    m = bd.bdg2_meta(split="dev")
    assert len(m) == 52 and len(bd.bdg2_meta()) == 1636 and set(m.split) == {"dev"}
    e = bd.bdg2("electricity", split="dev")
    assert e.shape == (17552, 52) and str(e.index.tz) == "UTC" and 0.9 < e.notna().mean().mean() <= 1.0
    assert len(bd.bdg2_quality()) == 3053 and len(bd.bdg2_specs("dev")) == 52
    w = bd.site_weather("Hog")
    assert len(w) == 17552 and -40 < w["T_out"].min() < -15 and 30 < w["T_out"].max() < 40


@pytest.mark.skipif(not HAVE_OTHER, reason="HUE/WPuQ data not present (mlratchet data building --from-zip ...)")
def test_buildingdata_reads_the_other_homes():
    import buildingdata as bd
    cat = bd.other("list")
    assert {"hue", "wpuq"} <= set(cat)
    assert len(bd.other("hue")) > 500_000 and len(bd.other("wpuq")) > 2_000_000


# ------------------------------------------------------------------ Phase 2 wrappers (opt-in; shared/building/NOTEBOOK.md)
def _synthetic_weather(start, days, T0, amp, tz="UTC", lat=41.9, lon=-87.6):
    import pandas as pd
    from buildingsim.weather import SIGMA
    idx = pd.date_range(start, periods=int(days * 24), freq="1h", tz="UTC")
    ph = np.arange(len(idx)) * 2 * np.pi / 24
    T = T0 + amp * np.sin(ph)
    ghi = np.clip(600.0 * np.sin(ph - np.pi / 2), 0, None)
    w = pd.DataFrame(dict(T_out=T, RH=60.0, GHI=ghi, wind_speed=3.0, wind_dir=180.0, pressure=101325.0,
                          IR_horiz=SIGMA * (T + 273.15) ** 4 * 0.8, cloud=0.4), index=idx)
    return w, dict(lat=lat, lon=lon, elev=200.0, tz=tz)


def test_phase2_flags_off_is_v1():
    """options=None, {} and a US calendar give the v1.0 arrays exactly; unknown options raise."""
    import buildingsim as bs
    w = _synthetic_weather("2019-07-01", 4, 24.0, 6.0, tz="America/Chicago")
    spec = dict(primary_use="office", floor_area_m2=2000, year_built=2000, climate_zone="5A", system="packaged_rtu")
    a = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h")
    for opts in ({}, {"holidays": "auto"}, {"holidays": "US"}):
        b = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h", options=opts)
        for c in a["ts"]:
            if a["ts"][c].dtype.kind == "f":
                assert np.array_equal(a["ts"][c].to_numpy(), b["ts"][c].to_numpy(), equal_nan=True), (opts, c)
    with pytest.raises(ValueError):
        bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-02", options={"nope": 1})
    assert bs.PHYSICS_VERSION == "1.0" and bs.__version__ == "1.2.0"


def _campus_pair(days=4, tmean=24.0, start="2019-07-01", **kw):
    w = _synthetic_weather(start, days, tmean, 6.0, tz="America/Chicago")
    spec = dict(primary_use="office", floor_area_m2=5000, year_built=2000, climate_zone="5A",
                system="district_steam_chw", hvac=dict(district_heat="hot_water"))
    return spec, w


def test_physics_default_is_v1(monkeypatch):
    """Physics 1.0 is the default: None, {'physics': '1.0'} and an unset BUILDINGSIM_PHYSICS give the v1.0 arrays bit
    for bit; the environment variable selects a version and the option overrides it; unknown versions raise."""
    import buildingsim as bs
    spec, w = _campus_pair()
    monkeypatch.delenv("BUILDINGSIM_PHYSICS", raising=False)
    a = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h")
    assert a["physics"] == "1.0" and "district_plant_kWh" not in a["ledger"]
    for opts in ({"physics": "1.0"}, {"seed": 5}):
        b = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h", options=opts)
        for c in a["ts"]:
            if a["ts"][c].dtype.kind == "f":
                assert np.array_equal(a["ts"][c].to_numpy(), b["ts"][c].to_numpy(), equal_nan=True), (opts, c)
    monkeypatch.setenv("BUILDINGSIM_PHYSICS", "1.1")
    e = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h")
    o = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h", options={"physics": "1.0"})
    assert e["physics"] == "1.1" and o["physics"] == "1.0"
    assert np.array_equal(o["ts"]["meter_chilled_water_kW"].to_numpy(), a["ts"]["meter_chilled_water_kW"].to_numpy())
    with pytest.raises(ValueError):
        bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-02", options={"physics": "2.0"})
    assert bs.PHYSICS_VERSIONS == ("1.0", "1.1", "1.2", "1.3", "1.4") and bs.PHYSICS_LATEST == "1.4"


def test_physics_11_district_plant():
    """1.1: the loop base is its own end use on the district meters (meter = systems + base, ledgers close), the
    hot-water loop locks out in warm weather, summer reheat adds heat that the chilled water removes."""
    import buildingsim as bs
    C = dict(q_hw=4.0, q_steam=3.0, q_chw=20.0, f_rh=0.0, T_hw_off=15.6, T_chw_off=10.0)
    spec, w = _campus_pair()
    v1 = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h")
    warm = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h", options={"physics": "1.1", "campus": C})
    L = bs.ledger(warm)
    assert L["ok"] and L["district_plant_kWh"]["closure_rel"] < 1e-9
    ts = warm["ts"]
    assert np.allclose(ts["plant_cooling_district_kW"], 20.0 * 5000 / 1000)     # 7-day mean > 10 C: loop on
    assert np.allclose(ts["plant_heating_district_kW"], 0.0)                    # 7-day mean > 15.6 C: loop off
    assert np.allclose(ts["meter_chilled_water_kW"], v1["ts"]["meter_chilled_water_kW"] + 100.0)
    rh = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-04", dt="1h",
                     options={"physics": "1.1", "campus": dict(C, f_rh=0.5, T_hw_off=None)})
    assert bs.ledger(rh)["ok"]
    k_rh, k_w = rh["summary"]["kWh_by_end_use"], warm["summary"]["kWh_by_end_use"]
    assert k_rh["heating_district"] > k_w["heating_district"] + 1.0              # summer reheat on the heat meter
    assert k_rh["cooling_district"] > k_w["cooling_district"]                    # removed again by chilled water
    sc, wc = _campus_pair(tmean=-2.0, start="2019-01-07")
    cold = bs.simulate(sc, wc, "2019-01-07 12:00", "2019-01-10", dt="1h", options={"physics": "1.1", "campus": C})
    assert np.allclose(cold["ts"]["plant_heating_district_kW"], 4.0 * 5000 / 1000)
    assert np.allclose(cold["ts"]["plant_cooling_district_kW"], 0.0)            # chilled-water loop locked out
    hosp = dict(sc, primary_use="healthcare")
    h = bs.simulate(hosp, wc, "2019-01-07 12:00", "2019-01-10", dt="1h", options={"physics": "1.1", "campus": C})
    assert np.allclose(h["ts"]["plant_cooling_district_kW"], 100.0)              # process cooling: never locked out


def test_physics_13_heat_pumps():
    """1.3: the emitter store keeps the heat-pump energy and smooths it, the field COP factor raises air-source
    electricity, the heating limit stops heating in warm weather; resistance homes are unchanged."""
    import buildingsim as bs
    from buildingsim.campus import emitter_store
    x = np.r_[np.zeros(10), np.ones(5) * 3.0, np.zeros(33)]
    y = emitter_store(x, 3600.0, 6.0)
    assert y.sum() == pytest.approx(x.sum(), rel=1e-12) and y.max() < x.max() and np.array_equal(emitter_store(x, 3600.0, 0.0), x)
    w = _synthetic_weather("2019-01-07", 6, 0.0, 6.0, tz="America/Chicago")
    hp = dict(primary_use="residential_single", floor_area_m2=150, year_built=2000, climate_zone="5A",
              system="heat_pump_air")
    run = lambda spec, wx, o: bs.simulate(spec, wx, "2019-01-07 12:00", "2019-01-12", dt="1h", options=o)  # noqa: E731
    a = run(hp, w, {"physics": "1.2"})
    b = run(hp, w, {"physics": "1.3", "campus": dict(tau_e=12.0, f_cop=0.7)})
    c = run(hp, w, {"physics": "1.3", "campus": dict(tau_e=12.0, f_cop=1.0)})
    ea, eb, ec = (o["ts"]["heating_electricity_kW"] for o in (a, b, c))
    assert bs.ledger(b)["ok"] and eb.sum() > 1.2 * ea.sum()
    assert ec.sum() == pytest.approx(ea.sum(), rel=0.02) and ec.std() < ea.std()
    warm = _synthetic_weather("2019-07-01", 4, 27.0, 4.0, tz="America/Chicago")
    h = bs.simulate(hp, warm, "2019-07-01 12:00", "2019-07-04", dt="1h", options={"physics": "1.3"})
    assert (h["ts"]["heat_sp"] < -50).mean() > 0.9 and h["ts"]["heating_electricity_kW"].sum() < 0.01
    res = dict(hp, system="baseboard_resistance")
    r2, r3 = run(res, w, {"physics": "1.2"}), run(res, w, {"physics": "1.3"})
    assert np.array_equal(r2["ts"]["meter_electricity_kW"].to_numpy(), r3["ts"]["meter_electricity_kW"].to_numpy())


@pytest.mark.skipif(not HAVE_PRIOR, reason="ResStock prior not present (mlratchet data building --from-zip ...)")
def test_physics_14_generator_intensity(monkeypatch):
    """1.4: same draws, filled specs with intensities x the use factor; the default generator is unchanged."""
    import buildingsim as bs
    monkeypatch.delenv("BUILDINGSIM_PHYSICS", raising=False)
    a = bs.generate_buildings(4, seed=7, primary_use="lab")
    assert a == bs.generate_buildings(4, seed=7, primary_use="lab", physics="1.3")
    b = bs.generate_buildings(4, seed=7, primary_use="lab", physics="1.4", intensity={"lab": 0.5})
    for x, y in zip(a, b):
        f = bs.fill_spec(x)
        assert y["internal_gains"]["plug"] == pytest.approx(0.5 * f["internal_gains"]["plug"])
        assert y["floor_area_m2"] == x["floor_area_m2"] and y["_physics_generator"] == "1.4"
    monkeypatch.setenv("BUILDINGSIM_PHYSICS", "1.4")
    assert bs.generate_buildings(1, seed=7, primary_use="lab")[0]["_physics_generator"] == "1.4"


def test_physics_12_weather_insensitive_electricity():
    """1.2: the always-on share keeps the run's lighting and plug energy and raises night/day; the daily activity
    draws are mean-preserving in expectation, deterministic per seed; homes are unchanged."""
    import buildingsim as bs
    from buildingsim.campus import activity
    spec, w = _campus_pair(days=14)
    P = dict(s_on=0.4, sigma_d=0.0, rho_d=0.0)
    a = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-14", dt="1h", options={"physics": "1.1"})
    b = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-14", dt="1h", options={"physics": "1.2", "campus": P})
    for k in ("interior_lighting_electricity_kW", "interior_equipment_electricity_kW"):
        ra, rb = a["ts"][k], b["ts"][k]
        assert rb.sum() == pytest.approx(ra.sum(), rel=0.02)          # preserved over the run incl. warm-up week
        night = ra.index.tz_convert("America/Chicago").hour < 5
        assert rb[night].mean() / rb[~night].mean() > ra[night].mean() / ra[~night].mean()
    cal = dict(doy=np.repeat(np.arange(1, 2001), 24).astype(float))
    m = activity(cal, 0.2, 0.7, 1)
    d = m[::24]
    assert abs(np.log(d).std() - 0.2) < 0.02 and abs(d.mean() - 1.0) < 0.03
    assert np.corrcoef(np.log(d[1:]), np.log(d[:-1]))[0, 1] == pytest.approx(0.7, abs=0.05)
    assert np.array_equal(m, activity(cal, 0.2, 0.7, 1)) and not np.array_equal(m, activity(cal, 0.2, 0.7, 2))
    home = dict(primary_use="residential_single", floor_area_m2=150, year_built=2000, climate_zone="5A",
                system="heat_pump_air")
    h1 = bs.simulate(home, w, "2019-07-01 12:00", "2019-07-04", dt="1h", options={"physics": "1.1"})
    h2 = bs.simulate(home, w, "2019-07-01 12:00", "2019-07-04", dt="1h",
                     options={"physics": "1.2", "campus": dict(s_on=0.4, sigma_d=0.2, rho_d=0.7)})
    assert np.array_equal(h1["ts"]["meter_electricity_kW"].to_numpy(), h2["ts"]["meter_electricity_kW"].to_numpy())


def test_phase2_country_holidays():
    import pandas as pd
    import buildingsim as bs
    from buildingsim import holidays as H
    uk = H.holiday_dates("UK", 2016)
    assert pd.Timestamp("2016-03-25") in uk and pd.Timestamp("2016-03-28") in uk
    assert pd.Timestamp("2017-03-17") in H.holiday_dates("IE", 2017)
    assert pd.Timestamp("2016-05-23") in H.holiday_dates("CA", 2016)
    assert H.resolve("auto", "Europe/Dublin") == "IE" and H.resolve("auto", "America/Denver") is None
    w = _synthetic_weather("2016-03-26", 4, 8.0, 4.0, tz="Europe/London", lat=51.5, lon=-0.1)
    spec = dict(primary_use="office", floor_area_m2=2000, year_built=2000, climate_zone="4A", system="packaged_rtu")
    v1 = bs.simulate(spec, w, "2016-03-27", "2016-03-29", dt="1h")
    on = bs.simulate(spec, w, "2016-03-27", "2016-03-29", dt="1h", options={"holidays": "auto"})
    monday = np.asarray(v1["ts"].index.tz_convert("Europe/London").day == 28)
    assert (v1["sim"].wx["cal"]["daytype"][monday] == 0).all() and (on["sim"].wx["cal"]["daytype"][monday] == 2).all()
    assert on["ts"]["meter_electricity_kW"][monday].sum() < v1["ts"]["meter_electricity_kW"][monday].sum()


def test_phase2_district_eff_chw_scales_the_chilled_water_meter():
    """Detective W2 in the plant: chilled-water meter / district_eff_chw, heat unchanged, ledgers still close."""
    import buildingsim as bs
    w = _synthetic_weather("2019-07-01", 3, 27.0, 6.0, tz="America/Chicago")
    spec = dict(primary_use="office", floor_area_m2=5000, year_built=2000, climate_zone="5A",
                system="district_steam_chw", hvac=dict(district_eff=0.5))
    a = bs.simulate(spec, w, "2019-07-01 12:00", "2019-07-03", dt="1h")
    b = bs.simulate(dict(spec, hvac=dict(district_eff=0.5, district_eff_chw=0.25)), w, "2019-07-01 12:00",
                    "2019-07-03", dt="1h")
    ka, kb = a["summary"]["kWh_by_meter"], b["summary"]["kWh_by_meter"]
    assert ka["chilled_water"] > 0 and kb["chilled_water"] == pytest.approx(2 * ka["chilled_water"], rel=1e-9)
    assert kb["steam"] == pytest.approx(ka["steam"], rel=1e-12)
    assert kb["electricity"] == pytest.approx(ka["electricity"], rel=1e-12)
    assert bs.ledger(b)["ok"] and "district_eff_chw" not in bs.fill_spec(spec)["hvac"]


def test_phase2_transforms():
    import pandas as pd
    from buildingsim import phase2 as P, fill_spec
    base = dict(primary_use="education", floor_area_m2=8000, year_built=1970, climate_zone="4A")
    assert P.spec_from_meters(base, ["electricity", "chilledwater", "steam"])["system"] == "district_steam_chw"
    s = P.spec_from_meters(base, ["electricity", "hotwater"], P.CAMPUS_KNOBS)
    assert s["system"] == "vav_chiller_boiler" and s["hvac"]["heat_source"] == "district"
    assert s["occupancy_schedule"]["type"] == "24x7" and s["ventilation"]["economizer"] is False
    y = P.scale_meters({"steam": np.ones(4), "_x": None}, {"a_steam": 2.0, "b_steam": 5.0}, 1000.0)
    assert list(y) == ["steam"] and np.allclose(y["steam"], 7.0)
    rng = np.random.default_rng(0)
    m = {"steam": rng.gamma(1, 10, 1000), "chilled_water": rng.gamma(2, 5, 1000), "gas": np.ones(1000)}
    out = P.flat_base_loads(m, np.random.default_rng(1))
    for k in ("steam", "chilled_water"):
        assert out[k].sum() == pytest.approx(m[k].sum(), rel=1e-12) and out[k].std() < m[k].std()
    assert np.array_equal(out["gas"], m["gas"])
    c = P.clean_labels(dict(base_share=-8e12, Tb_heat=-80.0, Tb_cool=12.0))
    assert np.isnan(c["base_share"]) and c["Tb_heat"] == -25 and c["Tb_cool"] == 12
    f = fill_spec(dict(base, system="vav_chiller_boiler"))
    L0 = f["internal_gains"]["lighting"]
    P.apply_intensity(f, 0.5)
    assert f["internal_gains"]["lighting"] == pytest.approx(0.5 * L0)
    f, kind = P.make_campus(f, np.random.default_rng(2), kind="both")
    assert f["system"] == "district_steam_chw" and 0.05 <= f["hvac"]["district_eff_chw"] <= 1.0
    tr = pd.DataFrame(dict(OA_damper_pct=np.linspace(0, 100, 11), SAT=13.0, MAT=18.0, RAT=23.0, OAT=30.0, ZAT=23.0,
                           fan_status=1.0, cool_cmd_pct=50.0, heat_cmd_pct=0.0))
    for seed in range(5):
        d = P.installed_damper(tr, np.random.default_rng(seed))["OA_damper_pct"].to_numpy()
        assert d[0] >= 0 and d[-1] == pytest.approx(100.0, abs=1e-3) and np.all(np.diff(d) >= -1e-4)
    t2, off = P.static_offsets(tr, np.random.default_rng(3))
    assert set(off) == {"SAT", "MAT", "RAT", "OAT", "ZAT"} and t2["MAT"].iloc[0] == pytest.approx(18.0 + off["MAT"])
    h = tr.copy()
    h["cool_cmd_pct"] = 80.0
    rng = np.random.default_rng(1)
    assert rng.random() >= 0.2                                 # this seed is a unit without position feedback
    lc = P.log_commands(h, tr, dict(type="cooling_valve_stuck", position=0.5), np.random.default_rng(1))
    assert np.allclose(lc["cool_cmd_pct"], 100.0)              # healthy 80 % > stuck 50 % + 5 %: the loop winds up


@pytest.mark.skipif(not HAVE_SURVEYS, reason="building data not present (mlratchet data building --from-zip ...)")
def test_phase2_tariffs_are_dst_safe():
    import pandas as pd
    import buildingdata as bd
    for key in ("pge_e_tou_c", "xcel_mn_a15_tod"):
        for tz in ("America/Los_Angeles", "America/Chicago"):
            u = bd.tariff_prices(key, 2017, tz=tz, utc=True)
            r = bd.tariff_prices(key, 2017, tz=tz).reindex(u.index.tz_convert(tz))
            assert len(u) == 8760 and np.array_equal(r.to_numpy(), u.to_numpy())
            pa = bd.tariff_arrays(key, u.index, tz)
            assert np.array_equal(pa["price"], u["energy_$per_kWh"].to_numpy())
    naive = bd.tariff_prices("pge_e_tou_c", 2017)          # unchanged default: naive local clock time
    assert naive.index.tz is None and len(naive) == 8760
    with pytest.raises(Exception):
        naive.index.tz_localize("America/Los_Angeles")      # the spring hour does not exist: use tz= or utc=True
    t = pd.date_range("2017-01-01", "2017-02-01", freq="h", tz="UTC", inclusive="left")
    pa = bd.tariff_arrays("pge_b19_sec", t, "America/Los_Angeles")
    e, dem = bd.bill_arrays(pa, np.ones(len(t)), np.ones(len(t), int))
    assert e == pytest.approx(pa["price"].sum()) and dem > 0


@pytest.mark.skipif(not (HAVE_SURVEYS and HAVE_PRIOR), reason="RECS / ResStock prior not present")
def test_phase2_recs_respondent_twin():
    import buildingdata as bd
    from buildingsim import stock, fill_spec
    x = bd.recs(split="dev")
    x = x[x.system.isin(["heat_pump_air", "split_ac_furnace"])].iloc[:5]
    got = [stock.recs_to_spec(r, seed=1) for _, r in x.iterrows()]
    got = [g for g in got if g is not None]
    assert got
    for spec, band in got:
        assert spec["provenance"]["recs_id"] == spec["id"] and 17.0 <= band["lo"] < band["hi"] <= 28.0
        assert fill_spec(spec)["system"] in stock.RECS_SYSTEMS
