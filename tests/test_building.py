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
