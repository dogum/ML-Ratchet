"""The shared fire platform (shared/fire: FireSim + FireData) is intact and importable.

The data are not in git (`mlratchet data fire --from-zip firesim-firedata-v1.0.zip`, or the rebuild). Tests that
need them skip when they are absent, including when $ML_RATCHET_FIRE points at an empty folder. The physics subset
below (analytic cases, crown, moisture, FWI, a FireSim smoke run) needs no data and runs without numba; FireData
itself needs shapely (`pip install -e .[fire]`), so its loader tests skip without it."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIRE = ROOT / "shared" / "fire"                                     # code (tracked)
CHECKS = FIRE / "checks"
DATA_ROOT = Path(os.environ.get("ML_RATCHET_FIRE") or FIRE)          # data (git-ignored)
PROC = DATA_ROOT / "data" / "proc"
for p in (FIRE / "firesim", FIRE):                                   # as the models' training/_paths.py do
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
pd = pytest.importorskip("pandas")
HAVE_NUMBA = importlib.util.find_spec("numba") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
HAVE_TABLES = (PROC / "fires.parquet").exists()
HAVE_LANDSCAPES = HAVE_TABLES and any((PROC / "landscapes").glob("*.npz")) if (PROC / "landscapes").is_dir() else False
HAVE_BASELINES = (PROC / "baselines" / "spread_summary.json").exists()
HAVE_REFDATA = (DATA_ROOT / "firesim" / "refdata" / "gtr371_pages.txt").exists()
HAVE_PYRE = (DATA_ROOT / "checks" / "out" / "pyre_values.json").exists()
SPLITS_FILE_SHA = "70f655969586d73dfbe6af5e5c3c7c1bd11ac96e011451acfe1bc3132d017fd2"
SPLITS_FIELD_SHA = "d73e2d9f79e62c48ee17047c1e5e17e4212899c71c656603c27f9b6601b08fe0"
SKIP_KEYS = {"generated", "python", "wall_s", "numba", "threads"}


def same_numbers(want, got, path="", rel=1e-7, abs_=1e-9):
    """Every key and value of the shipped JSON is reproduced (run metadata aside)."""
    if isinstance(want, dict):
        for k, v in want.items():
            if k in SKIP_KEYS:
                continue
            assert k in got, path + "/" + k
            same_numbers(v, got[k], path + "/" + k, rel, abs_)
    elif isinstance(want, list):
        assert len(want) == len(got), path
        for i, (u, v) in enumerate(zip(want, got)):
            same_numbers(u, v, f"{path}[{i}]", rel, abs_)
    elif isinstance(want, float) and isinstance(got, (int, float)):
        assert got == pytest.approx(want, rel=rel, abs=abs_), path
    else:
        assert want == got, path


def run_check(script, tmp_path, timeout=600, physics="p1"):
    """Run a check script into tmp_path. physics='p1' (default) reproduces the shipped checks/*.json."""
    env = dict(os.environ, FIRESIM_CHECKS_OUT=str(tmp_path), OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1", FIRESIM_PHYSICS=physics)
    r = subprocess.run([sys.executable, script], cwd=CHECKS, env=env, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


# ---------------------------------------------------------------- frozen files and wiring
def test_frozen_splits_unchanged():
    """Every fire exam depends on this file; it must never change (data/DATA.md section 3)."""
    raw = (FIRE / "data" / "proc" / "splits.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SPLITS_FILE_SHA
    sp = json.loads(raw)
    stored = sp.pop("sha256")                       # make_splits.py: sha256 of json.dumps(sort_keys=True) without it
    assert stored == SPLITS_FIELD_SHA == hashlib.sha256(json.dumps(sp, sort_keys=True, default=str).encode()).hexdigest()
    assert sp["meta"]["seed"] == 20261118 and sp["meta"]["frozen"] == "2026-11-18T23:44Z"
    f = sp["fires"]
    assert {k: len(v) for k, v in f.items()} == {"dev": 40, "exam": 4197, "aux_rx": 2859, "excluded_nonCONUS": 643}
    assert not set(f["dev"]) & set(f["exam"])
    assert {k: len(v) for k, v in sp["exam_periods"].items()} == {"early": 1647, "main": 1859, "recent": 691}
    assert sorted(v["split"] for v in sp["gofer"].values()).count("dev") == 4 and len(sp["gofer"]) == 28
    ws = [v["split"] for v in sp["wsts"].values()]
    assert (ws.count("dev"), ws.count("exam"), ws.count("excluded_mixed")) == (4, 568, 35)
    assert sp["firesize"]["counts"] == {"exam": 655869, "dev": 8913} and len(sp["firesize"]["dev_fod_ids"]) == 8913


def test_manifest_lists_the_shipped_files():
    man = json.loads((FIRE / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["n_files"] == len(man["files"]) == 230
    assert man["files"]["data/proc/splits.json"][0] == SPLITS_FILE_SHA
    prefixes = ("data/proc/", "checks/out/", "firesim/refdata/", "firesim/demo_out/")
    assert all(f.startswith(prefixes) and not f.endswith(".part") for f in man["files"])
    assert sum(f.startswith("data/proc/landscapes/") for f in man["files"]) == 40     # the 40 dev fires
    assert not any(f.startswith(("data/proc/feds/", "data/proc/wsts/")) for f in man["files"])


def test_no_sandbox_paths_or_private_links():
    bad = []
    for f in FIRE.rglob("*"):
        rel = f.relative_to(FIRE)
        if not f.is_file() or f.suffix not in (".py", ".sh", ".md", ".json") or f.stat().st_size > 2_000_000:
            continue
        if rel.parts[0] == "data" and rel.as_posix() not in ("data/DATA.md", "data/MANIFEST.json") \
                and rel.parts[1] != "scripts" and rel.name != "build.py":
            continue                                              # git-ignored data (except the frozen splits)
        if rel.parts[:2] in (("checks", "out"), ("firesim", "refdata"), ("firesim", "demo_out"), ("firesim", ".refenv")):
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        if "/home/claude" in t or "/mnt/user-data" in t or re.search(r"claude\.ai/(artifact|chat|project)", t):
            bad.append(rel.as_posix())
    assert not bad, bad


def test_data_command_is_registered():
    from mlratchet.cli import DATA_TARGETS
    assert ROOT.joinpath(*DATA_TARGETS["fire"]).exists()
    r = subprocess.run([sys.executable, "-I", str(FIRE / "data" / "build.py"), "--list"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and "verify" in r.stdout and "--from-zip" in r.stdout


# ---------------------------------------------------------------- physics checks vs the shipped JSONs
@contextlib.contextmanager
def physics(version):
    """FireSim's default physics for in-process checks (the shipped checks/*.json are p1)."""
    from firesim.simulate import DEFAULTS            # `firesim.simulate` the attribute is the function
    old = DEFAULTS["physics"]
    DEFAULTS["physics"] = version
    try:
        yield
    finally:
        DEFAULTS["physics"] = old


def _analytic_namespace():
    """checks/check_analytic.py up to the sweep: its imports, wx() and ellipse_case(), nothing run."""
    src = (CHECKS / "check_analytic.py").read_text(encoding="utf-8")
    head, rest = src.split("\nell = []\n", 1)
    other = sys.modules.pop("_common", None)                         # other platforms' checks have a _common too
    sys.path.insert(0, str(CHECKS))
    try:
        ns = {"__name__": "check_analytic_subset"}
        exec(compile(head, "check_analytic.py", "exec"), ns)
    finally:
        sys.path.remove(str(CHECKS))
        sys.modules.pop("_common", None)
        if other is not None:
            sys.modules["_common"] = other
    return ns,{s.split("\n", 1)[0].split(".")[0]: "# " + s.split("\nwrite(")[0] for s in rest.split("\n# ---- ")[1:]}


@pytest.fixture(scope="module")
def analytic():
    return _analytic_namespace(), json.loads((CHECKS / "analytic.json").read_text())


@pytest.mark.parametrize("fuel,ws,wd", [(102, 8.0, 270.0), (102, 8.0, 225.0), (165, 6.0, 247.5), (102, 13.0, 270.0)])
def test_ellipse_cases_reproduce(analytic, fuel, ws, wd):
    """Four of the 55 default-solver ellipse cases (90 m; about 0.5-2.5 s each without numba), exactly as shipped.
    The 13 m/s case is the minimum area ratio, 0.940."""
    (ns, _), want = analytic
    with physics("p1"):
        r = ns["ellipse_case"](fuel, ws, wd, 90.0, ns["METHODS"]["mtt16+simplex (default)"], length_cells=60)
    w = [c for c in want["ellipse_cases"] if c["solver"] == "mtt16+simplex (default)" and c["fuel"] == r["fuel"]
         and c["wind_ms"] == ws and c["wind_from"] == wd and c["res_m"] == 90.0]
    assert len(w) == 1
    r = dict({k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}, solver=w[0]["solver"])
    same_numbers(w[0], r)


@pytest.mark.parametrize("fuel,ws,wd", [(102, 13.0, 270.0)])
def test_p2_ellipse_case_reproduces_and_keeps_the_baseline(analytic, fuel, ws, wd):
    """Physics p2 (22 Nov 2026) is p1 in constant weather: checks/p2/analytic.json has the same ellipse summary and
    hand-solution errors (largest relative change 1.5e-5), and one case re-runs exactly under p2."""
    (ns, _), p1 = analytic
    p2 = json.loads((CHECKS / "p2" / "analytic.json").read_text())
    for solver, s in p1["ellipse_summary"].items():
        same_numbers(s, p2["ellipse_summary"][solver], rel=1e-4, abs_=1e-9)
    assert p2["fuel_break"]["cells_burned_beyond_solid_break"] == 0
    with physics("p2"):
        r = ns["ellipse_case"](fuel, ws, wd, 90.0, ns["METHODS"]["mtt16+simplex (default)"], length_cells=60)
    w = [c for c in p2["ellipse_cases"] if c["solver"] == "mtt16+simplex (default)" and c["fuel"] == r["fuel"]
         and c["wind_ms"] == ws and c["wind_from"] == wd and c["res_m"] == 90.0]
    same_numbers(w[0], dict({k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()},
                            solver=w[0]["solver"]))


def test_ellipse_summary_is_the_reported_baseline(analytic):
    """HANDOFF section 3: default solver mean area ratio 0.981, minimum 0.940, orientation bias <= 3.4 %;
    level_set 0.992 / 0.973."""
    _, want = analytic
    s = want["ellipse_summary"]
    d, ls = s["mtt16+simplex (default)"], s["mtt32+simplex (level_set)"]
    assert round(d["area_ratio_mean"], 3) == 0.981 and round(d["area_ratio_min"], 3) == 0.940
    assert round(d["orientation_bias_area_ratio_range_max"], 3) == 0.034
    assert round(ls["area_ratio_mean"], 3) == 0.992 and round(ls["area_ratio_min"], 3) == 0.973
    assert sum(v["n"] for v in s.values()) == 4 * 55


@pytest.mark.parametrize("section,key", [("2", "strip"), ("5", "tv")])
def test_hand_solutions_reproduce(analytic, section, key):
    """Two-fuel strip (arrival error 1.5e-7 h) and the time-varying rate, as shipped (about 5 s without numba)."""
    (ns, secs), want = analytic
    with physics("p1"):
        exec(compile("res = 30.0\n" + secs[section], "check_analytic.py", "exec"), ns)
    name = {"strip": "two_fuel_strip", "tv": "time_varying"}[key]
    same_numbers(want[name], {k: (float(v) if isinstance(v, np.floating) else v) for k, v in ns[key].items()})


@pytest.mark.skipif(not HAVE_NUMBA, reason="fuel break and refraction take about 90 s without numba")
@pytest.mark.parametrize("section,key", [("3", "fb"), ("4", "refr")])
def test_fuel_break_and_refraction_reproduce(analytic, section, key):
    (ns, secs), want = analytic
    with physics("p1"):
        exec(compile("res = 30.0\n" + secs[section], "check_analytic.py", "exec"), ns)
    name = {"fb": "fuel_break", "refr": "refraction"}[key]
    got = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in ns[key].items()}
    same_numbers(want[name], got)
    if key == "fb":
        assert got["cells_burned_beyond_solid_break"] == 0


@pytest.mark.parametrize("script,name", [("check_crown.py", "crown.json"), ("check_moisture.py", "moisture.json")])
def test_crown_and_moisture_checks_reproduce(tmp_path, script, name):
    """Crown hand calculations (exact) and the dead-fuel moisture checks (Simard isotherm RMSE 0.41 points)."""
    run_check(script, tmp_path)
    want = json.loads((CHECKS / name).read_text())
    same_numbers(want, json.loads((tmp_path / name).read_text()))
    if name == "moisture.json":
        assert round(want["isotherm"]["rmse_pct_points"], 2) == 0.41


def test_fwi_check_reproduces(tmp_path):
    """The standard FWI test row within 0.005 (max 0.0049 on DMC); the xclim season too when its output is there."""
    out = run_check("check_fwi.py", tmp_path)
    want = json.loads((CHECKS / "fwi.json").read_text())
    got = json.loads((tmp_path / "fwi.json").read_text())
    same_numbers(want["test_row"], got["test_row"])
    assert max(v["abs_err"] for v in got["test_row"].values()) < 0.005
    if "skipped" not in got["xclim_season"]:
        same_numbers(want["xclim_season"], got["xclim_season"])
    else:
        assert "skipped" in out


@pytest.mark.skipif(not HAVE_REFDATA, reason="firesim/refdata (GTR PDFs + text) not unpacked (mlratchet data fire --from-zip)")
def test_published_tables_reproduce(tmp_path):
    """436 of 436 published BehavePlus/Rothermel values within half the last published digit."""
    run_check("check_published.py", tmp_path)
    got = json.loads((tmp_path / "published_examples.json").read_text())
    assert got["n_cases"] == got["n_within_tolerance"] == 436
    same_numbers(json.loads((CHECKS / "published_examples.json").read_text()), got)


@pytest.mark.skipif(not HAVE_PYRE, reason="checks/out/pyre_values.json not unpacked (mlratchet data fire --from-zip)")
def test_pyretechnics_crosscheck_reproduces(tmp_path):
    """FireSim vs the shipped pyretechnics values (6,720 cases; ROS median 4.9e-5): no pyretechnics install needed."""
    run_check("check_pyretechnics.py", tmp_path)
    same_numbers(json.loads((CHECKS / "pyretechnics_crosscheck.json").read_text()),
                 json.loads((tmp_path / "pyretechnics_crosscheck.json").read_text()))


# ---------------------------------------------------------------- FireSim
def test_firesim_smoke_run():
    """A small generated landscape, generated weather, 12 h of spread with crown fire, and a VIIRS overpass."""
    import pandas as pd
    import firesim as fs
    b = fs.behave("GR2", m1=6, m10=7, m100=8, mherb=60, wind_speed=4.0)
    assert 0 < b["ros_back_mmin"] < b["ros_flank_mmin"] < b["ros_mmin"] and b["lw_ratio"] > 1
    ls = fs.generate_landscape((60, 60), 90.0, rng=7, region="mixed_conifer", relief_m=300)
    w = fs.generate_weather("2021-08-01", hours=72, climate="west_dry_summer", rng=3, wind=6.0, rh_min=14.0,
                            p_wet=0.0, lat=40.0, lon=-121.0)
    out = fs.simulate(ls, w, {"rowcol": [(30, 30)]}, "2021-08-02 18:00", "2021-08-03 06:00", rng=1,
                      return_perimeters=False)
    a = out["arrival_time"]
    assert np.isfinite(a).sum() > 10 and np.nanmax(a) <= 12.0 + 1e-6
    assert out["area"].iloc[-1] > 0 and "ledger" in out
    obs = fs.observe(out, sensor="viirs", overpasses="nominal", rng=0, lat=40.0, lon=-121.0)
    assert {"detections", "overpasses"} <= set(obs) and len(obs["overpasses"]) > 0


def _u1_ring(n=31, radius=5.0, hours=12.0, physics="p2"):
    """FirePilot's leak test (training/leak_test.py), shrunk: uniform GR2 grass, flat, a closed 4-connected ring around
    a point ignition. Returns cells burned outside the ring with no line, a timed line at t=0, and non-burnable fuel."""
    import pandas as pd
    from scipy import ndimage as ndi
    import firesim as fs
    ls = fs.generate_landscape((n, n), 90.0, kind="parametric", region="great_basin", rng=1)
    ls["fuel"][:] = 102
    ls["slope"][:] = 0
    start = pd.Timestamp("2020-08-01 18:00", tz="UTC")
    w = fs.generate_weather(start - pd.Timedelta(hours=72), hours=120, climate="west_dry_summer", lon=-110, rng=3,
                            physics=physics)
    yy, xx = np.mgrid[:n, :n]
    inside = np.hypot(yy - n // 2, xx - n // 2) < radius
    ring4 = ndi.binary_dilation(inside) & ~inside                  # diagonal neighbours touch only at corners
    outside = ~(inside | ndi.binary_dilation(inside, structure=np.ones((3, 3), bool)))
    kw = dict(spotting=False, return_perimeters=False, rng=0, params={"physics": physics})
    end = start + pd.Timedelta(hours=hours)

    def out_cells(o):
        a = o["arrival_time"]
        return int((np.isfinite(a) & (a <= hours) & outside).sum())
    res = {"no_line": out_cells(fs.simulate(ls, w, {"rowcol": [(n // 2, n // 2)]}, start, end, **kw)),
           "timed_line": out_cells(fs.simulate(ls, w, {"rowcol": [(n // 2, n // 2)]}, start, end,
                                               suppression={"lines": [{"mask": ring4, "time": 0.0}]}, **kw))}
    ls["fuel"][ring4] = 91
    res["nonburnable"] = out_cells(fs.simulate(ls, w, {"rowcol": [(n // 2, n // 2)]}, start, end, **kw))
    return res


@pytest.fixture(scope="module")
def u1():
    return _u1_ring()


@pytest.fixture(scope="module")
def u1_p1():
    return _u1_ring(physics="p1")


def test_mask_ignition_interior_is_burned_at_start():
    """PR #7 review, 21 Nov 2026: parse_ignition seeds only a mask's edge, so the kernel gave the interior positive
    arrival times; the whole mask must count as burned at t = 0, and cells outside it must not change."""
    import pandas as pd
    import firesim as fs
    n = 41
    ls = fs.generate_landscape((n, n), 90.0, kind="parametric", region="great_basin", rng=1)
    start = pd.Timestamp("2021-08-02 18:00", tz="UTC")
    w = fs.generate_weather(start - pd.Timedelta(hours=72), hours=96, climate="west_dry_summer", lon=-110, rng=3)
    m = np.zeros((n, n), bool)
    m[15:26, 15:26] = True
    a = fs.simulate(ls, w, {"mask": m}, start, start + pd.Timedelta(hours=6), rng=1,
                    return_perimeters=False)["rasters"]["arrival_time"]
    assert np.all(a[m] <= 0.0)
    assert np.isfinite(a[~m]).any()
    # a scheduled mask ignition keeps its time; one scheduled after the end does not burn
    late = fs.simulate(ls, w, {"mask": m, "time": 3.0}, start, start + pd.Timedelta(hours=6), rng=1,
                       return_perimeters=False)["rasters"]["arrival_time"]
    assert np.allclose(late[m], 3.0)
    after = fs.simulate(ls, w, {"mask": m, "time": 10.0}, start, start + pd.Timedelta(hours=6), rng=1,
                        return_perimeters=False)["rasters"]["arrival_time"]
    assert np.isnan(after[m]).all()


def test_hourly_from_daily_keeps_every_hour_at_half_hour_offsets():
    """PR #7 review, 21 Nov 2026: at lon = -112.5 every UTC time fell on :30 and round-half-to-even merged alternate
    hours, so half the series was dropped."""
    import pandas as pd
    from firesim import weather as W
    d = pd.DataFrame({"tmmx": [30.0] * 4, "tmmn": [12.0] * 4, "rmin": [15.0] * 4, "rmax": [60.0] * 4, "vs": [4.0] * 4,
                      "th": [250.0] * 4, "pr": [0.0] * 4}, index=pd.date_range("2021-08-01", periods=4, freq="D"))
    for lon in (-112.5, -110.0, -97.5):
        h = W.hourly_from_daily(d, 40.0, lon)
        assert len(h) == 96 and (pd.Series(h.index).diff().dropna() == pd.Timedelta(hours=1)).all(), lon


@pytest.mark.skipif(not (HAVE_SHAPELY and HAVE_TABLES), reason="needs firedata (shapely) and the dev HRRR packages")
def test_real_weather_covers_the_requested_window():
    """PR #7 review, 21 Nov 2026: with `start`, a whole-day shift could leave the series starting after the spin-up
    start or ending before the horizon."""
    import pandas as pd
    from firesim import realdata as R
    s = pd.Timestamp("2020-08-10 18:00", tz="UTC")
    for seed in range(4):
        w = R.random_weather(start=s, hours=72, rng=seed)
        assert w.index[0] <= s - pd.Timedelta(hours=72) and w.index[-1] >= s + pd.Timedelta(hours=72), seed


def test_custom_fuels_keep_the_standard_fm10_crown_bundle():
    """PR #7 review, 21 Nov 2026: overriding a fuel numbered 1-10 reordered the table, and the Rothermel 1991
    crown-spread reference bundle was read from the wrong row (FM9 when fuel 1 was overridden)."""
    import pandas as pd
    import firesim as fs
    import importlib
    S = importlib.import_module("firesim.simulate")
    ls = fs.generate_landscape((20, 20), 90.0, kind="parametric", region="great_basin", rng=1)
    start = pd.Timestamp("2021-08-02 18:00", tz="UTC")
    w = fs.generate_weather(start - pd.Timedelta(hours=72), hours=96, climate="west_dry_summer", lon=-110, rng=3)
    end = start + pd.Timedelta(hours=3)
    custom = {1: dict(w1=0.05, w10=0.0, w100=0.0, wherb=0.0, wwoody=0.0, sav1=3000.0, depth=1.2, mx=0.15)}
    base = S.prepare(ls, w, start, end, {}, "nelson")
    over = S.prepare(ls, w, start, end, {"custom_fuels": custom}, "nelson")
    assert np.array_equal(base["bundle10"], over["bundle10"])


def test_generated_weather_covers_a_fractional_start():
    """PR #7 review, 21 Nov 2026: a start at 18:30 dropped the 18:00 row, and simulate() refused the weather."""
    import pandas as pd
    import firesim as fs
    s = pd.Timestamp("2020-07-01 18:30", tz="UTC")
    w = fs.generate_weather(s, hours=72, rng=1)
    assert w.index[0] == s.floor("h")
    ls = fs.generate_landscape((20, 20), 90.0, kind="parametric", region="great_basin", rng=1)
    fs.simulate(ls, w, {"rowcol": [(10, 10)]}, s, s + pd.Timedelta(hours=3), rng=1, return_perimeters=False)
    assert len(fs.generate_weather(s.floor("h"), hours=72, rng=1)) == 72      # on-the-hour starts unchanged


def test_initial_attack_at_ignition_time_does_not_contain_an_empty_fire():
    """PR #7 review, 21 Nov 2026: with initial_attack starting at the ignition hour, the first heap entry saw zero
    burned cells, a zero perimeter, and 'contained' the fire before it ignited (even at line rate 0)."""
    import pandas as pd
    import firesim as fs
    ls = fs.generate_landscape((25, 25), 90.0, kind="parametric", region="great_basin", rng=1)
    ls["fuel"][:] = 102
    s = pd.Timestamp("2020-08-01 18:00", tz="UTC")
    w = fs.generate_weather(s - pd.Timedelta(hours=72), hours=96, climate="west_dry_summer", lon=-110, rng=3)
    out = fs.simulate(ls, w, {"rowcol": [(12, 12)]}, s, s + pd.Timedelta(hours=2), rng=1, return_perimeters=False,
                      suppression={"initial_attack": dict(start_h=0.0, line_rate_m_per_h=0.0)})
    assert np.isfinite(out["rasters"]["arrival_time"]).sum() > 1


def test_fwi_daily_inputs_are_taken_at_local_noon():
    """PR #7 review, 21 Nov 2026: daily_from_hourly converted UTC to solar time with the longitude sign reversed."""
    import pandas as pd
    from firesim import fwi
    idx = pd.date_range("2021-07-01", periods=72, freq="h", tz="UTC")
    w = pd.DataFrame({"T": 0.0, "RH": 50.0, "wind_speed": 1.0, "precip": 0.0}, index=idx)
    w.loc[w.index.hour == 20, "T"] = 30.0          # local noon at lon = -120 is 20:00 UTC
    d = fwi.daily_from_hourly(w, lon=-120.0)
    assert (d["T"] == 30.0).all() and len(d) == 3


def test_u1_setup_is_meaningful(u1):
    """Without a line the fire leaves the ring; a ring of non-burnable fuel holds it (the corner rule works there)."""
    assert u1["no_line"] > 100 and u1["nonburnable"] == 0


def test_u1_timed_line_holds_like_nonburnable_fuel(u1):
    """Physics p2 (22 Nov 2026): the corner rule counts a side cell as blocked once its line is built."""
    assert u1["timed_line"] == u1["nonburnable"] == 0


def test_u1_p1_still_leaks(u1_p1):
    """physics='p1' reproduces the solver the v1.0 models were scored with, bug included (282 of 852 cells)."""
    assert u1_p1["timed_line"] == 282 and u1_p1["nonburnable"] == 0


def _grass(n=31, res=30.0):
    import firesim as fs
    ls = fs.generate_landscape((n, n), res, kind="parametric", region="great_basin", rng=1)
    ls["fuel"][:] = 102
    ls["slope"][:] = 0
    ls["cbd"][:] = 0
    return ls


def _given_weather(start, hours, **cols):
    """Hourly weather with given dead-fuel moisture (%), from an hour before `start`."""
    import pandas as pd
    idx = pd.date_range(start - pd.Timedelta(hours=1), periods=hours + 2, freq="h")
    base = dict(T=30.0, RH=15.0, wind_speed=6.0, wind_dir=270.0, precip=0.0, cloud=0.0, m1h=4.0, m10h=5.0,
                m100h=6.0, mherb=60.0, mwoody=90.0)
    base.update({k: (v(idx) if callable(v) else v) for k, v in cols.items()})
    return pd.DataFrame(base, index=idx)


def test_u11_knight_step_does_not_cross_a_line_built_mid_step():
    """U11 (p2): a (1, 2) step checked its middle cell's line against the departure time, so a line finished
    0.00001 h after ignition was crossed. A line the fire reaches only after it is built must hold."""
    import pandas as pd
    import firesim as fs
    ls = _grass()
    s = pd.Timestamp("2021-08-02 18:00", tz="UTC")
    w = _given_weather(s, 4)
    line = np.zeros(ls.shape, bool)
    line[:, 16] = True

    def east(physics, t_line):
        a = fs.simulate(ls, w, {"rowcol": [(15, 15)]}, s, s + pd.Timedelta(hours=2), moisture="given",
                        params={"physics": physics}, rng=0, return_perimeters=False,
                        suppression={"lines": [{"mask": line, "time": t_line}]})["rasters"]["arrival_time"]
        return int(np.isfinite(a[:, 17:]).sum())
    assert east("p2", 1e-5) == 0
    assert east("p1", 1e-5) > 100                 # the p1 leak, kept for reproduction
    assert east("p2", 0.05) > 100                 # the fire reaches the line cell (about 0.03 h) before it is built


def test_u12_a_line_built_after_a_cell_burns_does_not_change_that_cell():
    """U12 (p2, 22 Nov 2026): when the straight step to a cell met that cell's line before arriving, p1 counted the
    step as blocked and skipped the two-point update, which can arrive hours earlier. A line could then stop fire
    that reached its cell before the line was built (FirePilot's direct attack: a cell reached at 0.995 h stayed
    unburned under a line built at 5 h). Here: each of 40 burned cells gets a one-cell line finished 0.02 h after
    the no-line arrival; that cell's arrival must not change. p1 fails 3 of 40."""
    import pandas as pd
    import firesim as fs
    ls = fs.generate_landscape((60, 60), 90.0, kind="parametric", region="great_basin", rng=1)
    s = pd.Timestamp("2021-08-02 18:00", tz="UTC")
    w = fs.generate_weather(s - pd.Timedelta(hours=72), hours=120, lon=-110, rng=3, physics="p2")
    e = s + pd.Timedelta(hours=24)

    def changed(physics):
        kw = dict(rng=0, return_perimeters=False, params={"physics": physics})
        a = fs.simulate(ls, w, {"rowcol": [(30, 30)]}, s, e, **kw)["rasters"]["arrival_time"]
        cells = np.argwhere(np.isfinite(a) & (a > 2))
        cells = cells[np.random.default_rng(0).choice(len(cells), 40, replace=False)]
        n = 0
        for y, x in cells:
            m = np.zeros(a.shape, bool)
            m[y, x] = True
            b = fs.simulate(ls, w, {"rowcol": [(30, 30)]}, s, e, suppression={"lines": [{"mask": m,
                            "time": float(a[y, x]) + 0.02}]}, **kw)["rasters"]["arrival_time"]
            n += not (np.isfinite(b[y, x]) and abs(b[y, x] - a[y, x]) < 1e-9)
        return n
    assert changed("p2") == 0
    assert changed("p1") == 3


def test_ignition_on_a_finished_line_does_not_spread():
    """PR #10 review: p2 skipped the line check on a segment's source cell, so an ignition placed on a line already
    built (or a scheduled one after the line) spread outward. As in p1, such a source burns but does not spread."""
    import pandas as pd
    import firesim as fs
    ls = _grass()
    s = pd.Timestamp("2021-08-02 18:00", tz="UTC")
    w = _given_weather(s, 4)
    line = np.zeros(ls.shape, bool)
    line[15, 15] = True
    for physics, t_ign in (("p2", 0.0), ("p2", 1.0), ("p1", 0.0)):
        a = fs.simulate(ls, w, {"rowcol": [(15, 15)], "time": t_ign}, s, s + pd.Timedelta(hours=2),
                        moisture="given", params={"physics": physics}, rng=0, return_perimeters=False,
                        suppression={"lines": [{"mask": line, "time": 0.0}]})["rasters"]["arrival_time"]
        assert int(np.isfinite(a).sum()) <= 1, (physics, t_ign)


def test_check_physics_switch_reaches_the_weather_generator():
    """PR #10 review: FIRESIM_PHYSICS=p1 (and physics('p1') here) must give p1 generated weather too, or the FWI,
    moisture and speed checks compare p2 weather against shipped p1 results."""
    import pandas as pd
    import firesim as fs
    s = pd.Timestamp("2021-07-01 00:00", tz="UTC")
    with physics("p1"):
        w = fs.generate_weather(s, hours=120, lon=-120.0, rng=0, p_wet=0.3)
    assert w.attrs["physics"] == "p1"
    assert w.equals(fs.generate_weather(s, hours=120, lon=-120.0, rng=0, p_wet=0.3, physics="p1"))


def test_u9_two_point_update_follows_a_weather_slowdown():
    """U9 (p2): the two-point update used the departure hour's spread rates for the whole trip. With fast spread in
    the first hour and almost none after (1-h moisture just under extinction, no wind: 0.009 m/min), p1 still
    reached about 60 cells after 1 h; integrated through the hours, only cells a metre or so short of the front
    can be."""
    import pandas as pd
    import firesim as fs
    ls = _grass(81)
    s = pd.Timestamp("2021-08-02 18:00", tz="UTC")
    late = lambda idx, a, b: np.where(idx >= s + pd.Timedelta(hours=1), b, a)  # noqa: E731
    w = _given_weather(s, 6, wind_speed=lambda i: late(i, 8.0, 0.0), m1h=lambda i: late(i, 3.0, 14.9),
                       m10h=lambda i: late(i, 4.0, 14.9), m100h=lambda i: late(i, 5.0, 14.9))

    def after_1h(physics):
        a = fs.simulate(ls, w, {"rowcol": [(40, 15)]}, s, s + pd.Timedelta(hours=3), moisture="given",
                        params={"physics": physics}, rng=0, return_perimeters=False)["rasters"]["arrival_time"]
        return int((np.isfinite(a) & (a > 1.0)).sum())
    assert after_1h("p2") <= 5
    assert after_1h("p1") > 30


def test_u10_generated_flags_follow_the_local_day():
    """U10 (p2): generate_weather applied its daily event and wet flags by UTC date; hourly_from_daily lays out
    local solar days, so at lon = -120 a day's last 8 hours took the next day's flags. Each hour's cloud now matches
    its local day's wet flag. physics='p1' keeps the old mapping."""
    import pandas as pd
    import firesim as fs
    from firesim.weather import _local_days
    s = pd.Timestamp("2021-07-01 00:00", tz="UTC")
    for lon in (-120.0, -112.5, -75.0):
        for seed in range(6):
            w = fs.generate_weather(s, hours=240, climate="west_dry_summer", lon=lon, rng=seed, p_wet=0.3)
            wet = set(w.attrs["wet_days"])
            days = pd.date_range((s - pd.Timedelta(days=1)).tz_convert(None), periods=13)
            day = pd.DatetimeIndex(_local_days(w.index, days, lon))
            want = np.array([str(d.date()) in wet for d in day])
            assert np.array_equal(w["cloud"].to_numpy() > 0.5, want), (lon, seed)
    old = fs.generate_weather(s, hours=240, lon=-120.0, rng=0, p_wet=0.3, physics="p1")
    new = fs.generate_weather(s, hours=240, lon=-120.0, rng=0, p_wet=0.3)
    assert (old["cloud"] != new["cloud"]).any() and np.array_equal(old["T"], new["T"])


# ---------------------------------------------------------------- data (skip when absent)
@pytest.mark.skipif(not HAVE_BASELINES, reason="fire baselines not unpacked (mlratchet data fire --from-zip)")
def test_baselines_are_the_ones_the_models_report():
    """data/DATA.md sections 4-6: next-day spread fl_prev AP 0.252 (3,030 fires, 25,105 growth days), FireSize GBM
    escape AUC 0.777, live moisture GBM MAE 24.7."""
    s = json.loads((PROC / "baselines" / "spread_summary.json").read_text())["exam"]["all"]["fl_prev"]
    assert (s["n_fires"], s["n_fire_days"]) == (3030, 25105) and round(s["ap_mean"], 3) == 0.252
    assert round(s["f1_mean"], 3) == 0.184
    g = json.loads((PROC / "baselines" / "firesize_exam.json").read_text())["results"]["all"]["gbm"]
    assert round(g["escape"]["auc"], 3) == 0.777 and round(g["large"]["auc"], 3) == 0.803
    m = json.loads((PROC / "baselines" / "moisture_exam.json").read_text())["results"]["gbm"]
    assert round(m["mae"], 1) == 24.7
    rs = json.loads((PROC / "baselines" / "replay_subset.json").read_text())
    exam = set(json.loads((FIRE / "data" / "proc" / "splits.json").read_text())["fires"]["exam"])
    assert rs["n"] == len(rs["fires"]) == 48 and set(rs["fires"]) <= exam


@pytest.mark.skipif(not HAVE_TABLES, reason="fire tables not unpacked (mlratchet data fire --from-zip)")
def test_fire_table_matches_the_splits():
    f = pd.read_parquet(PROC / "fires.parquet")
    sp = json.loads((FIRE / "data" / "proc" / "splits.json").read_text())["fires"]
    for k in ("dev", "exam"):
        assert set(f.Event_ID[f.split == k]) == set(sp[k]), k


@pytest.mark.skipif(not (HAVE_LANDSCAPES and HAVE_SHAPELY), reason="dev landscapes not unpacked, or shapely missing")
def test_loader_and_guard_on_dev_fires():
    import firedata as fd
    dev = fd.fires("dev")
    assert len(dev) == 40
    fid = sorted(dev.Event_ID)[0]
    L = fd.landscape(fid)
    assert {"fuel", "cc", "ch", "cbh", "cbd", "elev", "slope", "aspect"} <= set(L["layers"])
    assert L["layers"]["fuel"].shape == tuple(L["shape"])
    ex = sorted(fd.fires("exam").Event_ID)[0]
    with pytest.raises(fd.ExamLeakError):
        fd.landscape(ex)
    r = fd.fires("exam").set_index("Event_ID").loc[ex]
    with pytest.raises(fd.ExamLeakError):
        fd.guard.check_landscape((r.xmin, r.ymin, r.xmax, r.ymax))


@pytest.mark.skipif(not (HAVE_LANDSCAPES and HAVE_SHAPELY and importlib.util.find_spec("pyproj")),
                    reason="dev landscapes not unpacked, or shapely/pyproj missing")
def test_realdata_adapters_on_a_dev_fire():
    import firedata as fd
    import firesim as fs
    fid = sorted(fd.fires("dev").Event_ID)[0]
    ls = fs.realdata.landscape_from_firedata(fd.landscape(fid))
    W = fd.weather(fid)
    x0, dx, _, y0, _, dy = ls.meta["transform"]
    w = fs.realdata.weather_series(W, x0 + dx * ls.shape[1] / 2, y0 + dy * ls.shape[0] / 2)
    assert {"T", "RH", "wind_speed", "wind_dir"} <= set(w.columns) and len(w) > 24
