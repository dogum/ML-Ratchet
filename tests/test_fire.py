"""The shared fire platform (shared/fire: FireSim + FireData) is intact and importable.

The data are not in git (`mlratchet data fire --from-zip firesim-firedata-v1.0.zip`, or the rebuild). Tests that
need them skip when they are absent, including when $ML_RATCHET_FIRE points at an empty folder. The physics subset
below (analytic cases, crown, moisture, FWI, a FireSim smoke run) needs no data and runs without numba; FireData
itself needs shapely (`pip install -e .[fire]`), so its loader tests skip without it."""
from __future__ import annotations

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


def run_check(script, tmp_path, timeout=600):
    env = dict(os.environ, FIRESIM_CHECKS_OUT=str(tmp_path), OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
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
    r = ns["ellipse_case"](fuel, ws, wd, 90.0, ns["METHODS"]["mtt16+simplex (default)"], length_cells=60)
    w = [c for c in want["ellipse_cases"] if c["solver"] == "mtt16+simplex (default)" and c["fuel"] == r["fuel"]
         and c["wind_ms"] == ws and c["wind_from"] == wd and c["res_m"] == 90.0]
    assert len(w) == 1
    r = dict({k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}, solver=w[0]["solver"])
    same_numbers(w[0], r)


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
    exec(compile("res = 30.0\n" + secs[section], "check_analytic.py", "exec"), ns)
    name = {"strip": "two_fuel_strip", "tv": "time_varying"}[key]
    same_numbers(want[name], {k: (float(v) if isinstance(v, np.floating) else v) for k, v in ns[key].items()})


@pytest.mark.skipif(not HAVE_NUMBA, reason="fuel break and refraction take about 90 s without numba")
@pytest.mark.parametrize("section,key", [("3", "fb"), ("4", "refr")])
def test_fuel_break_and_refraction_reproduce(analytic, section, key):
    (ns, secs), want = analytic
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


def _u1_ring(n=31, radius=5.0, hours=12.0):
    """FirePilot's leak test (training/leak_test.py), shrunk: uniform GR2 grass, flat, a closed 4-connected ring around
    a point ignition. Returns cells burned outside the ring with no line, a timed line at t=0, and non-burnable fuel."""
    import pandas as pd
    from scipy import ndimage as ndi
    import firesim as fs
    ls = fs.generate_landscape((n, n), 90.0, kind="parametric", region="great_basin", rng=1)
    ls["fuel"][:] = 102
    ls["slope"][:] = 0
    start = pd.Timestamp("2020-08-01 18:00", tz="UTC")
    w = fs.generate_weather(start - pd.Timedelta(hours=72), hours=120, climate="west_dry_summer", lon=-110, rng=3)
    yy, xx = np.mgrid[:n, :n]
    inside = np.hypot(yy - n // 2, xx - n // 2) < radius
    ring4 = ndi.binary_dilation(inside) & ~inside                  # diagonal neighbours touch only at corners
    outside = ~(inside | ndi.binary_dilation(inside, structure=np.ones((3, 3), bool)))
    kw = dict(spotting=False, return_perimeters=False, rng=0)
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


def test_u1_setup_is_meaningful(u1):
    """Without a line the fire leaves the ring; a ring of non-burnable fuel holds it (the corner rule works there)."""
    assert u1["no_line"] > 100 and u1["nonburnable"] == 0


@pytest.mark.xfail(strict=True, reason="Bug U1 (HANDOFF section 4): the kernel's corner rule ignores cells blocked by a "
                                       "timed suppression line, so fire steps diagonally through a 4-connected line. "
                                       "Fixing it changes physics p1; the fix flips this test.")
def test_u1_timed_line_holds_like_nonburnable_fuel(u1):
    assert u1["timed_line"] == u1["nonburnable"] == 0              # today: 282 of 852 outside cells burn


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
