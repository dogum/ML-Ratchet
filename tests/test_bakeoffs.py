"""The two bake-off records (experiments/bakeoff-river, experiments/bakeoff-vehicle) are consistent with themselves.

- The exam ran once: EXAM_DONE.json names the result file, and the arm hashes in it match the exam result.
- The headline numbers in each write-up (docs/experiments/bakeoff-*.md) match the shipped analysis.json.
- The site serves each bake-off page.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RIVER = ROOT / "experiments" / "bakeoff-river"
VEHICLE = ROOT / "experiments" / "bakeoff-vehicle"


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("bk", [RIVER, VEHICLE], ids=["river", "vehicle"])
def test_exam_ran_once(bk):
    done = _load(bk / "results" / "EXAM_DONE.json")
    result = bk / done["result"]
    assert result.exists(), result
    E = _load(result)
    assert E["run_id"] == done["run_id"]
    assert E["mode"] == "exam"
    for arm, h in done["arms"].items():
        a = E["arms"].get(arm)
        assert a is not None, arm
        if isinstance(a, dict) and a.get("hash"):
            assert a["hash"] == h, arm


def test_river_headline_matches_analysis():
    A = _load(RIVER / "results" / "analysis.json")
    doc = (ROOT / "docs" / "experiments" / "bakeoff-river.md").read_text(encoding="utf-8")
    board = A["boards"]["main/ungauged"]
    for arm in ("tab_gbm", "flowcast_v1", "seq_lstm", "seq_tcn", "seq_ssm", "seq_transformer", "tab_gp"):
        assert f"{board[arm]['nse']:.3f}" in doc, arm


def test_vehicle_headline_matches_analysis():
    A = _load(VEHICLE / "results" / "analysis.json")
    doc = (ROOT / "docs" / "experiments" / "bakeoff-vehicle.md").read_text(encoding="utf-8")
    rows = {"table": ("enet_v21", "tab_mlp", "tab_gbm", "formula", "phys_anchor"),
            "trace": ("trace_transformer", "trace_gru", "trace_cnn", "trace_s4d")}
    for track, arms in rows.items():
        for arm in arms:
            assert f"{100 * A['boards'][track][arm]['E16']:.1f} %" in doc, (track, arm)


@pytest.mark.parametrize("page", ["river-bakeoff", "vehicle-bakeoff"])
def test_site_serves_page(page):
    p = ROOT / "site" / "public" / "experiments" / page / "index.html"
    assert p.exists() and p.stat().st_size > 100_000
    src = (ROOT / "site" / "src" / "pages" / "mental-models.astro").read_text(encoding="utf-8")
    assert f"experiments/{page}/" in src
