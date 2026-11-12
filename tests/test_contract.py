"""Every model folder honours the contract, its README is in sync with its YAML, and its code imports
in isolation (two models' pipeline.py never collide)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlratchet import list_models  # noqa: E402
from mlratchet.readme import render  # noqa: E402
from mlratchet.validate import check  # noqa: E402

MODELS = list_models()
IDS = [m.id for m in MODELS]


def test_there_are_models():
    assert len(MODELS) >= 9


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_contract(model):
    assert check(model) == []


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_readme_in_sync(model):
    assert (model.path / "README.md").read_text(encoding="utf-8") == render(model), \
        "README.md is stale: run `mlratchet readme --write`"


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_every_ladder_has_one_released_rung(model):
    for lad in model.ladder["ladders"]:
        released = [r for r in lad["rungs"] if r["decision"] == "released"]
        assert len(released) <= 1, f"{lad['name']}: more than one released rung"
    assert model.released_rung() is not None


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_no_cloud_sandbox_paths_left(model):
    bad = []
    for f in model.path.rglob("*"):
        if f.suffix in (".py", ".sh") and f.is_file():
            t = f.read_text(encoding="utf-8", errors="replace")
            if "/home/claude" in t or "/mnt/user-data" in t:
                bad.append(str(f.relative_to(model.path)))
    assert not bad, bad


def test_inference_modules_import_in_isolation():
    loaded = {}
    for m in MODELS:
        for name in m.card["files"]["inference"]:
            mod = m.module(Path(name).stem)
            loaded[(m.id, name)] = mod
    # several models ship a pipeline.py; each must be its own module object
    pipes = [mod for (mid, n), mod in loaded.items() if n == "pipeline.py"]
    assert len(pipes) >= 2 and len({id(p) for p in pipes}) == len(pipes)
    assert "pipeline" not in sys.modules or not str(getattr(sys.modules["pipeline"], "__file__", "")).startswith(
        str(Path(__file__).resolve().parents[1] / "models"))
