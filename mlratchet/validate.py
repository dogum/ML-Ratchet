"""Check that a model folder honours the contract (docs/model-contract.md).

Schema checks need ``jsonschema`` (``pip install -e ".[dev]"``); the structural checks below
run without it.
"""
from __future__ import annotations

import json
from pathlib import Path

from .registry import Model, list_models

SCHEMA_DIR = Path(__file__).parent / "schema"


def _schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def _json_safe(x):
    """YAML gives dates as date objects; JSON Schema wants strings."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return x


def check(model: Model) -> list[str]:
    """Return a list of problems (empty = the folder passes)."""
    problems: list[str] = []
    p = model.path
    try:
        card, ladder = _json_safe(model.card), _json_safe(model.ladder)
    except Exception as e:  # YAML errors
        return [f"cannot parse YAML: {e}"]

    try:
        import jsonschema

        for doc, name in ((card, "model"), (ladder, "ratchet")):
            v = jsonschema.Draft202012Validator(_schema(name))
            for err in sorted(v.iter_errors(doc), key=lambda e: list(e.path)):
                where = "/".join(str(x) for x in err.path) or "(root)"
                problems.append(f"{name}.yaml {where}: {err.message}")
    except ImportError:
        pass

    if card.get("id") != model.id:
        problems.append(f"model.yaml id '{card.get('id')}' does not match folder '{model.id}'")
    if ladder.get("model") != model.id:
        problems.append(f"ratchet.yaml model '{ladder.get('model')}' does not match folder '{model.id}'")

    files = card.get("files", {})
    for key in ("demo", "card", "lab", "notebook"):
        if files.get(key) and not (p / files[key]).exists():
            problems.append(f"files.{key}: {files[key]} is missing")
    for key in ("inference", "weights"):
        for f in files.get(key, []):
            if not (p / f).exists():
                problems.append(f"files.{key}: {f} is missing")
    if files.get("examples") and not any((p / files["examples"]).glob("*")):
        problems.append("examples/ is empty — ship a few real exam examples for the demo")
    for must in ("README.md", "MODEL_CARD.md", "training"):
        if not (p / must).exists():
            problems.append(f"{must} is missing")

    released = [r for lad in ladder.get("ladders", []) for r in lad.get("rungs", []) if r.get("decision") == "released"]
    if card.get("status") in ("released", "partial") and not released:
        problems.append("status is released but no rung in ratchet.yaml is marked 'released'")
    for lad in ladder.get("ladders", []):
        cols = lad.get("columns")
        for r in lad.get("rungs", []):
            if cols and r.get("metrics") is not None and len(r["metrics"]) != len(cols):
                problems.append(f"ladder '{lad.get('name')}' rung {r.get('id')}: {len(r['metrics'])} metrics for {len(cols)} columns")
    return problems


def check_all() -> dict[str, list[str]]:
    return {m.id: check(m) for m in list_models()}
