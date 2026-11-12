"""Find the models in the repository and load them.

Every model lives in ``models/<id>/`` with the same contract (see docs/model-contract.md):
a ``model.yaml`` card, a ``ratchet.yaml`` ladder, a numpy-only inference module, weights,
``demo.py`` and ``examples/``. This module discovers them and imports their code in
isolation, so two models that both ship a ``pipeline.py`` never collide.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

ENV_HOME = "ML_RATCHET_HOME"


def repo_root(start: str | os.PathLike | None = None) -> Path:
    """Locate the repository root (the folder holding ``models/``).

    Order: ``$ML_RATCHET_HOME``; then walk up from ``start`` (default: the current directory);
    then walk up from this file (an editable install inside the repo).
    """
    env = os.environ.get(ENV_HOME)
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "models").is_dir():
            return p
        raise FileNotFoundError(f"{ENV_HOME}={env} has no models/ folder")
    for base in (Path(start or os.getcwd()).resolve(), Path(__file__).resolve().parent):
        for p in (base, *base.parents):
            if (p / "models").is_dir() and (p / "mlratchet").is_dir():
                return p
    raise FileNotFoundError(
        "Could not find the ML Ratchet repository (a folder with models/ and mlratchet/). "
        f"Run from inside a clone, or set {ENV_HOME}."
    )


@dataclass
class Model:
    """One model folder: its card, its ladder, and a way to import its code."""

    id: str
    path: Path
    _modules: dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- metadata -------------------------------------------------------------------------
    @cached_property
    def card(self) -> dict:
        with open(self.path / "model.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @cached_property
    def ladder(self) -> dict:
        p = self.path / "ratchet.yaml"
        if not p.exists():
            return {"model": self.id, "ladders": []}
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def name(self) -> str:
        return self.card.get("name", self.id)

    @property
    def version(self) -> str:
        return str(self.card.get("version", "0"))

    @property
    def status(self) -> str:
        return self.card.get("status", "unknown")

    def released_rung(self) -> dict | None:
        """The rung marked ``released`` in the first ladder that has one."""
        for lad in self.ladder.get("ladders", []):
            for r in lad.get("rungs", []):
                if r.get("decision") == "released":
                    return r
        return None

    def verdicts(self) -> dict[str, int]:
        """Count exam verdicts (win / loss / tie / mixed / info) from the card."""
        out: dict[str, int] = {}
        for row in (self.card.get("exam") or {}).get("results", []):
            v = row.get("verdict", "info")
            out[v] = out.get(v, 0) + 1
        return out

    # ---- code -----------------------------------------------------------------------------
    def module(self, name: str | None = None):
        """Import one of the model's Python modules in isolation and return it.

        ``name`` defaults to the model id (e.g. ``quakenet`` -> ``models/quakenet/quakenet.py``).
        The model folder is put on ``sys.path`` only while importing; any sibling modules it
        pulls in (``pipeline``, ``transitsim`` ...) are re-registered under a private
        namespace so other models can load their own versions.
        """
        name = name or self.id
        if name in self._modules:
            return self._modules[name]
        file = self.path / f"{name}.py"
        if not file.exists():
            raise FileNotFoundError(file)
        before = set(sys.modules)
        sys.path.insert(0, str(self.path))
        try:
            spec = importlib.util.spec_from_file_location(name, file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(self.path))
            # move anything imported from this folder into a private namespace
            for key in set(sys.modules) - before:
                m = sys.modules.get(key)
                origin = getattr(m, "__file__", None) or ""
                if origin and Path(origin).resolve().parent == self.path.resolve():
                    sys.modules[f"mlratchet_models.{self.id}.{key}"] = sys.modules.pop(key)
        self._modules[name] = mod
        return mod

    def demo(self, *args: str, capture: bool = False, timeout: float | None = None) -> subprocess.CompletedProcess:
        """Run the model's ``demo.py`` in its own folder with the current interpreter."""
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        return subprocess.run(
            [sys.executable, "demo.py", *args],
            cwd=self.path,
            env=env,
            text=True,
            encoding="utf-8",
            capture_output=capture,
            timeout=timeout,
        )


def list_models(root: str | os.PathLike | None = None) -> list[Model]:
    """All models, in the order they were built (``order`` in model.yaml)."""
    base = repo_root(root) / "models"
    models = [Model(p.name, p) for p in sorted(base.iterdir()) if (p / "model.yaml").exists()]
    return sorted(models, key=lambda m: (m.card.get("order", 999), m.id))


def get(model_id: str, root: str | os.PathLike | None = None) -> Model:
    p = repo_root(root) / "models" / model_id
    if not (p / "model.yaml").exists():
        known = ", ".join(m.id for m in list_models(root))
        raise KeyError(f"No model '{model_id}'. Known: {known}")
    return Model(model_id, p)


def load(model_id: str, module: str | None = None):
    """Shortcut: import a model's inference module. ``mlratchet.load('quakenet').QuakeNet()``."""
    return get(model_id).module(module)
