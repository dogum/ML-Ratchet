"""Portable paths for the TreeTwin + ConformalRatchet cross-platform experiment.

Everything downloaded or computed during a run goes to ``WORK`` = ``$ML_RATCHET_WORK/treetwin/``
else ``experiments/treetwin/work/`` (git-ignored). Committed records live next to this file:
``PREREG.md``, ``prereg_log.json``, ``results.json``, ``RESULTS.md``.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = Path(
    os.path.join(os.environ["ML_RATCHET_WORK"], "treetwin")
    if os.environ.get("ML_RATCHET_WORK")
    else HERE / "work"
).resolve()


def work(*parts: str, mkdir: bool = False) -> Path:
    p = WORK.joinpath(*parts)
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def add_sys_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
