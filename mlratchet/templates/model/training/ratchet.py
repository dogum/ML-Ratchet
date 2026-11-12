"""Judge a trained rung: keep it only if it beats the incumbent on the FROZEN simulated set by more than
the margin AND passes the dev gate. Every rung is logged in ../ratchet.yaml, kept or not.

    python ratchet.py r2 "slower surface waves, measured on dev"

The exam is NOT read here. It is run once, at the end, by exam.py.
"""
import sys
from pathlib import Path

import yaml

import _paths as _P
from mlratchet import Ratchet, Rung, gates

LADDER = Path(_P.MODEL) / "ratchet.yaml"
FROZEN = "val1"  # bump when the physics version changes, then rescore every rung (Ratchet.rescore)


def score_frozen(ckpt: str) -> float:
    """TODO: the single objective on the frozen simulated set (higher is better)."""
    raise NotImplementedError


def score_dev(ckpt: str) -> dict:
    """TODO: dev metrics for the gate, e.g. {"dist_mae": ..., "mw_mae": ...} (lower is better)."""
    raise NotImplementedError


def load() -> Ratchet:
    doc = yaml.safe_load(LADDER.read_text(encoding="utf-8"))
    lad = doc["ladders"][0]
    r = Ratchet(doc["model"], margin=lad.get("margin", 0.01), dev_gate=gates.no_worse_than(1.05),
                objective=lad.get("objective", ""), rule=lad.get("rule", ""), frozen_set=FROZEN, name=lad["name"])
    r.rungs = [Rung(**{k: v for k, v in x.items() if k in Rung.__dataclass_fields__}) for x in lad.get("rungs", [])]
    return r


if __name__ == "__main__":
    rung_id, change = sys.argv[1], sys.argv[2]
    ckpt = str(Path(_P.WORK) / "ckpt" / f"{rung_id}.pt")
    r = load()
    rung = r.judge(rung_id, change, sim=score_frozen(ckpt), dev=score_dev(ckpt))
    r.save(LADDER)
    print(f"{rung.id}: {rung.decision} — {rung.why}")
