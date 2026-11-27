"""Freeze experiments/treetwin/PREREG.md with SHA-256 before opening any exam split."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys

import tt_paths as P

PREREG = P.HERE / "PREREG.md"
LOG = P.HERE / "prereg_log.json"


def sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(note: str = "frozen before exam") -> dict:
    if not PREREG.exists():
        raise FileNotFoundError(f"missing {PREREG}")
    h = sha256_file(PREREG)
    rec = {
        "file": "PREREG.md",
        "sha256": h,
        "frozen_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": note,
    }
    LOG.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"Frozen PREREG.md sha256={h}")
    return rec


def verify_frozen() -> str:
    if not PREREG.exists() or not LOG.exists():
        raise RuntimeError("PREREG.md must be frozen with freeze.py before running the exam stage")
    rec = json.loads(LOG.read_text())
    actual = sha256_file(PREREG)
    if actual != rec["sha256"]:
        raise RuntimeError(
            f"PREREG.md modified after freeze! expected {rec['sha256']}, got {actual}"
        )
    return actual


if __name__ == "__main__":
    note = " ".join(sys.argv[1:]) or "frozen before opening any exam split"
    freeze(note)
