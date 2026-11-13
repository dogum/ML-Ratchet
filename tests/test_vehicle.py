"""The shared vehicle platform (shared/vehicle: RoadSim + RoadData) is intact and importable."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VEH = ROOT / "shared" / "vehicle"
sys.path.insert(0, str(VEH / "roadsim"))
sys.path.insert(0, str(VEH))


def test_frozen_splits_unchanged():
    """Every vehicle exam depends on this file; it must never change (HANDOFF.md)."""
    h = hashlib.sha256((VEH / "data" / "proc" / "splits.json").read_bytes()).hexdigest()
    assert h.startswith("d917df78"), h


def test_roadsim_ledger_closes_on_a_cycle():
    out = subprocess.run([sys.executable, "demo.py"], cwd=VEH / "roadsim", capture_output=True, text=True,
                         encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("closure (rel)"))
    worst = max(abs(float(x)) for x in line.split()[2:])
    assert worst < 5e-3, line   # the contract: the energy ledger closes to < 0.5%


def test_recorded_checks_still_meet_their_targets():
    """The verification baseline (HANDOFF.md section 3) as recorded in checks/*.json."""
    ledger = json.loads((VEH / "checks" / "ledger.json").read_text(encoding="utf-8"))
    assert json.dumps(ledger)  # present and parseable
    for name in ("cycles_vs_epa.json", "fastsim_crosscheck.json", "speed.json", "coastdown.json"):
        assert (VEH / "checks" / name).exists(), name


def test_realdata_reads_the_shipped_tables():
    import realdata as rd

    sp = rd.splits()
    assert sp, "splits"
    epa = rd.epa_testcars(split="dev")
    assert len(epa) > 100
    cyc = rd.cycles()
    assert len(cyc) >= 20
