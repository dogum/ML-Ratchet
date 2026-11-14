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


def test_rebuilt_ved_dev_trips_load():
    """After `mlratchet data vehicle`: dev VED trips carry grade, weather and a fuel label (skipped without the data)."""
    import pytest

    if not (VEH / "data" / "proc" / "ved" / "index.parquet").exists():
        pytest.skip("VED trips not built here (run `mlratchet data vehicle`)")
    import numpy as np
    import realdata as rd

    trips = list(rd.load_trips("dev", arch="conv", label_ok=True, limit=3, seed=0))
    assert len(trips) == 3
    for meta, a in trips:
        assert meta["split"] == "dev" and meta["vehicle_id"] in rd.splits()["VED"]["vehicles"]
        assert np.isfinite(a["grade"]).all() and np.abs(a["grade"]).max() <= 0.15 + 1e-6
        assert np.isfinite(a["T_amb"]).all() and np.isfinite(a["rho_air"]).all()
        assert np.nansum(a["fuel_rate"]) > 0 and meta["fuel_g"] > 0
    assert (VEH / "data" / "proc" / "dem_annarbor_2m.npz").exists()


def test_ztbus_full_split_and_legs():
    """ZTBus full (1,409 missions): the split file is frozen and agrees with splits.json; legs load (skipped without data)."""
    import pytest

    sf = VEH / "data" / "proc" / "hd" / "ztbus_full_splits.json"
    if not sf.exists():
        pytest.skip("ZTBus full not built here (run `mlratchet data vehicle --only ztbus_full`)")
    import numpy as np
    import realdata as rd

    sp = json.loads(sf.read_text(encoding="utf-8"))
    sha = sp.pop("sha256_canonical_content_without_this_field")
    canon = json.dumps(sp, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(canon).hexdigest() == sha
    frozen_dev = set(rd.splits()["HD"]["ZTBus"]["dev_missions"])
    assert {m for m, s in sp["missions"].items() if s == "dev"} == frozen_dev and len(sp["missions"]) == 1409
    legs = rd.ztbus_full()
    assert set(legs.split) <= {"dev", "exam"} and set(legs[legs.split == "dev"].mission) <= frozen_dev
    if not (VEH / "data" / "proc" / "hd" / "ztbus_full" / "trips").exists():
        pytest.skip("ZTBus full leg arrays not built here")
    for meta, a in rd.load_trips("dev", source="ZTBus_full", label_ok=True, limit=2, seed=0):
        assert meta["mission"] in frozen_dev and meta["split"] == "dev"
        assert np.isfinite(a["grade"]).all() and np.isfinite(a["elec_power"][a["valid"] == 1]).all()
        assert abs(np.nansum(a["elec_power"]) / 3.6e6 - meta["elec_kWh"]) < 1e-3 * max(1.0, abs(meta["elec_kWh"]))
