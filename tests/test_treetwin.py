"""Integrity tests for the TreeTwin + ConformalRatchet experiment (experiments/treetwin).

- PREREG.md SHA-256 matches prereg_log.json and every platform's scores/<platform>.json.
- results.json matches every platform's scores/<platform>.json.
- RESULTS.md and docs/experiments/treetwin.md exist and cite the frozen PREREG hash.
- site/src/lib/model-families.ts no longer uses made-up TreeSHAP numbers and matches
  the real measured EnergyNet dev TreeSHAP spotlight cases in results.json.

Standard library only (no lightgbm, banks, or raw exam files required).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TT = ROOT / "experiments" / "treetwin"
PLATFORMS = ("buildingdetective", "airfoilnet", "energynet")

pytestmark = pytest.mark.skipif(
    not (TT / "results.json").exists(),
    reason="experiments/treetwin/results.json not present yet",
)


def _results() -> dict:
    return json.loads((TT / "results.json").read_text(encoding="utf-8"))


def test_prereg_hash_matches_log_and_scores():
    prereg_bytes = (TT / "PREREG.md").read_bytes()
    assert b"\r" not in prereg_bytes, "PREREG.md must keep LF line endings"
    sha = hashlib.sha256(prereg_bytes).hexdigest()
    log = json.loads((TT / "prereg_log.json").read_text(encoding="utf-8"))
    assert log["sha256"] == sha
    R = _results()
    assert R["prereg_log"]["sha256"] == sha
    for p in PLATFORMS:
        sc = json.loads((TT / "scores" / f"{p}.json").read_text(encoding="utf-8"))
        assert sc["prereg_sha256"] == sha
        assert R["platforms"][p]["prereg_sha256"] == sha
        assert R["platforms"][p]["paired_bootstrap"] == sc["paired_bootstrap"]


def test_docs_and_results_md_present():
    log = json.loads((TT / "prereg_log.json").read_text(encoding="utf-8"))
    short_sha = log["sha256"][:12]
    res_md = (TT / "RESULTS.md").read_text(encoding="utf-8")
    doc_md = (ROOT / "docs" / "experiments" / "treetwin.md").read_text(encoding="utf-8")
    assert short_sha in res_md
    assert short_sha in doc_md
    for p in PLATFORMS:
        title = _results()["platforms"][p]["title"]
        assert title in res_md


def test_site_shap_cases_are_real_measured_values():
    mf = ROOT / "site" / "src" / "lib" / "model-families.ts"
    astro = ROOT / "site" / "src" / "pages" / "model-families.astro"
    if not mf.exists() or not astro.exists():
        pytest.skip("site sources not present")
    mf_txt = mf.read_text(encoding="utf-8")
    astro_txt = astro.read_text(encoding="utf-8")
    assert "made-up numbers" not in mf_txt.lower()
    assert "made-up numbers" not in astro_txt.lower()

    # Check that each spotlight case's first twin/real values from results.json appear in model-families.ts
    cases = _results()["platforms"]["energynet"]["treeshap_twin_dev"]["spotlight_cases"]
    assert len(cases) >= 3
    for c in cases:
        twin_str = ", ".join(f"{v}" for v in c["twin_mean"])
        real_str = ", ".join(f"{v}" for v in c["real_mean"])
        assert twin_str in mf_txt, f"missing measured twin vector for {c['name']}"
        assert real_str in mf_txt, f"missing measured real vector for {c['name']}"
