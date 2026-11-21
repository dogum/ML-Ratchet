"""The Prior Wars record (experiments/priorwars) is consistent with itself.

- Every freeze in prereg_log.json still hashes to the shipped PREREG.md (the harness's own check, run offline).
- The cross-collection table in RESULTS.md and in docs/experiments/prior-wars.md matches results.json: headlines to
  3 decimals, the C1-C3 verdicts, k* and the context sizes.
- The headline counts quoted on the site (mental-models page) match results.json.

Standard library only: TabPFN, the models' banks and the exam data are not needed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PW = ROOT / "experiments" / "priorwars"
pytestmark = pytest.mark.skipif(not (PW / "results.json").exists(), reason="experiments/priorwars not present")

FLIP = {"win": "loss", "loss": "win", "tie": "tie"}


def results():
    return json.loads((PW / "results.json").read_text(encoding="utf-8"))


def test_prereg_hashes_match():
    txt = (PW / "PREREG.md").read_text(encoding="utf-8")
    assert "\r" not in txt, "PREREG.md must keep LF line endings (the hashes were taken on the LF file)"
    log = json.loads((PW / "prereg_log.json").read_text(encoding="utf-8"))
    frozen = [m for e in log for m in e["models"]]
    assert sorted(frozen) == sorted(results()["models"]), "every scored model was frozen once"
    for e in log:
        assert hashlib.sha256(txt[: e["n_chars"]].encode("utf-8")).hexdigest() == e["sha256"], e["models"]
    assert log[-1]["n_chars"] <= len(txt)


def summary_rows(md: str) -> dict[str, list[str]]:
    sec = md.split("## Cross-collection summary", 1)[1].split("\n## ", 1)[0]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in sec.splitlines() if l.startswith("| ")]
    return {r[0]: r[1:] for r in rows[1:]}       # skip the header row


@pytest.mark.parametrize("doc", ["experiments/priorwars/RESULTS.md", "docs/experiments/prior-wars.md"])
def test_summary_table_matches_results_json(doc):
    R = results()
    rows = summary_rows((ROOT / doc).read_text(encoding="utf-8"))
    head = {(r["model"], r["arm"]): r["value"] for r in R["rows"] if r["item"] == "_headline"}
    con = {(c["model"], c["contrast"]): c["verdict_for_arm"] for c in R["contrasts"]}
    assert len(rows) == len(R["models"]) == 6
    for mid, m in R["models"].items():
        k, p, gdev, gsim, gsd, c1, c2, c3, kstar = rows[m["title"]]
        assert int(k) == m["k_in_g_dev_context"]
        for cell, arm in ((p, "P"), (gdev, "G-dev"), (gsim, "G-sim"), (gsd, "G-sim+dev")):
            assert cell == f"{head[(mid, arm)]:.3f}", (mid, arm)
        assert c1 == f"P {FLIP[con[(mid, 'C1')]]}"
        assert c2 == f"P {FLIP[con[(mid, 'C2')]]}"
        assert c3 == f"G-sim+dev {con[(mid, 'C3')]}"
        assert kstar == str(m["k_star"])


def test_headline_counts():
    v = results()["verdict"]
    # C1 from P's side: TabPFN with the dev rows wins 3, ties 2, loses 1
    assert (v["C1_P_losses"], v["C1_ties"], v["C1_P_wins"]) == (3, 2, 1)
    assert v["sentence"] == "the generic prior beats the domain simulator"
    # C2 is stated for G-sim: P wins 4, ties 1, loses 1; C3: simulated rows help 1, tie 3, hurt 2
    assert (v["C2"]["loss"], v["C2"]["tie"], v["C2"]["win"]) == (4, 1, 1)
    assert (v["C3"]["win"], v["C3"]["tie"], v["C3"]["loss"]) == (1, 3, 2)
    best0 = [w["best_0_labels"] for w in results()["where_wins"].values()]
    assert best0.count("P") == 5


def test_site_quotes_results():
    page = ROOT / "site" / "src" / "pages" / "mental-models.astro"
    if not page.exists():
        pytest.skip("site sources not present")
    src = page.read_text(encoding="utf-8")
    if "prior-wars" not in src:
        pytest.skip("the mental-models page has no Prior Wars entry")
    m = re.search(r"TabPFN wins on (\d), ties on (\d) and loses on (\d)", src)
    assert m and tuple(map(int, m.groups())) == (3, 2, 1)
