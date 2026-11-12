"""The CLI runs, and `mlratchet new` produces a folder that validates (as in-progress)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlratchet import cli, get  # noqa: E402
from mlratchet.validate import check  # noqa: E402


def test_list_info_ladder_scoreboard(capsys):
    assert cli.main(["list"]) == 0
    assert cli.main(["info", "quakenet"]) == 0
    assert cli.main(["ladder", "chirpnet"]) == 0
    assert cli.main(["scoreboard"]) == 0
    out = capsys.readouterr().out
    assert "QuakeNet" in out and "Neutron stars" in out and "wins" in out


def test_new_scaffolds_a_valid_in_progress_model():
    dest = ROOT / "models" / "zz-test-model"
    shutil.rmtree(dest, ignore_errors=True)
    try:
        assert cli.main(["new", "zz-test-model", "--name", "ZzNet"]) == 0
        assert (dest / "zz-test-model.py").exists()
        m = get("zz-test-model")
        assert m.card["name"] == "ZzNet" and m.status == "in-progress"
        assert check(m) == []
        assert "__ID__" not in (dest / "training" / "ratchet.py").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(dest, ignore_errors=True)
