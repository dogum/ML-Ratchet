"""Every model's demo must still run on its real exam examples and reproduce the recorded numbers.

The golden outputs in tests/golden/ were recorded from the released code. Numbers are compared with a
small tolerance (different numpy/BLAS builds round differently); everything else must match exactly.
Regenerate after an intentional change:  python tests/test_demos.py --regen
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlratchet import list_models  # noqa: E402

GOLDEN = Path(__file__).parent / "golden"
NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
TIMING = re.compile(r"\b\d+(?:\.\d+)?\s?s\b")
MODELS = [m for m in list_models()]


def _normalise(text: str) -> list[str]:
    return [TIMING.sub("<t> s", line.rstrip()) for line in text.strip().splitlines()]


def _close(a: str, b: str) -> bool:
    x, y = float(a.replace(",", "")), float(b.replace(",", ""))
    decimals = len(b.split(".")[1]) if "." in b else 0
    tol = max(2 * 10 ** -decimals, 0.005 * abs(y))
    return abs(x - y) <= tol


def compare(got: str, want: str) -> list[str]:
    g, w = _normalise(got), _normalise(want)
    if len(g) != len(w):
        return [f"{len(g)} lines vs {len(w)} expected"]
    problems = []
    for i, (lg, lw) in enumerate(zip(g, w), 1):
        if NUM.sub("#", lg) != NUM.sub("#", lw):
            problems.append(f"line {i} text differs:\n  got  {lg}\n  want {lw}")
            continue
        for a, b in zip(NUM.findall(lg), NUM.findall(lw)):
            if not _close(a, b):
                problems.append(f"line {i}: {a} vs {b}\n  got  {lg}\n  want {lw}")
                break
    return problems


@pytest.mark.parametrize("model", [pytest.param(m, id=m.id, marks=[pytest.mark.slow] if m.card.get("demo_seconds", 0) > 30 else [])
                                   for m in MODELS])
def test_demo_reproduces_golden_output(model):
    golden = GOLDEN / f"{model.id}.txt"
    assert golden.exists(), f"no golden output for {model.id}; run python tests/test_demos.py --regen"
    r = model.demo(capture=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    problems = compare(r.stdout, golden.read_text(encoding="utf-8"))
    assert not problems, "\n".join(problems[:10])


def test_compare_tolerates_rounding_but_not_changes():
    want = "QuakeNet  Mw 6.27 +-0.12   depth 155 km\ndone in 0.8 s"
    assert compare("QuakeNet  Mw 6.28 +-0.12   depth 155 km\ndone in 3.1 s", want) == []
    assert compare("QuakeNet  Mw 6.45 +-0.12   depth 155 km\ndone in 0.8 s", want)
    assert compare("QuakeNet  Mw 6.27 +-0.12   depth 155 m\ndone in 0.8 s", want)


if __name__ == "__main__" and "--regen" in sys.argv:
    for m in MODELS:
        r = m.demo(capture=True, timeout=900)
        r.check_returncode()
        (GOLDEN / f"{m.id}.txt").write_text(r.stdout, encoding="utf-8", newline="\n")
        print("wrote", m.id)
