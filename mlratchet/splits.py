"""Dev / exam splits and example draws that can't be cherry-picked.

* Split by UNIT (a star, a cell, a patient, a machine file, an event) — never by window —
  so nothing from a dev unit leaks into the exam.
* Use a fixed seed and write it down.
* Pick demo examples by a fixed random draw within strata, not by how well the model does.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence

import numpy as np


def split_units(units: Sequence[Hashable], n_dev: int, seed: int,
                strata: Sequence[Hashable] | None = None) -> tuple[list, list]:
    """Return ``(dev, exam)`` lists of unique units.

    With ``strata``, ``n_dev`` units are drawn per stratum (as SwellCast, StarQuake and BeatNet did).
    """
    uniq = list(dict.fromkeys(units))
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.permutation(len(uniq))[:n_dev]
        dev = [uniq[i] for i in sorted(idx)]
    else:
        first = dict(zip(units, strata))
        groups = defaultdict(list)
        for u in uniq:
            groups[first[u]].append(u)
        dev = []
        for key in sorted(groups, key=str):
            g = groups[key]
            take = rng.permutation(len(g))[:n_dev]
            dev += [g[i] for i in sorted(take)]
    dset = set(dev)
    return dev, [u for u in uniq if u not in dset]


def fixed_draw(items: Sequence, k: int, seed: int, strata: Sequence[Hashable] | None = None) -> list:
    """A reproducible random draw of ``k`` items (per stratum when ``strata`` is given)."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return [items[i] for i in sorted(rng.permutation(len(items))[:k])]
    groups = defaultdict(list)
    for it, s in zip(items, strata):
        groups[s].append(it)
    out = []
    for key in sorted(groups, key=str):
        g = groups[key]
        out += [g[i] for i in sorted(rng.permutation(len(g))[:k])]
    return out
