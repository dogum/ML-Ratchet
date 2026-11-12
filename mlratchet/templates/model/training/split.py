"""Dev / exam split by UNIT (star, cell, patient, machine, event), with a fixed seed. Run once; commit split.json."""
import json

import _paths as _P  # noqa: F401
from mlratchet.splits import split_units

SEED = 2026


def main():
    units: list[str] = []      # TODO: every real unit with labels
    strata = None              # TODO: optional stratum per unit (e.g. magnitude band)
    dev, exam = split_units(units, n_dev=3, seed=SEED, strata=strata)
    with open("split.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "dev": dev, "exam": exam}, f, indent=1)
    print(f"{len(dev)} dev units, {len(exam)} exam units")


if __name__ == "__main__":
    main()
