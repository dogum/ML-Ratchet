"""The final exam: run ONCE, after the ratchet has chosen the release, on real data that never chose anything.

Report, side by side:
  * the model, with error bars widened by one factor per output fitted on DEV only
    (``mlratchet.calibrate.widen_factor``) — then the exam coverage you actually get;
  * a classical / standard method on the same data;
  * a dumb baseline (``mlratchet.metrics.median_guess`` / ``majority_guess``, fitted on dev);
  * where it fails, by stratum.

Copy the headline rows into ../model.yaml (exam.results) with an honest verdict for each.
"""
import json

import _paths as _P  # noqa: F401
from mlratchet import calibrate, metrics  # noqa: F401


def main():
    results = {}
    # TODO
    with open("exam.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
