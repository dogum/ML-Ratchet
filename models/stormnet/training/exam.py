"""One primary exam. Reproduction writes to work/, with source and release hashes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from _paths import MODEL, work
from baselines import forecast
from export import examples
from noise import HERE, PROTOCOL, write_json
from split import period
from stormnet import StormNet

ROWS = (("All eligible hours", None), ("Storm-strength minimum", -50),
        ("Intense minimum", -100), ("Extreme minimum", -150), ("Quiet false alarms", "quiet"))


def score_row(cases, meta, predictions, sigma, threshold, rng):
    y = cases["truth"]
    quiet = threshold == "quiet"
    if quiet:
        take = y > -50
    elif threshold is None:
        take = np.ones(len(y), dtype=bool)
    else:
        take = y <= threshold
    groups, inv = np.unique(cases["group"][take], return_inverse=True)
    n = len(groups)
    if n == 0:
        return dict(issues=0, blocks=0, verdict="info", unit="%" if quiet else "nT")
    counts = np.bincount(inv)
    sums = []
    for p in predictions.values():
        loss = (100. * (p[take] <= -50)) if quiet else abs(p[take] - y[take])
        sums.append(np.bincount(inv, weights=loss))
    covered = np.abs(predictions["StormNet"][take] - y[take]) <= sigma[take]
    sums.append(np.bincount(inv, weights=100. * covered))
    sums = np.asarray(sums).T
    means = sums.sum(axis=0) / counts.sum()
    kinds = np.array([meta[g]["kind"] for g in groups])
    # Every column uses the same sampled blocks. Re-select the best comparator in each draw.
    sample = []
    for start in range(0, PROTOCOL["bootstrap_repeats"], 100):
        batches = min(100, PROTOCOL["bootstrap_repeats"] - start)
        idx = np.column_stack([rng.choice(np.flatnonzero(kinds == k), size=(batches, (kinds == k).sum()))
                               for k in np.unique(kinds)])
        sample.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None])
    boot = np.concatenate(sample)
    diff = boot[:, 0] - boot[:, 1:-1].min(axis=1)
    bounds = np.quantile(diff, [.025, .975]).tolist() if n >= 2 else None
    verdict = "info"
    if n >= 5:
        verdict = "win" if bounds[1] < 0 else "loss" if bounds[0] > 0 else "tie"
    return dict(issues=int(take.sum()), blocks=n, storm_blocks=int((kinds == "storm").sum()),
                unit="%" if quiet else "nT", scores=dict(zip(predictions, map(float, means[:-1]))),
                difference=float(means[0] - means[1:-1].min()), difference_ci95=bounds, verdict=verdict,
                coverage_pct=float(means[-1]),
                coverage_ci95=np.quantile(boot[:, -1], [.025, .975]).tolist() if n >= 2 else None)


def main():
    release = Path(MODEL) / "stormnet_weights.npz"
    model = StormNet(release)
    cal = json.loads((HERE / "calibration.json").read_text())
    cases, meta, counts, hashes = period(PROTOCOL["primary_years"])
    if len(np.unique(cases["time"])) != len(cases["time"]):
        raise ValueError("Duplicate primary issue keys")
    reading = model.predict(cases["wind"], cases["dst"])
    # Exam 1 was scored with v1.0's single widening factor. v1.0.1 changed only the bars (binned by the last Dst),
    # so reproduce v1.0's bars here; exam_reserve.py reports both versions on exam 2.
    sigma_v10 = model.predict(cases["wind"], cases["dst"], calibrated=False)["sigma"] * model.meta["widen_factor"]
    predictions = {"StormNet": reading["min_dst"], **forecast(cases, cal)}
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    rows = {name: score_row(cases, meta, predictions, sigma_v10, cut, rng) for name, cut in ROWS}
    result = dict(released=model.meta["rung"], weights_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),
                  first_issue_utc=str(cases["time"][0]), last_issue_utc=str(cases["time"][-1]),
                  eligibility=counts, distinct_storm_blocks=sum(m["kind"] == "storm" for m in meta),
                  rows=rows)
    dest = Path(work("exam"))
    write_json(dest / "exam.json", result)
    write_json(dest / "sources_primary.json", hashes)
    examples(cases, meta, dest / "examples")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
