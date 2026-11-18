"""Run the released network on real dev cases and their simulations under identical observed wind."""

import numpy as np

from noise import RESULTS, read_result, write_json
from split import dev_cases
from simulator import simulate
from stormphysics import PUBLISHED, HISTORY
from stormnet import StormNet


def paired_draws(net, full, wind, center, year):
    errors, levels = [], []
    for seed in range(4):
        twin = simulate(full, center, np.random.default_rng(int(year) * 10 + seed))
        pred = net.predict(wind, twin[:, HISTORY - 1])["min_dst"]
        truth = twin[:, HISTORY:].min(axis=1)
        errors.append(pred - truth)
        levels.append(truth)
    return np.asarray(errors), np.asarray(levels)


def main():
    cal = read_result("calibration.json")
    net = StormNet()
    report = {}
    for year, d in dev_cases().items():
        good = np.isfinite(d["future_wind"]).all(axis=(1, 2))
        wind = d["wind"][good]
        full = np.concatenate((wind, d["future_wind"][good]), axis=1)
        real = net.predict(wind, d["dst"][good])["min_dst"] - d["truth"][good]
        rows = {}
        for label, center in (("single", PUBLISHED), ("two_state", cal["double"])):
            errors, levels = paired_draws(net, full, wind, center, year)
            paired = np.mean(errors, axis=0)
            rows[label] = dict(twin_mae_nT=float(np.abs(errors).mean()),
                               real_mae_nT=float(np.abs(real).mean()),
                               paired_error_r=float(np.corrcoef(real, paired)[0, 1]),
                               real_minus_twin_minimum_median_nT=float(np.median(d["truth"][good] - np.mean(levels, axis=0))))
        report[year] = dict(paired_issues=int(good.sum()), total_issues=len(good), physics=rows)
    write_json(RESULTS / "twin.json", report)
    print(report)


if __name__ == "__main__":
    main()
