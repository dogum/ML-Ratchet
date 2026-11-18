"""Dev-fit classical competitors. All scored paths forecast the same next three complete hours."""
import sys

import numpy as np
from scipy.optimize import least_squares

from _paths import MODEL
from noise import PROTOCOL, RESULTS, read_year, wind_law, wind_period, write_json
from split import dev_cases

sys.path.insert(0, MODEL)
from stormphysics import PUBLISHED, drivers, projected

NAMES = ("Persistence", "Published O'Brien–McPherron", "Dev-tuned O'Brien–McPherron", "Dev-tuned two-state")


def forecast(cases, cal):
    field, pressure = drivers(cases["wind"])
    result = {NAMES[0]: cases["dst"]}
    for name, params in zip(NAMES[1:], (PUBLISHED, cal["single"], cal["double"])):
        path, _, _ = projected(field, pressure, cases["dst"], params, cal["wind_law"])
        result[name] = path.min(axis=1)
    return result


def fit(dev, law, double):
    wind = np.concatenate([d["wind"] for d in dev.values()])
    e, p = drivers(wind)
    dst = np.concatenate([d["dst"] for d in dev.values()])
    truth = np.concatenate([d["truth"] for d in dev.values()])
    # Equal total influence per dev year and extra weight for the active-target stratum.
    weights = []
    for d in dev.values():
        active = d["truth"] <= -50
        weights.extend(1 / len(active) + active / max(1, active.sum()))
    weights = np.sqrt(np.array(weights) / np.mean(weights))
    start = [4.4, 0.5, 1.7, 0.55, 7.26] if double else [4.4, 1., 7.26]
    lower = [1., 0.15, 1., 0.1, 2.] if double else [1., 0.25, 2.]
    upper = [12., 1., 4., 0.9, 15.] if double else [12., 4., 15.]

    def unpack(x):
        if double:
            return [*x, 11.]
        return [x[0], x[1], x[1], 1., x[2], 11.]

    def residual(x):
        pred, _, _ = projected(e, p, dst, unpack(x), law)
        return (pred.min(axis=1) - truth) * weights

    fitted = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1", f_scale=3.,
                           max_nfev=80, ftol=1e-6, xtol=1e-6)
    return list(map(float, unpack(fitted.x))), int(fitted.nfev)


if __name__ == "__main__":
    wind, hashes = wind_period(PROTOCOL["training_wind_years"])
    _, val_hash = wind_period(PROTOCOL["frozen_wind_years"])
    hashes.update(val_hash)
    for y in PROTOCOL["dev_years"]:
        hashes[str(y)] = read_year(y, index=True)[3]
    write_json(RESULTS / "sources_dev.json", hashes)
    dev = dev_cases()
    law = wind_law(wind)
    single, ns = fit(dev, law, False)
    double, nd = fit(dev, law, True)
    cal = dict(wind_law=law, single=single, double=double, evaluations=[ns, nd])
    cal["dev_mae_nT"] = {yr: {k: float(np.mean(abs(x - d["truth"]))) for k, x in forecast(d, cal).items()}
                         for yr, d in dev.items()}
    write_json(RESULTS / "calibration.json", cal)
    print(cal)
