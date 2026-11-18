"""ring-3 physics candidates, fitted on exactly the dev blocks (17 Nov 2026; plan in NOTEBOOK.md).

  double         the v1.0 dev-tuned two-state model, refitted here as the reference
  double_assim   a. fast/slow split assimilated from the window's observed Dst
  double_wang    b. injection x (P / 3 nPa)^gamma
  double_both    a + b
A candidate is kept only if it lowers dev MAE on all issues and on storm targets (<= -50 nT) in every dev year.
Writes calibration_v3.json.
"""
import json
import sys

import numpy as np
from scipy.optimize import least_squares

from _paths import MODEL
from noise import HERE
from split import dev_cases

sys.path.insert(0, MODEL)
from stormphysics import drivers, projected  # noqa: E402

CAL = json.loads((HERE / "calibration.json").read_text())
LAW = CAL["wind_law"]


def forecast(d, params, assim):
    e, p = drivers(d["wind"])
    pd = d["past_dst"] if assim else None
    if assim:
        _, pp = drivers(d["wind"])
    path, _, _ = projected(e, p, d["dst"], params, LAW, past_dst=pd)
    return path.min(axis=1)


def fit(dev, assim, wang):
    start = [4.4, 0.5, 1.7, 0.55, 7.26] + ([0.2] if wang else [])
    lower = [1., 0.15, 1., 0.1, 2.] + ([-0.5] if wang else [])
    upper = [12., 1., 4., 0.9, 15.] + ([1.0] if wang else [])
    weights = []
    for d in dev.values():
        active = d["truth"] <= -50
        weights.append(np.sqrt((1 / len(active) + active / max(1, active.sum())) * len(active) / 2))
    weights = np.concatenate(weights)
    truth = np.concatenate([d["truth"] for d in dev.values()])

    def unpack(x):
        return [*x[:5], 11.] + ([float(x[5])] if wang else [])

    def residual(x):
        pred = np.concatenate([forecast(d, unpack(x), assim) for d in dev.values()])
        return (pred - truth) * weights

    r = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1", f_scale=3., max_nfev=80,
                      ftol=1e-6, xtol=1e-6)
    return [float(v) for v in unpack(r.x)], int(r.nfev)


def scores(dev, params, assim):
    out = {}
    for yr, d in dev.items():
        f = forecast(d, params, assim)
        a = d["truth"] <= -50
        out[yr] = dict(all=round(float(np.mean(np.abs(f - d["truth"]))), 3),
                       storm=round(float(np.mean(np.abs(f[a] - d["truth"][a]))), 3))
    return out


dev = dev_cases()
res = {"double": dict(params=CAL["double"], dev=scores(dev, CAL["double"], False))}
for name, assim, wang in (("double_assim", True, False), ("double_wang", False, True), ("double_both", True, True)):
    params, nfev = fit(dev, assim, wang)
    res[name] = dict(params=params, assimilate=assim, evaluations=nfev, dev=scores(dev, params, assim))
ref = res["double"]["dev"]
for name in ("double_assim", "double_wang", "double_both"):
    d = res[name]["dev"]
    res[name]["kept"] = all(d[y]["all"] < ref[y]["all"] and d[y]["storm"] < ref[y]["storm"] for y in ref)
(HERE / "calibration_v3.json").write_text(json.dumps(res, indent=2) + "\n")
for k, v in res.items():
    print(k, v.get("kept", "ref"), [round(x, 3) for x in v["params"]], v["dev"])
