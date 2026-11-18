"""Dev-only diagnostic (17 Nov 2026, before any v1.1 rung): where does the tuned two-state physics miss?

Splits the dev error of the dev-tuned two-state forecast into (a) not knowing the next three hours of wind and
(b) the physics itself, by re-running the same physics with the actual future wind (an oracle no forecast has),
and tabulates residuals by issue-time electric field, Dst level and storm phase. Writes dev_diagnostic.json.
"""
import json
import sys

import numpy as np

from _paths import MODEL
from noise import HERE
from split import dev_cases

sys.path.insert(0, MODEL)
from stormphysics import HORIZON, drivers, projected, reconstruct, step  # noqa: E402

cal = json.loads((HERE / "calibration.json").read_text())


def oracle(case, params):
    """Same conditioning as projected(), but stepping with the actual next three hours of wind."""
    gain, tf, ts, share, b, c = params
    e, p = drivers(case["wind"])
    fast, slow = reconstruct(e, params)
    total = fast + slow
    frac = np.divide(fast, total, out=np.full_like(total, share), where=np.abs(total) > 1e-8)
    corrected = case["dst"] - b * p[:, -1] + c
    fast, slow = frac * corrected, (1 - frac) * corrected
    fe, fp = drivers(case["future_wind"])
    path = []
    for k in range(HORIZON):
        ek = np.where(np.isfinite(fe[:, k]), fe[:, k], e[:, -1])
        pk = np.where(np.isfinite(fp[:, k]), fp[:, k], p[:, -1])
        fast = step(fast, ek, gain, tf, share)
        slow = step(slow, ek, gain, ts, 1 - share)
        path.append(fast + slow + b * pk - c)
    return np.stack(path, 1).min(axis=1)


def table(mask, err_f, err_o, res):
    if mask.sum() < 20:
        return None
    return dict(n=int(mask.sum()), mae_forecast=round(float(np.mean(np.abs(err_f[mask]))), 2),
                mae_with_true_wind=round(float(np.mean(np.abs(err_o[mask]))), 2),
                mean_residual=round(float(np.mean(res[mask])), 2))


out = {}
allc = {k: np.concatenate([d[k] for d in dev_cases().values()]) for k in ("wind", "dst", "truth", "future_wind", "past_dst")}
e, p = drivers(allc["wind"])
path, _, _ = projected(e, p, allc["dst"], cal["double"], cal["wind_law"])
fc = path.min(axis=1)
orc = oracle(allc, cal["double"])
res = allc["truth"] - fc
ef, eo = fc - allc["truth"], orc - allc["truth"]
trend = allc["dst"] - allc["past_dst"][:, -4]
strata = {
    "all": np.ones(len(fc), bool),
    "storm_targets_le_-50": allc["truth"] <= -50,
    "E_at_issue_<1": e[:, -1] < 1, "E_1-3": (e[:, -1] >= 1) & (e[:, -1] < 3), "E_>=3": e[:, -1] >= 3,
    "main_phase (Dst fell >10 nT in 3 h)": trend < -10, "recovery (rose >5 nT in 3 h)": trend > 5,
    "Dst_<-100": allc["dst"] < -100,
}
out = {k: table(m, ef, eo, res) for k, m in strata.items()}
(HERE / "dev_diagnostic.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=1))
