"""v1.0.1 error bars (17 Nov 2026; NOTEBOOK.md): one widen-only factor per bin of the last observed Dst
(<= -50, -50 to -30, > -30 nT), fitted on the released rung's dev residuals, replacing v1.0's single factor 1.167.
The plan named electric-field bins; on dev those bins were already covered (64-69 %), while issues at Dst <= -50 nT
were covered 55 %, so the bins follow Dst instead (disclosed before the reserve was opened).
Writes the factors into ../stormnet_weights.npz (meta.widen_by_dst) and training/calibration_sigma.json.
"""
import json
from pathlib import Path

import numpy as np
from mlratchet.calibrate import coverage, widen_factor

import sys

from _paths import MODEL

sys.path.insert(0, MODEL)
from noise import HERE
from split import dev_cases
from stormnet import StormNet  # noqa: E402
from stormphysics import drivers  # noqa: E402

EDGES = [-50.0, -30.0]
path = Path(MODEL) / "stormnet_weights.npz"
model = StormNet(path)
res, sig, e, efield = [], [], [], []
for d in dev_cases().values():
    raw = model.predict(d["wind"], d["dst"], calibrated=False)
    res.extend(raw["min_dst"] - d["truth"])
    sig.extend(raw["sigma"])
    e.extend(d["dst"])
    efield.extend(drivers(d["wind"])[0][:, -1])
res, sig, e, efield = map(np.asarray, (res, sig, e, efield))
b = np.searchsorted(EDGES, e, side="right")
factors, report = [], []
for k in range(len(EDGES) + 1):
    m = b == k
    f = widen_factor(res[m], sig[m])
    factors.append(round(f, 4))
    report.append(dict(n=int(m.sum()), single_factor_coverage=round(100 * coverage(res[m], sig[m], model.meta["widen_factor"]), 1),
                       factor=round(f, 4), coverage=round(100 * coverage(res[m], sig[m], f), 1)))
field_bins = [dict(bin=lab, n=int(m.sum()), coverage_single_factor=round(100 * coverage(res[m], sig[m], model.meta["widen_factor"]), 1))
              for lab, m in (("E < 1", efield < 1), ("E 1-3", (efield >= 1) & (efield < 3)), ("E >= 3", efield >= 3))]
meta = {k: v for k, v in model.meta.items() if k != "widen_by_field"}
meta.update(widen_by_dst=dict(edges=EDGES, factors=factors), version="1.0.1")
with np.load(path) as z:
    arrays = {k: z[k] for k in z if k != "meta"}
np.savez_compressed(path, meta=json.dumps(meta), **arrays)
(HERE / "calibration_sigma.json").write_text(json.dumps(dict(
    variable="last observed Dst (nT); bins <= -50, -50 to -30, > -30", edges=EDGES, bins=report,
    electric_field_bins_rejected=field_bins), indent=2) + "\n")
print(json.dumps(report, indent=1))
