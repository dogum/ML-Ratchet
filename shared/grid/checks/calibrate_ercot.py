"""Dev-year calibration of the ERCOT dispatch (2019 only): coal offer adder [$/MWh] with coal self-commitment and
observed nuclear, fitted so that simulated coal energy matches EIA-930 on 12 sample weeks (first Monday-week of each
month, LP mode).  Writes gridsim/gridsim/data/calibration.json -> used by the 'calibrated' config.
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, sys, os
sys.path.insert(0, f"{GRID_CODE}/gridsim")
sys.path.insert(0, f"{GRID_CODE}/checks")
from gridsim.fleet import calibration_path  # noqa: E402  (data/calibration.json for physics 1.0, calibration_v1_1.json for 1.1)
import numpy as np, pandas as pd
from check_ercot_dispatch import real_inputs, FLEET
from gridsim.dispatch import dispatch
YEAR = 2019
weeks = []
for m in range(1, 13):
    d0 = pd.Timestamp(f"{YEAR}-{m:02d}-01")
    d0 = d0 + pd.Timedelta(days=(7 - d0.dayofweek) % 7)
    weeks.append((str(d0.date()), str((d0 + pd.Timedelta(days=6, hours=23)).strftime("%Y-%m-%d %H:%M"))))
res = []
for adder in [0.0, -5.0, -10.0, -15.0, -20.0]:
    tot = {"COL": [0, 0], "NG": [0, 0]}
    for a, b in weeks:
        y = real_inputs(YEAR, a, b)
        r = dispatch(FLEET, y.d, wind_MW=y.ng_WND, solar_MW=y.ng_SUN, imports_MW=-y.ti, hydro_MW=y.ng_WAT,
                     fixed_MW={"nuclear": y.ng_NUC}, method="lp", must_run=("coal_steam",),
                     offer_adders={"coal_steam": adder})
        g = r.gen_by_fuel.sum()
        for f in tot:
            tot[f][0] += g.get(f, 0) / 1e6; tot[f][1] += y["ng_" + f].sum() / 1e6
    row = dict(adder=adder, **{f"{f}_sim_TWh": v[0] for f, v in tot.items()}, **{f"{f}_real_TWh": v[1] for f, v in tot.items()})
    row["COL_err_pct"] = (row["COL_sim_TWh"] - row["COL_real_TWh"]) / row["COL_real_TWh"] * 100
    res.append(row); print(row, flush=True)
df = pd.DataFrame(res).sort_values("adder")
# interpolate the adder giving zero coal error
assert (df.COL_err_pct.min() < 0 < df.COL_err_pct.max()), 'scan does not bracket zero coal error'
best = float(np.interp(0.0, df.COL_err_pct.values[::-1] if df.COL_err_pct.iloc[0] > df.COL_err_pct.iloc[-1] else df.COL_err_pct.values,
                       df.adder.values[::-1] if df.COL_err_pct.iloc[0] > df.COL_err_pct.iloc[-1] else df.adder.values))
p = calibration_path()
cal = json.load(open(p)) if os.path.exists(p) else {}
cal["ercot_dispatch"] = {"fitted_on": f"{YEAR} (dev), 12 sample weeks, LP", "must_run": ["coal_steam"],
                         "offer_adders": {"coal_steam": round(best, 2)}, "scan": res,
                         "note": "coal self-commitment + offer adder stand for captive-mine lignite economics and "
                                 "self-scheduling that public cost data do not show"}
json.dump(cal, open(p, "w"), indent=1)
print("best adder", best)
