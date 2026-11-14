"""Weather-driven demand model vs ERCOT EIA-930 demand.

Population-weighted NASA POWER temperature / humidity / irradiance over agent B's 8 ERCOT population points.
Fit (tau, T_h, T_c grid + least squares) on 2018-2019; test on 2021 (without the 10-20 Feb 2021 load shed) and 2022.
Also reports ERCOT's own day-ahead forecast error (EIA-930 DF) on the same hours for scale.
Writes grid/checks/demand.json
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, os, sys
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np, pandas as pd
from gridsim.demand import fit_demand, demand_model

W = f"{GRID_ROOT}/data/raw/weather/power"
pts = [p for p in json.load(open(f"{GRID_ROOT}/data/proc/weather/points.json"))["ERCO"]["points"]
       if p["kind"] == "pop"]
frames = []
for p in pts:
    d = pd.read_parquet(f"{W}/{p['pid']}.parquet")
    d["time_utc"] = pd.to_datetime(d.time_utc, utc=True)
    d = d.set_index("time_utc").sort_index().replace(-999.0, np.nan).interpolate(limit_direction="both")
    frames.append((p["w"], d))
wsum = sum(w for w, _ in frames)
wx = pd.DataFrame({"t2m": sum(w * d.T2M for w, d in frames) / wsum,
                   "rh": sum(w * d.RH2M for w, d in frames) / wsum / 100.0,
                   "ghi": sum(w * d.ALLSKY_SFC_SW_DWN.clip(lower=0) for w, d in frames) / wsum})
e = pd.read_parquet(f"{GRID_ROOT}/data/proc/eia930/ba/ERCO.parquet").set_index("time_utc").sort_index()
load = e.d.astype(float)
dfc = e.df.astype(float)


def sl(s, a, b):
    return s.loc[a:b]


train = wx.loc["2018-01-01":"2019-12-31 23:00"]
params = fit_demand(train, load)
out = {"params": params, "test": {}}
for name, (a, b) in {"train_2018_2019": ("2018-01-01", "2019-12-31 23:00"), "test_2021": ("2021-01-01", "2021-12-31 23:00"),
                     "test_2022": ("2022-01-01", "2022-12-31 23:00")}.items():
    w = wx.loc[a:b]
    pred = demand_model(w, params)
    y = load.reindex(w.index)
    m = y.notna()
    if name == "test_2021":
        m &= ~((w.index >= "2021-02-10") & (w.index < "2021-02-21"))
    err = (pred - y)[m]
    rec = {"MAPE_pct": float((err.abs() / y[m]).mean() * 100), "bias_pct": float(err.sum() / y[m].sum() * 100),
           "RMSE_MW": float(np.sqrt((err ** 2).mean())), "peak_real_MW": float(y[m].max()), "peak_sim_MW": float(pred[m].max()),
           "corr": float(np.corrcoef(pred[m], y[m])[0, 1])}
    # same with the annual level re-anchored (load growth is not weather): bias removed
    k = y[m].sum() / pred[m].sum()
    rec["MAPE_level_adjusted_pct"] = float(((pred[m] * k - y[m]).abs() / y[m]).mean() * 100)
    f = dfc.reindex(w.index)
    mm = m & f.notna() & (f > 0)
    rec["ercot_day_ahead_forecast_MAPE_pct"] = float(((f[mm] - y[mm]).abs() / y[mm]).mean() * 100)
    out["test"][name] = rec
    print(name, rec, flush=True)
json.dump(out, open(f"{GRID_CODE}/checks/demand.json", "w"), indent=1)
