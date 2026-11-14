"""Wind and solar from weather vs ERCOT EIA-930 hourly output.

Fleet: ERCOT wind/solar units online each hour (fleet_ERCO, EIA-860).  Weather: NASA POWER hourly (MERRA-2 based,
0.5 x 0.625 deg) at agent B's ERCOT points; each farm uses its nearest point.  Wind: WS10M/WS50M shear to the hub,
density from T2M/PS/RH2M, farm power curve with specific power by online year.  Solar: pvlib (Erbs + Perez +
single-axis tracking, DC/AC 1.3).  Real: EIA-930 ERCO WND / SUN (after curtailment).
    python3 check_renewables.py 2019 2022
Year 1 is used to fit ONE scale factor per technology (dev calibration, reported); year 2 tests it.
Writes grid/checks/renewables.json
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, os, sys, glob
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np, pandas as pd
from gridsim.fleet import prepare_fleet
from gridsim.renewables import fleet_profiles, turbine_curve, farm_curve

W = f"{GRID_ROOT}/data/raw/weather/power"


def weather(year):
    out = {}
    pts = json.load(open(f"{GRID_ROOT}/data/proc/weather/points.json"))["ERCO"]["points"]
    for p in pts:
        f = f"{W}/{p['pid']}.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        d["time_utc"] = pd.to_datetime(d.time_utc, utc=True)
        d = d.set_index("time_utc").sort_index().loc[f"{year}-01-01":f"{year}-12-31 23:00"]
        d = d.replace(-999.0, np.nan).interpolate(limit_direction="both")
        w = pd.DataFrame({"ws10": d.WS10M, "ws50": d.WS50M, "t2m": d.T2M, "ps": d.PS * 10.0, "rh": d.RH2M / 100.0,
                          "ghi": d.ALLSKY_SFC_SW_DWN.clip(lower=0)})
        w.attrs.update(lat=p["lat"], lon=p["lon"])
        out[p["pid"]] = w
    return out


def stats(sim, real):
    e = sim - real
    return {"real_TWh": float(real.sum() / 1e6), "sim_TWh": float(sim.sum() / 1e6),
            "annual_bias_pct": float((sim.sum() - real.sum()) / real.sum() * 100),
            "hourly_corr": float(np.corrcoef(sim, real)[0, 1]),
            "hourly_nMAE_pct": float(np.abs(e).mean() / real.mean() * 100),
            "daily_corr": float(np.corrcoef(pd.Series(sim).groupby(np.arange(len(sim)) // 24).mean(),
                                            pd.Series(real).groupby(np.arange(len(real)) // 24).mean())[0, 1])}


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2019, 2022]
    res = {"what": __doc__.split("\n")[0], "years": {}}
    e930 = pd.read_parquet(f"{GRID_ROOT}/data/proc/eia930/ba/ERCO.parquet").set_index("time_utc").sort_index()
    scale = {}
    for k, y in enumerate(years):
        wx = weather(y)
        fl = prepare_fleet(f"{GRID_ROOT}/data/proc/fleet/fleet_ERCO.parquet", start=f"{y}-01-01",
                           end=f"{y}-12-31", hr_source="spec")
        real = e930.loc[f"{y}-01-01":f"{y}-12-31 23:00"]
        rec = {}
        for tech, col, kind in (("wind", "ng_WND", "wind"), ("solar_pv", "ng_SUN", "solar")):
            u = fl[fl.tech == tech]
            sim = fleet_profiles(u, wx, kind=kind).reindex(real.index).interpolate(limit_direction="both").values
            r = real[col].astype(float).interpolate(limit_direction="both").values
            raw = stats(sim, r)
            if k == 0:
                scale[tech] = float(r.sum() / sim.sum())
            cal = stats(sim * scale[tech], r)
            rec[tech] = {"raw": raw, "dev_scaled": cal, "scale_from_dev_year": scale[tech],
                         "capacity_MW_end": float(u.pmax_MW.sum()), "n_units": int(len(u))}
            print(y, tech, json.dumps(rec[tech])[:300], flush=True)
        res["years"][str(y)] = rec
    # turbine curve sanity: capacity factor of the generic curve vs Rayleigh winds (published US fleet CF ~ 35-42 %)
    v = np.linspace(0, 30, 3001)
    cfs = {}
    for sp in (220, 300, 400):
        for vm in (6.5, 7.5, 8.5):
            k_ = 2.0; c_ = vm / 0.8862
            pdf = (k_ / c_) * (v / c_) ** (k_ - 1) * np.exp(-(v / c_) ** k_)
            cfs[f"SP{sp}_vmean{vm}"] = float(np.trapezoid(farm_curve(v, sp) * pdf, v))
    res["farm_curve_cf_rayleigh_no_losses"] = cfs
    json.dump(res, open(f"{GRID_CODE}/checks/renewables.json", "w"), indent=1)
