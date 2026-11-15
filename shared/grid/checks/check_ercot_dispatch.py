"""ERCOT year of hourly dispatch vs EIA-930 generation by fuel, CAMPD CO2 and ERCOT prices.

    python3 check_ercot_dispatch.py <year> <config> [<config> ...]
configs (see CONFIGS): physics, obs_nuc, selfcommit, calibrated, merit, lp
Real inputs: EIA-930 ERCO demand, wind, solar, interchange, hydro (grid/data/proc/eia930, agent B) and the ERCOT fleet
(grid/data/proc/fleet/fleet_ERCO.parquet).  Fuel prices: EIA-923 monthly plant/state delivered costs + Henry Hub daily.
Writes grid/checks/out/ercot_<year>_<config>.parquet (hourly) and grid/checks/ercot_dispatch_<year>.json (summary,
merged with earlier configs).
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, os, sys, time
sys.path.insert(0, f"{GRID_CODE}/gridsim")
from gridsim.fleet import calibration_path  # noqa: E402  (data/calibration.json for physics 1.0, calibration_v1_1.json for 1.1)
import numpy as np
import pandas as pd
from gridsim.dispatch import dispatch

OUT = f"{GRID_ROOT}/checks/out"
os.makedirs(OUT, exist_ok=True)
FLEET = f"{GRID_ROOT}/data/proc/fleet/fleet_ERCO.parquet"
FUELS = ["COL", "NG", "NUC", "WND", "SUN", "WAT", "OTH"]

# calibration constants fitted on the dev year 2019 only (checks/calibrate_ercot.py -> gridsim/data/calibration.json)
CAL = {}
calp = calibration_path()
if os.path.exists(calp):
    CAL = json.load(open(calp)).get("ercot_dispatch", {})

CONFIGS = {
    # final set (physics 1.0): "economic" (pure economics + observed nuclear), "calibrated" (+ coal self-commitment and
    # the dev-year 2019 offer adder), "merit"; prices from the LP relaxation unless pricing="fixed".
    "economic": dict(method="milp", fixed="nuclear"),
    # pure data-driven economics: costs from EIA-923, outages expected, nuclear modelled, hydro budget observed
    "physics": dict(method="milp", pricing="fixed"),
    # + observed nuclear output (EIA-930 NUC) as a fixed injection (refuelling outages are not predictable)
    "obs_nuc": dict(method="milp", fixed="nuclear", pricing="fixed"),
    # + coal self-commitment (units always committed when available)
    "selfcommit": dict(method="milp", fixed="nuclear", must_run=("coal_steam",), pricing="fixed"),
    # + dev-calibrated offer adders
    "calibrated": dict(method="milp", fixed="nuclear", must_run=tuple(CAL.get("must_run", ["coal_steam"])),
                       offer_adders=CAL.get("offer_adders", {})),
    "calibrated_fixedprice": dict(method="milp", fixed="nuclear", must_run=tuple(CAL.get("must_run", ["coal_steam"])),
                                  offer_adders=CAL.get("offer_adders", {}), pricing="fixed"),
    # same, prices from the LP relaxation (the default since the dev-week comparison; 'calibrated' used fixed pricing)
    "calibrated_relaxed": dict(method="milp", fixed="nuclear", must_run=tuple(CAL.get("must_run", ["coal_steam"])),
                               offer_adders=CAL.get("offer_adders", {}), pricing="relaxed"),
    # 'calibrated' with soft history constraints (HANDOFF P0.6): penalised slack instead of re-solving a day without
    # its carried history; target: fewer than 0.5 % of days re-solved
    "calibrated_soft": dict(method="milp", fixed="nuclear", must_run=tuple(CAL.get("must_run", ["coal_steam"])),
                            offer_adders=CAL.get("offer_adders", {}), soft_history=True),
    "calibrated_lp": dict(method="lp", fixed="nuclear", must_run=tuple(CAL.get("must_run", ["coal_steam"])),
                          offer_adders=CAL.get("offer_adders", {})),
    "merit": dict(method="merit", fixed="nuclear"),
}


def real_inputs(year, start=None, end=None):
    d = pd.read_parquet(f"{GRID_ROOT}/data/proc/eia930/ba/ERCO.parquet").set_index("time_utc").sort_index()
    y = d.loc[start or f"{year}-01-01":end or f"{year}-12-31 23:00"].copy()
    for c in ["d", "ti"] + ["ng_" + f for f in FUELS]:
        y[c] = pd.to_numeric(y[c], errors="coerce").astype(float).interpolate(limit_direction="both").fillna(0)
    # EIA-930 OTH has a few corrupted hours (27 GW); clip to the plausible range
    y["ng_OTH"] = y.ng_OTH.clip(0, 2000)
    return y


def run(year, name, start=None, end=None, save=True):
    cfg = dict(CONFIGS[name])
    y = real_inputs(year, start, end)
    fixed = None
    if cfg.pop("fixed", None) == "nuclear":
        fixed = {"nuclear": y.ng_NUC}
    t0 = time.time()
    r = dispatch(FLEET, y.d, wind_MW=y.ng_WND, solar_MW=y.ng_SUN, imports_MW=-y.ti, hydro_MW=y.ng_WAT,
                 fixed_MW=fixed, verbose=True, **cfg)
    wall = time.time() - t0
    fr = r.frame()
    fr["co2_campd_units_t"] = r.co2_t[:, [c["tech"] not in ("biomass", "other") for c in r.cf.clusters]].sum(1)
    if save:
        fr.to_parquet(f"{OUT}/ercot_{year}_{name}.parquet")
    return r, y, wall


def metrics(fr, y):
    out = {}
    for f in FUELS:
        sim = fr.get("MW_" + f, pd.Series(0.0, index=fr.index)).values
        real = y["ng_" + f].values
        e = sim - real
        out[f] = {"real_TWh": float(real.sum() / 1e6), "sim_TWh": float(sim.sum() / 1e6),
                  "annual_err_pct": float((sim.sum() - real.sum()) / max(real.sum(), 1) * 100),
                  "hourly_MAE_MW": float(np.abs(e).mean()), "hourly_RMSE_MW": float(np.sqrt((e ** 2).mean())),
                  "hourly_nMAE_pct_of_mean": float(np.abs(e).mean() / max(real.mean(), 1) * 100),
                  "hourly_corr": float(np.corrcoef(sim, real)[0, 1]) if sim.std() > 0 and real.std() > 0 else None,
                  "daily_nMAE_pct": float(np.abs(pd.Series(e, index=fr.index).resample("D").mean()).mean() /
                                          max(real.mean(), 1) * 100)}
        mon = pd.DataFrame({"s": sim, "r": real}, index=fr.index).resample("MS").sum() / 1e6
        out[f]["monthly_TWh_sim"] = [round(v, 3) for v in mon.s]
        out[f]["monthly_TWh_real"] = [round(v, 3) for v in mon.r]
    return out


def prices_real(year):
    """Real ERCOT hub-average real-time price from agent B's files, if present."""
    cands = [f"{GRID_ROOT}/data/proc/prices/ercot_hub_rt_hourly.parquet",
             f"{GRID_ROOT}/data/proc/prices/ERCO.parquet", f"{GRID_ROOT}/data/proc/prices/ercot.parquet"]
    for p in cands + sorted(__import__("glob").glob(f"{GRID_ROOT}/data/proc/prices/*rcot*.parquet")):
        if os.path.exists(p):
            return p
    return None


def load_prices(year, index):
    p = f"{GRID_ROOT}/data/proc/prices/ercot_dam.parquet"
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    d = d[d.node == "HB_HUBAVG"].set_index("time_utc").price.sort_index()
    d = d[~d.index.duplicated()]
    return d.reindex(index).values.astype(float)


def campd_co2(year):
    p = f"{GRID_ROOT}/data/proc/campd/hourly/tx_{year}.parquet"
    if not os.path.exists(p):
        return None
    fl = pd.read_parquet(FLEET)
    erc = set(fl.plant_id.astype(int))
    c = pd.read_parquet(p, columns=["plant_id_epa", "time_utc", "co2_short_tons", "gross_load_MW"])
    c = c[c.plant_id_epa.isin(erc)]
    return {"co2_Mt": float(c.co2_short_tons.sum() * 0.90718474 / 1e6), "n_plants": int(c.plant_id_epa.nunique()),
            "gross_TWh": float(c.gross_load_MW.sum() / 1e6),
            "hourly": (c.groupby("time_utc").co2_short_tons.sum() * 0.90718474)}


if __name__ == "__main__":
    year = int(sys.argv[1])
    names = sys.argv[2:] or ["physics"]
    jpath = f"{GRID_CODE}/checks/ercot_dispatch_{year}.json"
    summary = json.load(open(jpath)) if os.path.exists(jpath) else {"year": year, "configs": {}}
    camp = campd_co2(year)
    for name in names:
        r, y, wall = run(year, name)
        fr = pd.read_parquet(f"{OUT}/ercot_{year}_{name}.parquet")
        m = metrics(fr, y)
        rec = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIGS[name].items()},
               "wall_s": wall, "by_fuel": m, "summary": r.summary,
               "max_balance_residual_MW": float(np.abs(r.balance_residual_MW).max()),
               "n_days_relaxed_history": int(getattr(r, "n_days_relaxed", 0)),
               "solver_status_counts": pd.Series([str(i.get("status")) for i in r.solver_stats]).value_counts().to_dict()
               if getattr(r, "solver_stats", None) else None,
               "solve_s_per_day_median": float(np.median([i.get("solve_s", np.nan) for i in r.solver_stats]))
               if getattr(r, "solver_stats", None) else None,
               "price_mean": float(fr["price_$_per_MWh"].mean()),
               "price_p5_p50_p95": [float(np.percentile(fr["price_$_per_MWh"], q)) for q in (5, 50, 95)]}
        if camp is not None:
            sim_co2 = fr.co2_campd_units_t.sum() / 1e6
            rec["co2"] = {"sim_Mt_fossil_units": float(sim_co2), "campd_Mt_ercot_plants": camp["co2_Mt"],
                          "err_pct": float((sim_co2 - camp["co2_Mt"]) / camp["co2_Mt"] * 100),
                          "campd_plants": camp["n_plants"]}
            h = camp["hourly"].reindex(fr.index).fillna(0)
            rec["co2"]["hourly_corr"] = float(np.corrcoef(h.values, fr.co2_campd_units_t.values)[0, 1])
        pr = load_prices(year, fr.index)
        if pr is not None:
            sp = fr["price_$_per_MWh"].values
            ok = np.isfinite(pr)
            rec["price_vs_ercot_dam_hubavg"] = {
                "real_mean": float(pr[ok].mean()), "sim_mean": float(sp[ok].mean()),
                "real_median": float(np.median(pr[ok])), "sim_median": float(np.median(sp[ok])),
                "MAE": float(np.abs(sp[ok] - pr[ok]).mean()),
                "MAE_hours_real_below_100": float(np.abs(sp[ok] - pr[ok])[pr[ok] < 100].mean()),
                "corr": float(np.corrcoef(sp[ok], pr[ok])[0, 1]),
                "corr_hours_real_below_100": float(np.corrcoef(sp[ok][pr[ok] < 100], pr[ok][pr[ok] < 100])[0, 1]),
                "real_p95": float(np.percentile(pr[ok], 95)), "sim_p95": float(np.percentile(sp[ok], 95)),
                "real_hours_above_200": int((pr[ok] > 200).sum()), "sim_hours_above_200": int((sp[ok] > 200).sum())}
        summary["configs"][name] = rec
        json.dump(summary, open(jpath, "w"), indent=1)
        print(name, "wall", round(wall), {f: round(m[f]["annual_err_pct"], 1) for f in FUELS},
              rec.get("co2", {}).get("err_pct"), flush=True)
