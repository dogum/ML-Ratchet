"""Post-process the ERCOT dispatch runs: price metrics vs ERCOT day-ahead hub average for every saved config (some
runs started before the price comparison was added), and a figure of monthly energy by fuel, simulated vs EIA-930.
    python3 summarize_ercot.py 2019 2022
Updates grid/checks/ercot_dispatch_<year>.json and writes out/ercot_<year>_monthly.png
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, os, sys, glob
sys.path.insert(0, f"{GRID_CODE}/checks")
import numpy as np, pandas as pd
from check_ercot_dispatch import load_prices, real_inputs, FUELS

for year in [int(a) for a in sys.argv[1:]] or [2019]:
    jp = f"{GRID_CODE}/checks/ercot_dispatch_{year}.json"
    d = json.load(open(jp))
    y = real_inputs(year)
    for name, rec in d["configs"].items():
        p = f"{GRID_ROOT}/checks/out/ercot_{year}_{name}.parquet"
        if not os.path.exists(p):
            continue
        fr = pd.read_parquet(p)
        pr = load_prices(year, fr.index)
        sp = fr["price_$_per_MWh"].values
        ok = np.isfinite(pr)
        lo = ok & (pr < 100)
        rec["price_vs_ercot_dam_hubavg"] = {
            "real_mean": float(pr[ok].mean()), "sim_mean": float(sp[ok].mean()),
            "real_median": float(np.median(pr[ok])), "sim_median": float(np.median(sp[ok])),
            "MAE": float(np.abs(sp[ok] - pr[ok]).mean()), "MAE_hours_real_below_100": float(np.abs(sp[lo] - pr[lo]).mean()),
            "corr": float(np.corrcoef(sp[ok], pr[ok])[0, 1]), "corr_hours_real_below_100": float(np.corrcoef(sp[lo], pr[lo])[0, 1]),
            "real_p95": float(np.percentile(pr[ok], 95)), "sim_p95": float(np.percentile(sp[ok], 95)),
            "real_hours_above_200": int((pr[ok] > 200).sum()), "sim_hours_above_200": int((sp[ok] > 200).sum()),
            "spearman": float(pd.Series(sp[ok]).rank().corr(pd.Series(pr[ok]).rank())),
            "median_abs_err": float(np.median(np.abs(sp[ok] - pr[ok]))),
            "share_hours_sim_le_0": float(np.mean(sp[ok] <= 0)), "share_hours_real_le_0": float(np.mean(pr[ok] <= 0)),
            "diurnal_median_profile_corr": float(np.corrcoef(
                pd.Series(sp[ok]).groupby(fr.index.tz_convert("America/Chicago").hour[ok]).median(),
                pd.Series(pr[ok]).groupby(fr.index.tz_convert("America/Chicago").hour[ok]).median())[0, 1]),
            "monthly_mean_corr": float(pd.DataFrame({"s": sp, "r": pr}, index=fr.index)[ok].resample("MS").mean().corr().iloc[0, 1]),
            "daily_mean_corr": float(pd.DataFrame({"s": sp, "r": pr}, index=fr.index)[ok].resample("D").mean().corr().iloc[0, 1])}
        if "inertia_MWs" in fr:
            rec["inertia_GWs_min_p50_max"] = [float(fr.inertia_MWs.min() / 1e3), float(fr.inertia_MWs.median() / 1e3),
                                              float(fr.inertia_MWs.max() / 1e3)]
    json.dump(d, open(jp, "w"), indent=1)
    best = "calibrated" if "calibrated" in d["configs"] else list(d["configs"])[0]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        m = d["configs"][best]["by_fuel"]
        mon = np.arange(1, 13)
        for f, c in (("NG", "C2"), ("COL", "C1"), ("NUC", "C0"), ("WND", "C4")):
            ax[0].plot(mon, m[f]["monthly_TWh_real"], "-", color=c, label=f + " real")
            ax[0].plot(mon, m[f]["monthly_TWh_sim"], "--", color=c, label=f + " sim")
        ax[0].set_xlabel("month"); ax[0].set_ylabel("TWh"); ax[0].legend(ncol=2, fontsize=7)
        ax[0].set_title(f"ERCOT {year} monthly energy ({best})")
        fr = pd.read_parquet(f"{GRID_ROOT}/checks/out/ercot_{year}_{best}.parquet")
        wk = fr.index[(fr.index >= f"{year}-07-15") & (fr.index < f"{year}-07-22")]
        ax[1].plot(wk, y.loc[wk, "ng_NG"] / 1e3, "C2-", label="gas real"); ax[1].plot(wk, fr.loc[wk, "MW_NG"] / 1e3, "C2--", label="gas sim")
        ax[1].plot(wk, y.loc[wk, "ng_COL"] / 1e3, "C1-", label="coal real"); ax[1].plot(wk, fr.loc[wk, "MW_COL"] / 1e3, "C1--", label="coal sim")
        ax[1].set_ylabel("GW"); ax[1].legend(fontsize=7); ax[1].tick_params(axis="x", rotation=30); ax[1].set_title("one July week")
        pr = load_prices(year, fr.index)
        ax[2].plot(fr.index, pd.Series(pr, index=fr.index).rolling(24 * 7).mean(), "k-", label="ERCOT DAM hub avg (7-day mean)")
        ax[2].plot(fr.index, fr["price_$_per_MWh"].rolling(24 * 7).mean(), "C3--", label="simulated")
        ax[2].set_ylabel("$/MWh"); ax[2].legend(fontsize=7); ax[2].tick_params(axis="x", rotation=30); ax[2].set_title("price")
        fig.tight_layout(); fig.savefig(f"{GRID_ROOT}/checks/out/ercot_{year}_monthly.png", dpi=100)
    except Exception as e:  # noqa
        print("plot", e)
    for name, rec in d["configs"].items():
        print(year, name, {k: round(v, 1) if isinstance(v, float) else v for k, v in rec.get("price_vs_ercot_dam_hubavg", {}).items()})
