"""GridSim demo: dispatch one real ERCOT day, then trip a 1,300 MW nuclear unit at the evening peak of that day.

Inputs: EIA-930 ERCOT demand / wind / solar / interchange / nuclear for 15 Aug 2019 and the ERCOT fleet (agent B's
files in grid/data).  Prints the energy balance and the event metrics; writes demo_day.png.
    python3 demo.py [YYYY-MM-DD]
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from gridsim import dispatch, frequency_event

DAY = sys.argv[1] if len(sys.argv) > 1 else "2019-08-15"
FLEET = f"{GRID_ROOT}/data/proc/fleet/fleet_ERCO.parquet"
e = pd.read_parquet(f"{GRID_ROOT}/data/proc/eia930/ba/ERCO.parquet").set_index("time_utc").sort_index()
# one local day (Central time) = 05:00 UTC to 04:00 UTC next day
y = e.loc[pd.Timestamp(DAY + " 05:00", tz="UTC"):pd.Timestamp(DAY + " 05:00", tz="UTC") + pd.Timedelta(hours=23)]
y = y.apply(lambda c: pd.to_numeric(c, errors="coerce")).interpolate(limit_direction="both")

import json
CAL = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gridsim", "data", "calibration.json")))["ercot_dispatch"]
t0 = time.time()
res = dispatch(FLEET, y.d, wind_MW=y.ng_WND, solar_MW=y.ng_SUN, imports_MW=-y.ti.fillna(0), hydro_MW=y.ng_WAT,
               fixed_MW={"nuclear": y.ng_NUC}, method="milp", must_run=tuple(CAL["must_run"]),
               offer_adders=CAL["offer_adders"])          # dev-year (2019) calibration: coal self-commitment + adder
wall = time.time() - t0
g = res.gen_by_fuel
print(f"ERCOT {DAY}: {len(y)} hours dispatched in {wall:.1f} s ({res.cf.C} unit clusters, MILP, calibrated config)")
print("\nEnergy balance (GWh):")
bal = pd.Series({"demand": y.d.sum() / 1e3, "net imports": -y.ti.sum() / 1e3})
for f in g.columns:
    bal[f"gen {f}"] = g[f].sum() / 1e3
print(bal.round(1).to_string())
print(f"generation + imports - demand = {res.balance_residual_MW.sum() / 1e3:.2e} GWh "
      f"(max hourly residual {np.abs(res.balance_residual_MW).max():.1e} MW)")
real = y[["ng_COL", "ng_NG", "ng_NUC", "ng_WND", "ng_SUN"]].sum() / 1e3
print("\nsimulated vs real (EIA-930) GWh:", {k[3:]: (round(g[k[3:]].sum() / 1e3), round(v)) for k, v in real.items()})
print(f"CO2 {res.co2_t.sum() / 1e3:.1f} kt   mean price ${res.price.mean():.1f}/MWh   peak price ${res.price.max():.0f}/MWh")

# frequency event at the evening peak hour
h = int(np.argmax(y.d.values))
st = res.system_state(h, region="ERCO")
st["ffr"] = [dict(MW=1100.0, trigger_Hz=59.7, delay_s=0.5, ramp_s=0.1)]   # load resources on UF relays (ERCOT RRS)
ev = frequency_event(st, 1300.0, t_end=60.0, dt=0.02)
print(f"\nTrip 1,300 MW at {y.index[h]} (demand {y.d.iloc[h]:,.0f} MW, inertia {ev['inertia_MWs'] / 1e3:.0f} GW s):")
print(f"  RoCoF (0-0.5 s) {ev['rocof_Hz_s']:.3f} Hz/s   nadir {ev['nadir_Hz']:.3f} Hz at {ev['t_nadir_s']:.1f} s   "
      f"settling {ev['settling_Hz']:.3f} Hz   UFLS stages {len(ev['ufls'])}   FFR {ev['ffr_MW'].max():.0f} MW")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    loc = y.index.tz_convert("America/Chicago").hour
    order = ["NUC", "COL", "NG", "WAT", "WND", "SUN", "OTH"]
    ax[0].stackplot(loc, *[g[c].values / 1e3 for c in order if c in g], labels=[c for c in order if c in g])
    ax[0].plot(loc, y.d.values / 1e3, "k", lw=2, label="demand")
    ax[0].set_xlabel("hour (Central)"); ax[0].set_ylabel("GW"); ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    ax[0].set_title(f"ERCOT {DAY}: simulated dispatch")
    ax[1].plot(ev["t"], ev["f"]); ax[1].axhline(59.3, color="r", ls=":", lw=1)
    ax[1].set_xlabel("s"); ax[1].set_ylabel("Hz"); ax[1].set_title("1,300 MW trip at the evening peak")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_day.png"), dpi=110)
    print("wrote demo_day.png")
except Exception as ex:  # noqa
    print("plot skipped:", ex)
