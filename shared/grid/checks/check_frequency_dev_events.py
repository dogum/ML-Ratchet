"""Frequency model vs the DEV events of agent B's frozen splits (grid/data/proc/splits.json, events.*.dev); exam events
are not touched.
GB (4 dev events, NESO 1 s data): 9 Aug 2019 with the ESO report inputs (see check_frequency_events.py); the other
three with B's loss estimate (Elexon REMIT/B1610), NESO outturn inertia and NESO generation as demand, and the
9 Aug 2019 response-holding mix at its validated 89-103 % delivery.  Compared: nadir deviation, time of nadir,
frequency at 60 s, RMS over 0-60 s.
ERCOT (5 dev events, ERCOT operations reports): system state from a one-day LP dispatch of that day (real EIA-930
inputs, calibrated config), inertia rescaled to ERCOT's reported value, loss = reported MW; compared: nadir deviation.
Writes grid/checks/frequency_dev_events.json and out/gb_dev_events.png
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, sys
sys.path.insert(0, f"{GRID_CODE}/gridsim"); sys.path.insert(0, f"{GRID_CODE}/checks")
import numpy as np, pandas as pd
from gridsim.frequency import frequency_event, event_metrics
from check_frequency_events import gb_20190809

P = f"{GRID_ROOT}/data/proc/"
splits = json.load(open(P + "splits.json"))["events"]
out = {"GB": [], "ERCO": []}
sys.path.insert(0, GRID_CODE)
from griddata import apply_label_corrections   # cited loss-label corrections (3 Dec 2019: 1,788 -> 854.7 MW)
gb = apply_label_corrections(pd.read_parquet(P + "frequency/gb_events.parquet")).set_index("event_id")
mix = pd.read_parquet(P + "frequency/GB/genmix.parquet").set_index("time_utc").sort_index()
traces = {}
for eid in splits["GB"]["dev"]:
    meta = json.load(open(P + f"frequency/GB/events/{eid}.json"))
    z = np.load(P + f"frequency/GB/events/{eid}.npz")
    t0 = pd.Timestamp(meta["event_t0_utc"]); start = pd.Timestamp(meta["start_utc"])
    tm = z["t"] + (start - t0).total_seconds()
    fm = z["f"]
    row = gb.loc[eid]
    if eid == "GB_20190809T155233":
        st, losses = gb_20190809(D_pct_per_Hz=1.5)
        # the 1 s data has t0 at 15:52:33; the report's lightning strike is 15:52:33.490
        losses = [(t + 0.49, mw) for t, mw in losses]
        what = "ESO report inputs (D fitted on nadir 1)"
    else:
        dem = float(mix.GENERATION.asof(t0))
        st, _ = gb_20190809(D_pct_per_Hz=1.5, inertia_GVAs=float(row.inertia_outturn_GVAs), load_MW=dem)
        losses = [(0.0, float(row.loss_MW))]
        what = f"loss {row.loss_MW:.0f} MW ({row.loss_confidence}), inertia {row.inertia_outturn_GVAs:.0f} GVA s, demand {dem:.0f} MW"
    r = frequency_event(st, losses, t_end=110.0 if eid == "GB_20190809T155233" else 60.0, dt=0.1)
    pre = float(np.mean(fm[(tm < 0) & (tm > -60)]))
    sim = r["f"] - 50.0 + pre                       # start the simulation from the measured pre-event frequency
    win = (tm >= 0) & (tm <= (100 if eid == "GB_20190809T155233" else 60))
    fs = np.interp(tm[win], r["t"], sim)
    me = event_metrics(tm, fm)
    i60 = np.argmin(np.abs(tm - 60))
    rec = dict(event=eid, inputs=what, measured_nadir_dev_Hz=float(fm[win].min() - pre),
               sim_nadir_dev_Hz=float(sim[r["t"] <= tm[win].max()].min() - pre),
               measured_t_nadir_s=float(tm[win][np.argmin(fm[win])]), sim_t_nadir_s=float(r["t"][np.argmin(sim)]),
               measured_f60_dev_Hz=float(fm[i60] - pre), sim_f60_dev_Hz=float(np.interp(60, r["t"], sim) - pre),
               rms_Hz=float(np.sqrt(np.mean((fs - fm[win]) ** 2))),
               rms_0_75s_Hz=float(np.sqrt(np.mean(((fs - fm[win]) ** 2)[tm[win] <= 75]))))
    rec["nadir_dev_err_pct"] = (rec["sim_nadir_dev_Hz"] - rec["measured_nadir_dev_Hz"]) / abs(rec["measured_nadir_dev_Hz"]) * 100
    out["GB"].append(rec); traces[eid] = (tm[win], fm[win], r["t"], sim)
    print(rec, flush=True)

# ERCOT dev events
from check_ercot_dispatch import real_inputs, FLEET, CAL
from gridsim.dispatch import dispatch
ev = pd.read_parquet(P + "frequency/ercot_events.parquet").set_index("event_id")
ercot_states = {}
for eid in splits["ERCO"]["dev"]:
    row = ev.loc[eid]
    t0 = pd.Timestamp(row.time_utc)
    day0 = t0.floor("D") - pd.Timedelta(hours=2)
    y = real_inputs(t0.year, str(day0), str(day0 + pd.Timedelta(hours=26)))
    res = dispatch(FLEET, y.d, wind_MW=y.ng_WND, solar_MW=y.ng_SUN, imports_MW=-y.ti, hydro_MW=y.ng_WAT,
                   fixed_MW={"nuclear": y.ng_NUC}, method="lp", must_run=tuple(CAL.get("must_run", ["coal_steam"])),
                   offer_adders=CAL.get("offer_adders", {}), lookahead_h=0)
    h = int(np.argmin(np.abs(y.index - t0.floor("h"))))
    st = res.system_state(h, region="ERCO")
    E_sim = sum((u.get("H_s") or 0) * (u.get("mva") or 0) for u in st["units"] if u["tech"] not in ("wind", "solar_pv", "battery"))
    k = float(row.inertia_MWs) / max(E_sim, 1.0)
    for u in st["units"]:
        if u.get("H_s"):
            u["H_s"] = u["H_s"] * k
    st["load_MW"] = float(row.load_MW)
    st["ffr"] = [dict(MW=1100.0, trigger_Hz=59.7, delay_s=0.5, ramp_s=0.1)]       # load resources (UF relays)
    ercot_states[eid] = (st, float(row.mw_loss), -float(row.delta_f_Hz))
    r = frequency_event(st, float(row.mw_loss), t_end=60, dt=0.05)
    rec = dict(event=eid, mw_loss=float(row.mw_loss), load_MW=float(row.load_MW), inertia_GWs=float(row.inertia_MWs) / 1e3,
               sim_inertia_before_rescale_GWs=E_sim / 1e3, measured_nadir_dev_Hz=-float(row.delta_f_Hz),
               sim_nadir_dev_Hz=float(r["nadir_Hz"] - 60.0), sim_t_nadir_s=r["t_nadir_s"],
               sim_settle_dev_Hz=float(r["settling_Hz"] - 60.0),
               sim_headroom_MW=float(sum((u.get("governor") or {}).get("up_MW", 0) for u in st["units"])))
    rec["nadir_dev_err_pct"] = (rec["sim_nadir_dev_Hz"] - rec["measured_nadir_dev_Hz"]) / abs(rec["measured_nadir_dev_Hz"]) * 100
    rec["measured_MW_per_0.1Hz"] = rec["mw_loss"] / (abs(rec["measured_nadir_dev_Hz"]) * 10)
    rec["sim_MW_per_0.1Hz"] = rec["mw_loss"] / (abs(rec["sim_nadir_dev_Hz"]) * 10)
    out["ERCO"].append(rec)
    print(rec, flush=True)
# one dev-fitted ERCOT parameter: the effective load damping D (stands for all fast response the model lacks)
import copy
grid = np.round(np.arange(1.0, 8.01, 0.25), 2)
sse = []
for D in grid:
    errs = []
    for eid, (st, loss, meas) in ercot_states.items():
        s2 = copy.deepcopy(st); s2["D"] = float(D)
        q = frequency_event(s2, loss, t_end=30, dt=0.05, record=False)
        errs.append((q["nadir_Hz"] - 60.0 - meas) / abs(meas))
    sse.append((float(np.mean(np.square(errs))), float(D), [float(e) * 100 for e in errs]))
best = min(sse)
out["ERCO_damping_fit"] = {"D_pct_per_pct": best[1], "D_pct_per_Hz": best[1] * 100 / 60.0,
                           "nadir_err_pct_after_fit": best[2], "grid_rms_err_pct": [(d, float(np.sqrt(m)) * 100) for m, d, _ in sse]}
print("ERCOT D fit", out["ERCO_damping_fit"]["D_pct_per_pct"], best[2], flush=True)
for k in ("GB", "ERCO"):
    e = np.array([x["nadir_dev_err_pct"] for x in out[k]])
    out[k + "_summary"] = {"n": len(e), "nadir_dev_err_pct_mean": float(e.mean()), "nadir_dev_err_pct_mae": float(np.abs(e).mean())}
json.dump(out, open(f"{GRID_CODE}/checks/frequency_dev_events.json", "w"), indent=1, default=float)
# the fitted ERCOT damping belongs to this physics version: store it in the version's calibration file
from gridsim.fleet import calibration_path, physics_version
_cp = calibration_path(); _cal = json.load(open(_cp)) if _os.path.exists(_cp) else {}
_cal.setdefault("frequency", {}).update(ERCO_D_pct_per_pct=out["ERCO_damping_fit"]["D_pct_per_pct"],
                                        ERCO_fitted_on="5 ERCOT dev events of splits.json, states from the calibrated "
                                                       f"dispatch of physics {physics_version()}")
json.dump(_cal, open(_cp, "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (eid, (tm, fm, ts, fs)) in zip(axs, traces.items()):
        ax.plot(tm, fm, "k", lw=1, label="NESO 1 s"); ax.plot(ts, fs, "C0--", label="gridsim")
        ax.set_title(eid, fontsize=9); ax.set_xlabel("s"); ax.grid(alpha=.3)
    axs[0].set_ylabel("Hz"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{GRID_ROOT}/checks/out/gb_dev_events.png", dpi=100)
except Exception as e:  # noqa
    print(e)
