"""Reproduce published frequency events with gridsim.frequency (no fitting: inputs from the operator reports).

GB 9 August 2019 (NG ESO Technical Report, 6 Sep 2019; grid/data/raw/reports/fd3584a2.pdf):
  demand 29 GW, inertia 210 GVA.s, response held 1,022 MW primary / 1,314 MW secondary with the delivered shares of
  Table 5, losses Hornsea 737 MW, Little Barford ST 244 MW, ~150 MW embedded (vector shift), ~350 MW embedded on RoCoF
  (0.125 Hz/s), ~200 MW net at 49 Hz, GT1A 210 MW at 16:53:31, GT1B 187 MW at 16:53:58, LFDD 931 MW at 48.8 Hz.
  Reported trace points: nadir 49.1 Hz at 16:52:58, 49.2 Hz at 16:53:18, 48.8 Hz (LFDD) at 16:53:49.398.
  If agent B's 1 s NESO frequency for that afternoon exists, the whole trace is compared too.
Sensitivity: load damping 1-2.5 %/Hz, inertia +-10 %, RoCoF relay timing.
Writes grid/checks/frequency_events.json (+ out/gb_20190809.png)
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import glob, json, os, sys
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np
import pandas as pd
from gridsim.frequency import frequency_event, make_system_state, event_metrics

T0 = pd.Timestamp("2019-08-09 15:52:33.490", tz="UTC")     # 16:52:33.490 BST


def gb_20190809(D_pct_per_Hz=2.5, inertia_GVAs=210.0, rocof_delay=0.5, rocof_window=0.2, uf49_delay=0.5,
                include_stor=False, load_MW=29000.0, losses=None):
    f0 = 50.0
    resp = 1.0 / (0.5 - 0.015)          # GB dynamic services: full delivery at -0.5 Hz, deadband 0.015 Hz
    R = 0.05

    def dyn(name, up_MW, T_s, tech="other", LL_A=(0, 0)):
        K = up_MW * resp                   # MW/Hz
        pmax = K * R * f0                  # rating that gives that gain at droop R
        return dict(name=name, tech=tech, MW=pmax - up_MW, pmax_MW=pmax, H_s=0.0, mva=1.0,
                    governor=dict(enabled=True, droop=R, deadband_Hz=0.015, T_s=T_s, up_MW=up_MW, dn_MW=up_MW,
                                  LL_A=LL_A, LL_B=(0, 0)))
    units = [
        # mandatory response from BM generators: 284 MW primary (10 s) -> 103 % ; 325 MW secondary (30 s) -> 102 %
        dyn("mandatory_gen", 0.98 * 325, 3.0, LL_A=(0.85 * 15.0, 15.0)),
        # dynamic FFR: 74 % of 259 primary, 81 % of 270 secondary
        dyn("dynamic_ffr", 0.81 * 270, 2.0, LL_A=(0.88 * 15.0, 15.0)),
        # enhanced frequency response (batteries, full within 1 s): 94 % of 227 MW
        dyn("efr_battery", 0.94 * 227, 0.5, tech="battery"),
        # all synchronous inertia (no further governor response)
        dict(name="sync_inertia", tech="coal_steam", MW=20000, pmax_MW=20000, H_s=inertia_GVAs * 1e3 / 25000,
             mva=25000, governor=dict(enabled=False)),
    ]
    ffr = [  # static services
        dict(MW=1.00 * 200, trigger_Hz=49.6, delay_s=0.5, ramp_s=2.0),     # interconnectors, 100 % of 200 MW
        dict(MW=0.71 * 31, trigger_Hz=49.7, delay_s=1.0, ramp_s=1.0),      # static low-frequency response
        dict(MW=0.67 * 261, trigger_Hz=49.7, delay_s=2.0, ramp_s=25.0),    # static FFR (secondary, 30 s)
    ]
    protection = [
        dict(MW=350.0, kind="rocof", threshold=0.125, delay_s=rocof_delay, window_s=rocof_window),  # LoM RoCoF
        dict(MW=200.0, kind="uf", threshold=49.0, delay_s=uf49_delay),                              # 49 Hz trips (net)
    ]
    ufls = [dict(f_Hz=48.8, MW=931.0, delay_s=0.2), dict(f_Hz=48.75, frac=0.05, delay_s=0.2),
            dict(f_Hz=48.7, frac=0.10, delay_s=0.2)]
    st = make_system_state(units, load_MW=load_MW, region="GB", D=D_pct_per_Hz * f0 / 100.0, ffr=ffr,
                           protection=protection, ufls=ufls,
                           secondary=dict(MW=400.0, T_s=120.0) if include_stor else None)
    if losses is None:
        losses = [(0.0, 150.0), (0.29, 737.0), (0.51, 244.0), (57.51, 210.0), (84.51, 187.0)]
    return st, losses


def times_of(r, level, after=0.0):
    i = np.where((r["t"] >= after) & (r["f"] <= level))[0]
    return float(r["t"][i[0]]) if len(i) else None


def load_neso_1s():
    """Agent B's NESO 1 s frequency for 9 Aug 2019, if present (npz with t, f + meta json, or parquet)."""
    cands = (glob.glob(f"{GRID_ROOT}/data/proc/frequency/GB/events/GB_20190809T155233.npz") +
             glob.glob(f"{GRID_ROOT}/data/proc/frequency/*2019*08*09*") +
             glob.glob(f"{GRID_ROOT}/data/proc/neso/*2019*08*09*") +
             glob.glob(f"{GRID_ROOT}/data/proc/frequency/gb/*2019-08-09*") +
             glob.glob(f"{GRID_ROOT}/data/proc/**/*20190809*", recursive=True))
    for p in cands:
        try:
            if p.endswith(".npz"):
                z = np.load(p)
                meta = {}
                mj = p.replace(".npz", ".json")
                if os.path.exists(mj):
                    meta = json.load(open(mj))
                t0 = pd.Timestamp(meta.get("start_utc")) if meta.get("start_utc") else None
                if t0 is not None and t0.tzinfo is None:
                    t0 = t0.tz_localize("UTC")
                return p, z["t"], z["f"], t0
            if p.endswith(".parquet"):
                d = pd.read_parquet(p)
                tc = [c for c in d.columns if "time" in c or c == "t"][0]
                fc = [c for c in d.columns if c in ("f", "f_Hz", "freq", "frequency")][0]
                tt = pd.to_datetime(d[tc], utc=True)
                return p, (tt - T0).dt.total_seconds().values, d[fc].values, None
        except Exception:  # noqa
            continue
    return None


if __name__ == "__main__":
    out = {"gb_2019_08_09": {}}
    # the one free parameter: load damping (literature 1-2.5 %/Hz), fitted on the FIRST nadir only (49.1 Hz);
    # the later timeline (recovery to 49.2 Hz, LFDD time) is then a check.
    fit = []
    for Dp in np.round(np.arange(1.0, 2.55, 0.1), 2):
        s_, l_ = gb_20190809(D_pct_per_Hz=Dp)
        q = frequency_event(s_, l_, t_end=40.0, dt=0.05, record=False)
        fit.append((abs(q["f"].min() - 49.1), Dp, float(q["f"].min())))
    D_fit = min(fit)[1]
    out["gb_2019_08_09"]["damping_fit"] = {"D_pct_per_Hz": float(D_fit), "grid": [(float(b), c) for a, b, c in fit]}
    st, losses = gb_20190809(D_pct_per_Hz=D_fit)
    r = frequency_event(st, losses, t_end=110.0, dt=0.02)
    n1 = r["t"] <= 50
    i1 = int(np.argmin(np.where(n1, r["f"], 99)))
    rep = {"nadir1_Hz": 49.1, "t_nadir1_s": 24.5, "t_49p2_recovered_s": 44.5, "t_lfdd_s": 75.9, "lfdd_MW": 931,
           "rocof_trip_MW": 350, "uf49_trip_MW": 200}
    sim = {"nadir1_Hz": float(r["f"][i1]), "t_nadir1_s": float(r["t"][i1]),
           "t_49p2_recovered_s": next((float(t) for t, f in zip(r["t"], r["f"]) if r["t"][i1] < t < 57.5 and f >= 49.2), None),
           "max_Hz_between_nadir1_and_GT1A": float(r["f"][(r["t"] > r["t"][i1]) & (r["t"] < 57.5)].max()),
           "t_lfdd_s": next((u["t_s"] for u in r["ufls"] if u["stage"] == 0), None),
           "lfdd_stages": len(r["ufls"]), "min_Hz": float(r["f"].min()),
           "rocof_0_0p5s_Hz_s": r["rocof_Hz_s"], "trips": r["trips"], "inertia_MWs": r["inertia_MWs"]}
    out["gb_2019_08_09"]["reported"] = rep
    out["gb_2019_08_09"]["simulated"] = sim
    out["gb_2019_08_09"]["errors"] = {"nadir1_Hz": sim["nadir1_Hz"] - rep["nadir1_Hz"],
                                      "t_nadir1_s": sim["t_nadir1_s"] - rep["t_nadir1_s"],
                                      "t_lfdd_s": (sim["t_lfdd_s"] - rep["t_lfdd_s"]) if sim["t_lfdd_s"] else None}
    # sensitivity
    sens = []
    for D in (1.0, 1.5, 2.5):
        for H in (190.0, 210.0, 230.0):
            for rd in (0.0, 0.5):
                s2, l2 = gb_20190809(D_pct_per_Hz=D, inertia_GVAs=H, rocof_delay=rd)
                q = frequency_event(s2, l2, t_end=110.0, dt=0.05, record=False)
                j = int(np.argmin(np.where(q["t"] <= 50, q["f"], 99)))
                sens.append(dict(D_pct_per_Hz=D, inertia_GVAs=H, rocof_delay_s=rd, nadir1_Hz=float(q["f"][j]),
                                 t_nadir1_s=float(q["t"][j]), lfdd=bool(q["ufls"]),
                                 t_lfdd_s=q["ufls"][0]["t_s"] if q["ufls"] else None, min_Hz=float(q["f"].min()),
                                 n_rocof_trip=sum(1 for x in q["trips"] if x["index"] == 0)))
    out["gb_2019_08_09"]["sensitivity"] = sens
    # compare with 1 s data
    meas = load_neso_1s()
    if meas is not None:
        p, tm, fm, t0 = meas
        if t0 is not None:
            tm = tm + (t0 - T0).total_seconds()
        m = (tm >= -5) & (tm <= 110)
        fs = np.interp(tm[m], r["t"], r["f"], left=50.0)
        sel = tm[m] >= 0
        out["gb_2019_08_09"]["vs_neso_1s"] = {"file": p, "rmse_Hz_0_110s": float(np.sqrt(np.mean((fs[sel] - fm[m][sel]) ** 2))),
                                             "measured": event_metrics(tm[m], fm[m]),
                                             "measured_min_Hz": float(fm[m].min())}
    else:
        out["gb_2019_08_09"]["vs_neso_1s"] = "1 s data not available yet (agent B); compared with report timeline only"
    # plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
        ax[0].plot(r["t"], r["f"], label="gridsim (report inputs; load damping fitted on nadir 1)", color="C0")
        if meas is not None:
            ax[0].plot(tm[m], fm[m], color="k", lw=1, label="NESO 1 s measured")
        for k, (tt, ff) in {"nadir 49.1 Hz": (24.5, 49.1), "49.2 Hz": (44.5, 49.2), "LFDD 48.8 Hz": (75.9, 48.8)}.items():
            ax[0].plot(tt, ff, "o", color="C3"); ax[0].annotate(k, (tt, ff), textcoords="offset points", xytext=(5, -12))
        ax[0].set_ylabel("Hz"); ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)
        ax[0].set_title("GB 9 Aug 2019: simulated vs reported (ESO technical report)")
        ax[1].plot(r["t"], r["gov_total_MW"], label="dynamic response")
        ax[1].plot(r["t"], r["ffr_MW"], label="static response")
        ax[1].plot(r["t"], r["damping_MW"], label="load damping")
        ax[1].plot(r["t"], r["shed_MW"], label="LFDD")
        ax[1].set_ylabel("MW"); ax[1].set_xlabel("s after 16:52:33.490 BST"); ax[1].legend(ncol=4, fontsize=8)
        ax[1].grid(alpha=.3)
        os.makedirs(f"{GRID_ROOT}/checks/out", exist_ok=True)
        fig.tight_layout(); fig.savefig(f"{GRID_ROOT}/checks/out/gb_20190809.png", dpi=110)
    except Exception as e:  # noqa
        out["plot_error"] = repr(e)
    json.dump(out, open(f"{GRID_CODE}/checks/frequency_events.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out["gb_2019_08_09"].items() if k != "sensitivity"}, indent=1, default=float))
    for s in sens:
        print(s)
