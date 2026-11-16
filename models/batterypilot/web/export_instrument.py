"""Export BatteryPilot's live instrument for the website: the network in float16, the three real exam node-years in
examples/ (hourly prices, demand forecast, temperature, local calendar), and reference outputs for the parity test.

    python models/batterypilot/web/export_instrument.py      # writes site/public/instruments/batterypilot/

The browser rebuilds the 135 causal features from the hourly arrays, runs the network for every hour of the chosen
month, and runs the battery against the real RT prices with the offer rule. It also runs GridData's daily-spread rule
and settles both with rainflow wear. The perfect-hindsight bound (an LP on the month's RT prices) is computed here in
Python for every efficiency x wear-cost setting the sliders snap to, and the page labels it as precomputed.
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import batterypilot as bp  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("batterypilot")
EXAMPLES = ["ERCOT_HB_HOUSTON_2023", "ERCOT_HB_WEST_2024", "CAISO_TH_SP15_2025"]
NAMES = {"ERCOT_HB_HOUSTON_2023": ("ERCOT", "HB_HOUSTON", "Houston hub", 2023),
         "ERCOT_HB_WEST_2024": ("ERCOT", "HB_WEST", "West hub", 2024),
         "CAISO_TH_SP15_2025": ("CAISO", "TH_SP15_GEN-APND", "SP15 trading hub", 2025)}
DURS = [1, 2, 4]
ETAS = [0.75, 0.80, 0.85, 0.90, 0.95]          # round-trip efficiency grid (as built: 0.85)
COSTS = [0.0, 65.0, 130.0, 195.0, 260.0]       # replacement cost, $/kWh nameplate (as built: 130)
FLOAT = ("rt", "da", "df", "T")
INT = ("hour", "day", "dow", "doy")


def load(name):
    cols = {}
    with open(os.path.join(MODEL, "examples", f"{name}.csv")) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    p = {k: np.array(cols[k], float) for k in FLOAT}
    for k in INT:
        p[k] = np.array(cols[k], int)
    p["time_utc"] = cols["time_utc"]
    return p


def months(p, year):
    """Index of the first hour of each local calendar month of `year` (12 starts, then the end), from the local day
    number. The year is in UTC, so its first hours are still 31 Dec of the year before in local time; they are left out."""
    base = np.datetime64("2000-01-01")
    mon = (base + p["day"].astype("timedelta64[D]")).astype("datetime64[M]")
    starts = [int(np.argmax(mon == np.datetime64(f"{year}-{m:02d}"))) for m in range(1, 13)] + [len(mon)]
    assert all(b > a for a, b in zip(starts, starts[1:])), starts
    return starts


def battery(d, eta=0.85, cost=130.0):
    return bp.Battery(P=1.0, duration_h=float(d), eta_rt=eta, cost_per_kWh=cost)


def values(net, p, bat, cl):
    """Raw network outputs (z knots, asinh of the value curve over the price scale) and the value curve V for every hour."""
    n = len(p["rt"])
    ps = bp.rolling_scale(p["da"])
    X, ps = bp.build(p, np.arange(n), bat.d, cl, ps=ps)
    if net.hybrid:      # v1.3: the day-ahead backbone curve, as run_policy appends it
        lam = bp.value_curves(p["da"], p["hour"], p["day"], cl, bat.d, P=bat.P, eta_rt=bat.eta_rt)
        X = np.concatenate([X, np.arcsinh(lam / ps[:, None])], 1).astype(np.float32)
    return net.z_knots(X), net.values(X, ps)


def window_run(bat, p, V, cl, a, b):
    """BatteryPilot over hours [a, b), starting empty: the offer rule against the real RT price."""
    ch, dis, soc = bp.offer_policy_run(bat, p["rt"][a:b], V[a:b], cl[a:b], 0.0)
    return ch, dis, soc


def main():
    t0 = time.time()
    W = dict(np.load(os.path.join(MODEL, "batterypilot_weights.npz")))
    W16 = roundtrip(W, "f16")
    W16.update({k: W[k] for k in ("hybrid", "min_step") if k in W})    # flags, not weights: kept exact (meta)
    net, net16 = bp.PolicyNet(W), bp.PolicyNet(W16)
    paths = [load(n) for n in EXAMPLES]
    tensors, meta = {}, []
    rng = np.random.default_rng(20261114)
    cases = []
    worst_f16 = 0.0
    for i, (name, p) in enumerate(zip(EXAMPLES, paths)):
        for k in FLOAT:
            assert np.isfinite(p[k]).all(), (name, k)
            tensors[f"x{i}_{k}"] = p[k].astype(np.float32)
        for k in INT:
            tensors[f"x{i}_{k}"] = p[k].astype(np.int32)
        # store exactly what the browser reads (f32), and run Python on the same values
        for k in FLOAT:
            p[k] = p[k].astype(np.float32).astype(np.float64)
        mkt, node, label, year = NAMES[name]
        ms = months(p, year)
        # perfect hindsight per month for every (duration, efficiency, wear cost) on the slider grid; spread rule and
        # BatteryPilot at the as-built setting (the browser recomputes both live; these are for the page's notes)
        pf = np.zeros((len(DURS), len(ETAS), len(COSTS), 12))
        annual = {}
        for a_, d in enumerate(DURS):
            for b_, eta in enumerate(ETAS):
                for c_, cost in enumerate(COSTS):
                    bat = battery(d, eta, cost)
                    cl = np.broadcast_to(bat.c_lin(p["T"]), (len(p["rt"]),))
                    for m in range(12):
                        a, b = ms[m], ms[m + 1]
                        ch, dis = bp.perfect_foresight(p["rt"][a:b], bat, cl[a:b])
                        pf[a_, b_, c_, m] = bp.settle(bat, p["rt"][a:b], ch, dis, 0.0, p["T"][a:b])["net"]
            bat = battery(d)
            cl = bat.c_lin(p["T"])
            out = {}
            ch, dis, _, V = bp.run_policy(net, p, bat, cl)
            out["bp"] = bp.settle(bat, p["rt"], ch, dis, 0.0, p["T"])
            ch, dis = bp.perfect_foresight(p["rt"], bat, cl)
            out["pf"] = bp.settle(bat, p["rt"], ch, dis, 0.0, p["T"])
            ch, dis = bp.rolling_da_lp(p["da"], p["hour"], p["day"], bat, cl)
            out["rolling"] = bp.settle(bat, p["rt"], ch, dis, 0.0, p["T"])
            ch, dis = bp.daily_spread(p["da"], p["day"], bat, cl)
            out["spread"] = bp.settle(bat, p["rt"], ch, dis, 0.0, p["T"])
            annual[str(d)] = {s: round(r["net"] / 1000 * 8760 / len(p["rt"]), 2) for s, r in out.items()}
            annual[str(d)]["bp_cycles"] = round(out["bp"]["cycles"], 1)
            # how much float16 weights move the value curve (full network vs the browser's)
            z32, _ = values(net, p, bat, cl)
            z16, _ = values(net16, p, bat, cl)
            worst_f16 = max(worst_f16, float(np.abs(z32 - z16).max()))
        tensors[f"pf{i}"] = pf.astype(np.float32)
        imax = int(np.argmax(p["rt"]))
        meta.append(dict(id=name, market=mkt, node=node, label=label, year=year, n=len(p["rt"]),
                         t0=p["time_utc"][0], tz="America/Chicago" if mkt == "ERCOT" else "America/Los_Angeles",
                         month_start=ms, annual=annual, peak=dict(i=imax, rt=float(p["rt"][imax]), utc=p["time_utc"][imax])))
        # parity cases (float16 weights as the browser has them): raw z knots at 16 seeded hours + the peak hour, and
        # the state of charge over the peak's month; as built, and perturbed variants that change the inputs
        hours = sorted(set(rng.choice(np.arange(24 * 8, len(p["rt"])), 16, replace=False).tolist() + [imax]))
        m_peak = int(np.searchsorted(ms, imax, side="right") - 1)
        for variant, d, eta, cost in (("as built, 2 h", 2, 0.85, 130.0), ("1 h", 1, 0.85, 130.0),
                                      ("4 h, 75 % round trip, wear $260/kWh", 4, 0.75, 260.0)):
            bat = battery(d, eta, cost)
            cl = bat.c_lin(p["T"])
            z, V = values(net16, p, bat, cl)
            a, b = ms[m_peak], ms[m_peak + 1]
            _, _, soc = window_run(bat, p, V, cl, a, b)
            cases.append(dict(example=i, variant=variant, d=d, eta=eta, cost=cost, hours=hours, window=[a, b],
                              outputs=np.r_[z[hours].ravel(), soc].tolist()))
        print(f"{name}: {len(p['rt'])} h, annual {annual}  ({time.time() - t0:.0f} s)", flush=True)

    n_params = int(sum(v.size for k, v in W.items() if k[0] in "Wb"))
    wmeta = dict(n_params=n_params, knots=bp.SOC_KNOTS.tolist(), H_DA=bp.H_DA, H_RT=bp.H_RT, H_DF=bp.H_DF,
                 publish_hour=bp.PUBLISH_HOUR, n_feat=bp.N_FEAT, layers=sum(1 for k in W if k.startswith("W")),
                 f16_max_abs_z_diff=worst_f16, hybrid=bool(net.hybrid), min_step=float(net.min_step))
    write_pack(OUT / "weights", {k: W[k] for k in ["x_mu", "x_sd"] + sorted(k for k in W if k[0] in "Wb")},
               dtype="f16", meta=wmeta)
    keep = {k: ("i32" if k.split("_")[-1] in INT else "f32") for k in tensors}
    write_pack(OUT / "examples", tensors, dtype="f32", meta=dict(examples=meta, durations=DURS, etas=ETAS, costs=COSTS),
               keep=keep)
    write_json(OUT / "reference.json", dict(cases=cases, tolerance=2e-3))
    size = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"wrote {OUT} ({size / 1e6:.2f} MB), {len(cases)} parity cases; f16 weights move z by at most {worst_f16:.1e}"
          f" ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
