"""Network-constrained dispatch (dispatch(..., network=...)) vs the single-period DC-OPF (itself checked against
PYPOWER in powerflow.json), on ACTIVSg200 and ACTIVSg2000 with the same congestion (lines tightened greedily to 85 % of
their unconstrained flow while feasible).  Every unit is committed (must-run) and costs are linear (c1 + c2 pmax), so
the two problems are identical for one hour: costs, dispatch and LMPs must match.
Then a 24-hour MILP run on ACTIVSg2000 with a summer-day load shape (unit commitment + line limits each hour).
Writes grid/checks/network_dispatch.json
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, sys, time
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np, pandas as pd
from gridsim.network import load_matpower, Network, dc_opf
from gridsim.dispatch import dispatch, fleet_from_case

REF = f"{GRID_ROOT}/gridsim/refdata/matpower/"
out = {}
for case in ["case_ACTIVSg200", "case_ACTIVSg2000"]:
    ppc = load_matpower(REF + case + ".m")
    N = Network.from_ppc(ppc)
    base = dc_opf(N)
    deg = np.bincount(np.r_[N.branch[:, 0], N.branch[:, 1]].astype(int), minlength=N.nb)
    meshed = (deg[N.branch[:, 0].astype(int)] > 1) & (deg[N.branch[:, 1].astype(int)] > 1)
    H = N.ptdf(); gb = N.gen[N.gen[:, 7] > 0, 0].astype(int)
    cand = np.where(meshed & (np.abs(H[:, gb]).max(1) > 0.05) & (np.abs(base["Pf_MW"]) > 20))[0]
    order = cand[np.argsort(-np.abs(base["Pf_MW"][cand]) / N.rate_MW[cand])][:60]
    br = ppc["branch"]
    pick = []
    for l in order:
        keep = br[l, 5]; br[l, 5] = 0.85 * abs(base["Pf_MW"][l])
        if dc_opf(Network.from_ppc(ppc))["status"] == "Optimal":
            pick.append(l)
        else:
            br[l, 5] = keep
        if len(pick) >= 8:
            break
    N2 = Network.from_ppc(ppc)
    fleet, nwk = fleet_from_case(N2)
    fleet["tech"] = "gas_ct"                                   # every generator is its own dispatchable unit
    cost = np.zeros((N2.gen.shape[0], 3))
    on = N2.gen[:, 7] > 0
    for g in np.where(on)[0]:
        gc = N2.gencost[g]; n = int(gc[3]); coef = gc[4:4 + n]
        c1 = coef[-2] if n >= 2 else 0; c2 = coef[-3] if n >= 3 else 0
        cost[g, 1] = c1 + c2 * N2.gen[g, 8]
    ref = dc_opf(N2, cost=cost)
    t = pd.date_range("2020-01-01", periods=1, freq="h", tz="UTC")
    Pd = N2.bus[:, 2].sum() + N2.bus[:, 4].sum()
    t0 = time.time()
    r = dispatch(fleet, pd.Series([Pd], index=t), network=nwk, method="lp", reserves=0.0, must_run=("gas_ct",),
                 hr_source="spec", prices="constant", outages="none", lookahead_h=0)
    wall = time.time() - t0
    lmp_err = np.abs(r.lmp[0] - ref["lmp"])
    out[case] = {"n_tightened": len(pick), "opf_cost": ref["cost"], "dispatch_cost_energy": float(
        (r.p[0] * np.array([fleet.set_index("id").loc[c["ids"][0], "vom_$_per_MWh"] for c in r.cf.clusters])).sum()),
        "opf_binding": int(len(ref["binding"])), "lines_monitored": int(len(r.solver_stats[0].get("n_monitored_lines", 0) and [1] or [])),
        "n_monitored_lines": r.solver_stats[0].get("n_monitored_lines"), "max_dLMP": float(lmp_err.max()),
        "lmp_range_dispatch": [float(r.lmp[0].min()), float(r.lmp[0].max())],
        "lmp_range_opf": [float(ref["lmp"].min()), float(ref["lmp"].max())],
        "max_overload_pct": float(np.max(np.abs(r.line_flow_MW[0]) / N2.rate_MW[r.monitored_lines] - 1) * 100), "wall_s": wall}
    print(case, out[case], flush=True)
    if case == "case_ACTIVSg2000":
        # 24 h MILP with a summer load shape (ERCOT-like: 0.62 at night to 1.0 at 17:00 local)
        shape = np.array([.66, .63, .62, .62, .63, .67, .72, .77, .81, .85, .89, .93, .96, .98, 1.0, 1.0, 1.0, .99,
                          .97, .94, .9, .84, .77, .7])
        times = pd.date_range("2020-08-01 05:00", periods=24, freq="h", tz="UTC")
        fleet2, nwk2 = fleet_from_case(N2)
        fleet2.loc[fleet2.tech.isin(["coal_steam", "nuclear"]), "min_up_h"] = 8.0
        fleet2.loc[fleet2.tech.isin(["coal_steam", "nuclear"]), "min_down_h"] = 8.0
        fleet2["start_cost_$"] = 20.0 * fleet2.pmax_MW
        wcap = fleet2[fleet2.tech == "wind"].pmax_MW.sum(); scap = fleet2[fleet2.tech == "solar_pv"].pmax_MW.sum()
        t0 = time.time()
        r2 = dispatch(fleet2, pd.Series(Pd * 0.95 * shape, index=times), wind_MW=pd.Series(0.35 * wcap, index=times),
                      solar_MW=pd.Series(scap * np.clip(np.sin((np.arange(24) - 1) / 13 * np.pi), 0, None), index=times),
                      network=nwk2, method="milp", hr_source="spec", prices="constant", outages="none", lookahead_h=0)
        out["ACTIVSg2000_24h_milp"] = {"wall_s": time.time() - t0,
                                       "lines_monitored": r2.solver_stats[0].get("n_monitored_lines"),
                                       "max_overload_pct": float(np.max(np.abs(r2.line_flow_MW) / N2.rate_MW[r2.monitored_lines] - 1) * 100),
                                       "lmp_spread_by_hour": [float(np.ptp(r2.lmp[h])) for h in range(24)],
                                       "price_by_hour": [float(p) for p in r2.price],
                                       "units_on_by_hour": [float(np.round(r2.u[h]).sum()) for h in range(24)],
                                       "max_balance_residual_MW": float(np.abs(r2.balance_residual_MW).max())}
        print(out["ACTIVSg2000_24h_milp"], flush=True)
json.dump(out, open(f"{GRID_CODE}/checks/network_dispatch.json", "w"), indent=1)
