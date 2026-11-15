"""Power-flow and DC-OPF check: gridsim vs PYPOWER (MATPOWER port) and pandapower on IEEE and TAMU ACTIVSg cases.

Run with the reference venv (pandapower/PYPOWER pin pandas 2.x):
    <refenv python> shared/grid/checks/check_powerflow.py   (or run_all.py --ref <refenv python>)
Writes grid/checks/powerflow.json
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np
from pypower.api import runpf, rundcpf, rundcopf, ppoption
from gridsim.network import load_matpower, Network, ac_powerflow, dc_powerflow, dc_opf

REF = f"{GRID_ROOT}/gridsim/refdata/matpower/"
CASES = ["case9", "case14", "case30", "case57", "case118", "case300", "case_ACTIVSg200", "case_ACTIVSg2000"]
OPT = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10)


def to_pyp(ppc):
    p = {k: ppc[k].copy() for k in ["bus", "gen", "branch", "gencost"]}
    p["baseMVA"], p["version"] = ppc["baseMVA"], "2"
    p["branch"], p["bus"], p["gen"] = p["branch"][:, :13], p["bus"][:, :13], p["gen"][:, :21]
    return p


def timeit(f, n=5):
    t = time.perf_counter()
    for _ in range(n):
        out = f()
    return out, (time.perf_counter() - t) / n


out = {"what": "gridsim power flow vs PYPOWER 5.1 (independent Python port of MATPOWER) and pandapower 3.5",
       "reference_env": ".refenv (pandas 2.3, numpy 2.4)", "cases": {}}
for case in CASES:
    ppc = load_matpower(REF + case + ".m")
    N = Network.from_ppc(ppc)
    rec = {"n_bus": int(N.nb), "n_branch": int(N.nl), "n_gen": int(N.gen.shape[0])}
    # AC
    (res, ok), t_ref = timeit(lambda: runpf(to_pyp(ppc), OPT), 3)
    r, t_me = timeit(lambda: ac_powerflow(Network.from_ppc(ppc), tol=1e-10), 3)
    rec["ac"] = {"converged": bool(r["converged"] and ok), "iterations": r["iterations"],
                 "max_dVm_pu": float(np.abs(res["bus"][:, 7] - r["Vm"]).max()),
                 "max_dVa_deg": float(np.abs(res["bus"][:, 8] - r["Va_deg"]).max()),
                 "max_dPf_MW": float(np.abs(res["branch"][:, 13] - r["Pf_MW"]).max()),
                 "max_dQf_MVAr": float(np.abs(res["branch"][:, 14] - r["Qf_MVAr"]).max()),
                 "losses_MW": r["losses_MW"],
                 "losses_ref_MW": float(np.sum(res["branch"][:, 13] + res["branch"][:, 15])),
                 "t_gridsim_s": t_me, "t_pypower_s": t_ref}
    # DC
    (res, ok), t_ref = timeit(lambda: rundcpf(to_pyp(ppc), OPT), 3)
    r, t_me = timeit(lambda: dc_powerflow(Network.from_ppc(ppc)), 3)
    rec["dc"] = {"max_dVa_deg": float(np.abs(res["bus"][:, 8] - r["Va_deg"]).max()),
                 "max_dPf_MW": float(np.abs(res["branch"][:, 13] - r["Pf_MW"]).max()),
                 "t_gridsim_s": t_me, "t_pypower_s": t_ref}
    # DC vs AC (how wrong the DC approximation is, for information)
    ac = ac_powerflow(N)
    big = np.abs(ac["Pf_MW"]) > 1
    rec["dc_vs_ac"] = {"median_abs_flow_err_pct": float(np.median(np.abs(r["Pf_MW"][big] - ac["Pf_MW"][big]) /
                                                                np.abs(ac["Pf_MW"][big])) * 100),
                       "p95_abs_flow_err_MW": float(np.percentile(np.abs(r["Pf_MW"] - ac["Pf_MW"]), 95))}
    out["cases"][case] = rec
    print(case, json.dumps(rec)[:400], flush=True)

# pandapower on the IEEE cases (its from_ppc turns ACTIVSg transformers into impedances and drops taps, so not used there)
try:
    import pandapower as pp
    from pandapower.converter.pypower import from_ppc
    pp_res = {}
    np.seterr(all='warn')  # PYPOWER leaves numpy in raise mode
    for case in ["case9", "case14", "case30", "case57", "case118", "case300"]:
      try:
        ppc = load_matpower(REF + case + ".m")
        net = from_ppc(to_pyp(ppc), f_hz=60, validate_conversion=False)
        pp.runpp(net, tolerance_mva=1e-9, numba=False, calculate_voltage_angles=True)
        r = ac_powerflow(Network.from_ppc(ppc), tol=1e-10)
        dva = net.res_bus.va_degree.values - r["Va_deg"]
        pp_res[case] = {"max_dVm_pu": float(np.abs(net.res_bus.vm_pu.values - r["Vm"]).max()),
                        "max_dVa_deg": float(np.abs(dva - dva[0]).max()),
                        "losses_pp_MW": float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()),
                        "losses_gridsim_MW": r["losses_MW"]}
        pp.rundcpp(net, numba=False)
        d = dc_powerflow(Network.from_ppc(ppc))
        dva = net.res_bus.va_degree.values - d["Va_deg"]
        pp_res[case]["dc_max_dVa_deg"] = float(np.abs(dva - dva[0]).max())
        print("pandapower", case, pp_res[case], flush=True)
      except Exception as e:  # noqa
        import traceback; traceback.print_exc(); pp_res[case] = {"error": repr(e)}
    out["pandapower"] = pp_res
except Exception as e:  # noqa
    out["pandapower"] = {"error": repr(e)}

# DC-OPF with congestion: tighten the 20 most-loaded meshed lines to 90 % of their unconstrained flow
opf = {}
for case in ["case_ACTIVSg200", "case_ACTIVSg2000"]:
    ppc = load_matpower(REF + case + ".m")
    N = Network.from_ppc(ppc)
    base = dc_opf(N)
    # meshed = not a bridge: removing it keeps the graph connected (test by B matrix rank cheaply via degree>1 on both ends)
    deg = np.bincount(np.r_[N.branch[:, 0], N.branch[:, 1]].astype(int), minlength=N.nb)
    meshed = (deg[N.branch[:, 0].astype(int)] > 1) & (deg[N.branch[:, 1].astype(int)] > 1)
    H = N.ptdf()
    on = N.gen[:, 7] > 0
    gb = N.gen[on, 0].astype(int)
    sens = np.abs(H[:, gb]).max(axis=1)  # can generation move this flow?
    cand = np.where(meshed & (sens > 0.05) & (np.abs(base["Pf_MW"]) > 20))[0]
    order = cand[np.argsort(-np.abs(base["Pf_MW"][cand]) / N.rate_MW[cand])][:60]
    ppc2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ppc.items()}
    br = ppc2["branch"]
    pick = []
    for l in order:  # greedy: tighten a line to 85 % of its unconstrained flow if the OPF stays feasible
        keep = br[l, 5]
        br[l, 5] = 0.85 * abs(base["Pf_MW"][l])
        if dc_opf(Network.from_ppc(ppc2))["status"] == "Optimal":
            pick.append(l)
        else:
            br[l, 5] = keep
        if len(pick) >= 8:
            break
    pick = np.array(pick, int)
    N2 = Network.from_ppc(ppc2)
    r, t_me = timeit(lambda: dc_opf(Network.from_ppc(ppc2)), 2)
    p = to_pyp(ppc2)
    p["branch"][:, 11], p["branch"][:, 12] = -360, 360
    res, t_ref = timeit(lambda: rundcopf(p, OPT), 1)
    on = ppc["gen"][:, 7] > 0
    lam_ref = res["bus"][:, 13]
    opf[case] = {"status": r["status"], "ref_success": bool(res["success"]), "n_tightened": int(len(pick)),
                 "n_binding_gridsim": int(len(r["binding"])),
                 "cost_gridsim": r["cost"], "cost_pypower": float(res["f"]),
                 "cost_rel_diff": float(abs(r["cost"] - res["f"]) / res["f"]),
                 "max_dPg_MW": float(np.abs(res["gen"][on, 1] - r["Pg_MW"][on]).max()),
                 "max_dLMP_per_MWh": float(np.abs(lam_ref - r["lmp"]).max()),
                 "lmp_range_gridsim": [float(r["lmp"].min()), float(r["lmp"].max())],
                 "lmp_range_pypower": [float(lam_ref.min()), float(lam_ref.max())],
                 "t_gridsim_s": t_me, "t_pypower_s": t_ref}
    print("opf", case, opf[case], flush=True)
out["dc_opf_congested"] = opf
json.dump(out, open(f"{GRID_CODE}/checks/powerflow.json", "w"), indent=1)
print("wrote powerflow.json")
