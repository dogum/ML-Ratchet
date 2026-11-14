"""Speed of each gridsim component, single thread (OMP_NUM_THREADS=1), on the shared 2-core Xeon 2.1 GHz.
Writes grid/checks/speed.json.  Year-long dispatch timings are taken from ercot_dispatch_<year>.json (same machine,
measured while other jobs shared the CPU, so they are upper bounds).
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, os, sys, time, glob
sys.path.insert(0, f"{GRID_CODE}/gridsim"); sys.path.insert(0, f"{GRID_CODE}/checks")
import numpy as np, pandas as pd


def tm(f, n=3):
    f()
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n


out = {"machine": "Intel Xeon 2.1 GHz, 1 thread, shared with other jobs (load average ~3 on 2 cores)"}
from gridsim.network import Network, ac_powerflow, dc_powerflow, dc_opf
N = Network.from_matpower(f"{GRID_ROOT}/gridsim/refdata/matpower/case_ACTIVSg2000.m")
out["powerflow_ACTIVSg2000"] = {"ac_nr_s": tm(lambda: ac_powerflow(N)), "dc_s": tm(lambda: dc_powerflow(N)),
                                "ptdf_build_s": tm(lambda: Network.from_matpower(
                                    f"{GRID_ROOT}/gridsim/refdata/matpower/case_ACTIVSg2000.m").ptdf(), 1),
                                "dc_opf_s": tm(lambda: dc_opf(N), 1)}
from gridsim.frequency import frequency_event
from gridsim.generator import generate_states
S, L, lab = generate_states(200, seed=3)
out["frequency"] = {"single_60s_event_dt0.02_h0.01_s": tm(lambda: frequency_event(S[0], L[0], t_end=60), 3)}
g60 = [i for i, s in enumerate(S) if s["f0"] == 60]
t = time.perf_counter(); frequency_event([S[i] for i in g60], [L[i] for i in g60], t_end=60, record=False)
out["frequency"]["batch_events"] = len(g60)
out["frequency"]["batch_60s_s_per_event"] = (time.perf_counter() - t) / len(g60)
t = time.perf_counter(); generate_states(100, seed=4)
out["generator_states_s_per_state"] = (time.perf_counter() - t) / 100
from check_ercot_dispatch import real_inputs, FLEET
from gridsim.dispatch import dispatch
y = real_inputs(2019, "2019-07-01", "2019-07-07 23:00")
for m in ("merit", "lp", "milp"):
    t = time.perf_counter()
    r = dispatch(FLEET, y.d, wind_MW=y.ng_WND, solar_MW=y.ng_SUN, imports_MW=-y.ti, hydro_MW=y.ng_WAT,
                 fixed_MW={"nuclear": y.ng_NUC}, method=m, must_run=("coal_steam",))
    out[f"dispatch_ercot_week_{m}_s"] = time.perf_counter() - t
    if m != "merit":
        out[f"dispatch_ercot_{m}_solve_s_per_day_median"] = float(np.median([i["solve_s"] for i in r.solver_stats]))
years = {}
for p in glob.glob(f"{GRID_CODE}/checks/ercot_dispatch_20??.json"):
    d = json.load(open(p))
    years[d["year"]] = {k: round(v["wall_s"], 1) for k, v in d["configs"].items()}
out["dispatch_ercot_year_wall_s"] = years
json.dump(out, open(f"{GRID_CODE}/checks/speed.json", "w"), indent=1)
print(json.dumps(out, indent=1))
