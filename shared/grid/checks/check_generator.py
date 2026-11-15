"""Generated fleets and system states vs real ranges.  Writes grid/checks/generator.json

- inertia per MW of demand (s): ERCOT 2019 minimum 134.6 GW s / 29.4 GW = 4.6 s (Texas RE), GB 9 Aug 2019
  210 GVA s / 29 GW = 7.2 s and GB minimum-inertia policy 140 GVA s (FRCR 2021) at ~20-25 GW;
- generated heat rates per technology vs the EIA-923 operating heat rates of the real ERCOT fleet;
- frequency outcomes of 2,000 random events: share with UFLS, nadir distribution, RoCoF vs loss/inertia (must follow
  RoCoF = dP f0 / 2E exactly at t = 0).
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, sys
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np, pandas as pd
from gridsim.generator import generate_states, generate_fleet
from gridsim.frequency import frequency_event
from gridsim.fleet import prepare_fleet

out = {}
S, L, lab = generate_states(2000, seed=11)
out["n_states"] = len(S)
for sysn in ("ERCO", "GB"):
    m = lab.system == sysn
    idx = np.where(m)[0]
    r = frequency_event([S[i] for i in idx], [L[i] for i in idx], t_end=30, record=False)
    nad = np.array([x["nadir_Hz"] for x in r]); f0 = 60.0 if sysn == "ERCO" else 50.0
    roc = np.array([x["rocof_init_Hz_s"] for x in r])
    ana = -lab.loss_MW.values[idx] * f0 / (2 * lab.inertia_MWs.values[idx])
    out[sysn] = {"n": int(m.sum()),
                 "inertia_s_per_demand_p5_p50_p95": [float(np.percentile(lab.inertia_s_per_demand[m], q)) for q in (5, 50, 95)],
                 "inertia_GWs_p5_p50_p95": [float(np.percentile(lab.inertia_MWs[m] / 1e3, q)) for q in (5, 50, 95)],
                 "ibr_share_mean": float(lab.ibr_share[m].mean()),
                 "nadir_dev_Hz_p5_p50_p95": [float(np.percentile(nad - f0, q)) for q in (5, 50, 95)],
                 "share_with_ufls": float(np.mean([len(x["ufls"]) > 0 for x in r])),
                 "rocof_t0_vs_formula_max_rel_err": float(np.max(np.abs(roc - ana) / np.abs(ana)))}
out["reference"] = {"ERCOT_2019_min_inertia_s_per_demand": 134.6 / 29.426, "GB_2019_08_09_s_per_demand": 210 / 29.0,
                    "ERCOT_inertia_range_GWs": "~115-360 (Texas RE 2015-2025)", "GB_min_policy_GVAs": 140}
fl = prepare_fleet(f"{GRID_ROOT}/data/proc/fleet/fleet_ERCO.parquet", start="2019-01-01", end="2023-12-31")
hr = {}
gens = pd.concat([generate_fleet(seed=s) for s in range(40)])
for t in ("coal_steam", "gas_cc", "gas_ct", "gas_steam"):
    g = [c[-1][1] / 1.0 for c in gens[gens.tech == t].heat_rate_curve if c]
    real = fl[fl.tech == t].hr_avg_eia923.dropna()
    hr[t] = {"generated_fullload_p10_p50_p90": [float(np.percentile(g, q)) for q in (10, 50, 90)],
             "ercot_eia923_operating_p10_p50_p90": [float(np.percentile(real, q)) for q in (10, 50, 90)] if len(real) else None}
out["heat_rates"] = hr
json.dump(out, open(f"{GRID_CODE}/checks/generator.json", "w"), indent=1)
print(json.dumps(out, indent=1))
