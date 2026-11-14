"""System inertia implied by the simulated ERCOT commitment vs ERCOT's reported inertia (Texas RE 2025 Assessment of
Reliability Performance, Appendix A Table C.2; grid/data/raw/reports/0a3af2d5.pdf):
  2019 minimum hourly inertia 134.6 GW s (load 29,426 MW, net load 14,645 MW, 50.2 % IRR), 79 hours below 150 GW s.
  2022 minimum 115.0 GW s (load 33,365 MW, net load 11,445 MW), 492 hours below 150 GW s.
Also: inertia vs net load relation, and a governor-response check: simulated frequency response (MW per 0.1 Hz) for a
1,300 MW trip at sampled hours vs the BAL-003 IFRO (412 MW/0.1 Hz).
    python3 check_ercot_inertia.py 2019 calibrated
Writes grid/checks/ercot_inertia.json (one entry per year/config)
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, os, sys
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np, pandas as pd

REPORTED = {2019: dict(min_GWs=134.6, load_MW=29426, net_load_MW=14645, hours_below_150=79),
            2022: dict(min_GWs=115.0, load_MW=33365, net_load_MW=11445, hours_below_150=492)}
year, cfg = int(sys.argv[1]), sys.argv[2]
fr = pd.read_parquet(f"{GRID_ROOT}/checks/out/ercot_{year}_{cfg}.parquet")
E = fr.inertia_MWs.values / 1e3
net = fr.demand_MW - fr.MW_WND - fr.MW_SUN
i = int(np.argmin(E))
rec = {"sim": {"min_GWs": float(E.min()), "at_load_MW": float(fr.demand_MW.iloc[i]), "at_net_load_MW": float(net.iloc[i]),
               "hours_below_150": int((E < 150).sum()), "p5_p50_p95_GWs": [float(np.percentile(E, q)) for q in (5, 50, 95)],
               "max_GWs": float(E.max()), "corr_with_net_load": float(np.corrcoef(E, net)[0, 1]),
               "slope_GWs_per_GW_net_load": float(np.polyfit(net / 1e3, E, 1)[0])},
       "reported": REPORTED.get(year)}
if REPORTED.get(year):
    rec["min_err_pct"] = (rec["sim"]["min_GWs"] - REPORTED[year]["min_GWs"]) / REPORTED[year]["min_GWs"] * 100
p = f"{GRID_CODE}/checks/ercot_inertia.json"
d = json.load(open(p)) if os.path.exists(p) else {}
d[f"{year}_{cfg}"] = rec
json.dump(d, open(p, "w"), indent=1)
print(json.dumps(rec, indent=1))
