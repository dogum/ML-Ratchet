"""v1.1 demo: the reference days of the four demo days (examples/labels.json). For each, the same dataset and unit's
exam days with an earlier date within 7 days, the most recent 3, chosen by date only (their labels are not read or
saved). Written as CSV like examples.py into examples/references/, with examples/references.json listing them.
Run from training/:  python examples_refs.py"""
import os
import sys
import json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import MODEL  # noqa: E402  (puts the building platform on sys.path)
import buildingdata as bd  # noqa: E402

COLS = ['time_local', 'T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_sa_sp_cool', 'T_sa_sp_heat', 'T_zone', 'damper_oa',
        'valve_cc', 'valve_hc', 'fan_sa_speed', 'fan_sa_power_kW', 'x_fan_sa_current_A', 'fan_sa_status', 'occ',
        'comp1_status', 'comp2_status']          # as examples.py
out = os.path.join(MODEL, 'examples', 'references')
os.makedirs(out, exist_ok=True)
labels = json.load(open(os.path.join(MODEL, 'examples', 'labels.json')))
r = bd.faults(split='exam')
r['date'] = pd.to_datetime(r.date)
index = {}
for fn, m in sorted(labels.items()):
    day = pd.Timestamp(m['date'])
    g = r[(r.dataset == m['dataset']) & (r.unit == m['unit']) & (r.date < day) &
          (r.date >= day - pd.Timedelta(days=7))].sort_values(['date', 'run_id'])
    files = []
    for _, row in g.tail(3).iterrows():
        d = bd.fault_data(row.dataset)
        d = d[d.run_id == row.run_id]
        d = d[[c for c in COLS if c in d and d[c].notna().any()]]
        f = f'{row.run_id}.csv'
        d.to_csv(os.path.join(out, f), index=False, float_format='%.4g')
        files.append(f'references/{f}')
    index[fn] = files
    print(fn, files)
json.dump(index, open(os.path.join(MODEL, 'examples', 'references.json'), 'w'), indent=1)
