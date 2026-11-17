"""Pick the release demo days: one exam run per source (fixed seed 7, uniform within the source), saved as CSV with
the contract columns FaultFinder reads, plus the dataset label in examples/labels.json."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK  # noqa: E402  (ML Ratchet paths; puts the building platform on sys.path)
import buildingdata as bd

COLS = ['time_local', 'T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_sa_sp_cool', 'T_sa_sp_heat', 'T_zone', 'damper_oa',
        'valve_cc', 'valve_hc', 'fan_sa_speed', 'fan_sa_power_kW', 'x_fan_sa_current_A', 'fan_sa_status', 'occ',
        'comp1_status', 'comp2_status']
GROUPS = {'iowa_ers': ['ers_rp1312', 'lbnl_mzvav2_ers_exp'], 'flexlab': ['lbnl_szcav_flexlab', 'lbnl_szvav_flexlab'],
          'ornl_rtu': ['lbnl_rtu_ornl'], 'nesbitt': ['nesbitt']}
out = os.path.join(WORK, 'release', 'examples')
os.makedirs(out, exist_ok=True)
r = bd.faults(split='exam')
rng = np.random.default_rng(7)
labels = {}
for name, dss in GROUPS.items():
    g = r[r.dataset.isin(dss)].sort_values('run_id')
    row = g.iloc[int(rng.integers(len(g)))]
    d = bd.fault_data(row.dataset)
    d = d[d.run_id == row.run_id]
    d = d[[c for c in COLS if c in d and d[c].notna().any()]]
    fn = f'{name}_{row.run_id}.csv'
    d.to_csv(os.path.join(out, fn), index=False, float_format='%.4g')
    labels[fn] = dict(run_id=row.run_id, dataset=row.dataset, facility=row.facility, unit=row.unit, date=str(row.date)[:10],
                      fault_type=row.fault_type, severity=None if str(row.severity) == 'nan' else str(row.severity))
    print(fn, row.fault_type, len(d))
json.dump(labels, open(os.path.join(out, 'labels.json'), 'w'), indent=1)
