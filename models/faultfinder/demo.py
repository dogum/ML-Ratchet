"""FaultFinder on four real exam days, one per source (drawn with a fixed seed, never used for training or choosing).
Two of them have earlier days of the same unit within 7 days (examples/references/; their labels are not used): they
are read once alone (single-day mode, v1.0's log-odds) and once compared with those days (v1.1).
Runs in a few seconds:  python demo.py"""
import os
import sys
import glob
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from faultfinder import FaultFinder  # noqa: E402
from pipeline import report  # noqa: E402

ff = FaultFinder()
labels = json.load(open(os.path.join(HERE, 'examples', 'labels.json')))
references = json.load(open(os.path.join(HERE, 'examples', 'references.json')))
for f in sorted(glob.glob(os.path.join(HERE, 'examples', '*.csv'))):
    m = labels[os.path.basename(f)]
    df = pd.read_csv(f)
    r = ff.read_day(df)
    truth = 'no fault' if m['fault_type'] == 'none' else m['fault_type'].replace('_', ' ') + (f" ({m['severity']})" if m['severity'] else '')
    print(f"\n{m['facility']}, {m['unit']}, {m['date']}")
    print(f"  dataset label: {truth}")
    print('  ' + report(r).replace('\n', '\n  '))
    refs = references.get(os.path.basename(f), [])
    if refs:
        rr = ff.read_day(df, ref_days=[pd.read_csv(os.path.join(HERE, 'examples', g)) for g in refs])
        print(f"  with the unit's earlier days ({', '.join(os.path.basename(g)[:-4] for g in refs)}):")
        print('  ' + report(rr).replace('\n', '\n  '))
