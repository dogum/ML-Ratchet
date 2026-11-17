"""FaultFinder on four real exam days, one per source (drawn with a fixed seed, never used for training or choosing).
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
for f in sorted(glob.glob(os.path.join(HERE, 'examples', '*.csv'))):
    m = labels[os.path.basename(f)]
    r = ff.read_day(pd.read_csv(f))
    truth = 'no fault' if m['fault_type'] == 'none' else m['fault_type'].replace('_', ' ') + (f" ({m['severity']})" if m['severity'] else '')
    print(f"\n{m['facility']}, {m['unit']}, {m['date']}")
    print(f"  dataset label: {truth}")
    print('  ' + report(r).replace('\n', '\n  '))
