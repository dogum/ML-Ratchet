"""Read one day of an air handler's trend log (CSV or parquet with contract column names and a 'time_local' column)
and print FaultFinder's verdict.  Usage: python pipeline.py trend.csv [YYYY-MM-DD] [--ref earlier_day.csv ...]
(--ref: the same unit's earlier days, oldest first, at most 7 days before; the last 3 are used)"""
import sys
import pandas as pd
from faultfinder import FaultFinder


def read(path):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    return df


def report(r, top=3):
    n = r.get('n_reference_days', 0)
    ref = f", compared with {n} earlier day{'s' if n > 1 else ''} of the unit" if n else ', single-day mode'
    lines = [f"P(fault) = {r['p_fault']:.2f}  (calibrated on 20 real dev days{ref})"]
    if r['p_fault'] > 0.5:
        lines.append('Most likely faults (given a fault):')
    else:
        lines.append('If it is faulty, the most likely faults would be:')
    for c, lab, p in r['diagnosis'][:top]:
        lines.append(f'  {p:5.2f}  {lab}')
    return '\n'.join(lines)


if __name__ == '__main__':
    ff = FaultFinder()
    a = sys.argv[1:]
    refs = [read(f) for f in a[a.index('--ref') + 1:]] if '--ref' in a else None
    a = a[:a.index('--ref')] if '--ref' in a else a
    print(report(ff.read_day(read(a[0]), a[1] if len(a) > 1 else None, ref_days=refs)))
