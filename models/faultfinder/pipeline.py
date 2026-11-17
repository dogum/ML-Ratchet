"""Read one day of an air handler's trend log (CSV or parquet with contract column names and a 'time_local' column)
and print FaultFinder's verdict.  Usage: python pipeline.py trend.csv [YYYY-MM-DD]"""
import sys
import pandas as pd
from faultfinder import FaultFinder


def read(path):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    return df


def report(r, top=3):
    lines = [f"P(fault) = {r['p_fault']:.2f}  (calibrated on 20 real dev days)"]
    if r['p_fault'] > 0.5:
        lines.append('Most likely faults (given a fault):')
    else:
        lines.append('If it is faulty, the most likely faults would be:')
    for c, lab, p in r['diagnosis'][:top]:
        lines.append(f'  {p:5.2f}  {lab}')
    return '\n'.join(lines)


if __name__ == '__main__':
    ff = FaultFinder()
    df = read(sys.argv[1])
    print(report(ff.read_day(df, sys.argv[2] if len(sys.argv) > 2 else None)))
