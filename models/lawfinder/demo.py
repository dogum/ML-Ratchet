"""LawFinder demo: every released law on a fixed random draw of REAL exam examples (5 per law, drawn by seed within
strata, never picked by result), beside the truth and the textbook rival tuned on the same dev data.
Runs in about a second:  python demo.py
"""
import csv, os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawfinder import LawFinder

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(name):
    with open(os.path.join(HERE, 'examples', f'{name}.csv')) as f:
        return list(csv.DictReader(f))


def main():
    lf = LawFinder()
    for name in lf.names():
        law = lf[name]
        d = law.d
        print(f"\n== {d['title']} ({d['platform']})   {d['target']['symbol']} = {d['text']}")
        print(f"   released rung {d['rung']}; 68% bar +-{law.sigma:.3g}; textbook: {d['textbook']['label']}")
        hdr = '   ' + '  '.join(f'{v:>8s}' for v in law.vars) + f"  {'truth':>8s} {'law':>8s} {'textbook':>8s}  id"
        print(hdr)
        for r in rows(name):
            kw = {v: float(r[v]) for v in law.vars}
            mu, sig = law.predict(**kw)
            tb = float(r['textbook'])
            print('   ' + '  '.join(f'{kw[v]:8.4g}' for v in law.vars) +
                  f"  {float(r['truth']):8.3g} {float(mu):8.3g} {tb:8.3g}  {r['id']}")


if __name__ == '__main__':
    main()
