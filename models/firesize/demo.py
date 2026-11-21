"""FireSize demo: eight real exam ignitions (a seeded draw, two per size group), features precomputed in
examples/demo_examples.csv exactly as in the exam. Runs in well under a second with numpy alone.
    python demo.py"""
import csv
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import firesize  # noqa: E402

m = firesize.FireSize()
with open(os.path.join(here, 'examples', 'demo_examples.csv'), newline='', encoding='utf-8') as fh:
    rows = list(csv.DictReader(fh))
d = {f: np.array([float(r[f]) for r in rows]) for f in m.features}
out = m.predict(d)
print(f"{'fire':18s} {'year':>4s} {'st':>3s} {'real acres':>11s} {'P(escape)':>9s} {'median ac':>10s} {'80% range (acres)':>22s}")
for i, r in enumerate(rows):
    lo, hi = out['interval80_acres'][i]
    print(f"{r['fire_name'][:18]:18s} {int(r['year']):4d} {r['state']:>3s} {float(r['real_acres']):11.2f} "
          f"{out['p_escape'][i]:9.3f} {out['median_acres'][i]:10.2f} {lo:10.2f} - {hi:9.1f}")
exam_p = np.array([float(r['exam_p_escape']) for r in rows])
assert np.allclose(out['p_escape'], exam_p, atol=1e-4), 'differs from the exam run'
print('\nP(escape) matches the exam run for all rows.')
# the exam's coverage test: is the real size class between the classes of the 10 % and 90 % points?
real = np.searchsorted(firesize.EDGES_AC, [float(r['real_acres']) for r in rows], side='right')
cdf = np.cumsum(out['probs'], 1)
inside = (real >= np.argmax(cdf >= 0.1, 1)) & (real <= np.argmax(cdf >= 0.9, 1))
print(f'Real size class inside the 80 % range: {int(inside.sum())} of {len(rows)} (exam: 90 % of 655,869).')
# the full distribution of the first example
print('class probabilities, first example:',
      ', '.join(f'{k} {v:.3f}' for k, v in zip(firesize.CLASS_LABELS, out['probs'][0])))
