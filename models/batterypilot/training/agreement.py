"""Does the frozen simulated set rank policies the way the real dev years do?  The test of a simulator fix.

    python agreement.py <out_name> <frozen_a>[,<frozen_b>...] <policy>[,<policy>...]
        -> training/checks/<out_name>.json

For each frozen set: the policies' frozen captures (WORK/data/scores/<policy>__<frozen>.json) against their dev
captures (real prices, the same for every frozen set): Spearman rank correlation, the share of policy pairs ordered
the same way (concordant pairs), and the best policy by each.  Dev never changes between frozen sets, so a better
simulator shows as a higher agreement.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

import _paths

ROOT = _paths.WORK


def main(name, frozens, policies):
    out = dict(policies=policies, sets={})
    for fz in frozens:
        rows = {}
        for p in policies:
            fn = f'{ROOT}/data/scores/{p}__{fz}.json'
            if os.path.exists(fn):
                s = json.load(open(fn))
                rows[p] = (s['frozen_capture'], s['dev_capture'])
        ps = list(rows)
        f = np.array([rows[p][0] for p in ps]); d = np.array([rows[p][1] for p in ps])
        pairs = list(itertools.combinations(range(len(ps)), 2))
        conc = [np.sign(f[i] - f[j]) == np.sign(d[i] - d[j]) for i, j in pairs]
        out['sets'][fz] = dict(
            n=len(ps), frozen={p: rows[p][0] for p in ps}, dev={p: rows[p][1] for p in ps},
            spearman=round(float(stats.spearmanr(f, d).correlation), 3) if len(ps) > 2 else None,
            concordant_pairs=f'{int(sum(conc))} of {len(pairs)}',
            concordant_share=round(float(np.mean(conc)), 3) if pairs else None,
            best_frozen=ps[int(np.argmax(f))], best_dev=ps[int(np.argmax(d))],
            frozen_rank=[ps[i] for i in np.argsort(-f)], dev_rank=[ps[i] for i in np.argsort(-d)])
        print(fz, json.dumps({k: v for k, v in out['sets'][fz].items() if k not in ('frozen', 'dev')}))
    os.makedirs(os.path.join(_paths.SRC, 'checks'), exist_ok=True)
    json.dump(out, open(os.path.join(_paths.SRC, 'checks', f'{name}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(','))
