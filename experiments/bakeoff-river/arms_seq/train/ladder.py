"""Collect every rung's harness evaluation for one arm (or all) and print the ladder with the paired bootstrap.
usage: python3 ladder.py [arm ...]   ->  prints, and writes <arm>/work/ladder.json
Sim val: per-run NSE on all 800 frozen runs (harness parquet); dev: per-basin NSE on the 52 dev basins.
Paired bootstrap of the difference in medians (10,000 resamples, seed 20261118), as the harness does.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SEQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def boot(a, b, n=10000, seed=20261118):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a = a[ok]; b = b[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = np.nanmedian(a[idx], 1) - np.nanmedian(b[idx], 1)
    return float(np.median(a) - np.median(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def rung_tables(arm):
    ev = os.path.join(SEQ, arm, 'work', 'eval')
    out = {}
    for f in sorted(glob.glob(os.path.join(ev, '*_summary.json'))):
        tag = os.path.basename(f)[:-len('_summary.json')]
        s = json.load(open(f))
        sv = pd.read_parquet(os.path.join(ev, f'{tag}_simval_ungauged.parquet'))
        dv = pd.read_parquet(os.path.join(ev, f'{tag}_dev_ungauged.parquet'))
        tr = os.path.join(SEQ, arm, 'work', f'train_{tag}.json')
        out[tag] = {'summary': s, 'sim': sv, 'dev': dv, 'train': json.load(open(tr)) if os.path.exists(tr) else None}
    return out


def row(tag, t):
    s, d = t['summary']['simval']['ungauged'], t['summary']['dev']['ungauged']
    fc = d['vs'].get('flowcast_v1', {}).get('paired', {})
    r = {'rung': tag, 'simval_nse': round(s['nse'], 4), 'simval_kge': round(s['kge'], 4),
         'simval_cov90': s.get('cov90'), 'dev_nse': round(d['nse'], 4), 'dev_kge': round(d['kge'], 4),
         'dev_fhv': round(d['fhv'], 1), 'dev_cov50': d.get('cov50'), 'dev_cov90': d.get('cov90'),
         'dev_crps_rel': d.get('crps_rel'), 'dev_by_class_nse': d.get('by_class_nse'),
         'train_cpu_h': t['train']['train_cpu_h'] if t['train'] else None,
         'n_params': t['train']['n_params'] if t['train'] else None}
    return r


def compare(t_new, t_inc):
    a = t_new['sim'].nse; b = t_inc['sim'].nse.reindex(a.index)
    c = t_new['dev'].nse; e = t_inc['dev'].nse.reindex(c.index)
    return {'sim_diff_ci': boot(a.values, b.values), 'dev_diff_ci': boot(c.values, e.values)}


def main(arms):
    for arm in arms:
        T = rung_tables(arm)
        rows = [row(k, v) for k, v in T.items()]
        res = {'arm': arm, 'rows': rows, 'pairs': {}}
        tags = list(T)
        for x in tags:
            for y in tags:
                if x != y and x > y:
                    res['pairs'][f'{x}_vs_{y}'] = compare(T[x], T[y])
        json.dump(res, open(os.path.join(SEQ, arm, 'work', 'ladder.json'), 'w'), indent=1, default=float)
        print(f'== {arm}')
        print(pd.DataFrame(rows).drop(columns=['dev_by_class_nse']).to_string(index=False))
        for k, v in res['pairs'].items():
            print(f'  {k}: sim {v["sim_diff_ci"][0]:+.3f} ({v["sim_diff_ci"][1]:+.3f}..{v["sim_diff_ci"][2]:+.3f})  '
                  f'dev {v["dev_diff_ci"][0]:+.3f} ({v["dev_diff_ci"][1]:+.3f}..{v["dev_diff_ci"][2]:+.3f})')


if __name__ == '__main__':
    main(sys.argv[1:] or ['lstm', 'transformer', 'ssm', 'tcn'])
