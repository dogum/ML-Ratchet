"""Where does each sequence arm win? (sim val + dev only; no exam data.)
usage: python3 analyze.py lstm:r1 transformer:r1 ssm:r1 tcn:r1    ->  arms_seq/results/compare.json + printout
1. Frozen sim val (800 runs): median per-run NSE by the run's hidden truth (runoff model, snow routine, routing,
   reservoir, prior) and by class; paired bootstrap of each arm vs the LSTM control.
2. Dev gate (52 basins): median NSE by class, paired bootstrap vs the LSTM control.
3. Memory diagnostic: impulse response. On 60 frozen val windows, add 20 mm of rain on target day 120 (a warm day is
   not enforced) and record the change in predicted mean flow over the next 180 days: total extra flow (fraction of the
   20 mm returned within 180 days) and the lag by which half / 90 % of it has come out.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.dirname(HERE)
sys.path.insert(0, '/home/claude/bakeoff/river/harness')
sys.path.insert(0, os.path.join(SEQ, 'common'))
import bake  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seqlib  # noqa: E402
from ladder import boot  # noqa: E402


RELEASED = {'lstm:r1', 'transformer:r1', 'transformer:r3', 'ssm:r2', 'tcn:r1'}


def main(specs):
    arms = [s.split(':') for s in specs]
    B = bake.load_bank('val', ['F', 'qo', 'attrs', 'st', 'lat', 'lon', 'start', 'model', 'snow', 'routing', 'prior', 'reg'])
    info = bake.bank_info()
    lab = pd.DataFrame({'model': [info['models'][i] for i in B['model']], 'snow': [info['snows'][i] for i in B['snow']],
                        'routing': np.where(np.asarray(B['routing']) == 1, 'muskingum', 'uh'),
                        'reservoir': np.asarray(B['reg']) == 1, 'prior': np.where(np.asarray(B['prior']) == 1, 'camels', 'param')})
    lab['classes'] = [bake._sim_classes(B, i) for i in range(len(lab))]
    sim, dev = {}, {}
    for a, r in arms:
        sim[f'{a}:{r}'] = pd.read_parquet(os.path.join(SEQ, a, 'work', 'eval', f'{r}_simval_ungauged.parquet')).nse
        dev[f'{a}:{r}'] = pd.read_parquet(os.path.join(SEQ, a, 'work', 'eval', f'{r}_dev_ungauged.parquet')).nse
    S = pd.DataFrame(sim).join(lab)
    out = {'arms': specs, 'sim': {}, 'dev': {}, 'impulse': {}}
    names = [f'{a}:{r}' for a, r in arms]
    base = names[0]                     # the control (the LSTM's released rung): list it first
    groups = {'all': np.ones(len(S), bool)}
    for col in ('model', 'snow', 'routing', 'prior'):
        for v in sorted(S[col].unique()):
            groups[f'{col}={v}'] = (S[col] == v).values
    groups['reservoir'] = S.reservoir.values
    for c in ['snow', 'arid', 'humid', 'regulated', 'large', 'other']:
        groups[f'class={c}'] = np.array([c in x for x in S.classes])
    for g, m in groups.items():
        row = {'n': int(m.sum())}
        for a in names:
            row[a] = round(float(np.nanmedian(S[a][m])), 4)
            if a != base:
                row[f'{a}-ctrl'] = [round(x, 4) for x in boot(S[a][m].values, S[base][m].values, n=2000)]
        out['sim'][g] = row
    D = pd.DataFrame(dev)
    cls = {b: bake.basin_classes(b) for b in D.index}
    dgroups = {'all': np.ones(len(D), bool)}
    for c in ['snow', 'arid', 'humid', 'regulated', 'large']:
        dgroups[c] = np.array([c in cls[b] for b in D.index])
    for g, m in dgroups.items():
        row = {'n': int(m.sum())}
        for a in names:
            row[a] = round(float(np.nanmedian(D[a][m])), 4)
            if a != base:
                row[f'{a}-ctrl'] = [round(x, 4) for x in boot(D[a][m].values, D[base][m].values, n=2000)]
        out['dev'][g] = row
    # wins per dev basin
    rel = [n for n in names if n in RELEASED]
    out['dev_best_arm_count_released'] = D[rel].idxmax(1).value_counts().to_dict()
    # impulse response
    rng = np.random.default_rng(20261123)
    runs = rng.choice(len(lab), 60, replace=False)
    t0, H, add = 120, 180, 20.0
    for a, r in arms:
        M = seqlib.Model(os.path.join(SEQ, a, 'work', f'cand_{r}', 'weights.npz'))
        tot, half, p90 = [], [], []
        for i in runs:
            b = bake.sim_window(B, int(i), 0)
            b2 = dict(b); F = b['forcing'].copy(); F[365 + t0, 0] += add; b2['forcing'] = F
            o1, o2 = M.predict_many([b, b2])
            d = (o2['mean'] - o1['mean'])[t0:t0 + H]
            c = np.cumsum(d)
            tot.append(c[-1] / add)
            if c[-1] > 0.02 * add:
                half.append(int(np.argmax(c >= 0.5 * c[-1]))); p90.append(int(np.argmax(c >= 0.9 * c[-1])))
        out['impulse'][f'{a}:{r}'] = {'n': len(runs), 'frac_returned_180d_median': round(float(np.median(tot)), 3),
                             'lag_half_days_median': float(np.median(half)) if half else None,
                             'lag_90_days_median': float(np.median(p90)) if p90 else None,
                             'n_with_response': len(half)}
    os.makedirs(os.path.join(SEQ, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(SEQ, 'results', 'compare.json'), 'w'), indent=1, default=float)
    print(pd.DataFrame(out['sim']).T[['n'] + names].to_string())
    print(pd.DataFrame(out['dev']).T[['n'] + names].to_string())
    print('dev best-arm count (released rungs)', out['dev_best_arm_count_released'])
    print(json.dumps(out['impulse'], indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
