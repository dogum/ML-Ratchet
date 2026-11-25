"""PREREG section 3: judge each bank variant with the 3-seed LSTM control. Reads work/eval/<bank>_lstm_r1_s<seed>/.

usage: python judge_banks.py b2a b2b b2c [...]          (each against p1) -> results/bank_rungs.json
Pass mark: mean over seeds 0-2 of the dev median NSE (52 basins) on the variant > the same on p1 + 0.01.
Reported, not gating: sim-val NSE on the variant's own val and on p1's val, KGE, |FHV|, per class, and a paired
bootstrap over dev basins of the per-basin NSE averaged over the 3 seeds (10,000 resamples, default_rng(20261118)).
"""
import json
import os
import sys

import r2
import numpy as np
import pandas as pd
import bake
import score as S

SEEDS = (0, 1, 2)
MARGIN = 0.01


def load(bank):
    out = {'dev': [], 'dev_kge': [], 'dev_fhv': [], 'sim_own': [], 'sim_p1': [], 'per_basin': [], 'cpu_h': []}
    for s in SEEDS:
        d = os.path.join(r2.WORK, 'eval', f'{bank}_lstm_r1_s{s}')
        sm = json.load(open(os.path.join(d, 'own_summary.json')))
        out['dev'].append(sm['dev']['ungauged']['nse'])
        out['dev_kge'].append(sm['dev']['ungauged']['kge'])
        out['dev_fhv'].append(sm['dev']['ungauged']['fhv'])
        out['sim_own'].append(sm['simval']['ungauged']['nse'])
        if bank == 'p1':
            out['sim_p1'].append(sm['simval']['ungauged']['nse'])
        else:
            out['sim_p1'].append(json.load(open(os.path.join(d, 'p1val_summary.json')))['simval']['ungauged']['nse'])
        out['per_basin'].append(pd.read_parquet(os.path.join(d, 'own_dev_ungauged.parquet'))['nse'])
        tr = json.load(open(os.path.join(r2.WORK, 'cand', f'{bank}_lstm_r1_s{s}', 'train.json')))
        out['cpu_h'].append(tr['train_cpu_h'])
    pb = pd.concat(out['per_basin'], axis=1).mean(1)
    out['per_basin'] = pb
    return out


def main(variants):
    classes = {b: bake.basin_classes(b) for b in bake.dev_basins()}
    inc = load('p1')
    res = {'rule': f'keep if mean over seeds {SEEDS} of control dev median NSE > p1 + {MARGIN}',
           'p1': {k: v for k, v in inc.items() if k != 'per_basin'}, 'variants': {}}
    res['p1']['dev_mean'] = float(np.mean(inc['dev']))
    res['p1']['by_class_seedavg'] = S.by_class(pd.DataFrame({'nse': inc['per_basin']}), classes)
    rng = np.random.default_rng(20261118)
    for v in variants:
        try:
            x = load(v)
        except FileNotFoundError as e:
            print(v, 'not complete:', e)
            continue
        d = float(np.mean(x['dev']) - np.mean(inc['dev']))
        a, b = x['per_basin'], inc['per_basin']
        diffs = []
        idx = np.arange(len(a))
        for _ in range(10000):
            j = rng.integers(0, len(a), len(a))
            diffs.append(np.median(a.values[j]) - np.median(b.values[j]))
        rec = {k: v2 for k, v2 in x.items() if k != 'per_basin'}
        rec.update({'dev_mean': float(np.mean(x['dev'])), 'dev_gain_vs_p1': d, 'kept': bool(d > MARGIN),
                    'sim_own_mean': float(np.mean(x['sim_own'])), 'sim_p1val_mean': float(np.mean(x['sim_p1'])),
                    'p1_sim_mean': float(np.mean(inc['sim_own'])),
                    'seedavg_basin_median_diff_ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                    'by_class_seedavg': S.by_class(pd.DataFrame({'nse': x['per_basin']}), classes)})
        res['variants'][v] = rec
        print(f"{v}: dev {np.round(x['dev'], 4)} mean {rec['dev_mean']:.4f} vs p1 {res['p1']['dev_mean']:.4f} "
              f"gain {d:+.4f} -> {'KEPT' if rec['kept'] else 'rejected'}; sim own {rec['sim_own_mean']:.4f} "
              f"p1val {rec['sim_p1val_mean']:.4f} (p1 control {rec['p1_sim_mean']:.4f}); CI {rec['seedavg_basin_median_diff_ci95']}")
    print('p1: dev', np.round(inc['dev'], 4), 'mean', round(res['p1']['dev_mean'], 4), 'sim', np.round(inc['sim_own'], 4))
    os.makedirs(r2.RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(r2.RESULTS, 'bank_rungs.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1:])
