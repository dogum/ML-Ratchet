"""Analysis of the TabPFN supplementary exam (results/supp_tabpfn/), beside the main board, never folded into it.

usage: python phase3/supp_tabpfn.py   -> results/supp_tabpfn/analysis.json + leaderboard.md
1. Same records: every non-TabPFN table-track row re-scored in the supplement against the main exam's scores.parquet
   (per unit, set and row: prediction and truth).
2. E16 board of the table track, with TabPFN, by phase3/analyze.py's own code (shared cluster resamples per set),
   and TabPFN's paired E16 differences against EnergyNet v2.1, the MLP control, trees, the GP and the classical
   composite.
Reads only the two exams' outputs.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P3 = os.path.dirname(os.path.abspath(__file__))
BK = os.path.dirname(P3)
sys.path.insert(0, P3)
import analyze as A  # noqa: E402

SUPP = os.path.join(BK, 'results', 'supp_tabpfn')
MAIN = os.path.join(BK, 'results', 'exam_20261123T204251', 'scores.parquet')


def main():
    done = json.load(open(os.path.join(SUPP, 'EXAM_DONE.json')))
    fn = os.path.join(BK, done['result'])
    E, sc, od = A.load(fn)
    sc = sc[sc.track == 'table']
    out = {'run_id': E['run_id'], 'source': os.path.relpath(fn, BK).replace(os.sep, '/'),
           'main_exam': '20261123T204251', 'note': E.get('supplement', {}).get('note')}
    # 1. same records
    rows = sorted(r for r in sc.row.unique() if not str(r).startswith('tab_tabpfn'))
    m = pq.read_table(MAIN, columns=['unit', 'y', 'pred', 'set', 'row', 'track'],
                      filters=[('track', '=', 'table'), ('row', 'in', rows)]).to_pandas()
    k = ['row', 'set', 'unit']
    s = sc[sc.row.isin(rows)][k + ['y', 'pred']].astype({'row': str, 'set': str, 'unit': str})
    m = m.astype({'row': str, 'set': str, 'unit': str})
    j = s.merge(m, on=k, how='outer', suffixes=('', '_main'), indicator=True)
    both = j[j._merge == 'both']
    rel = lambda a, b: float(np.nanmax(np.abs(a / b - 1))) if len(a) else None  # noqa: E731
    out['same_records'] = {'rows_checked': rows, 'units_both': int(len(both)),
                           'only_in_supplement': int((j._merge == 'left_only').sum()),
                           'only_in_main': int((j._merge == 'right_only').sum()),
                           'max_rel_diff_y': rel(both.y.values, both.y_main.values),
                           'max_rel_diff_pred': rel(both.pred.values, both.pred_main.values)}
    print(out['same_records'], flush=True)
    # 2. E16 board
    A.index_scores(sc)
    board_rows = [r for r in ['enet_v21', 'tab_mlp', 'tab_gbm', 'tab_gp', 'tab_tabpfn', 'tab_tabpfn_conformal',
                              'tab_mlp_conformal', 'tab_gbm_conformal', 'tab_gp_conformal', 'formula', 'phys_anchor',
                              'climatology', 'ephys', 'classical'] if r == 'classical' or r in set(sc.row.unique())]
    eb = A.e16_board(sc, 'table', board_rows)
    out['E16'] = {r: {'E16': v['E16'], 'E16_crps': v['E16_crps'], 'n_sets': v['n_sets']} for r, v in eb.items()}
    out['pairs'] = {}
    for a in ('tab_tabpfn', 'tab_tabpfn_conformal'):
        for b in ('enet_v21', 'tab_mlp', 'tab_gbm', 'tab_gp', 'classical'):
            out['pairs'][f'{a} vs {b}'] = A.e16_pairs(eb, a, b)
    per = {}
    for st in A.E16:
        u = A.units_of(sc, 'table', 'tab_tabpfn', st)
        if len(u):
            per[st] = {'n': int(len(u)), 'MdAPE': A.r4(float(np.median(np.abs(u.pred / u.y - 1)))),
                       'bias': A.r4(float(np.median(u.pred / u.y - 1)))}
    out['tabpfn_per_set'] = per
    json.dump(out, open(os.path.join(SUPP, 'analysis.json'), 'w'), indent=1, default=float)
    L = [f"# TabPFN supplementary exam (run {E['run_id']}), table track, beside the main exam 20261123T204251", '',
         'E16 = mean exam MdAPE over the 16 D16 sets (lower is better); paired cluster bootstrap with shared resamples, '
         'as phase3/analyze.py. The main board is unchanged; these rows are a supplement.', '',
         '| Row | E16 | E16 CRPS(log) |', '|---|---|---|']
    for r, v in sorted(out['E16'].items(), key=lambda kv: (kv[1]['E16'] is None, kv[1]['E16'] or 9)):
        L.append(f"| {r} | {v['E16']} | {v['E16_crps']} |")
    L += ['', '| Pair | E16 diff (95 % CI) | verdict |', '|---|---|---|']
    for p, v in out['pairs'].items():
        if v:
            L.append(f"| {p} | {v['E16']['diff']} ({v['E16']['ci95'][0]}..{v['E16']['ci95'][1]}) | {v['E16']['verdict']} |")
    L += ['', f"Same records: {out['same_records']}"]
    open(os.path.join(SUPP, 'leaderboard.md'), 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
