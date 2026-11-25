"""Vehicle Bake-off, Phase 3: extra comparisons for the River question ("families converge; the simulator is the
lever"). Reads only the exam's scores.parquet, analysis.json and the arms' pre-exam numbers. Chooses nothing.
usage: python3 extra.py  -> phase3/results/extra_comparisons.json

1. spread: per track, the E16 range and sd across the raw learner families, against the gaps E_phys -> best learner,
   best learner -> EnergyNet v2.1 (3x data) and best learner -> the classical composite (dev-fitted);
2. shared errors: per exam set, the median pairwise Pearson correlation of per-unit log errors between learners of
   different families (high = the errors come from what they share: bank, inputs, anchor), and the share of sets in
   which every learner's median bias has the same sign and exceeds 3 %;
3. transfer: Spearman rank correlation between the arms' sim val skill / dev D16 and their exam E16.
"""
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'harness'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

P3 = os.path.dirname(os.path.abspath(__file__))  # cloud: /home/claude/bakeoff/vehicle/phase3
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # results/ sits beside harness/ in the repo (cloud: /home/claude/bakeoff/vehicle/harness)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work', 'phase3')  # cloud: phase3/results
R = json.load(open(f'{OUT}/analysis.json'))
import pyarrow.parquet as pq  # noqa: E402
_rows = ['tab_mlp', 'tab_gbm', 'tab_gp', 'formula', 'trace_gru', 'trace_cnn', 'trace_s4d', 'trace_transformer', 'enet_v21', 'ephys']
sc = pq.read_table(os.path.join(H, R['source'][:-5], 'scores.parquet'), columns=['unit', 'y', 'pred', 'set', 'row', 'track'],
                   filters=[('row', 'in', _rows)]).to_pandas(strings_to_categorical=True)
_G = {k: g.drop_duplicates('unit').set_index('unit') for k, g in sc.groupby(['track', 'row', 'set'], observed=True)}
E16 = ['VED_ICE', 'VED_HEV', 'VED_EV', 'VED_PHEV', 'EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06', 'EPA_conv_FTPcold',
       'EPA_conv_SC03', 'EPA_hev_FTP', 'EPA_hev_HWY', 'EPA_bev_FTP', 'EPA_bev_HWY', 'Mersin_diesel', 'Mersin_cng', 'ICCT']
LEARN = {'table': ['tab_mlp', 'tab_gbm', 'tab_gp', 'formula'], 'trace': ['trace_gru', 'trace_cnn', 'trace_s4d', 'trace_transformer']}
ALL = [('table', k) for k in LEARN['table']] + [('trace', k) for k in LEARN['trace']]
FAM = {'tab_mlp': 'mlp', 'tab_gbm': 'gbm', 'tab_gp': 'gp', 'formula': 'symbolic', 'trace_gru': 'ssm', 'trace_cnn': 'cnn1d',
       'trace_s4d': 'ssm', 'trace_transformer': 'attn', 'enet_v21': 'mlp'}


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def logerr(track, row, st):
    g = _G.get((track, row, st))
    if g is None:
        return pd.Series(dtype=float)
    s = np.log(g.pred / g.y)
    s.index = s.index.astype(str)
    return s


def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    return float(a.rank().corr(b.rank()))


out = {'spread': {}, 'shared_errors': {}, 'transfer': {}}
B = R['boards']
for track, L in LEARN.items():
    e = {k: B[track][k]['E16'] for k in L if B[track].get(k, {}).get('E16') is not None}
    nets = {k: v for k, v in e.items() if k != 'formula'}
    best = min(e, key=e.get)
    out['spread'][track] = {
        'E16': e, 'best': best, 'range_all': r4(max(e.values()) - min(e.values())),
        'range_networks_and_trees': r4(max(nets.values()) - min(nets.values())) if nets else None,
        'sd_networks_and_trees': r4(np.std(list(nets.values()))) if nets else None,
        'gap_ephys_to_best': r4(B[track]['ephys']['E16'] - e[best]),
        'gap_best_to_enet': r4(e[best] - B[track]['enet_v21']['E16']),
        'gap_best_to_classical': r4(e[best] - B[track]['classical']['E16']) if B[track].get('classical', {}).get('E16') else None,
        'pairwise_ties': {k: v['verdict'] for k, v in R['pairwise'][track].items()}}

# shared errors
corr_rows = []
bias_sets = {}
for st in E16 + ['ZTBus_legs']:
    errs = {f'{t}:{k}': logerr(t, k, st) for t, k in ALL}
    errs['table:enet_v21'] = logerr('table', 'enet_v21', st)
    errs['table:ephys'] = logerr('table', 'ephys', st)
    keys = [k for k in errs if len(errs[k]) >= 5]
    cross, same = [], []
    for a, b in combinations([k for k in keys if k.split(':')[1] in FAM], 2):
        ia = errs[a].index.intersection(errs[b].index)
        if len(ia) < 5:
            continue
        c = float(np.corrcoef(errs[a].loc[ia], errs[b].loc[ia])[0, 1])
        (same if FAM[a.split(':')[1]] == FAM[b.split(':')[1]] else cross).append(c)
    ce = []
    for a in keys:
        if a.split(':')[1] in FAM and a != 'table:enet_v21':
            ia = errs[a].index.intersection(errs['table:ephys'].index)
            ce.append(float(np.corrcoef(errs[a].loc[ia], errs['table:ephys'].loc[ia])[0, 1]))
    biases = {k: float(np.median(np.exp(errs[k]) - 1)) for k in keys if k.split(':')[1] in LEARN['table'] + LEARN['trace']}
    signs = set(np.sign(list(biases.values())))
    shared = len(signs) == 1 and min(abs(v) for v in biases.values()) > 0.03
    bias_sets[st] = {'shared_sign_over_3pct': shared, 'sign': int(next(iter(signs))) if len(signs) == 1 else 0,
                     'median_bias_by_learner': {k: r4(v) for k, v in biases.items()},
                     'enet_bias': r4(float(np.median(np.exp(errs['table:enet_v21']) - 1))) if len(errs['table:enet_v21']) else None}
    corr_rows.append({'set': st, 'n': int(len(errs['table:enet_v21'])), 'median_corr_cross_family': r4(np.median(cross)) if cross else None,
                      'median_corr_same_family': r4(np.median(same)) if same else None,
                      'median_corr_with_ephys_error': r4(np.median(ce)) if ce else None})
out['shared_errors'] = {'by_set': corr_rows, 'bias': bias_sets,
                        'n_sets_shared_bias': int(sum(v['shared_sign_over_3pct'] for k, v in bias_sets.items() if k in E16)),
                        'median_cross_family_corr_over_E16': r4(np.median([r['median_corr_cross_family'] for r in corr_rows if r['set'] in E16 and r['median_corr_cross_family'] is not None]))}

# transfer: sim val / dev -> exam
sv = R['simval']
rows = []
for t, k in ALL:
    if k in sv and B[t].get(k, {}).get('E16') is not None:
        rows.append({'arm': k, 'track': t, 'simval_skill': sv[k]['simval_skill'], 'dev_D16': sv[k]['dev_D16'], 'exam_E16': B[t][k]['E16']})
rows.append({'arm': 'enet_v21', 'track': 'table', 'simval_skill': 0.680, 'dev_D16': 0.1133, 'exam_E16': B['table']['enet_v21']['E16']})
df = pd.DataFrame(rows).drop_duplicates('arm')
out['transfer'] = {'rows': rows, 'spearman_simskill_vs_exam': r4(spearman(-df.simval_skill, df.exam_E16)),
                   'spearman_devD16_vs_exam': r4(spearman(df.dev_D16, df.exam_E16)),
                   'note': 'positive = the pre-exam ranking carries over (sim skill negated so that lower = better on both)'}
json.dump(out, open(f'{OUT}/extra_comparisons.json', 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('spread', 'transfer')}, indent=1, default=float)[:4000])
print('shared bias sets', out['shared_errors']['n_sets_shared_bias'], 'cross-family corr', out['shared_errors']['median_cross_family_corr_over_E16'])
