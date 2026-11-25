"""River Bake-off, Phase 3: analysis of the once-only exam (reads only the exam's outputs; chooses nothing).

usage: python3 analyze.py [results/exam_<id>.json]      (default: the run named in results/EXAM_DONE.json)
       python3 analyze.py --rehearsal results/rehearsal_<id>.json   (code test on the dev rehearsal)
Writes phase3/results/analysis.json and phase3/results/leaderboards.md.
"""
import json
import os
import sys

HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'harness')  # cloud: /home/claude/bakeoff/river/harness
sys.path.insert(0, HARNESS)
BASE = HARNESS if os.path.isdir(os.path.join(HARNESS, 'results')) else os.path.dirname(HARNESS)  # repo: results/ sits beside harness/
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import config as C  # noqa: E402
import score as S  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work', 'phase3')  # cloud: phase3/results; shipped copies in results/

LEARNERS = ['flowcast_v1', 'seq_lstm', 'seq_ssm', 'seq_tcn', 'seq_transformer', 'tab_gbm', 'tab_gp', 'tab_dpl',
            'tab_formula']
NEW_LEARNERS = LEARNERS[1:]
CONF = [k + '_conformal' for k in LEARNERS]
UNG_REFS = ['udev', 'uk12', 'gr4j', 'nwm', 'clim']
GAUGED_ARMS = ['enkf_watersim', 'enkf_watersim_conformal', 'flowcast_v1', 'flowcast_v1_conformal']
GAUGED_REFS = ['persistence', 'gr4j', 'nwm']
CLASSES = ['snow', 'arid', 'humid', 'regulated', 'large']

LABEL = {
    'flowcast_v1': 'FlowCast v1.0 (LSTM, as released)', 'seq_lstm': 'LSTM (FlowCast net, retrained)',
    'seq_ssm': 'State space (S4D + encoder)', 'seq_tcn': 'TCN (dilated 1-D CNN)', 'seq_transformer': 'Transformer',
    'tab_gbm': 'Gradient-boosted trees', 'tab_gp': 'Sparse Gaussian process', 'tab_dpl': 'Physics in the loop (HBV)',
    'tab_formula': '17 formulas in HBV', 'enkf_watersim': 'Ensemble Kalman filter (HBV)', 'climatology': 'Climatology (Budyko-Fu)',
    'udev': 'U-dev regionalisation', 'uk12': 'U-k12 regionalisation', 'gr4j': 'GR4J calibrated per basin',
    'nwm': 'NOAA NWM v3.0', 'clim': 'Own day-of-year climatology', 'persistence': 'Persistence'}
for k in LEARNERS + ['enkf_watersim']:
    LABEL[k + '_conformal'] = LABEL[k] + ' + conformal'
FAMILY = {'flowcast_v1': 'ssm', 'seq_lstm': 'ssm', 'seq_ssm': 'ssm', 'seq_tcn': 'cnn1d', 'seq_transformer': 'attn',
          'tab_gbm': 'gbm', 'tab_gp': 'gp', 'tab_dpl': 'diffsim', 'tab_formula': 'symbolic', 'enkf_watersim': 'assim'}
# numbers stored for inference (counted from the registered weight files, 23 Nov 2026)
PARAMS = {'flowcast_v1': 24710, 'seq_lstm': 24710, 'seq_tcn': 26266, 'seq_ssm': 30438, 'seq_transformer': 26854,
          'tab_gbm': 87876, 'tab_gp': 260585, 'tab_dpl': 6481, 'tab_formula': None, 'enkf_watersim': 0}
PARAMS_NOTE = {'tab_gbm': '1,172 trees, 87,876 nodes', 'tab_gp': '500 inducing points x 20 inputs + a 500 x 500 matrix',
               'tab_dpl': 'MLP 18-64-64-17', 'tab_formula': '17 formulas', 'enkf_watersim': 'no learned weights; 8 tuned filter settings'}
METRICS = ['nse', 'kge', 'fhv', 'fhv_abs', 'cov50', 'cov90', 'cov50_mean', 'cov90_mean', 'crps', 'crps_rel',
           'width90_rel', 'peak_abs_err_pct_median', 'frac_nse_lt0', 'n', 'n_errors']


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def load(fn):
    E = json.load(open(fn))
    od = os.path.join(os.path.dirname(fn), os.path.basename(fn)[:-5])
    sc = pd.read_parquet(os.path.join(od, 'scores.parquet'))
    if 'error' in sc:
        sc = sc[sc['error'].isna()]
    return E, sc, od


def per_unit(sc, row, per, proto, lead):
    if row == 'persistence':
        g = sc[(sc.row == row) & (sc.period == per) & (sc.lead == lead)]
    elif row in UNG_REFS:
        g = sc[(sc.row == row) & (sc.period == per)]
    else:
        g = sc[(sc.row == row) & (sc.period == per) & (sc.protocol == proto) & (sc.lead == (lead or 0))]
    return g.set_index('basin')


def class_of(g):
    return {b: str(c).split(',') for b, c in g['classes'].items()}


def class_table(g):
    cl = class_of(g)
    out = {}
    for c in ['all'] + CLASSES:
        ids = list(g.index) if c == 'all' else [b for b in g.index if c in cl[b]]
        h = g.loc[ids]
        d = {'n': len(ids)}
        for m in ('nse', 'kge', 'crps_rel'):
            d[m] = r3(np.nanmedian(h[m].astype(float))) if len(ids) and m in h else None
        d['fhv_abs'] = r3(np.nanmedian(np.abs(h['fhv'].astype(float)))) if len(ids) else None
        d['cov90_mean'] = r3(h['cov90'].astype(float).mean()) if len(ids) and 'cov90' in h and h['cov90'].notna().any() else None
        out[c] = d
    return out


def boot(a, b, col='nse', ids=None, better='higher', stat=S.med):
    common = a.index.intersection(b.index)
    if ids is not None:
        common = common.intersection(pd.Index(ids))
    r = S.paired_bootstrap(a.loc[common, col].values, b.loc[common, col].values, stat,
                           rng=np.random.default_rng(C.SEED))
    r['verdict'] = S.verdict(r, better)
    return {k: (r3(v) if isinstance(v, float) else [r3(x) for x in v] if isinstance(v, list) else v) for k, v in r.items()}


def board(E, sc, per, kind, lead=None):
    key = f'{per}/{kind}' if kind == 'ungauged' else f'{per}/gauged_L{lead}'
    B = E['boards'][key]
    if kind == 'ungauged':
        rows = [k for k in LEARNERS + CONF + ['climatology'] if k in B['rows']] + [k for k in UNG_REFS if k in B['rows']]
        proto = 'ungauged'
    else:
        rows = [k for k in GAUGED_ARMS if k in B['rows']] + [k for k in GAUGED_REFS if k in B['rows']]
        proto = 'gauged'
    tab = {}
    units = {k: per_unit(sc, k, per, proto, lead) for k in rows}
    anchor = 'flowcast_v1'
    for k in rows:
        s = B['rows'][k]
        d = {m: (s.get(m) if m in ('n', 'n_errors') else r3(s.get(m))) for m in METRICS}
        d.update({'label': LABEL.get(k, k), 'kind': s['kind'], 'protocol': s['protocol'],
                  'conformal': k.endswith('_conformal'), 'base': k[:-10] if k.endswith('_conformal') else k,
                  'by_class': class_table(units[k])})
        P = B['paired'].get(k, {})
        d['vs'] = {}
        for anc, v in P.items():
            d['vs'][anc] = {m: {'diff': r3(v[m]['diff']), 'ci95': [r3(x) for x in v[m]['ci95']] if v[m].get('ci95') else None,
                                'verdict': v[m]['verdict']} for m in ('nse', 'kge', 'crps_rel', 'fhv') if m in v}
            d['vs'][anc]['share_better_nse'] = r3(v.get('share_units_a_better_nse'))
            d['vs'][anc]['by_class_nse'] = {c: {'diff': r3(x['diff']), 'ci95': x['ci95'] and [r3(y) for y in x['ci95']],
                                                 'verdict': x['verdict'], 'n': x['n']} for c, x in v.get('by_class_nse', {}).items()}
        if k not in B['paired'] and k != anchor and anchor in units and k in units:   # references vs FlowCast (for the table)
            d['vs'][anchor] = {'nse': boot(units[k], units[anchor])}
        tab[k] = d
    # bootstrap vs the in-bake-off LSTM control (same bank, same pipeline) for the new ungauged learners
    if kind == 'ungauged' and 'seq_lstm' in units:
        for k in NEW_LEARNERS + ['flowcast_v1']:
            if k in units and k != 'seq_lstm':
                tab[k]['vs_lstm'] = boot(units[k], units['seq_lstm'])
    return tab, units


def wins_map(units, cands, per, lead=None, band_cands=None, refs=()):
    """Per class: the best candidate by median NSE, every candidate tied with it (paired bootstrap CI covers 0), the
    best by CRPS/mean Q (band_cands), and the best reference row. Classes from the harness (FlowCast's strata)."""
    any_g = next(iter(units.values()))
    cl = class_of(any_g)
    out = {}
    for c in ['all'] + CLASSES:
        ids = list(any_g.index) if c == 'all' else [b for b in any_g.index if c in cl[b]]
        meds = {k: float(np.nanmedian(units[k].loc[units[k].index.intersection(ids), 'nse'])) for k in cands if k in units}
        best = max(meds, key=meds.get)
        tied, beaten = [], []
        for k in meds:
            if k == best:
                continue
            r = boot(units[best], units[k], ids=ids)
            (tied if r['verdict'] == 'tie' else beaten).append(k)
        d = {'n': len(ids), 'best_nse': best, 'best_nse_value': r3(meds[best]), 'tied_with_best': tied,
             'beaten_by_best': beaten, 'nse': {k: r3(v) for k, v in meds.items()}}
        if band_cands:
            cr = {k: float(np.nanmedian(units[k].loc[units[k].index.intersection(ids), 'crps_rel'])) for k in band_cands if k in units}
            d['best_crps'] = min(cr, key=cr.get)
            d['best_crps_value'] = r3(cr[d['best_crps']])
            d['crps_rel'] = {k: r3(v) for k, v in cr.items()}
        rr = {k: float(np.nanmedian(units[k].loc[units[k].index.intersection(ids), 'nse'])) for k in refs if k in units}
        if rr:
            br = max(rr, key=rr.get)
            d['best_ref'] = br
            d['best_ref_value'] = r3(rr[br])
            d['ref_nse'] = {k: r3(v) for k, v in rr.items()}
            d['best_vs_best_ref'] = boot(units[best], units[br], ids=ids)
        out[c] = d
    return out


def compute_table(E):
    out = {}
    for k, a in E['arms'].items():
        cpu = a.get('exam_cpu', {})
        out[k] = {'label': LABEL.get(k, k), 'family': a.get('family'), 'train_cpu_hours': a.get('compute', {}).get('train_cpu_hours'),
                  'exam_predict_cpu_s': round(sum(v.get('predict_cpu_s', 0) for v in cpu.values()), 1),
                  'exam_process_cpu_s': round(sum(v.get('process_cpu_s', 0) for v in cpu.values()), 1),
                  'jobs': {j: v.get('predict_cpu_s') for j, v in cpu.items()},
                  'params': PARAMS.get(k), 'params_note': PARAMS_NOTE.get(k), 'numpy_only': a.get('numpy_only'),
                  'released': a.get('released'), 'unchanged_after': a.get('unchanged_after'), 'hash': a.get('hash', '')[:16]}
    return out


def md_board(tab, title, gauged=False):
    L = [f'### {title}', '',
         '| Row | n | NSE | KGE | \\|FHV\\| % | cov50 / cov90 (mean over basins) | CRPS / mean Q | vs FlowCast v1.0, NSE (95 % CI) |',
         '|---|---|---|---|---|---|---|---|']
    for k, d in tab.items():
        v = d['vs'].get('flowcast_v1', {}).get('nse')
        vs = '' if not v or v.get('diff') is None else f"{v['diff']:+.3f} ({v['ci95'][0]:+.3f}..{v['ci95'][1]:+.3f}) {v['verdict']}"
        cov = '' if d.get('cov50_mean') is None else f"{d['cov50_mean']:.2f} / {d['cov90_mean']:.2f}"
        f = lambda x, p=3: '' if x is None else f'{x:.{p}f}'  # noqa: E731
        L.append(f"| {d['label']} | {d['n']} | {f(d['nse'])} | {f(d['kge'])} | {f(d['fhv_abs'], 1)} | {cov} | {f(d['crps_rel'])} | {vs} |")
    return '\n'.join(L) + '\n'


def main():
    a = sys.argv[1:]
    rehearsal = '--rehearsal' in a
    a = [x for x in a if x != '--rehearsal']
    if a:
        fn = a[0]
    else:
        fn = os.path.join(BASE, json.load(open(os.path.join(BASE, 'results', 'EXAM_DONE.json')))['result'])
    E, sc, od = load(fn)
    periods = ['dev'] if rehearsal else ['main', 'recent']
    R = {'source': os.path.relpath(fn, BASE).replace(os.sep, '/'), 'run_id': E['run_id'], 'mode': E['mode'], 'boards': {}, 'wins': {},
         'compute': compute_table(E), 'notes': E['notes']}
    md = [f"# River Bake-off exam leaderboards (run {E['run_id']})", '',
          'Medians over basins. Paired bootstrap over basins of the difference in medians (10,000 resamples); win/loss '
          'only when the 95 % interval excludes 0. Conformal rows share their base arm\'s mean (same NSE/KGE/FHV).', '']
    for per in periods:
        tab, units = board(E, sc, per, 'ungauged')
        R['boards'][f'{per}/ungauged'] = tab
        md.append(md_board(tab, f'{per}: ungauged'))
        R['wins'][f'{per}/ungauged'] = wins_map(units, [k for k in LEARNERS if k in units], per,
                                                band_cands=[k for k in LEARNERS + CONF if k in units],
                                                refs=[k for k in UNG_REFS if k in units])
        for L in C.LEADS:
            if f'{per}/gauged_L{L}' not in E['boards']:
                continue
            tab, units = board(E, sc, per, 'gauged', L)
            R['boards'][f'{per}/gauged_L{L}'] = tab
            md.append(md_board(tab, f'{per}: gauged, lead {L} d', True))
            R['wins'][f'{per}/gauged_L{L}'] = wins_map(units, [k for k in ['enkf_watersim', 'flowcast_v1', 'persistence'] if k in units],
                                                       per, band_cands=[k for k in GAUGED_ARMS if k in units],
                                                       refs=[k for k in ['nwm', 'gr4j'] if k in units])
    md.append('### Where each assumption wins (median NSE; best learner, and who is statistically tied with it)\n')
    md.append('| Board | Class | n | Best learner (NSE) | Tied with it | Best CRPS row | Best reference (NSE) | best learner vs best reference |')
    md.append('|---|---|---|---|---|---|---|---|')
    for b, W in R['wins'].items():
        for c, d in W.items():
            br = f"{LABEL.get(d.get('best_ref'), '')} ({d.get('best_ref_value')})" if d.get('best_ref') else ''
            v = d.get('best_vs_best_ref')
            vv = '' if not v or v.get('diff') is None else f"{v['diff']:+.3f} {v['verdict']}"
            md.append(f"| {b} | {c} | {d['n']} | {LABEL[d['best_nse']]} ({d['best_nse_value']}) | "
                      f"{', '.join(LABEL[k] for k in d['tied_with_best']) or 'none'} | {LABEL.get(d.get('best_crps'), '')} "
                      f"({d.get('best_crps_value')}) | {br} | {vv} |")
    md.append('\n### Compute\n')
    md.append('| Arm | Train CPU-h | Exam predict CPU-s | Stored numbers |')
    md.append('|---|---|---|---|')
    for k, d in sorted(R['compute'].items()):
        md.append(f"| {d['label']} | {d['train_cpu_hours']} | {d['exam_predict_cpu_s']} | {d['params'] if d['params'] is not None else ''} {d['params_note'] or ''} |")
    os.makedirs(OUT, exist_ok=True)
    tag = 'rehearsal_' if rehearsal else ''
    json.dump(R, open(os.path.join(OUT, f'{tag}analysis.json'), 'w'), indent=1, default=float)
    open(os.path.join(OUT, f'{tag}leaderboards.md'), 'w').write('\n'.join(md) + '\n')
    print('wrote', os.path.join(OUT, f'{tag}analysis.json'))


if __name__ == '__main__':
    main()
