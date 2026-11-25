"""Assemble the River Bake-off page: template.html + analysis.json + showcase_page.json + text.json -> river-bakeoff.html.
usage: python3 build_page.py
"""
import json
import os

P3 = os.path.dirname(os.path.abspath(__file__))  # cloud: /home/claude/bakeoff/river/phase3
R = json.load(open(f'{P3}/results/analysis.json'))
SH = json.load(open(f'{P3}/results/showcase_page.json'))
NAMES = {'01078000': 'Smith River near Bristol, NH', '09066200': 'Booth Creek near Minturn, CO',
         '08158810': 'Bear Creek below FM 1826 near Driftwood, TX', '13161500': 'Bruneau River at Rowland, NV',
         '01054200': 'Wild River at Gilead, ME', '12411000': "North Fork Coeur d'Alene River near Prichard, ID",
         '03237280': 'Upper Twin Creek at McGaw, OH', '05120500': 'Wintering River near Karlsruhe, ND',
         '04056500': 'Manistique River near Manistique, MI', '12414500': 'St. Joe River at Calder, ID'}
for _b in SH['basins']:     # display names (USGS station names, tidied by hand)
    _b['name'] = NAMES.get(_b['id'], _b['name'])
T = json.load(open(f'{P3}/page/text.json'))
FAMURL = 'https://dogum.github.io/ML-Ratchet/model-families/'

ARMS = {
    'flowcast_v1': ('FlowCast v1.0', 'FlowCast v1.0 (LSTM, as released)', '--s1', ''),
    'seq_lstm': ('LSTM retrained', 'LSTM, FlowCast net retrained', '--s1', '6 4'),
    'seq_ssm': ('S4D state space', 'State space (S4D + encoder)', '--s3', ''),
    'seq_tcn': ('TCN', 'TCN (dilated 1-D CNN)', '--s7', ''),
    'seq_transformer': ('Transformer', 'Transformer (daily tokens)', '--s5', ''),
    'tab_gbm': ('Trees', 'Gradient-boosted trees', '--s2', ''),
    'tab_gp': ('GP', 'Sparse Gaussian process', '--s4', ''),
    'tab_dpl': ('Physics in loop', 'Physics in the loop (HBV)', '--s6', ''),
    'tab_formula': ('Formulas', '17 formulas in HBV', '--s6', '6 4'),
    'climatology': ('Climatology', 'Climatology (Budyko-Fu, dumb)', '--ref', '2 3'),
    'enkf_watersim': ('EnKF', 'Ensemble Kalman filter (HBV)', '--s8', ''),
    'flowcast_now': ('FlowCast nowcast', 'FlowCast v1.0 nowcast', '--s1', ''),
    'persistence': ('Persistence', 'Persistence', '--ref', '2 3'),
    'udev': ('U-dev', 'U-dev regionalisation (GR4J donors)', '--ref', '7 3'),
    'uk12': ('U-k12', 'U-k12 regionalisation', '--ref', '1 3'),
    'gr4j': ('GR4J calibrated', 'GR4J calibrated per basin (gauged)', '--ink-3', '9 3 2 3'),
    'nwm': ('NWM', 'NOAA NWM v3.0', '--ink-2', '4 2'),
    'clim': ('DOY climatology', 'Own day-of-year climatology (gauged)', '--ref', '1 2'),
}
arms = {k: {'short': s, 'label': l, 'color': c, 'dash': d, 'kind': 'ref' if k in ('persistence', 'udev', 'uk12', 'gr4j', 'nwm', 'clim', 'climatology') else 'arm'}
        for k, (s, l, c, d) in ARMS.items()}
for k in list(ARMS):
    if k + '_conformal' not in arms and k not in ('flowcast_now', 'persistence', 'udev', 'uk12', 'gr4j', 'nwm', 'clim'):
        a = dict(arms[k]); a['short'] += ' + conformal'; a['label'] += ' + conformal'
        arms[k + '_conformal'] = a

boards, classtab, wins = {}, {}, {}
for key, tab in R['boards'].items():
    rows = []
    gauged = '/gauged' in key
    classtab[key] = {}
    for k, d in tab.items():
        v = d['vs'].get('flowcast_v1', {}).get('nse')
        dd = None if not v or v.get('diff') is None else {'diff': v['diff'], 'lo': v['ci95'][0], 'hi': v['ci95'][1], 'verdict': v['verdict']}
        label = arms['flowcast_now']['label'] if (gauged and k == 'flowcast_v1') else (arms['flowcast_now']['label'] + ' + conformal' if gauged and k == 'flowcast_v1_conformal' else arms.get(k, {'label': d['label']})['label'])
        rows.append({'id': 'flowcast_now' if (gauged and k == 'flowcast_v1') else k, 'label': label, 'kind': d['kind'], 'conformal': d['conformal'],
                     'n': d['n'], 'nse': d['nse'], 'kge': d['kge'], 'fhv_abs': d['fhv_abs'], 'crps_rel': d['crps_rel'],
                     'cov50_mean': d['cov50_mean'], 'cov90_mean': d['cov90_mean'], 'width90_rel': d['width90_rel'], 'd': dd})
        if not d['conformal']:
            classtab[key]['flowcast_now' if (gauged and k == 'flowcast_v1') else k] = {c: x['nse'] for c, x in d['by_class'].items()}
    # conformal rows in the gauged boards keep their base id for the raw-vs-conformal table
    for r in rows:
        if gauged and r['id'] == 'flowcast_v1_conformal':
            r['id'] = 'flowcast_now_conformal'
    boards[key] = rows
for key, W in R['wins'].items():
    gauged = '/gauged' in key
    ren = (lambda k: 'flowcast_now' if (gauged and k == 'flowcast_v1') else k)
    out = {}
    for c, w in W.items():
        out[c] = {'n': w['n'], 'best_nse': ren(w['best_nse']), 'best_nse_value': w['best_nse_value'],
                  'tied_with_best': [ren(k) for k in w['tied_with_best']], 'nse': {ren(k): v for k, v in w['nse'].items()},
                  'best_ref': w.get('best_ref'), 'best_ref_value': w.get('best_ref_value'), 'ref_nse': w.get('ref_nse', {}),
                  'best_vs_best_ref': w.get('best_vs_best_ref')}
    wins[key] = out
if 'flowcast_now_conformal' not in arms:
    a = dict(arms['flowcast_now']); a['short'] += ' + conformal'; a['label'] += ' + conformal'; arms['flowcast_now_conformal'] = a

# compute
nunits = {}
for key, tab in R['boards'].items():
    for k, d in tab.items():
        nunits.setdefault(k, 0)
comp = []
order = ['flowcast_v1', 'seq_lstm', 'seq_ssm', 'seq_tcn', 'seq_transformer', 'tab_gbm', 'tab_gp', 'tab_dpl', 'tab_formula', 'enkf_watersim', 'climatology']
for k in order + sorted(x for x in R['compute'] if x.endswith('_conformal')):
    c = R['compute'][k]
    njobs = len(c['jobs'])
    units = sum(R['boards'][f'{per}/{"ungauged" if j.endswith("ungauged") else "gauged_L" + j[-1]}'].get(k, {}).get('n', 0)
                for j in c['jobs'] for per in [j.split('_')[0]])
    ung = k in order[:9]
    comp.append({'id': k, 'train_h': c['train_cpu_hours'], 'exam_s': c['exam_predict_cpu_s'],
                 'exam_s_per_unit': round(c['exam_predict_cpu_s'] / max(units, 1), 3), 'units': units, 'jobs': njobs,
                 'params': c['params'], 'params_note': c['params_note'],
                 'numpy': bool(c['numpy_only']), 'ung': ung,
                 'nse_main': R['boards']['main/ungauged'].get(k, {}).get('nse') if ung else None,
                 'nse_recent': R['boards']['recent/ungauged'].get(k, {}).get('nse') if ung else None})

fams = []
for c in T['families']:
    c = dict(c)
    c['link'] = FAMURL + '#fam-' + c['fam']
    c['color'] = arms[c['armid']]['color']
    fams.append(c)

DATA = {'run_id': R['run_id'], 'arms': arms, 'boards': boards, 'classtab': classtab, 'wins': wins, 'compute': comp,
        'show': SH, 'families': fams, 'findings': T['findings'], 'text': T['text']}
html = open(f'{P3}/page/template.html').read().replace('/*__DATA__*/null', json.dumps(DATA, separators=(',', ':')))
out = f'{P3}/page/river-bakeoff.html'
open(out, 'w').write(html)
print('wrote', out, os.path.getsize(out), 'bytes')
