"""Assemble the Vehicle Bake-off page: template.html + analysis.json + the exam's showcase (json + npz) + text.json
-> vehicle-bakeoff.html. Reads only exam outputs, arm metadata and the frozen showcase draw.
usage: python build_page.py [--rehearsal]
Reads the newest analysis (work/phase3/, written by analyze.py; else the shipped results/) and writes
work/phase3/vehicle-bakeoff.html (the artifact page) and work/phase3/site-index.html (the same page with a document
head, as served at site/public/experiments/vehicle-bakeoff/index.html). Copy them there by hand once checked.
"""
import json
import os
import sys

import numpy as np

P3 = os.path.dirname(os.path.abspath(__file__))  # cloud: /home/claude/bakeoff/vehicle/phase3
BK = os.path.dirname(P3)                          # experiments/bakeoff-vehicle: results/ sits beside harness/ in the repo
H = os.path.join(BK, 'harness')                     # cloud: /home/claude/bakeoff/vehicle/harness
WK = os.path.join(BK, 'work', 'phase3')
REH = '--rehearsal' in sys.argv
_an = f'{"rehearsal_" if REH else ""}analysis.json'
R = json.load(open(os.path.join(WK, _an) if os.path.exists(os.path.join(WK, _an)) else os.path.join(BK, 'results', _an)))
od = os.path.join(BK, R['source'][:-5])
SHJ = json.load(open(os.path.join(od, 'showcase.json')))
SHN = np.load(os.path.join(od, 'showcase.npz'))
DRAW = {t['trip_id']: t for t in json.load(open(f'{H}/showcase.json'))['trips']}
T = json.load(open(f'{P3}/page_text.json', encoding='utf-8'))
FAMURL = 'https://dogum.github.io/ML-Ratchet/model-families/'

ARMS = {   # id: (short, label, color token)
    'enet_v21': ('EnergyNet v2.1', 'EnergyNet v2.1 (5 MLPs on 151 features, as released)', '--ink'),
    'tab_mlp': ('MLP', 'MLP, EnergyNet net retrained (table control)', '--s1'),
    'tab_gbm': ('Trees', 'Gradient-boosted trees (LightGBM)', '--s2'),
    'tab_gp': ('Sparse GP', 'Sparse Gaussian process (500 inducing points)', '--s3'),
    'formula': ('Formula', 'Trip-energy law (symbolic regression)', '--s4'),
    'trace_cnn': ('CNN', '1-D dilated CNN on the trace', '--s5'),
    'trace_gru': ('GRU', 'Bidirectional GRU on the trace (trace control)', '--s6'),
    'trace_s4d': ('S4D', 'S4D state space on the trace', '--s8'),
    'trace_transformer': ('Transformer', 'Transformer on the trace (20-s patches)', '--s7'),
    'phys_anchor': ('E_phys anchor', 'Physics anchor: E_phys alone + simulated bands', '--ref'),
    'climatology': ('Climatology', 'Climatology (sim median MJ/km x distance)', '--ref'),
    'ephys': ('E_phys', 'E_phys alone (road-load, no bands)', '--ref'),
    'moves_like': ('MOVES-like', 'MOVES-like VSP model (dev-fitted)', '--ref'),
    'label': ('Label x km', 'Window label x distance (dev-fitted)', '--ref'),
    'class_median': ('Class median', 'Class median per km (dev)', '--ref'),
    'regression': ('EPA regression', 'EPA regression (dev-fitted)', '--ref'),
    'dev_median': ('Dev median', 'Dev median (EPA / heavy duty)', '--ref'),
    'classical': ('Classical', 'Classical composite (MOVES-like on VED, regression on EPA, dev median on HD)', '--ref'),
}
MID = {'enet_v21': 'EnergyNet v2.1', 'tab_mlp': 'MLP retrained (control)', 'tab_gbm': 'Gradient-boosted trees',
       'tab_gp': 'Sparse GP', 'formula': 'Trip-energy law', 'trace_cnn': '1-D CNN', 'trace_gru': 'GRU (control)',
       'trace_s4d': 'S4D state space', 'trace_transformer': 'Transformer', 'phys_anchor': 'E_phys anchor + bands',
       'climatology': 'Climatology', 'ephys': 'E_phys alone', 'moves_like': 'MOVES-like (dev-fitted)',
       'label': 'Label x distance (dev-fitted)', 'class_median': 'Class median (dev)', 'regression': 'EPA regression (dev-fitted)',
       'dev_median': 'Dev median', 'classical': 'Classical composite (dev-fitted)'}
arms = {k: {'short': s, 'label': l, 'color': c, 'mid': MID.get(k, s)} for k, (s, l, c) in ARMS.items()}
for k in list(arms):
    if k not in ('ephys', 'moves_like', 'label', 'class_median', 'regression', 'dev_median', 'classical', 'climatology'):
        a = dict(arms[k]); a['short'] += ' + conf.'; a['label'] += ' + conformal'; a['mid'] += ' + conformal'; arms[k + '_conformal'] = a

SHOW_ORDER = ['enet_v21', 'tab_mlp', 'tab_gbm', 'tab_gp', 'trace_gru', 'trace_cnn', 'trace_s4d', 'trace_transformer',
              'formula', 'phys_anchor', 'climatology', 'moves_like', 'label']
SHOW_DEFAULT = ['enet_v21', 'tab_mlp', 'tab_gbm', 'trace_gru', 'formula', 'phys_anchor']
TRACK_OF = {'enet_v21': 'table', 'tab_mlp': 'table', 'tab_gbm': 'table', 'tab_gp': 'table', 'formula': 'table',
            'phys_anchor': 'table', 'climatology': 'table', 'trace_gru': 'trace', 'trace_cnn': 'trace', 'trace_s4d': 'trace',
            'trace_transformer': 'trace'}
STRATUM_TITLE = {'city': 'City errand, petrol car', 'highway': 'Highway run, petrol car', 'hilly': 'Hilly drive, hybrid',
                 'cold': 'Cold morning, petrol car', 'hybrid': 'Town trip, power-split hybrid', 'ev': 'Deep-cold drive, Nissan Leaf',
                 'plugin': 'Plug-in hybrid trip', 'bus': 'Zurich electric bus leg'}
ARCH = {'conv': 'conventional petrol car', 'hev_powersplit': 'power-split hybrid', 'bev': 'battery-electric',
        'phev_powersplit': 'plug-in power-split hybrid', 'phev_series': 'plug-in series hybrid (Volt-type)',
        'hev_parallel_p2': 'P2 hybrid', 'hev_series': 'series hybrid'}


# ---------------------------------------------------------------- showcase
def show():
    out = []
    for e in SHJ:
        tid = e['trip_id']
        dr = DRAW.get(tid, {})
        prof = SHN[f'profile|{tid}']
        rows = {}
        for k in SHOW_ORDER:
            for suf in ('', '_conformal'):
                kk = k + suf
                key = f'{kk}|{TRACK_OF.get(k, "ref")}' if k in TRACK_OF else f'{k}|ref'
                x = e['rows'].get(key)
                if not x:
                    continue
                r = {'pred': round(x['pred'], 4)}
                if 'q05' in x:
                    for q in ('q05', 'q25', 'q75', 'q95'):
                        r[q] = round(x[q], 4)
                c = x.get('components')
                if c and all(np.isfinite(c)) and suf == '':
                    r['comp'] = [round(v, 4) for v in c]
                rows[kk] = r
        arch = e.get('arch') or dr.get('arch', '')
        stratum = dr.get('stratum', 'trip')
        unit = 'kWh' if arch == 'bev' or stratum == 'bus' else ('MJ' if arch.startswith('phev') else 'L')
        dur = dr.get('dur_s', len(prof) * 5)
        dist = dr.get('dist_km', e.get('dist_km'))
        out.append({'trip_id': tid, 'stratum': stratum, 'title': STRATUM_TITLE.get(stratum, tid),
                    'vehicle': ARCH.get(arch, arch) if stratum != 'bus' else 'battery-electric articulated bus (ZTBus)',
                    'dist_km': float(dist), 'dur_s': float(dur), 'kmh': float(dr.get('kmh', dist / max(dur, 1) * 3600)),
                    'T': float(dr.get('T', e.get('T', np.nan))), 'climb': float(dr.get('climb_per_km', np.nan)) if dr.get('climb_per_km') is not None else 0.0,
                    'y': float(e['y_MJ']), 'vr': float(e.get('vr', 1)), 'unit': unit, 'note': T.get('trip_notes', {}).get(stratum, ''),
                    'profile': {'v': [round(float(x), 2) for x in prof[:, 0]], 'g': [round(float(x), 4) for x in prof[:, 1]],
                                'e': [round(float(x), 1) for x in prof[:, 2]], 'T': [round(float(x), 1) for x in prof[:, 3]]},
                    'rows': rows})
    return out


# ---------------------------------------------------------------- boards
PT_COLS = [('conventional', 'Conv.'), ('hybrid', 'Hybrid'), ('plugin', 'Plug-in'), ('bev', 'BEV'), ('heavy', 'Heavy')]
SETS = [('VED_ICE', 'VED petrol trips'), ('VED_HEV', 'VED hybrid trips'), ('VED_EV', 'VED Leaf trips'), ('VED_PHEV', 'VED plug-in trips'),
        ('VED_ICE_cars', 'VED petrol, car totals'), ('VED_HEV_cars', 'VED hybrid, car totals'), ('VED_EV_cars', 'VED Leaf, car totals'),
        ('VED_PHEV_cars', 'VED plug-in, car totals'),
        ('EPA_conv_FTP', 'EPA conventional FTP (city)'), ('EPA_conv_HWY', 'EPA conventional highway'), ('EPA_conv_US06', 'EPA conventional US06'),
        ('EPA_conv_FTPcold', 'EPA conventional 20 °F cold FTP'), ('EPA_conv_SC03', 'EPA conventional SC03 (A/C)'),
        ('EPA_hev_FTP', 'EPA hybrid FTP'), ('EPA_hev_HWY', 'EPA hybrid highway'), ('EPA_hev_US06', 'EPA hybrid US06'),
        ('EPA_hev_FTPcold', 'EPA hybrid cold FTP'), ('EPA_hev_SC03', 'EPA hybrid SC03'),
        ('EPA_bev_FTP', 'EPA BEV city'), ('EPA_bev_HWY', 'EPA BEV highway'),
        ('Mersin_diesel', 'Mersin diesel buses'), ('Mersin_cng', 'Mersin CNG buses'), ('ICCT', 'ICCT tractor-trailers'),
        ('ZTBus_legs', 'ZTBus legs (v1 subset)'), ('ZTBus_missions', 'ZTBus missions (v1 subset)')]


def verdict(v):
    return None if not v else {'diff': v['diff'], 'ci95': v['ci95'], 'verdict': v['verdict']}


def boards():
    B, SB, U = {}, {}, {}
    for track, tab in R['boards'].items():
        ctrl = 'tab_mlp' if track == 'table' else 'trace_gru'
        rows = []
        order = sorted(tab, key=lambda k: (tab[k]['kind'] == 'reference' and k != 'enet_v21' and not k.startswith('enet_v21'),
                                           k.endswith('_conformal'), tab[k]['E16'] if tab[k]['E16'] is not None else 9))
        # conformal rows directly under their base
        base_order = [k for k in order if not k.endswith('_conformal')]
        final = []
        for k in base_order:
            final.append(k)
            if k + '_conformal' in tab:
                final.append(k + '_conformal')
        for k in final:
            d = tab[k]
            rows.append({'id': k, 'base': d['base'], 'label': arms.get(k, {'label': d['label']})['label'], 'kind': d['kind'],
                         'conformal': d['conformal'], 'E16': d['E16'], 'vs_enet': verdict(d['vs_E16'].get('enet_v21', {}).get('E16')),
                         'vs_ctrl': verdict(d['vs_E16'].get(ctrl, {}).get('E16')), 'pt': d['powertrain'],
                         'cov50': d.get('E16_cov50'), 'cov90': d.get('E16_cov90'), 'crps': d['E16_crps']})
        B[track] = rows
        SB[track] = {}
        for st, _ in SETS:
            L = []
            for k in final:
                d = tab[k]; s = d['sets'].get(st)
                if not s:
                    continue
                vs = d['vs_set'].get(st, {}).get('enet_v21')
                L.append({'id': k, 'base': d['base'], 'label': arms.get(k, {'label': d['label']})['label'], 'kind': d['kind'],
                          'n': s['n'], 'MdAPE': s['MdAPE'], 'bias': s['bias'], 'cov50': s.get('cov50'), 'cov90': s.get('cov90'),
                          'crps_log': s['crps_log'], 'vs': None if not vs else {'d': vs['d_MdAPE'], 'ci': vs['ci'], 'verdict': vs['verdict']}})
            L.sort(key=lambda r: (r['kind'] == 'reference' and not r['id'].startswith('enet_v21'), r['MdAPE'] if r['MdAPE'] is not None else 9, r['id']))
            SB[track][st] = L
        U[track] = []
        for k in final:
            if not k.endswith('_conformal') or k[:-10] not in tab:
                continue
            b = tab[k[:-10]]; c = tab[k]
            U[track].append({'id': k[:-10], 'label': arms.get(k[:-10], {'label': b['label']})['label'], 'raw50': b.get('E16_cov50'),
                             'raw90': b.get('E16_cov90'), 'c50': c.get('E16_cov50'), 'c90': c.get('E16_cov90'),
                             'rawc': b['E16_crps'], 'cc': c['E16_crps'], 'd': verdict(c['vs_E16'].get('raw', {}).get('E16_crps'))})
    return B, SB, U


# ---------------------------------------------------------------- wins
WCOLS = [('VED_ICE', 'VED petrol', 'pt'), ('VED_HEV', 'VED hybrid', 'pt'), ('VED_EV', 'VED Leaf', 'pt'), ('VED_PHEV', 'VED plug-in', 'pt'),
         ('EPA_conv_FTP', 'EPA FTP', 'pt'), ('EPA_conv_HWY', 'EPA HWY', 'pt'), ('EPA_conv_US06', 'EPA US06', 'pt'),
         ('EPA_conv_FTPcold', 'EPA cold', 'pt'), ('EPA_conv_SC03', 'EPA SC03', 'pt'), ('EPA_hev_FTP', 'Hybrid FTP', 'pt'),
         ('EPA_hev_HWY', 'Hybrid HWY', 'pt'), ('EPA_bev_FTP', 'BEV city', 'pt'), ('EPA_bev_HWY', 'BEV HWY', 'pt'),
         ('Mersin_diesel', 'Bus diesel', 'pt'), ('Mersin_cng', 'Bus CNG', 'pt'), ('ICCT', 'Trucks', 'pt'), ('ZTBus_legs', 'E-bus legs', 'pt'),
         ('city', 'City < 40 km/h', 'cond'), ('highway', 'Highway > 55', 'cond'), ('cold', 'Cold < 0 °C', 'cond'), ('hot', 'Hot > 25 °C', 'cond'),
         ('hilly', 'Hilly ≥ 15 m/km', 'cond'), ('short', 'Short < 2 km', 'cond'), ('long', 'Long > 15 km', 'cond')]


def wins():
    W = R['wins']
    cols, rows, refs = [], [], []
    for k, lab, grp in WCOLS:
        if k not in W:
            continue
        w = W[k]
        long = dict(SETS).get(k, lab)
        cols.append({'key': k, 'label': lab, 'long': long, 'grp': grp, 'n': w['n'], 'md': w['MdAPE'], 'ref': w.get('ref_MdAPE', {}),
                     'best': w['best'], 'tied': w['tied'], 'best_ref': w.get('best_ref'), 'vsref': w.get('best_vs_best_ref')})
        for r in w['MdAPE']:
            if r not in rows:
                rows.append(r)
        for r in w.get('ref_MdAPE', {}):
            if r not in refs:
                refs.append(r)
    pref = ['enet_v21|table', 'tab_mlp|table', 'tab_gbm|table', 'tab_gp|table', 'trace_gru|trace', 'trace_cnn|trace', 'trace_s4d|trace',
            'trace_transformer|trace', 'formula|table', 'phys_anchor|table']
    rows = [r for r in pref if r in rows] + [r for r in rows if r not in pref]
    return {'cols': cols, 'rows': rows, 'refs': refs}


# ---------------------------------------------------------------- compute
LAB = {'enet_v21': (None, None)}


def compute():
    out = []
    order = ['tab_mlp', 'tab_gbm', 'tab_gp', 'formula', 'phys_anchor', 'trace_gru', 'trace_cnn', 'trace_s4d', 'trace_transformer',
             'enet_v21', 'climatology']
    for k in order:
        c = R['compute'].get(k)
        if not c:
            continue
        sv = R['simval'].get(k, {})
        tr = 'table' if k in ('tab_mlp', 'tab_gbm', 'tab_gp', 'enet_v21') else ('trace' if k.startswith('trace') else 'both')
        e16 = (c['E16'] or {}).get('table' if tr in ('table', 'both') else 'trace')
        es = c['exam_predict_cpu_s'] or {}
        out.append({'id': k, 'track': tr, 'train_h': c['train_cpu_hours'], 'exam_s': ' + '.join(f'{v:.0f}' for v in es.values() if v is not None),
                    'params': c['params'], 'sim_skill': sv.get('simval_skill'), 'd16': sv.get('dev_D16'), 'e16': e16,
                    'plot': k not in ('enet_v21', 'climatology') and e16 is not None and c['train_cpu_hours'] is not None,
                    **T.get('cvs_labels', {}).get(k, {})})
    return out


def families():
    out = []
    for c in T['families']:
        c = dict(c)
        c['link'] = FAMURL + '#fam-' + c['fam']
        out.append(c)
    return out


B, SB, U = boards()
DATA = {'run_id': R['run_id'], 'arms': arms, 'show': show(), 'show_order': SHOW_ORDER, 'show_default': SHOW_DEFAULT,
        'comp_names': ['aerodynamic drag', 'rolling resistance', 'friction brakes', 'auxiliaries / HVAC', 'powertrain losses'],
        'boards': B, 'control': {'table': 'tab_mlp', 'trace': 'trace_gru'}, 'pt_cols': PT_COLS, 'sets': SETS,
        'setboards': SB, 'set_notes': T.get('set_notes', {}), 'unc': U, 'wins': wins(), 'families': families(),
        'compute': compute(), 'enet_e16': R['boards']['table'].get('enet_v21', {}).get('E16'),
        'findings': T['findings'], 'river': T['river'], 'limits': T['limits'], 'text': T['text']}
html = open(f'{P3}/page_template.html', encoding='utf-8').read().replace('/*__DATA__*/null', json.dumps(DATA, separators=(',', ':'), allow_nan=False))
os.makedirs(WK, exist_ok=True)
out = os.path.join(WK, f'{"rehearsal-" if REH else ""}vehicle-bakeoff.html')
open(out, 'w', encoding='utf-8', newline='\n').write(html)
open(os.path.join(WK, 'site-index.html'), 'w', encoding='utf-8', newline='\n').write(
    '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n' + html)
print('wrote', out, os.path.getsize(out), 'bytes')
