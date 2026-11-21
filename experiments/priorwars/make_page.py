"""Build the self-contained 'Prior Wars' page from results.json (data inlined into the template).
python make_page.py -> WORK/page/prior-wars.html (pw_paths.WORK). Reads WORK/results.json if a rerun wrote one, else the
shipped results.json. The site serves the page as site/public/experiments/prior-wars/index.html: copy it there."""
import json
import os

import pw_paths as _P
ROOT = _P.WORK
R = json.load(open(_P.latest('results.json'), encoding='utf-8'))
NOTES = json.load(open(f'{_P.HERE}/page_notes.json', encoding='utf-8')) if os.path.exists(f'{_P.HERE}/page_notes.json') else {}
ARMS = ['P', 'G-dev', 'G-sim', 'G-sim+dev', 'C-dev']
models = []
for name, m in R['models'].items():
    rows = [r for r in R['rows'] if r['model'] == name]
    items, arms = [], []
    for r in rows:
        if r['item'] != '_headline' and r['item'] not in items:
            items.append(r['item'])
        if r['arm'] not in arms:
            arms.append(r['arm'])
    prim = {}
    for r in rows:
        if r['item'] != '_headline' and r['item'] not in prim:
            prim[r['item']] = r['metric']
    table = []
    for a in arms:
        cells = {}
        for r in rows:
            if r['arm'] == a and r['item'] != '_headline' and r['metric'] == prim[r['item']]:
                cells[r['item']] = [r['value'], r['ci95'][0], r['ci95'][1], r.get('verdict_vs_P')]
        hl = next((r for r in rows if r['arm'] == a and r['item'] == '_headline'), None)
        table.append({'arm': a, 'kind': rows[[x['arm'] for x in rows].index(a)]['kind'],
                      'labels': rows[[x['arm'] for x in rows].index(a)]['real_labels'], 'cells': cells,
                      'headline': [hl['value'], hl['ci95'][0], hl['ci95'][1]] if hl else None})
    con = {c['contrast']: {'arm': c['arm'], 'vs': c['vs'], 'd': c['headline_diff'], 'ci': c['ci95'],
                           'v': c['verdict_for_arm'],
                           'items': {k: v['verdict'] for k, v in c['per_item'].items()}}
           for c in R['contrasts'] if c['model'] == name}
    lc = [{'k': x['k'], 'h': x['headline_mean'], 'sd': x['headline_sd_over_3_subsets'], 'all': x['is_full_dev']}
          for x in R['lc'] if x['model'] == name]
    models.append({'id': name, 'title': m['title'], 'headline': m['headline'], 'rows': m['n_exam_rows'],
                   'units': m['n_units'], 'unit': m['unit'], 'dev': m['n_dev'], 'kstar': m['k_star'],
                   'items': items, 'metrics': prim, 'table': table, 'con': con, 'lc': lc,
                   'note': NOTES.get(name, '')})
data = {'models': models, 'verdict': R['verdict'], 'notes': NOTES, 'where_wins': R.get('where_wins', {})}
tpl = open(f'{_P.HERE}/page_template.html', encoding='utf-8').read()
os.makedirs(f'{ROOT}/page', exist_ok=True)
out = tpl.replace('/*__DATA__*/null', json.dumps(data, default=float).replace('</', '<\\/'))
open(f'{ROOT}/page/prior-wars.html', 'w', encoding='utf-8', newline='\n').write(out)
print('page', len(out), 'bytes;', len(models), 'models')
