"""Page/Python parity for the road law: the lab page's JS (the released rung and the road-load rival, as the page
builds them with new Function) against lawfinder.py and law_road.road_load on the page's 1,500 exam points.
Needs node on PATH.  python js_parity_road.py  -> results/js_parity_road.json"""
import json, os, subprocess, sys, tempfile, time
import numpy as np, pandas as pd
from _paths import MODEL, SRC
sys.path.insert(0, MODEL)
from lawfinder import LawFinder
import law_road as L

page = open(f'{MODEL}/lab/index.html', encoding='utf-8').read()
a = page.index('const DATA = ') + len('const DATA = ')
D = json.loads(page[a:page.index('\n', a)].rstrip().rstrip(';'))
R = next(x for x in D['laws'] if x['id'] == 'road')
V = [v['id'] for v in R['vars']]
P = R['exam_points']
js = f"""const R = {json.dumps({'vars': V, 'law': R['released']['js'], 'tb': R['textbook']['js'], 'TB': R['textbookArgs'], 'P': P})};
const f = new Function(...R.vars, 'return ' + R.law + ';'), g = new Function(...R.vars, 'TB', 'return ' + R.tb + ';');
const n = R.P[R.vars[0]].length, law = [], tb = [];
for (let i = 0; i < n; i++) {{ const x = R.vars.map(v => R.P[v][i]); law.push(f(...x)); tb.push(g(...x, R.TB)); }}
process.stdout.write(JSON.stringify({{law, tb}}));"""
with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as fh:
    fh.write(js)
out = json.loads(subprocess.run(['node', fh.name], capture_output=True, text=True, check=True).stdout)
os.unlink(fh.name)
law = LawFinder()['road']
kw = {v: np.asarray(P[v], float) for v in V}
py_law = np.asarray(law.predict(**kw)[0], float)
py_tb = L.road_load(pd.DataFrame(kw), R['textbookArgs'])
rel = lambda a, b: float(np.max(np.abs(np.asarray(a, float) / b - 1)))
res = {'law': 'road', 'rung': R['released']['id'], 'cases': len(py_law), 'law_max_rel_diff': rel(out['law'], py_law),
       'textbook_max_rel_diff': rel(out['tb'], py_tb), 'date': time.strftime('%Y-%m-%d'),
       'what': 'the page JS (released law and road-load rival) against lawfinder.py on the page exam points'}
json.dump(res, open(f'{SRC}/results/js_parity_road.json', 'w'))
print(res)
