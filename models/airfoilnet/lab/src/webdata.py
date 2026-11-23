"""Build the AirfoilNet page: inject weights, airfoils, measured polars, baselines and results into template.html.
usage: python webdata.py <weights.npz> <out.html>
"""
import os, sys, json, base64, glob, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
from _paths import WORK   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'release')); sys.path.insert(2, os.path.abspath(ROOT))   # ROOT: airfoilnet.py in the repository layout
import airfoilnet as an  # noqa
import baselines as bl  # noqa
import realeval as rev  # noqa
import xsolver as xs  # noqa
import shapes as sh  # noqa

W = sys.argv[1]; OUT = sys.argv[2]
units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
splits = json.load(open(os.path.join(ROOT, 'splits.json')))
z = np.load(os.path.join(WORK, 'shapes.npz'))
pmap = {k: p for k, p in zip(z['tested_keys'], z['tested_p'])}
fitrms = {k: r for k, r in zip(z['tested_keys'], z['tested_rms'])}
J = lambda f: json.load(open(os.path.join(WORK, f))) if os.path.exists(os.path.join(WORK, f)) else None
b64 = lambda a: base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()

# ---- weights
zz = np.load(W); meta = json.loads(str(zz['meta']))
members = []
for m in range(meta['n_members']):
    mem = {}
    for part in ('branch', 'trunk', 'net'):
        L = []; i = 0
        while f'm{m}/{part}.{i}.w' in zz.files:
            w = zz[f'm{m}/{part}.{i}.w']; L.append(dict(w=b64(w), b=b64(zz[f'm{m}/{part}.{i}.b']), o=int(w.shape[0]), i=int(w.shape[1]))); i += 1
        if L:
            mem[part] = L
    if f'm{m}/bias' in zz.files:
        mem['bias'] = b64(zz[f'm{m}/bias'])
    members.append(mem)
weights = dict(arch=meta['arch'], anchor=meta['anchor'], P=meta['p_latent'], widen=meta.get('widen', [1, 1, 1]), members=members)

# ---- airfoils and runs
NAME = {}
for aid, u in units.items():
    NAME[aid] = re.sub(r'\s*\([A-Ea-e]\)\s*$', '', u['runs'][0]['label'])
grid = np.arange(-8, 20.01, 0.5)
nf = bl.NeuralFoilMethod('xxxlarge')
phys = {}
for tag in ('p1_twin_dev', 'p1_phys_exam'):
    phys.update(bl.load_sweeps(tag))
explorer = []; library = []
for aid in splits['dev'] + splits['exam']:
    u = units[aid]; split = 'dev' if aid in splits['dev'] else 'exam'
    byk = {}
    for r in u['runs']:
        k = (r['model'], round(r['Re'] / 1e4) * 1e4)
        e = byk.setdefault(k, dict(Re=float(k[1]), model=r['model']))
        if r['kind'] == 'lift' and 'lift' not in e:
            L = rev.upsweep(r['data'])
            e['lift'] = [[round(a, 2), round(c, 3), None if m is None or not np.isfinite(m) else round(m, 4)] for a, c, m in L.tolist()]
        elif r['kind'] == 'drag' and 'drag' not in e:
            e['drag'] = [[round(a, 2), round(c, 3), round(d, 5)] for a, c, d in r['data']]
    models = sorted({k[0] for k in byk})
    mdl = sorted(models, key=lambda m: -sum(1 for k in byk if k[0] == m))[0]
    p = pmap[f'{aid}|{mdl}']
    runs = []
    for k in sorted(byk, key=lambda k: (k[0] != mdl, k[0], k[1])):
        e = byk[k]; pk = pmap[f'{aid}|{k[0]}']
        o = nf(pk, e['Re'], grid)
        e['nf'] = [np.round(o['cl'], 4).tolist(), np.round(o['cd'], 5).tolist(), np.round(o['cm'], 4).tolist()]
        q = phys.get((aid, k[0], float(k[1])))
        if q is not None and len(q) >= 2:
            e['phys'] = [[None if not np.isfinite(v) else round(float(v), 4) for v in bl._interp_gap(grid, q[:, 0], q[:, 1])],
                         [None if not np.isfinite(v) else round(float(v), 5) for v in np.exp(bl._interp_gap(grid, q[:, 0], np.log(q[:, 2])))]]
        if k[0] != mdl:
            e['p'] = [round(float(v), 6) for v in pk]
        runs.append(e)
    explorer.append(dict(id=aid, name=NAME[aid], family=u['family'], split=split, models=len(models),
                         p=[round(float(v), 6) for v in p], fit_rms=float(fitrms[f'{aid}|{mdl}']), runs=runs))
    lr = [dict(Re=e['Re'], lift=e.get('lift'), drag=e.get('drag')) for e in runs if e['model'] == mdl]
    library.append(dict(group='Tested: dev airfoils' if split == 'dev' else 'Tested: exam airfoils', name=NAME[aid],
                        p=[round(float(v), 6) for v in p], runs=lr, Re=lr[0]['Re'] if lr else 2e5))
DB = os.path.join(ROOT, 'data', 'raw', 'coord_seligFmt', 'coord_seligFmt')
for f, nm in (('naca2412', 'NACA 2412'), ('n0012', 'NACA 0012'), ('naca4412', 'NACA 4412'), ('sd7034', 'SD7034'), ('ag25', 'AG25'), ('mh60', 'MH 60')):
    pth = os.path.join(DB, f + '.dat')
    if os.path.exists(pth):
        p, _ = sh.cst_fit(xs.normalize(xs.read_dat(pth)))
        library.append(dict(group='Library shapes (no tunnel data)', name=nm, p=[round(float(v), 6) for v in p], runs=[], Re=2e5))
# the explorer: runs reference their own p if the build differs
for e in explorer:
    for r in e['runs']:
        r.setdefault('p', None)

# ---- results
exam = J('exam.json'); rat = J('ratchet.json'); twin = J('twin_dev.json'); ver = J('verify_physics.json'); cal = J('calibration.json')
ASSET = lambda n: os.path.join(HERE, 'web_' + n) if os.path.exists(os.path.join(HERE, 'web_' + n)) else os.path.join(HERE, n)   # training/ names them web_*, lab/src/ not
TXT = json.load(open(ASSET('page_text.json')))
METH = [('AirfoilNet', 'AirfoilNet'), ('NeuralFoil', 'NeuralFoil'), ('physics', 'raw solver'), ('thin_airfoil', 'thin airfoil'), ('mean_polar', 'mean dev polar')]
ROWS = [('cl_pre', 'Cl error before stall', 3), ('cl_post', 'Cl error after stall', 3), ('clmax', 'Cl max error', 3),
        ('a_stall', 'stall angle error (deg)', 2), ('lncd', 'Cd error (|ln ratio|)', 3), ('cd_design', 'Cd at Cl 0.5 (|ln ratio|)', 3),
        ('cm_pre', 'Cm error before stall', 4)]
def mk_table(block, names):
    head = ['metric (mean |error|, per airfoil)'] + [dict(METH)[n] for n in names]; rows = []; best = []
    for k, lab, d in ROWS:
        vals = [(None if (n == 'thin_airfoil' and k in ('cl_post', 'clmax', 'a_stall')) else block[n]['summary'].get(k)) if n in block else None for n in names]
        row = [lab] + [('–' if v is None or not np.isfinite(v) else f'{v:.{d}f}') for v in vals]
        fin = [(v, j + 1) for j, v in enumerate(vals) if v is not None and np.isfinite(v)]
        best.append(min(fin)[1] if fin else -1); rows.append(row)
    return dict(head=head, rows=rows, best=best)
if exam:
    examTable = mk_table(exam['all_runs'], ['AirfoilNet', 'NeuralFoil', 'thin_airfoil', 'mean_polar'])
    physTable = mk_table(exam['physics_subset'], ['AirfoilNet', 'NeuralFoil', 'physics', 'thin_airfoil', 'mean_polar'])
else:
    examTable = dict(head=['metric'], rows=[['exam not run yet']], best=[-1]); physTable = examTable
if rat:
    rat2 = {r['id']: r for r in J('ratchet_val2.json')['log']}
    rel = rat['incumbent']
    ratchet = dict(head=['rung', 'change', 'skill val1 (p1)', 'skill val2 (p1c)', 'dev error D', 'decision', 'why (val2 decision)'], pillCol=5,
                   rows=[[r['id'], r['change'], f"{r['sim']:.3f}", f"{rat2[r['id']]['sim']:.3f}", f"{r['dev_D']:.3f}",
                          'released' if r['id'] == rel else r['decision'], rat2[r['id']]['why']] for r in rat['log']])
else:
    ratchet = dict(head=['rung'], rows=[], pillCol=-1)
# twin chart: physics Clmax and stall errors vs Re on dev
tw_rows = []; tw_series = []
if twin:
    S = bl.load_sweeps('p1_twin_dev'); pts = []
    for aid in splits['dev']:
        for run in units[aid]['runs']:
            key = (aid, run['model'], float(round(run['Re'] / 1e4) * 1e4))
            if key not in S or run['kind'] != 'lift':
                continue
            P = bl.PhysicsMethod.__new__(bl.PhysicsMethod); P.S = S; P.inv = {tuple(np.round(pmap[f"{aid}|{run['model']}"], 10)): f"{aid}|{run['model']}"}
            r = rev.eval_run(P, pmap[f"{aid}|{run['model']}"], run)
            if r and 'clmax_signed' in r:
                pts.append((run['Re'] / 1000, r['clmax_signed'], r['a_stall_signed'], r.get('answered', np.nan), NAME[aid]))
    pts.sort()
    if pts:
        tw_series = [dict(kind='line', x=[q[0] for q in pts], y=[q[1] for q in pts], label='solver Cl max − tunnel', w=0)]
    for k, v in twin['twin_vs_real']['per'].items():
        tw_rows.append([NAME[k], f"{v['cl_pre']:.3f}" if np.isfinite(v['cl_pre']) else '–',
                        f"{v['clmax_signed']:+.2f}" if np.isfinite(v['clmax_signed']) else '–',
                        f"{v['a_stall_signed']:+.1f}" if np.isfinite(v['a_stall_signed']) else '–',
                        f"{v['lncd_signed']:+.2f}" if np.isfinite(v['lncd_signed']) else '–'])
twinChart = dict(title='Solver minus tunnel, Cl max (dev)', ylab='ΔCl max', x=[0, 520], y=[-1.0, 0.6],
                 series=[dict(kind='dots', x=tw_series[0]['x'], y=tw_series[0]['y'], label='solver − tunnel')] if tw_series else [])
twinTable = dict(head=['dev airfoil', 'Cl err pre-stall', 'Cl max bias', 'stall angle bias (deg)', 'ln Cd bias'], rows=tw_rows)
dE = [i for i, e in enumerate(explorer) if e['id'] == 'e387']
dE = dE[0] if dE else 0
dR = int(np.argmin([abs(np.log10(r['Re'] / 2e5)) + (0 if r['model'] == explorer[dE]['runs'][0]['model'] else 1) for r in explorer[dE]['runs']]))
dP = [i for i, a in enumerate(library) if a['name'] == 'E387']
DATA = dict(defaultRun=dR, weights=weights, library=library, explorer=explorer, grid=grid.tolist(), defaultPreset=dP[0] if dP else 0,
            defaultExplorer=dE, examTable=examTable, physTable=physTable, ratchet=ratchet, twinChart=twinChart,
            twinTable=twinTable, text=TXT)
s = json.dumps(DATA, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
html = open(ASSET('template.html')).read().replace('/*DATA*/', s)
open(OUT, 'w').write(html)
print('wrote', OUT, round(len(html) / 1e6, 2), 'MB')
