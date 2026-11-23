"""THE EXAM. Run once, at the end, after the released rung and its widen factors are frozen. Chooses nothing.

usage: python exam.py <released weights.npz>
Writes work/exam.json and work/exam_preds.json; refuses to run twice (work/EXAM_STARTED).
Methods: AirfoilNet, NeuralFoil (xxxlarge, ncrit 9), raw physics (mfoil, one Re per airfoil, paired table),
thin-airfoil theory, and the dumb mean dev polar.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
from _paths import WORK, SPLITS   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'release')); sys.path.insert(2, os.path.abspath(ROOT))   # ROOT: airfoilnet.py in the repository layout
import realeval as rev  # noqa
import baselines as bl  # noqa
import airfoilnet as an  # noqa

FLAG = os.path.join(WORK, 'EXAM_STARTED')
if os.path.exists(FLAG):
    raise SystemExit('exam already started at ' + open(FLAG).read())
open(FLAG, 'w').write(time.strftime('%Y-%m-%d %H:%M:%S'))

units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
splits = json.load(open(SPLITS))
z = np.load(os.path.join(WORK, 'shapes.npz'))
pmap = {k: p for k, p in zip(z['tested_keys'], z['tested_p'])}
EX = splits['exam']

net = an.AirfoilNet(sys.argv[1])
def f_net(p, Re, al):
    r = net.predict(p, Re, al)
    return dict(cl=r['cl'], cd=r['cd'], cm=r['cm'], cl_sd=r['cl_sd'], lncd_sd=r['lncd_sd'], cm_sd=r['cm_sd'])
def f_net_raw(p, Re, al):
    r = net.predict(p, Re, al, raw_sigma=True)
    return dict(cl=r['cl'], cd=r['cd'], cm=r['cm'], cl_sd=r['cl_sd'], lncd_sd=r['lncd_sd'], cm_sd=r['cm_sd'])
nf = bl.NeuralFoilMethod('xxxlarge')
phys = bl.PhysicsMethod('p1_phys_exam', pmap)
thin = bl.thin_airfoil
mean = bl.MeanPolar(units, splits['dev'])
M = dict(AirfoilNet=f_net, NeuralFoil=nf, physics=phys, thin_airfoil=thin, mean_polar=mean)

out = dict(time=time.strftime('%Y-%m-%d %H:%M'), weights=os.path.basename(sys.argv[1]), n_exam_airfoils=len(EX))
# 1. all exam runs (physics excluded: it was run on one Re per airfoil)
out['all_runs'] = {}
for name in ('AirfoilNet', 'NeuralFoil', 'thin_airfoil', 'mean_polar'):
    s, per = rev.evaluate(M[name], units, EX, pmap)
    out['all_runs'][name] = dict(summary=s, per=per)
    print(name, {k: round(v, 4) for k, v in s.items() if isinstance(v, float)}, flush=True)
s, _ = rev.evaluate(f_net_raw, units, EX, pmap)
out['coverage_raw_sigma'] = s['coverage']
# 2. paired table on the runs the physics was run on
only = lambda aid, run: phys.has(aid, run['model'], run['Re'])
out['physics_subset'] = {}
for name in M:
    s, per = rev.evaluate(M[name], units, EX, pmap, only)
    out['physics_subset'][name] = dict(summary=s, per=per)
    print('subset', name, {k: round(v, 4) for k, v in s.items() if isinstance(v, float)}, flush=True)
# physics answer rate on its subset
ans = []
for aid in EX:
    for run in units[aid]['runs']:
        if only(aid, run):
            r = rev.eval_run(phys, pmap[f"{aid}|{run['model']}"], run)
            if r and 'answered' in r:
                ans.append(r['answered'])
out['physics_answer_rate'] = float(np.mean(ans)) if ans else None
# 3. Reynolds strata (all runs)
for lab, flt in (('Re_le_1e5', lambda a, r: r['Re'] <= 1.05e5), ('Re_gt_1e5', lambda a, r: r['Re'] > 1.05e5)):
    out[lab] = {}
    for name in ('AirfoilNet', 'NeuralFoil', 'mean_polar'):
        s, _ = rev.evaluate(M[name], units, EX, pmap, flt)
        out[lab][name] = s
json.dump(out, open(os.path.join(WORK, 'exam.json'), 'w'), indent=1, default=float)

# predictions for the page explorer: every exam airfoil, its first build, all its Re
preds = {}
g = np.arange(-8, 20.01, 0.5)
for aid in EX:
    u = units[aid]
    for run in u['runs']:
        key = f"{aid}|{run['model']}|{int(round(run['Re'] / 1e4) * 1e4)}"
        if key in preds:
            continue
        p = pmap[f"{aid}|{run['model']}"]
        r = net.predict(p, run['Re'], g); o = nf(p, run['Re'], g)
        ph = phys(p, run['Re'], g) if phys.has(aid, run['model'], run['Re']) else None
        preds[key] = dict(net=[r['cl'].round(4).tolist(), r['cd'].round(5).tolist(), r['cm'].round(4).tolist(),
                               r['cl_sd'].round(4).tolist(), r['lncd_sd'].round(4).tolist()],
                          nf=[o['cl'].round(4).tolist(), o['cd'].round(5).tolist(), o['cm'].round(4).tolist()],
                          phys=None if ph is None else [np.round(ph['cl'], 4).tolist(), np.round(ph['cd'], 5).tolist()])
json.dump(dict(grid=g.tolist(), preds=preds), open(os.path.join(WORK, 'exam_preds.json'), 'w'))
print('exam done')
