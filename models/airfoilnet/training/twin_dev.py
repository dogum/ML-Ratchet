"""Twin test on the dev airfoils: the simulator's own polar (the 'twin') against the wind tunnel, and,
when a model is given, the network against both. Finds where XFOIL-class physics fails.
usage: python twin_dev.py [weights.npz]   -> work/twin_dev.json (+ printed table)
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
from _paths import WORK   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'release')); sys.path.insert(2, os.path.abspath(ROOT))   # ROOT: airfoilnet.py in the repository layout
import realeval as rev  # noqa
import baselines as bl  # noqa

units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
splits = json.load(open(os.path.join(ROOT, 'splits.json')))
z = np.load(os.path.join(WORK, 'shapes.npz'))
pmap = {k: p for k, p in zip(z['tested_keys'], z['tested_p'])}
phys = bl.PhysicsMethod('p1_twin_dev', pmap)
only = lambda aid, run: phys.has(aid, run['model'], run['Re'])
out = {}
summ, per = rev.evaluate(phys, units, splits['dev'], pmap, only)
out['twin_vs_real'] = dict(summary=summ, per=per)
print('TWIN (physics) vs REAL dev:', {k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items()})
for aid, v in per.items():
    print(f'  {aid:10s}', ' '.join(f'{k}={v[k]:.3f}' for k in rev.KEYS + ['clmax_signed', 'a_stall_signed', 'lncd_signed']))
# Reynolds-number breakdown of the physics gap
byre = {}
for aid in splits['dev']:
    for run in units[aid]['runs']:
        if not only(aid, run):
            continue
        r = rev.eval_run(phys, pmap[f"{aid}|{run['model']}"], run)
        if r:
            b = 'Re<=1e5' if run['Re'] <= 1.05e5 else 'Re>1e5'
            for k in ('cl_pre', 'clmax_signed', 'a_stall_signed', 'lncd_signed', 'answered', 'answered_d'):
                if k in r and np.isfinite(r[k]):
                    byre.setdefault(b, {}).setdefault(k, []).append(r[k])
out['by_re'] = {b: {k: float(np.mean(v)) for k, v in d.items()} for b, d in byre.items()}
print('by Re:', json.dumps(out['by_re'], indent=0))
if len(sys.argv) > 1:
    import airfoilnet as an
    net = an.AirfoilNet(sys.argv[1])
    def fnet(p, Re, al):
        r = net.predict(p, Re, al); return dict(cl=r['cl'], cd=r['cd'], cm=r['cm'])
    s_real, _ = rev.evaluate(fnet, units, splits['dev'], pmap, only)
    # network vs its twin: treat the twin sweep as 'measured'
    def twin_as_runs(aid):
        rs = []
        for (a, m, Re), q in phys.S.items():
            if a != aid or len(q) < 4:
                continue
            rs.append(dict(kind='lift', model=m, Re=Re, data=q[:, [0, 1, 3]].tolist()))
            rs.append(dict(kind='drag', model=m, Re=Re, data=q[:, [0, 1, 2]].tolist()))
        return rs
    tunits = {aid: dict(runs=twin_as_runs(aid)) for aid in splits['dev']}
    s_twin, _ = rev.evaluate(fnet, tunits, splits['dev'], pmap)
    out['net_vs_real'] = s_real; out['net_vs_twin'] = s_twin
    print('NET vs REAL:', {k: round(v, 4) for k, v in s_real.items() if isinstance(v, float)})
    print('NET vs TWIN:', {k: round(v, 4) for k, v in s_twin.items() if isinstance(v, float)})
json.dump(out, open(os.path.join(WORK, 'twin_dev.json'), 'w'), indent=1, default=float)
