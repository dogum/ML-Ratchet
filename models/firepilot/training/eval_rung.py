"""Score a planner on a frozen set (val = simulated, gate = real dev fires), resumable per case.

  python3 eval_rung.py <rung> <set> [n]      rung: r0 (proxy) | a model name in models/ | mpc (forecast-world CEM)
  -> results/eval_<set>_<physics>_<rung>.jsonl (physics: FIREPILOT_PHYSICS, default p1), summary printed (mean
     saved fraction of the no-action growth, with baselines on the same cases and the perfect-foresight bound where
     it exists).
"""
import os
import sys
import json
import time

import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
import numpy as np

from fp.planner import Context, ValueNet, plan_value, plan_proxy, cem_search, BASELINES
from bank import case_dir, rebuild, VERSION

ROOT = _paths.WORK


def cases(setname):
    d = case_dir(setname)
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.json') and not fn.endswith('.pf.json'):
            rec = json.load(open(os.path.join(d, fn)))
            if 'skip' not in rec:
                out.append(rec)
    return out


def choose(rung, ctx, seed, S=None):
    rs = np.random.default_rng(seed)
    if rung == 'r0':
        return plan_proxy(ctx, rs)
    if rung == 'mpc':
        p, info = cem_search(S, ctx.fr, 'forecast', rs, iters=3, pop=8)
        return p, info
    net = ValueNet(os.path.join(ROOT, 'models', rung + '.npz'))
    return plan_value(ctx, net, rs)


def run(rung, setname, n=None):
    d = case_dir(setname)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    fn = os.path.join(ROOT, 'results', f'eval_{setname}_{VERSION}_{rung}.jsonl')
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)['name'] for l in open(fn)}
    cs = cases(setname)[:n] if n else cases(setname)
    for k, rec in enumerate(cs):
        if rec['name'] in done:
            continue
        t0 = time.time()
        z = np.load(os.path.join(d, rec['name'] + '.npz'))
        ctx = Context.from_store(z, rec)
        S = None
        if rung == 'r0' and any(r['tag'] == 'r0' for r in rec['rows']):
            r = [r for r in rec['rows'] if r['tag'] == 'r0'][0]
            out = dict(name=rec['name'], plan=r['plan'], saved=r['saved'], area=r['area'], km=r['km'],
                       pred=r['s_proxy'], stored=True)
        else:
            S = rebuild(setname, rec)
            S._none = dict(arr=None, fl=z['fl_t'].astype(np.float64), area=rec['area_none'])
            t1 = time.time()
            p, info = choose(rung, ctx, k, S)
            tplan = time.time() - t1
            r = S.evaluate(ctx.fr, p)
            out = dict(name=rec['name'], plan=p, saved=r['saved'], area=r['area'], km=r['km'],
                       pred=info.get('pred'), sigma=info.get('sigma'), plan_s=tplan,
                       n_sims=info.get('n_sims', 0))
        out['wall_s'] = time.time() - t0
        with open(fn, 'a') as f:
            f.write(json.dumps(out) + '\n')
        print(setname, rung, rec['name'], 'saved %.3f' % out['saved'], '%.1fs' % out['wall_s'], flush=True)
    return summary(rung, setname, n)


def summary(rung, setname, n=None):
    fn = os.path.join(ROOT, 'results', f'eval_{setname}_{VERSION}_{rung}.jsonl')
    ev = {json.loads(l)['name']: json.loads(l) for l in open(fn)}
    cs = [c for c in (cases(setname)[:n] if n else cases(setname)) if c['name'] in ev]
    d = case_dir(setname)
    out = dict(rung=rung, set=setname, n=len(cs), saved_mean=float(np.mean([ev[c['name']]['saved'] for c in cs])),
               saved_median=float(np.median([ev[c['name']]['saved'] for c in cs])))
    for b in BASELINES:
        v = [[r['saved'] for r in c['rows'] if r['tag'] == 'base:' + b][0] for c in cs]
        out['base_' + b] = float(np.mean(v))
    pf = [json.load(open(os.path.join(d, c['name'] + '.pf.json')))['saved'] for c in cs
          if os.path.exists(os.path.join(d, c['name'] + '.pf.json'))]
    if pf:
        names = [c['name'] for c in cs if os.path.exists(os.path.join(d, c['name'] + '.pf.json'))]
        out['pf_n'] = len(pf)
        out['pf_mean'] = float(np.mean(pf))
        out['rung_on_pf_cases'] = float(np.mean([ev[k]['saved'] for k in names]))
    return out


if __name__ == '__main__':
    rung, setname = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(json.dumps(run(rung, setname, n), indent=1))
