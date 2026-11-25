"""LawFinder-style symbolic search for one powertrain group's trip-energy law on the simulated TRAIN bank only.

usage: python3 search_law.py <tag> <group> <seconds> [--vars p,n,t,m,T,c,d] [--maxsize 25] [--seed 0] [--ld-only]
Writes work/search/<tag>_<group>.json (Pareto front, constants refitted on every train row of the group) and
logs/search_<tag>_<group>.log. Resumable: an existing output is not recomputed.
Data: the 28,800-trip train bank seen with two independent tier draws (rng 11, 12), target = bake.target (EnergyNet
v2.1's version) per km; rows with y > 0.02 MJ and distance > 0.3 km. Loss: squared log error (lawsearch log_target).
Planted seeds: the road-load estimate's own forms (E_phys: e = p / eta + k m^0.7 t, with regen terms for hybrids
and BEVs), so the front starts at road-load and the search can only add to it.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import bake  # noqa: E402
import lawsearch as ls  # noqa: E402
import lawvars as LV  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'work', 'search')
ARCHS_OF = {'ice': [0], 'hev': [1, 2, 3], 'bev': [6], 'phev': [4, 5]}


def train_rows(group, ld_only=False, seeds=(11, 12)):
    B = bake.load_bank('train')
    Vs, ys = [], []
    for s in seeds:
        T = bake.draw_tiers(B, np.random.default_rng(s))
        D = bake.sim_table(B, T)
        V = LV.from_table(D['X'])
        ok = (D['y'] > 0.02) & (V['dist_km'] > 0.3) & np.isin(D['arch'], ARCHS_OF[group])
        if ld_only:
            ok &= D['cls'] < 4
        Vs.append({k: v[ok] for k, v in V.items()}); ys.append(D['y'][ok])
    V = {k: np.concatenate([v[k] for v in Vs]) for k in Vs[0]}
    y = np.concatenate(ys)
    return V, y


def road_load_seeds(names, group):
    ix = {k: i for i, k in enumerate(names)}
    x = lambda k: ('x', ix[k])
    out = []
    if all(k in ix for k in ('p', 't', 'm')):
        idle = ('*', ('*', ('c', 1), ('^', x('m'), ('c', 2))), x('t'))
        if group == 'ice':
            out.append((('+', ('*', ('c', 0), x('p')), idle), [3.5, 0.28, 0.7]))
        elif 'n' in ix:
            out.append((('+', ('*', ('c', 0), ('-', x('p'), ('*', ('c', 3), x('n')))), idle),
                        [3.3 if group == 'hev' else 1.12, 0.1 if group == 'hev' else 0.034, 0.7,
                         0.38 if group == 'hev' else 0.56]))
        out.append((('+', ('*', ('c', 0), x('p')), ('*', ('c', 1), x('t'))), [3.0, 0.2]))
    return [ls.renumber(t, np.array(c, float)) for t, c in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag'); ap.add_argument('group'); ap.add_argument('seconds', type=float)
    ap.add_argument('--vars', default=','.join(LV.VARS)); ap.add_argument('--maxsize', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ld-only', action='store_true')
    ap.add_argument('--n-fit', type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    fn = os.path.join(OUT, f'{a.tag}_{a.group}.json')
    if os.path.exists(fn):
        print('exists', fn); return
    c0 = time.process_time(); t0 = time.time()
    names = a.vars.split(',')
    V, y = train_rows(a.group, a.ld_only)
    d = dict(V); d['d'] = V['dist_km']
    X = np.stack([d[k] for k in names], 1)
    e = y / V['dist_km']
    print(f'{a.group}: {len(e)} train rows, vars {names}', flush=True)
    log = open(os.path.join(ROOT, 'logs', f'search_{a.tag}_{a.group}.log'), 'a')
    s = ls.Search(names, seed=a.seed, fixed_seeds=road_load_seeds(names, a.group), maxsize=a.maxsize,
                  n_fit=a.n_fit, log_target=True)
    s.fit(X, e, None, seconds=a.seconds, verbose=True, log=log)
    n = min(len(e), 30000)
    idx = np.random.default_rng(a.seed + 1).permutation(len(e))[:n]
    front = s.front(X[idx], e[idx], None, refit=True)
    laws = []
    for (cx, loss, t, c) in front:
        t2, c2 = ls.renumber(t, c)
        laws.append({'vars': names, 'tree': ls.to_json(t2), 'consts': [float(v) for v in c2], 'complexity': ls.size(t2),
                     'train_loss': float(loss), 'expr': ls.to_str(t2, c2, names, 4)})
    out = {'tag': a.tag, 'group': a.group, 'vars': names, 'ld_only': a.ld_only, 'n_train': int(len(e)),
           'seconds': time.time() - t0, 'cpu_s': time.process_time() - c0, 'n_eval': s.n_eval,
           'maxsize': a.maxsize, 'seed': a.seed, 'front': laws}
    json.dump(out, open(fn + '.tmp', 'w'), indent=1)
    os.replace(fn + '.tmp', fn)
    for L in laws:
        print(f"c={L['complexity']:2d} loss={L['train_loss']:.5f}  {L['expr']}", flush=True)
    print(f'done {fn} cpu {out["cpu_s"]:.0f} s', flush=True)


if __name__ == '__main__':
    main()
