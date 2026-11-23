"""Simulated polar banks for AirfoilNet (resumable, one thread, run as several workers).

usage: python simbank.py <set> <worker> <nworkers> [physics]
  set: train | val | twin_dev | phys_exam
  physics: p1 (default) ncrit = 9 everywhere
           p2 ncrit drawn per sweep from U(5, 11) (tunnel turbulence nuisance; dev-motivated, see NOTEBOOK)
Each task is one alpha sweep of one shape at one Reynolds number; output lines go to
work/bank/<physics>_<set>_w<worker>.jsonl and finished task ids are skipped on restart.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(HERE, '..', 'numba_cache'))
import xsolver as xs  # noqa
import shapes as sh  # noqa

ROOT = os.path.join(HERE, '..')
OUTD = os.path.join(ROOT, 'work', 'bank'); os.makedirs(OUTD, exist_ok=True)
RE_LO, RE_HI = 4e4, 6e5
N_TRAIN_SYN = 600
N_VAL = 50            # frozen simulated validation sweeps (val1)
EXAM_RE_PER_AIRFOIL = 1  # physics baseline on exam: 1 Reynolds number per airfoil, drawn by seed
DEV_RE_PER_AIRFOIL = 3   # twin tests: 3 Reynolds numbers per dev airfoil build, drawn by seed


def synth(pool, rng):
    """a new shape: blend of two pool shapes plus a small coefficient perturbation; must be valid"""
    sd = pool.std(0)
    for _ in range(200):
        i, j = rng.choice(len(pool), 2, replace=False)
        w = rng.uniform(0, 1)
        p = w * pool[i] + (1 - w) * pool[j] + rng.normal(0, 0.15, pool.shape[1]) * sd
        p[-1] = abs(p[-1]) * 0.5
        if sh.valid(p):
            return p
    return pool[i]


def tasks(setname, physics):
    z = np.load(os.path.join(ROOT, 'work', 'shapes.npz'))
    pool, isval = z['pool'], z['isval']
    rng = np.random.default_rng({'train': 11, 'val': 12, 'twin_dev': 13, 'phys_exam': 14}[setname] +
                                (100 if physics == 'p2' else 0))
    T = []
    if setname in ('train', 'val'):
        P = pool[~isval] if setname == 'train' else pool[isval]
        shapes_ = [p for p in P]
        if setname == 'train':
            shapes_ += [synth(P, rng) for _ in range(N_TRAIN_SYN)]
        order = rng.permutation(len(shapes_))
        if setname == 'val':
            order = order[:N_VAL]
        for k, i in enumerate(order):
            Re = float(10 ** rng.uniform(np.log10(RE_LO), np.log10(RE_HI)))
            a0 = float(rng.uniform(-1, 1))
            nc = 9.0 if physics == 'p1' else float(rng.uniform(5, 11))
            T.append(dict(id=f'{setname}{k:05d}', p=shapes_[i].tolist(), Re=Re, ncrit=nc,
                          up=list(a0 + np.arange(0, 21, 2.)), dn=list(a0 - np.arange(2, 9, 2.))))
    else:
        units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
        splits = json.load(open(os.path.join(ROOT, 'splits.json')))
        keys = list(z['tested_keys']); tp = z['tested_p']
        want = splits['dev'] if setname == 'twin_dev' else splits['exam']
        for aid in want:
            u = units[aid]
            seen = set()
            runs = u['runs']
            if setname in ('phys_exam', 'twin_dev'):   # one build, a few Reynolds numbers, fixed draw
                cnt = {}
                for r in runs:
                    cnt[r['model']] = cnt.get(r['model'], 0) + 1
                mdl = sorted(cnt, key=lambda m: (-cnt[m], m))[0]
                res_ = sorted({round(r['Re'] / 1e4) * 1e4 for r in runs if r['model'] == mdl})
                nre = EXAM_RE_PER_AIRFOIL if setname == 'phys_exam' else DEV_RE_PER_AIRFOIL
                if setname == 'twin_dev' and aid == 'a18':   # a18 twins (1-deg steps) were already run: keep them
                    nre = len(res_)
                pick = set(rng.choice(res_, min(nre, len(res_)), replace=False).tolist())
                runs = [r for r in runs if r['model'] == mdl and round(r['Re'] / 1e4) * 1e4 in pick]
            for r in runs:
                Re_r = round(r['Re'] / 1e4) * 1e4
                key = (r['model'], Re_r)
                if key in seen:
                    continue
                seen.add(key)
                p = tp[keys.index(f"{aid}|{r['model']}")]
                st = 1. if (setname == 'twin_dev' and aid == 'a18') else 2.
                T.append(dict(id=f"{aid}|{r['model']}|{int(Re_r)}", p=p.tolist(), Re=float(Re_r), ncrit=9.0,
                              up=list(np.arange(0, 21, st)), dn=list(-np.arange(st, 9, st))))
    return T


def main():
    setname, wk, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    physics = sys.argv[4] if len(sys.argv) > 4 else 'p1'
    T = tasks(setname, physics)
    out = os.path.join(OUTD, f'{physics}_{setname}_w{wk}.jsonl')
    done = set()
    import glob as _g
    for fo in _g.glob(os.path.join(OUTD, f'{physics}_{setname}_w*.jsonl')):   # any worker layout
        for line in open(fo):
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    mine = [t for i, t in enumerate(T) if i % nw == wk and t['id'] not in done]
    print(f'{setname} {physics}: {len(T)} tasks, worker {wk} has {len(mine)} to do', flush=True)
    with open(out, 'a') as f:
        for t in mine:
            t0 = time.time()
            Y = sh.cst_coords(np.array(t['p']))
            try:
                res = xs.polar_sweep(Y, t['Re'], alphas_up=np.array(t['up']), alphas_dn=np.array(t['dn']),
                                     ncrit=t['ncrit'], time_limit=240)
            except Exception as e:
                res = []
            rec = dict(id=t['id'], p=t['p'], Re=t['Re'], ncrit=t['ncrit'],
                       pts=[[r['alpha'], r['cl'], r['cd'], r['cm'], int(r['loose']), r['xtr_u'], r['xtr_l']]
                            for r in res], sec=round(time.time() - t0, 1))
            f.write(json.dumps(rec) + '\n'); f.flush()


if __name__ == '__main__':
    main()
