"""Tune EnKF settings on SIMULATED TRAIN runs only (no real data, no frozen val set).
usage: python3 tune.py <name> '<json of base hyper overrides>' '<json list of {override} variants>' [nruns]
Appends one JSON line per variant to work/tune/<name>.jsonl: median NSE/KGE per lead, cov90, CRPS/mean flow, CPU s."""
import json, os, sys, time
sys.path.insert(0, '/home/claude/bakeoff/river/harness')
ARM = '/home/claude/bakeoff/river/arms_filt/enkf_watersim'
sys.path.insert(0, ARM)
import numpy as np
import bake, score as S
import predict as P

name, base, variants = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
nruns = int(sys.argv[4]) if len(sys.argv) > 4 else 250
leads = json.loads(sys.argv[5]) if len(sys.argv) > 5 else [1, 3, 7]
out = f'/home/claude/bakeoff/river/arms_filt/work/tune/{name}.jsonl'
os.makedirs(os.path.dirname(out), exist_ok=True)
done = set()
if os.path.exists(out):
    for l in open(out):
        done.add(json.dumps(json.loads(l)['variant'], sort_keys=True))
B = bake.load_bank('train', ['F', 'qo', 'attrs', 'st', 'lat', 'lon', 'start'])
rng = np.random.default_rng(424242)
runs = rng.choice(len(B['qo']), nruns, replace=False)
offs = rng.integers(0, 1462 - 731, nruns)
cache = {}
for i, a in zip(runs, offs):
    for L in leads:
        b = bake.sim_window(B, int(i), int(a), 365, 'gauged', L)
        cache[(i, L)] = (b, bake.sim_target(B, int(i), b))
for v in variants:
    key = json.dumps(v, sort_keys=True)
    if key in done:
        continue
    H = dict(P.HYPER); H.update(base); H.update(v)
    t0 = time.process_time()
    res = {}
    for L in leads:
        rows = []
        for i in runs:
            b, y = cache[(i, L)]
            o = P.predict(b, H)
            Q = np.stack([o[k] for k in ('q05', 'q25', 'q50', 'q75', 'q95')], 1)
            r = S.simval_scores(y, o['mean'], Q)
            rows.append((r['nse'], r['kge'], r['cov90'], r['cov50'], r['crps_rel'], r['fhv']))
        A = np.array(rows, float)
        res[f'L{L}'] = dict(zip(['nse', 'kge', 'cov90', 'cov50', 'crps_rel', 'fhv'], np.round(np.nanmedian(A, 0), 4).tolist()))
    rec = {'variant': v, 'base': base, 'res': res, 'cpu_s': round(time.process_time() - t0, 1), 'nruns': nruns}
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(key, {k: (x['nse'], x['kge'], x['fhv'], x['crps_rel']) for k, x in res.items()}, rec['cpu_s'], flush=True)
