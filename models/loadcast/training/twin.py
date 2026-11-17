"""Twin test on DEV: each dev building's calibrated twin (its own fitted knobs, a, b) is simulated over 2016-2017,
given real noise from a different dev building, and pushed through the networks like the real meter.
If the network reads twin and real alike, their CV(RMSE) distributions match and correlate building by building.

    python3 twin.py <run> [noise]
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, lcsim as L, dataset as D, evaluate as E, lcnet as N


def main(run, noise='real'):
    kn = json.load(open(f'{C.ROOT}/data/dev_knobs.json'))
    nets = {m: N.load(f'{C.ROOT}/ckpt/{run}_{m}.pt')[0] for m in ('cold', 'warm')}
    real = D.real_units('dev')
    for u in real: D.prepare(u, ('full',))
    twins = []
    cache = {}
    for u in real:
        b = u['bid']; k = dict(kn[b]['best']['knobs'])
        for m, p in kn[b]['best']['per'].items(): k[f'a_{m}'] = p['a']; k[f'b_{m}'] = p['b']
        if b not in cache:
            spec = L.make_spec(C.bd.bdg2_spec(b), kn[b]['meters'], k)
            raw, _ = L.simulate_site(spec, u['site'])
            cache[b] = L.post(raw, k, u['area'])
        if u['meter'] not in k and f"a_{u['meter']}" not in k: continue
        t = dict(u); t['kind'] = 'twin'; t['id'] = u['id'] + ':twin'; t['exclude_bid'] = b
        y = cache[b][u['meter']] / (u['area'] * C.IREF[u['meter']] / 1000.0)
        rng = np.random.default_rng([4242, len(twins)])
        t['y'] = D.apply_noise(y, t, noise, rng).astype(np.float32) if noise else y.astype(np.float32)
        D.prepare(t, ('full',)); twins.append(t)
    keep = {t['id'][:-5] for t in twins}
    dr = E.eval_units([u for u in real if u['id'] in keep], nets)
    dt = E.eval_units(twins, nets); dt['id'] = dt['id'].str[:-5]
    d = dr.merge(dt, on=['id', 'meter'], suffixes=('_real', '_twin'))
    out = {}
    for m, g in d.groupby('meter'):
        out[m] = dict(n=len(g))
        for k in ('cold', 'warm'):
            out[m][k] = dict(cv_real=float(g[f'{k}_cv_real'].median()), cv_twin=float(g[f'{k}_cv_twin'].median()),
                             r_log_cv=float(np.corrcoef(np.log(g[f'{k}_cv_real']), np.log(g[f'{k}_cv_twin']))[0, 1]) if len(g) > 2 else None,
                             cov_real=float(g[f'{k}_cov_real'].mean()), cov_twin=float(g[f'{k}_cov_twin'].mean()))
    C.jdump(dict(run=run, noise=noise, by_meter=out), f'{C.ROOT}/data/twin_{run}.json')
    d.to_csv(f'{C.ROOT}/data/twin_{run}.csv', index=False)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
