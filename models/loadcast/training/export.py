"""Export a rung for release + web: widen factors fitted on DEV only (widen-only rule), numpy weights, numpy==torch check.

    python3 export.py <run>   -> data/widen_<run>.json, release/loadcast_weights.npz, web/weights_<run>.json
"""
import sys, os, json, base64
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, evaluate as E, lcnet as N
from exam import fit_widen


def main(run):
    nets = {m: N.load(f'{C.ROOT}/ckpt/{run}_{m}.pt')[0] for m in ('cold', 'warm')}
    wf = f'{C.ROOT}/data/widen_{run}.json'
    if os.path.exists(wf):
        W = json.load(open(wf))
    else:
        dev = D.real_units('dev')
        for u in dev: D.prepare(u, ('full', 'w8', 'w2'))
        W = fit_widen(dev, nets); C.jdump(W, wf)
    print('widen', W)
    arr = {}
    for mode, net in nets.items():
        for k, v in N.to_numpy(net).items(): arr[f'{mode}/{k}'] = v
    arr['widen_keys'] = np.array(list(W)); arr['widen_vals'] = np.array([W[k] for k in W], np.float32)
    np.savez(f'{C.ROOT}/release/loadcast_weights.npz', **arr)
    # numpy == torch check on random rows
    rng = np.random.default_rng(0); maxd = 0
    dev = D.real_units('dev', meters=['electricity', 'chilledwater'])[::6]
    for u in dev: D.prepare(u, ('full',))
    for mode, net in nets.items():
        R = [D.rows(u, u['tidx'][::50], mode) for u in dev]
        x = np.concatenate([r[0] for r in R]); b = np.concatenate([r[1] for r in R])
        mt, st = N.predict(net, x, b)
        Wn = {k: v.astype(np.float64) for k, v in N.to_numpy(net).items()}
        mn, ln = N.np_forward(Wn, x.astype(np.float64), b.astype(np.float64))
        maxd = max(maxd, float(np.abs(mt - mn).max() / max(np.abs(mn).max(), 1)), float(np.abs(np.log(st) - ln).max()))
    print('numpy vs torch max abs diff', maxd)
    # web: float32 little-endian base64 per tensor
    web = {}
    for mode, net in nets.items():
        web[mode] = {k: dict(shape=list(v.shape), b64=base64.b64encode(v.astype('<f4').tobytes()).decode())
                     for k, v in N.to_numpy(net).items()}
    web['widen'] = W
    os.makedirs(f'{C.ROOT}/web', exist_ok=True)
    json.dump(web, open(f'{C.ROOT}/web/weights_{run}.json', 'w'))
    C.jdump(dict(run=run, numpy_vs_torch=maxd), f'{C.ROOT}/data/export_{run}.json')


if __name__ == '__main__':
    main(sys.argv[1])
