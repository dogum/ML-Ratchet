"""Cache the day-ahead value-curve backbone (backbone.py) for every path of a bank, for 1, 2 and 4 h batteries.

    python bbcache.py <bank> [k/n]       -> WORK/data/bb/<bank>/<path file>.npz  (lam1, lam2, lam4: (n, 5) $/MWh)
    python bbcache.py dev                -> WORK/data/bb/dev.pkl  (dict key -> {1: lam, 2: lam, 4: lam})

The wear proxy c_lin is the battery's at the path's temperature, as in training and scoring.
"""
import glob, os, sys, time
import numpy as np
import pandas as pd
import _paths
from battery import Battery
from backbone import value_curves

ROOT = _paths.WORK
DUR = (1, 2, 4)


def curves(p):
    out = {}
    for d in DUR:
        bat = Battery(1.0, d)
        out[d] = value_curves(np.asarray(p['da'], float), np.asarray(p['hour']), np.asarray(p['day']),
                              bat.c_lin(np.asarray(p['T'], float)), float(d)).astype(np.float32)
    return out


def main():
    bank = sys.argv[1]
    if bank == 'dev':
        from ratchet import dev_paths, path_dict
        dev = dev_paths()
        res = {k: curves(path_dict(f)) for k, f in dev.items()}
        os.makedirs(f'{ROOT}/data/bb', exist_ok=True)
        pd.to_pickle(res, f'{ROOT}/data/bb/dev.pkl')
        return
    part = sys.argv[2] if len(sys.argv) > 2 else '0/1'
    k, n = map(int, part.split('/'))
    files = sorted(glob.glob(f'{ROOT}/data/banks/{bank}/*.npz'))[k::n]
    od = f'{ROOT}/data/bb/{bank}'
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    for fn in files:
        out = f'{od}/{os.path.basename(fn)}'
        if os.path.exists(out):
            continue
        z = np.load(fn, allow_pickle=True)
        c = curves({k_: z[k_] for k_ in ('da', 'hour', 'day', 'T')})
        np.savez(out + '.tmp.npz', **{f'lam{d}': c[d] for d in DUR})
        os.replace(out + '.tmp.npz', out)
        print(bank, os.path.basename(fn), '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
