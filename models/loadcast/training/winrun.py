"""Windows / no-shell runner for the slow LoadCast steps, sharded over a few processes (added 17 Nov 2026).

The build ran calib.py and bank.py from bash queues (`nice`, `python3`, restarts every 2 parts). This runner does the
same work with Python's multiprocessing so it runs on Windows too, and writes to the same WORK folder.

    python winrun.py calib [procs]                      # dev_knobs.json (calib.py's grid, buildings sharded)
    python winrun.py bank <phys> <name> <n> <seed> [procs] [part]   # bank.py's parts, sharded by part number
    python winrun.py noise [variant]                    # noise_bank.npz (noisebank.py); variant 'y16' -> see below

Noise variant 'y16' (dev-gate optimism test): writes data/noise_bank_y16.npz, the same bank with every series'
2017 residuals and gaps replaced by its own 2016 ones shifted by 364 days (same weekday). The 2017/2016 level ratios
are kept. dataset.py reads it when LC_NOISE_BANK=noise_bank_y16.npz.

Note: with phys v1/v0, bank.py's generator cache (_GEN) hands out specs in the order a process asks for them, so a sharded bank is not
bit-identical to a sequential one (neither was the build's, which restarted every 2 parts); phys v1d is (17 Nov 2026).
"""
import sys, os, json, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _calib_shard(args):
    k, n = args
    import lc_common as C, calib
    real_meta = C.bd.bdg2_meta
    def meta(split=None, **kw):
        d = real_meta(split=split, **kw)
        return d.iloc[k::n] if split == 'dev' else d
    C.bd.bdg2_meta = meta
    calib.OUT = f"{C.ROOT}/data/{os.environ.get('LC_KNOBS', 'dev_knobs.json')[:-5]}_shard{k}.json"
    calib.main()
    return calib.OUT


def calib_all(procs=3):
    import lc_common as C
    os.makedirs(f'{C.ROOT}/data', exist_ok=True)
    t0 = time.time()
    with mp.Pool(procs) as p:
        outs = p.map(_calib_shard, [(k, procs) for k in range(procs)])
    res = {}
    for f in outs: res.update(json.load(open(f)))
    C.jdump(res, f"{C.ROOT}/data/{os.environ.get('LC_KNOBS', 'dev_knobs.json')}")
    print(len(res), 'dev buildings calibrated', round(time.time() - t0), 's', flush=True)


def _bank_parts(args):
    phys, name, n, seed, parts = args
    import lc_common as C, bank
    d = f'{C.ROOT}/data/bank_{name}'
    info = bank.dev_info()
    if phys.startswith('v1') and not info[3]: raise SystemExit('need data/dev_knobs.json')
    t0 = time.time()
    for p in parts:
        f = f'{d}/part_{p:04d}.npz'
        if os.path.exists(f): continue
        recs, ys = [], {}
        for i in range(p * bank.PART, min((p + 1) * bank.PART, n)):
            try:
                rec, Y = bank.make_one(i, seed, phys, info)
            except Exception as e:
                print('ERR', i, repr(e)[:200], flush=True); continue
            recs.append(rec)
            for m, v in Y.items(): ys[f'{rec["i"]}_{m}'] = v
        np.savez(f + '.tmp.npz', meta=json.dumps(recs, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), **ys)
        os.replace(f + '.tmp.npz', f)
        print(f'{name}: part {p} ({len(recs)} buildings) {time.time() - t0:.0f} s', flush=True)
    return len(parts)


def bank_all(phys, name, n, seed, procs=3, part=None):
    import lc_common as C, bank
    if part: bank.PART = int(part)
    os.makedirs(f'{C.ROOT}/data/bank_{name}', exist_ok=True)
    nparts = (n + bank.PART - 1) // bank.PART
    t0 = time.time()
    jobs = [(phys, name, n, seed, list(range(k, nparts, procs))) for k in range(procs)]
    with mp.Pool(procs, maxtasksperchild=1) as p:
        p.map(_bank_parts, jobs)
    print(f'bank {name}: {n} buildings in {time.time() - t0:.0f} s', flush=True)


def noise(variant=''):
    import lc_common as C
    if not variant:
        import noisebank
        noisebank.main(); return
    z = dict(np.load(f'{C.ROOT}/data/noise_bank.npz'))
    E = z['E'].astype(np.float32); G = z['gaps'].copy()
    h16 = 366 * 24; sh = 364 * 24; NL = E.shape[1]
    src = np.arange(NL) - sh
    src = np.where(src >= h16, src - sh, src)  # the last days of local 2017 fall past 2016's end: shift once more
    sel = np.arange(h16, NL)
    E[:, sel] = E[:, src[sel]]; G[:, sel] = G[:, src[sel]]
    z['E'] = E.astype(np.float16); z['gaps'] = G
    np.savez_compressed(f'{C.ROOT}/data/noise_bank_{variant}.npz', **z)
    print('wrote', f'noise_bank_{variant}.npz', E.shape, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'calib': calib_all(int(a[1]) if len(a) > 1 else 3)
    elif a[0] == 'bank': bank_all(a[1], a[2], int(a[3]), int(a[4]), int(a[5]) if len(a) > 5 else 3, a[6] if len(a) > 6 else None)
    elif a[0] == 'noise': noise(a[1] if len(a) > 1 else '')
