"""Rung r11 (27 Nov 2026): r10's recipe (bank p12) with simulated district meters that have real zero runs (seasonal
plant shutdowns, long outages; features.quirk_hourly(district_zeros=True)). Hourly network only. Plan in NOTEBOOK.md,
entry of 27 Nov 2026 (written before the run). Dev and simulation only: no exam unit and no HEAPO home is read here.

    python r11.py feats [procs]   # training shards 0-59 of bank p12 -> feats/f4 (hourly); frozen shards 900-904 copied from f3
    python train.py r11_hourly hourly f4 --max_shard 59        # same settings as r10-hourly
    python r11.py decide          # rule against r10-hourly and v1.0's dev composite; bounded-input variant (info)

Output: WORK/r11.json (copied to training/r11.json).
"""
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import common as C            # noqa: E402
import evaluate as E          # noqa: E402
import features as F          # noqa: E402
import r10 as R10             # noqa: E402
from detective import Net     # noqa: E402
from _paths import MODEL, WORK  # noqa: E402

SRC, DST = 'f3', 'f4'
OUT = os.path.join(WORK, 'r11.json')
MARGIN, GATE_TOL = 0.01, 0.02
ZCOLS = (F.N_W_H + 2 * F.N_C_H + 291, F.N_W_H + 3 * F.N_C_H + 291)     # 904 dheat, 1197 dcool zero fraction


def _shard(fn):
    import featurize as FZ
    sid = int(os.path.basename(fn)[6:10])
    od = os.path.join(C.ROOT, 'feats', DST)
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f'hourly_{sid:04d}.npz')
    if os.path.exists(out):
        return sid, 0.0
    z = np.load(fn)
    X, S, Y, K = z['X'], z['S'], z['Y'], z['K']
    info = json.load(open(fn.replace('.npz', '.json')))
    rng = np.random.default_rng(20261127 * 1000 + sid)
    acc = dict(X=[], Y=[], K=[], idx=[])
    t0 = time.time()
    for i, inf in enumerate(info):
        if K[i, 0] < 0 or 'wx' not in inf:
            continue
        w = FZ.load_wx(inf['wx'])
        n = min(inf['n_hours'], len(w['T']))
        met = FZ.building_meters(X, S, i, n)
        if not met:
            continue
        T, G, loc = w['T'][:n], w['GHI'][:n], w['loc'][:n]
        day = np.asarray(loc.normalize().asi8)
        _, inv = np.unique(day, return_inverse=True)
        Th = (np.bincount(inv, weights=T) / np.bincount(inv))[inv]
        for a in range(2):                                         # n_aug = 2, as f3
            mq = F.quirk_hourly(met, rng, n, Th=Th, district_zeros=True)[0] or met
            acc['X'].append(F.feat_hourly(mq, T, G, loc, w['lat']))
            acc['Y'].append(Y[i]); acc['K'].append(K[i]); acc['idx'].append(sid * 1000 + i)
    np.savez(out + '.tmp.npz', X=np.array(acc['X'], np.float32), Y=np.array(acc['Y'], np.float32),
             K=np.array(acc['K'], np.int16), idx=np.array(acc['idx'], np.int64))
    os.replace(out + '.tmp.npz', out)
    return sid, time.time() - t0


def feats(procs=4):
    od = os.path.join(C.ROOT, 'feats', DST)
    os.makedirs(od, exist_ok=True)
    for f in glob.glob(os.path.join(C.ROOT, 'feats', SRC, 'hourly_09*.npz')):        # frozen set: unchanged files
        shutil.copy(f, os.path.join(od, os.path.basename(f)))
    fns = sorted(f for f in glob.glob(os.path.join(C.ROOT, 'bank', 'p12', 'shard_*.npz'))
                 if not f.endswith('.tmp.npz') and int(os.path.basename(f)[6:10]) < 60)
    print(len(fns), 'training shards', flush=True)
    t0 = time.time()
    with mp.Pool(procs) as p:
        for sid, dt in p.imap_unordered(_shard, fns):
            print('shard', sid, '%.0f s' % dt, 'total %.0f s' % (time.time() - t0), flush=True)


class Bounded(Net):
    """v1.0's network with the two district zero-fraction inputs clipped at training mean + 8 sd (variant, info)."""
    def raw(self, X):
        X = np.array(np.atleast_2d(X), np.float64)
        for c in ZCOLS:
            X[:, c] = np.minimum(X[:, c], self.xm[c] + 8 * self.xs[c])
        return super().raw(X)


def zinfo(net):
    d = np.load(os.path.join(C.ROOT, 'feats', 'real', 'dev_bdg2.npz'))['hourly']
    z = np.abs((d - net.xm) / net.xs)
    return dict(max_z_904=float(z[:, ZCOLS[0]].max()), max_z_1197=float(z[:, ZCOLS[1]].max()),
                dev_rows_any_z_gt8=int((z > 8).any(1).sum()), n_dev=int(len(d)))


def decide():
    r10 = json.load(open(os.path.join(WORK, 'r10.json')))
    inc_r10 = next(r for r in r10['rungs'] if r['name'] == 'r10-hourly')['val_skill']
    inc_r9 = r10['rebaseline']['hourly']['val_skill']
    v10 = r10['v10_dev_composite']
    X, Y, K, fs = R10.val_feats('hourly')
    assert fs == r10['rebaseline']['hourly']['files']
    out = dict(date='2026-11-27', frozen='p12 (f3 files, shards 900-904)', n_val=int(len(X)),
               incumbent_r10_hourly=inc_r10, r9_on_p12=inc_r9, v10_dev_composite=v10, floor=v10 - GATE_TOL)
    od = os.path.join(WORK, 'models', 'r11')
    os.makedirs(od, exist_ok=True)
    from detective import export_torch
    export_torch(os.path.join(WORK, 'models', 'r11_hourly', 'final.pt'), os.path.join(od, 'hourly.npz'))
    res = json.load(open(os.path.join(WORK, 'models', 'r11_hourly', 'result.json')))
    v1 = E.load_nets(os.path.join(MODEL, 'weights'))
    rows = {}
    for name, hourly in (('v1.0', v1['hourly']), ('r10-hourly', Net(os.path.join(WORK, 'models', 'r10', 'hourly.npz'))),
                         ('r11-hourly', Net(os.path.join(od, 'hourly.npz'))),
                         ('v1.0 bounded', Bounded(os.path.join(MODEL, 'weights', 'hourly.npz')))):
        mu, sg, lg = hourly.raw(X)
        S = R10.skills(mu, sg, lg, Y, K)
        g, det = E.dev_gate(dict(v1, hourly=hourly))
        b = det['bdg2']['hourly']
        rows[name] = dict(val_skill=S['_overall'], per_task=S, dev_composite=g,
                          bdg2_dev_hourly_area_mdape=b['area']['MdAPE'], bdg2_dev_hourly_use_f1=b['use']['macroF1'],
                          **zinfo(hourly))
        print(f"{name:14s} frozen {S['_overall']:.4f}  dev composite {g:.4f}  BDG2 dev area {b['area']['MdAPE']:.1f} %  "
              f"use F1 {b['use']['macroF1']:.3f}  max z 904/1197 {rows[name]['max_z_904']:.1f}/{rows[name]['max_z_1197']:.1f}",
              flush=True)
    r = rows['r11-hourly']
    assert abs(r['val_skill'] - res['val_skill']['_overall']) < 1e-4, (r['val_skill'], res['val_skill']['_overall'])
    dv = r['val_skill'] - inc_r10
    okg = r['dev_composite'] >= v10 - GATE_TOL
    keep = dv > MARGIN and okg
    out.update(rows=rows, cfg=res['cfg'], kept=bool(keep),
               reason=(f"val skill {r['val_skill']:.4f} vs r10-hourly {inc_r10:.4f} (delta {dv:+.4f}, need > {MARGIN}; "
                       f"vs R9 on p12 {inc_r9:.4f}: {r['val_skill'] - inc_r9:+.4f}); dev composite {r['dev_composite']:.4f} "
                       f"vs v1.0 {v10:.4f} (floor {v10 - GATE_TOL:.4f}, {'ok' if okg else 'FAIL'})"))
    print('r11-hourly', 'KEPT' if keep else 'rejected', '|', out['reason'])
    json.dump(out, open(OUT, 'w'), indent=1, default=float)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'feats':
        feats(int(a[1]) if len(a) > 1 else 4)
    elif a[0] == 'decide':
        decide()
