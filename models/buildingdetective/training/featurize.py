"""Bank shards -> feature matrices for the three granularities, with meter-quirk emulation.

usage: python3 featurize.py <bank_tag> <feat_tag> [n_hourly_aug] [quirks=1]
Writes feats/<feat_tag>/<mode>_<shard>.npz with X, Y, K, idx. Val shards (>= 900) use a fixed seed per building,
so the frozen validation features are reproducible.
"""
import glob
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import common as C
import features as F

_WX = {}


def load_wx(name):
    if name not in _WX:
        z = np.load(os.path.join(C.ROOT, 'bank', 'weather', name.replace(':', '_') + '.npz'))
        idx = pd.DatetimeIndex(z['utc'].astype('datetime64[us]'), tz='UTC')
        tz = str(z['tz'])
        T = z['T'].astype(float)
        G = z['GHI'].astype(float)
        loc = idx.tz_convert(tz)
        day = np.asarray(loc.normalize().asi8)
        ud, inv = np.unique(day, return_inverse=True)
        Td = np.bincount(inv, weights=T) / np.bincount(inv)
        hdd = float(np.maximum(F.HB - Td, 0).sum() * 365 / len(ud))
        cdd = float(np.maximum(Td - F.HB, 0).sum() * 365 / len(ud))
        _WX[name] = dict(T=T, GHI=G, loc=loc, lat=float(z['lat']), hdd=hdd, cdd=cdd)
    return _WX[name]


def building_meters(X, S, i, n):
    m = {}
    for j, ch in enumerate(C.CHANNELS):
        if S[i, j] > 0:
            m[ch] = X[i, j, :n].astype(np.float64) * S[i, j]
    return m


def featurize_shard(bank, tag, fn, n_aug=2, quirks=True, w3=False):
    sid = int(os.path.basename(fn)[6:10])
    od = os.path.join(C.ROOT, 'feats', tag)
    os.makedirs(od, exist_ok=True)
    outs = {m: os.path.join(od, f'{m}_{sid:04d}.npz') for m in ('hourly', 'monthly', 'annual')}
    if all(os.path.exists(o) for o in outs.values()):
        return
    z = np.load(fn)
    X, S, Y, K = z['X'], z['S'], z['Y'], z['K']
    info = json.load(open(fn.replace('.npz', '.json')))
    acc = {m: dict(X=[], Y=[], K=[], idx=[]) for m in outs}
    t0 = time.time()
    for i, inf in enumerate(info):
        if K[i, 0] < 0 or 'wx' not in inf:
            continue
        w = load_wx(inf['wx'])
        n = min(inf['n_hours'], len(w['T']))
        met = building_meters(X, S, i, n)
        if not met:
            continue
        T, G, loc = w['T'][:n], w['GHI'][:n], w['loc'][:n]
        base = (sid * 1000 + i) * 10 + (0 if sid >= 900 else int(time.time() * 1e3) % 7919)
        rng = np.random.default_rng(base)
        if w3:                                      # physics v2: always-on base loads (wrappers.W3)
            from wrappers import w3_base_loads
            met = w3_base_loads(met, rng, commercial=not inf['use'].startswith('residential'))
        for a in range(n_aug):
            mq = F.quirk_hourly(met, rng, n)[0] if quirks else met
            if not mq:
                mq = met
            acc['hourly']['X'].append(F.feat_hourly(mq, T, G, loc, w['lat']))
            acc['hourly']['Y'].append(Y[i]); acc['hourly']['K'].append(K[i]); acc['hourly']['idx'].append(sid * 1000 + i)
        # monthly: billing windows shifted 0-15 days, noise, missing months
        mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(met, T, loc, shift_days=int(rng.integers(0, 16)) if quirks else 0)
        if quirks:
            mm = F.quirk_monthly(mm, rng)
            if len(mm) > 1 and rng.random() < 0.25:
                mm.pop(rng.choice([c for c in mm if c != 'elec'] or list(mm)))
        acc['monthly']['X'].append(F.feat_monthly(mm, Tm, Hm, Cm, nd, w['lat']))
        acc['monthly']['Y'].append(Y[i]); acc['monthly']['K'].append(K[i]); acc['monthly']['idx'].append(sid * 1000 + i)
        kwh = {c: float(np.nansum(v)) * 8760.0 / n for c, v in met.items()}
        for a in range(2):
            kq = F.quirk_annual(kwh, rng) if quirks else kwh
            if not kq:
                kq = {'elec': kwh.get('elec', 1.0)}
            hdd = w['hdd'] * np.exp(rng.normal(0, 0.05)) if quirks else w['hdd']
            cdd = w['cdd'] * np.exp(rng.normal(0, 0.08)) if quirks else w['cdd']
            acc['annual']['X'].append(F.feat_annual(kq, hdd, cdd))
            acc['annual']['Y'].append(Y[i]); acc['annual']['K'].append(K[i]); acc['annual']['idx'].append(sid * 1000 + i)
    for m, o in outs.items():
        a = acc[m]
        np.savez(o + '.tmp.npz', X=np.array(a['X'], np.float32), Y=np.array(a['Y'], np.float32),
                 K=np.array(a['K'], np.int16), idx=np.array(a['idx'], np.int64))
        os.replace(o + '.tmp.npz', o)
    print(tag, sid, len(info), 'featurized %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    bank, tag = sys.argv[1], sys.argv[2]
    n_aug = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    quirks = (sys.argv[4] != '0') if len(sys.argv) > 4 else True
    w3 = '--w3' in sys.argv
    maxs = int(os.environ.get('MAXSHARD', '10000'))
    while True:
        fns = sorted(glob.glob(os.path.join(C.ROOT, 'bank', bank, 'shard_*.npz')))
        fns = [f for f in fns if not f.endswith('.tmp.npz')]
        fns = [f for f in fns if int(os.path.basename(f)[6:10]) >= 900 or int(os.path.basename(f)[6:10]) < maxs]
        todo = [f for f in fns if not os.path.exists(os.path.join(C.ROOT, 'feats', tag,
                                                                  f'annual_{int(os.path.basename(f)[6:10]):04d}.npz'))]
        for f in todo:
            featurize_shard(bank, tag, f, n_aug, quirks, w3)
        if '--once' in sys.argv or os.path.exists(os.path.join(C.ROOT, 'bank', bank, 'DONE')):
            fns2 = [f for f in sorted(glob.glob(os.path.join(C.ROOT, 'bank', bank, 'shard_*.npz'))) if not f.endswith('.tmp.npz')]
            fns2 = [f for f in fns2 if int(os.path.basename(f)[6:10]) >= 900 or int(os.path.basename(f)[6:10]) < maxs]
            if len(fns2) == len(fns):
                break
        if maxs < 10000 and sum(1 for f in fns if int(os.path.basename(f)[6:10]) < maxs) >= maxs and not todo:
            break
        if not todo:
            time.sleep(60)
