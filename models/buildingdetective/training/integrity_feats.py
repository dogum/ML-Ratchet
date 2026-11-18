"""Rebuild the real feature matrices the exam reads (integrity pass, 17 Nov 2026), in parallel.

Same per-building code as realfeats.bdg2 / surveys / hue, but only the matrices exam.py and evaluate.py read
(hourly, monthly, annual, hourly_elec; the 1/3/6-month and 2016 copies are skipped) and BDG2 split over up to
3 worker processes. Writes WORK/feats/real/<split>_<set>.npz + .json, the files evaluate.load() reads.
usage: python integrity_feats.py [n_proc=3]
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings('ignore')


def _bdg2_chunk(args):
    split, ids = args
    warnings.filterwarnings('ignore')
    import features as F
    import realdata as R
    meta = R.bdg2_buildings(split).set_index('id')
    out = []
    for bid in ids:
        row = meta.loc[bid].copy()
        row['id'] = bid
        c = R.bdg2_case(bid, 2017, row)
        if c is None:
            continue
        m, T, G, loc, lat = c['meters'], c['T'], c['GHI'], c['loc'], c['lat']
        h = F.feat_hourly(m, T, G, loc, lat)
        mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(m, T, loc)
        mo = F.feat_monthly(mm, Tm, Hm, Cm, nd, lat)
        kwh = {}
        for ch, x in m.items():
            v = np.isfinite(x)
            kwh[ch] = float(np.nanmean(x) * 8760) if v.mean() > 0.2 else 0.0
        d = np.asarray(loc.normalize().asi8)
        ud, inv = np.unique(d, return_inverse=True)
        Td = np.bincount(inv, weights=np.nan_to_num(T, nan=np.nanmean(T))) / np.bincount(inv)
        hdd, cdd = float(np.maximum(F.HB - Td, 0).sum()), float(np.maximum(Td - F.HB, 0).sum())
        an = F.feat_annual(kwh, hdd, cdd)
        he = (F.feat_hourly({'elec': m['elec']}, T, G, loc, lat) if 'elec' in m
              else np.full(F.N_W_H + 4 * F.N_C_H, np.nan, np.float32))
        lab = dict(c['labels'])
        lab.update(id=c['id'], kwh=kwh, hdd=hdd, cdd=cdd, channels=sorted(m), lat=float(lat))
        out.append((h, mo, an, he, lab))
    return out


def bdg2(split, n_proc):
    import realdata as R
    import realfeats as RF
    ids = list(R.bdg2_buildings(split)['id'])
    k = max(1, len(ids) // (n_proc * 8))
    chunks = [(split, ids[i:i + k]) for i in range(0, len(ids), k)]
    t0 = time.time()
    with ProcessPoolExecutor(n_proc) as ex:
        res = list(ex.map(_bdg2_chunk, chunks))
    rows = [r for ch in res for r in ch]
    mats = dict(hourly=[r[0] for r in rows], monthly=[r[1] for r in rows], annual=[r[2] for r in rows],
                hourly_elec=[r[3] for r in rows])
    RF._save(f'{split}_bdg2', mats, [r[4] for r in rows])
    print(split, 'bdg2', len(rows), 'of', len(ids), '%.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    import realfeats as RF
    for sp in ('dev', 'exam'):
        t = time.time()
        RF.surveys(sp)
        print(sp, 'surveys %.0f s' % (time.time() - t), flush=True)
    t = time.time()
    RF.hue()
    print('hue %.0f s' % (time.time() - t), flush=True)
    for sp in ('dev', 'exam'):
        bdg2(sp, n)
