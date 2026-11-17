"""Data for the showpiece page (web/page_data.json): real exam buildings for the explorer (fixed random draw within
use-type strata, seed 11: never cherry-picked), HUE homes (one fixed draw per heating type), their weather, the
network weights, the lab grid, exam / ratchet summaries. Python predictions are stored beside each unit so the page
can show JS == Python.
usage: python3 webdata.py [weights_dir=models/final]
"""
import base64
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import common as C
import features as F
import realdata as R
from detective import Net, decode
from verify_js import net_json

warnings.filterwarnings('ignore')
WEB = os.path.join(C.ROOT, 'web')


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def comp(x):
    x = np.asarray(x, float)
    nan = ~np.isfinite(x)
    x = np.where(nan, 0, np.maximum(x, 0))
    mx = float(x.max()) if x.size else 0.0
    q = np.round(np.sqrt(x / mx) * 254).astype(np.uint8) if mx > 0 else np.zeros(len(x), np.uint8)
    q[nan] = 255
    return q, mx


def decomp(q, mx):
    q = np.asarray(q)
    return np.where(q == 255, np.nan, (q.astype(float) / 254) ** 2 * mx)


def tz_changes(idx_utc, tz):
    off = np.array([t.utcoffset().total_seconds() / 60 for t in idx_utc.tz_convert(tz)], int)
    ch = [[0, int(off[0])]] + [[int(i), int(off[i])] for i in np.nonzero(np.diff(off))[0] + 1]
    return ch


def time_from(idx_utc, ch):
    """Replicates the page's local-time reconstruction (UTC ms + offset) to make the Python features from it."""
    off = np.zeros(len(idx_utc), int)
    for i, o in ch:
        off[i:] = o
    t = idx_utc.tz_localize(None) + pd.to_timedelta(off, unit='min')
    return t


def net_b64(fn):
    """Network weights as base64 float32 (little endian) for the page; JS decodes to Float64Array."""
    z = np.load(fn)
    n = int(z['n_layers'])
    e = lambda a: b64(np.asarray(a, '<f4').ravel())
    return dict(b64=True, W=[e(z[f'W{i}']) for i in range(n)], b=[e(z[f'b{i}']) for i in range(n)], Wr=e(z['Wr']),
                br=e(z['br']), Wc=e(z['Wc']), bc=e(z['bc']), xm=e(z['xm']), xs=e(z['xs']), ym=e(z['ym']), ys=e(z['ys']),
                widen=e(z['widen'] if 'widen' in z else np.ones(len(z['ym']))),
                temp=e(z['temp'] if 'temp' in z else np.ones(3)))


def unit_record(c, nets, kind):
    idx = c['loc'].tz_convert('UTC')
    met = {}
    md = {}
    for ch, x in c['meters'].items():
        q, mx = comp(x)
        met[ch] = dict(b64=b64(q), max=mx)
        md[ch] = decomp(q, mx)
    T = np.round(np.nan_to_num(c['T'], nan=-3276.8) * 10).astype(np.int16)
    Td = np.where(T == -32768, np.nan, T / 10.0)
    G = np.round(np.clip(c['GHI'], 0, 6000)).astype(np.uint16)
    tzc = tz_changes(idx, str(c['loc'].tz))
    loc = c['loc']
    pred = {}
    h = F.feat_hourly(md, Td, G.astype(float), loc, c['lat'])
    mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(md, Td, loc)
    mo = F.feat_monthly(mm, Tm, Hm, Cm, nd, c['lat'])
    for mode, X in (('hourly', h), ('monthly', mo)):
        if mode in nets:
            mu, sg, lg = nets[mode].raw(X)
            r = decode(mu, sg, lg, nets[mode].temp)[0]
            pred[mode] = dict(mu=mu[0].tolist(), sg=sg[0].tolist(), use=r['use'], heat=r['heat'], cool=r['cool'])
    lab = c['labels']
    return dict(id=c['id'], kind=kind, t0=int(idx[0].value // 10 ** 6), n=len(idx), tz=str(c['loc'].tz), tzc=tzc,
                lat=c['lat'], T=b64(T), GHI=b64(G), meters=met, pred=pred,
                truth={k: lab.get(k) for k in ('use', 'A', 'year', 'heat', 'codes', 'site', 'cz', 'meters', 'house_type',
                                               'has_ac') if k in lab})


def main(wdir):
    nets = {m: Net(os.path.join(wdir, f'{m}.npz')) for m in ('hourly', 'monthly', 'annual')}
    meta = R.bdg2_buildings('exam')
    meta = meta[meta.meters.astype(str).str.contains('electricity')]
    rng = np.random.default_rng(11)
    pick = []
    for u, g in sorted(meta.groupby('primary_use'), key=lambda t: t[0]):
        k = min(3, len(g))
        pick += list(g.id.to_numpy()[rng.choice(len(g), k, replace=False)])
    units = []
    for bid in pick:
        c = R.bdg2_case(bid, 2017, meta.set_index('id').loc[bid])
        if c is None:
            continue
        units.append(unit_record(c, nets, 'bdg2'))
        print(bid, flush=True)
    hue = R.hue_cases()
    rng = np.random.default_rng(12)
    byh = {}
    for c in hue:
        byh.setdefault(c['labels']['heat'], []).append(c)
    for hcls in sorted(byh):
        cs = byh[hcls]
        for i in rng.choice(len(cs), min(2, len(cs)), replace=False):
            units.append(unit_record(cs[i], nets, 'hue'))
            print(cs[i]['id'], hcls, flush=True)
    wx = {}
    for u in units:                                  # one weather record per (tz, start, length, T)
        key = f"{u['tz']}|{u['t0']}|{u['n']}|{hash(u['T']) & 0xffffffff:08x}"
        wx.setdefault(key, dict(T=u['T'], GHI=u['GHI']))
        u['wx'] = key
        del u['T'], u['GHI']
    D = dict(units=units, weather=wx, nets={m: net_b64(os.path.join(wdir, f'{m}.npz')) for m in nets})
    json.dump(D, open(os.path.join(WEB, 'explorer.json'), 'w'))
    print('units', len(units), 'MB %.2f' % (os.path.getsize(os.path.join(WEB, 'explorer.json')) / 1e6))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.ROOT, 'models', 'final'))
