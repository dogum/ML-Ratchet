"""Training / validation sets: windows of simulated unit records -> (features, targets).

Each unit contributes windows of 30-365 days (capped by its record), with random missing-hour gaps, optional net basis,
and price and context present or dropped at random.  Targets = the unit's observed character over its whole record
(pdcore.truth, the same function used on real CAMPD units).  Units are split train / frozen-val by a hash of the unit
key, so the frozen validation set of a bank version never changes.

usage: python trainset.py <bank_version> [windows_per_unit]
"""
from common import *
import json, glob, zlib
import numpy as np, pandas as pd
from pdcore import truth, features, context_vector, TECHS, REG
from bank_dp import ercot_background, gs_background

WIN_DAYS = [30, 61, 92, 182, 365]


def tgt_vector(tr):
    """Transformed regression targets (NaN = undefined) in REG order."""
    lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))  # noqa
    g = lambda k: tr.get(k) if tr.get(k) is not None else np.nan  # noqa
    hf, hm = g('hr_full'), g('hr_min')
    ok_hr = np.isfinite(hf) and 4 < hf < 30 and np.isfinite(hm) and 4 < hm < 40
    return np.array([lg(g('pmin_frac')), np.log(max(g('ramp_up'), 0.003)), np.log(max(g('ramp_dn'), 0.003)),
                     np.log1p(g('starts_yr')), np.log(g('min_up')), np.log(g('med_up')), np.log(g('min_dn')),
                     lg(g('hot')), lg(g('warm')), lg(g('cold')),
                     np.log(hf) if ok_hr else np.nan, np.log(hm / hf) if ok_hr else np.nan, lg(g('cf'))], np.float32)


def gaps(n, rng):
    """Missing-hour mask: none (60 %), or blocks covering 0.5-8 % of hours."""
    m = np.zeros(n, bool)
    if rng.random() < 0.6:
        return m
    frac = rng.uniform(0.005, 0.08)
    while m.mean() < frac:
        s = int(rng.integers(0, n)); m[s:s + int(rng.choice([1, 2, 3, 6, 24, 72]))] = True
    return m


def window_features(rec, t, price, gas, utc_off, ctx_raw, rng, n_days, p_price=0.7, p_ctx=0.5, p_net=0.15,
                    aux=0.03, start=None, gap=True, force=None):
    n = len(rec['op'])
    L = min(n, n_days * 24)
    s = int(rng.integers(0, n - L + 1)) if start is None else start
    sl = slice(s, s + L)
    x = np.round(np.nan_to_num(rec['P'][sl]) * np.nan_to_num(rec['op'][sl]))
    use_net = rng.random() < p_net if force is None else force.get('net', False)
    if use_net:
        x = np.round(x * (1 - aux))
    if gap:
        x = np.where(gaps(L, rng), np.nan, x)
    tt = pd.to_datetime(t[sl], unit='s')
    hl = ((tt.hour.values + utc_off) % 24)
    dow = (tt + pd.to_timedelta(utc_off, unit='h')).dayofweek.values
    use_p = (rng.random() < p_price if force is None else force.get('price', True)) and price is not None
    use_c = (rng.random() < p_ctx if force is None else force.get('ctx', True)) and ctx_raw is not None
    cv = None
    if use_c:
        yrs = sorted(set(tt.year))
        hr = [ctx_raw['hr923'].get(str(y), ctx_raw['hr923'].get(y)) for y in yrs]
        hr = [h for h in hr if h]
        c = dict(ctx_raw); c['hr923_window'] = float(np.mean(hr)) if hr else None
        c['ref_year'] = float(np.mean(yrs))
        x_ok = x[np.isfinite(x)]
        cv = context_vector(c, max(np.percentile(x_ok[x_ok > 0], 99.5) if (x_ok > 0).sum() > 10 else 1.0, 1.0))
    v, info = features(x, hl, dow, price[sl] if use_p else None, gas[sl] if use_p else None, cv)
    v = np.r_[v, float(use_net)].astype(np.float32)
    return v, info


def unit_key(shard, i):
    return f'{shard}:{i}'


def is_val(key):
    return zlib.crc32(key.encode()) % 100 < 15


def build(version, wpu=12, seed=0):
    rng = np.random.default_rng(seed)
    bgs = {}
    X, Y, C, V, K, W, G = [], [], [], [], [], [], []
    files = sorted(glob.glob(f'{DATA}/bank_{version}/shard_*.npz'))
    for f in files:
        z = np.load(f)
        U = json.loads(str(z['units']))
        for i, d in enumerate(U):
            key = unit_key(os.path.basename(f)[:-4], i)
            bk = d['bg']
            if bk not in bgs:
                if bk.startswith('ercot'):
                    bgs[bk] = ercot_background(int(bk[5:]))
                else:
                    g = gs_background(f'{DATA}/bank_gs/{bk}.npz'); g.pop('z'); bgs[bk] = g
            bg = bgs[bk]
            rec = {k: z[f'{k}_{i}'] for k in ('op', 'P', 'HI', 'meas')}
            tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
            if not np.isfinite(tr['pmax']):
                continue
            y = tgt_vector(tr)
            cls = TECHS.index(d['u']['tech'])
            t = fix_times(bg['t'])
            n_days = len(rec['op']) // 24
            for w in range(wpu):
                nd = int(rng.choice([wd for wd in WIN_DAYS if wd <= n_days] or [n_days]))
                v, _ = window_features(rec, t, bg['price'], bg['gas'], bg['utc_off'], d['ctx'], rng, nd,
                                       aux=d['rep']['aux'])
                X.append(v); Y.append(y); C.append(cls); V.append(is_val(key)); K.append(key); W.append(nd)
                G.append(d['mode'])
        print(f, len(X), flush=True)
    out = f'{DATA}/set_{version}.npz'
    np.savez_compressed(out, X=np.array(X, np.float32), Y=np.array(Y, np.float32), C=np.array(C), V=np.array(V),
                        K=np.array(K), W=np.array(W), G=np.array(G))
    print('saved', out, len(X))


if __name__ == '__main__':
    build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
