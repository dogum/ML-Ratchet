"""Scores: frozen simulated validation (ratchet score) and the real dev gate (10 dev plants, 40 units, dev years).

Skill score in [0, 1] = mean over [tech accuracy] + [per target max(0, 1 - MAE/MAE_dumb)], where MAE is in the
transformed target space and the dumb predictor is the median of the target per true technology.

usage: python evaluate.py <model> <set_version>
"""
from common import *
import json, glob
import numpy as np, pandas as pd
from pdcore import truth, TECHS, REG
from trainset import tgt_vector, window_features
from model import load_any as load
import realdata

DEV_YEARS = (2019, 2021)


def skill(pc, mean, Y, C, dumb=None):
    acc = float((pc.argmax(1) == C).mean())
    sk = []
    for j in range(Y.shape[1]):
        m = np.isfinite(Y[:, j])
        if m.sum() < 5:
            sk.append(np.nan); continue
        if dumb is None:
            med = np.array([np.nanmedian(Y[m & (C == c), j]) if (m & (C == c)).any() else np.nanmedian(Y[m, j]) for c in range(4)])
        else:
            med = dumb[:, j]
        mae = np.abs(mean[m, j] - Y[m, j]).mean(); mae0 = np.abs(med[C[m]] - Y[m, j]).mean()
        sk.append(max(0.0, 1 - mae / max(mae0, 1e-9)))
    return float(np.nanmean([acc] + sk)), acc, sk


def val_score(W, set_version):
    z = np.load(f'{DATA}/set_{set_version}.npz')
    V = z['V']; X, Y, C = z['X'][V], z['Y'][V], z['C'][V]
    tr = ~V
    dumb = np.array([[np.nanmedian(z['Y'][tr & (z['C'] == c), j]) for j in range(Y.shape[1])] for c in range(4)])
    p, m, s = W(X)
    return skill(p, m, Y, C, dumb)


# ------------------------------------------------------------------ real units
def real_unit(split, uid):
    f = f'{DATA}/real_{split}/{uid.replace("*", "s")}.npz'
    z = np.load(f)
    return {k: z[k] for k in ('op', 'P', 'HI', 'meas')}, z['t']


def windows_for(t, kind, years=None):
    """Start/length index pairs of the evaluation windows."""
    tt = pd.to_datetime(t, unit='s')
    out = []
    yrs = sorted(set(tt.year)) if years is None else years
    for y in yrs:
        if kind == 'year':
            m = np.flatnonzero(tt.year == y)
            if len(m) > 24 * 300: out.append((m[0], len(m)))
        elif kind == 'quarter':
            for q in range(4):
                m = np.flatnonzero((tt.year == y) & ((tt.month - 1) // 3 == q))
                if len(m) > 24 * 80: out.append((m[0], len(m)))
        elif kind == 'month':
            for mo in (1, 4, 7, 10):
                m = np.flatnonzero((tt.year == y) & (tt.month == mo))
                if len(m) > 24 * 25: out.append((m[0], len(m)))
    return out


def predict_windows(W, rec, t, ctx, wins, force, pr=None):
    pr = realdata.prices() if pr is None else pr
    tt = pd.to_datetime(t, unit='s', utc=True)
    price = pr.price.reindex(tt).values; gas = pr.gas.reindex(tt).values
    X = []
    rng = np.random.default_rng(0)
    for s, L in wins:
        v, _ = window_features(rec, t, price, gas, -6, ctx, rng, L // 24 + 1, start=s, gap=False, force=force)
        X.append(v)
    if not X:
        return None
    return W(np.array(X))


def pool(p, m, s):
    """Pool window predictions: tech by summed log-probabilities, targets by precision-weighted mean; sigma kept
    at the mean window sigma (window errors of one unit are correlated) -- widened later by dev-fitted factors."""
    lp = np.log(np.clip(p, 1e-6, 1)).sum(0); pc = np.exp(lp - lp.max()); pc /= pc.sum()
    w = 1 / s ** 2
    mean = (w * m).sum(0) / w.sum(0)
    sig = s.mean(0)
    return pc, mean, sig


def dev_table(W, kinds=('month', 'quarter', 'year', 'pooled'), force=dict(price=True, ctx=True, net=False)):
    """Per dev unit and window kind: predictions vs truth over the dev years."""
    U = realdata.units(); U = U[U.split == 'dev']
    rows = []
    for uid in U.unit_id:
        rec, t = real_unit('dev', uid)
        tt = pd.to_datetime(t, unit='s')
        m = np.isin(tt.year, DEV_YEARS)
        rec = {k: v[m] for k, v in rec.items()}; t = t[m]
        tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
        if not np.isfinite(tr['pmax']):
            continue
        y = tgt_vector(tr)
        ctx = realdata.context(uid)
        c = TECHS.index(realdata.truth_tech(realdata.units().set_index('unit_id').loc[uid]))
        for kind in kinds:
            wins = windows_for(t, 'year' if kind == 'pooled' else kind)
            out = predict_windows(W, rec, t, ctx, wins, force)
            if out is None:
                continue
            p, mm, ss = out
            if kind == 'pooled':
                p, mm, ss = [a[None] for a in pool(p, mm, ss)]
            for i in range(len(p)):
                rows.append(dict(unit=uid, kind=kind, C=c, p=p[i], m=mm[i], s=ss[i], y=y))
    return rows


def dev_score(rows, kind='year'):
    r = [x for x in rows if x['kind'] == kind]
    p = np.array([x['p'] for x in r]); m = np.array([x['m'] for x in r]); Y = np.array([x['y'] for x in r])
    C = np.array([x['C'] for x in r])
    return skill(p, m, Y, C)


if __name__ == '__main__':
    W = load(sys.argv[1])
    print('val', val_score(W, sys.argv[2]))
    rows = dev_table(W)
    for k in ('month', 'quarter', 'year', 'pooled'):
        s, acc, sk = dev_score(rows, k)
        print(k, round(s, 3), 'acc', round(acc, 2), dict(zip(REG, np.round(sk, 2))))
