"""Tables from a grade file (exam.py): Detective vs baselines per target, window kind, technology; coverage.

usage: python report.py <grade pkl> [sigma_factors.json]
"""
from common import *
import json
import numpy as np, pandas as pd
from pdcore import REG, TECHS
from exam import decode

FRAC = {'pmin_frac', 'hot', 'warm', 'cold', 'cf'}
NAMES = dict(pmin_frac='min stable load (frac of pmax)', ramp_up='ramp up (p99, frac/h)', ramp_dn='ramp down (p99, frac/h)',
             starts_yr='starts per year', min_up='min up time (h)', med_up='median run (h)', min_dn='min down time (h)',
             hot='hot-start share', warm='warm-start share', cold='cold-start share', hr_full='full-load heat rate',
             hr_min='part-load heat rate', cf='capacity factor')


def natural(v):
    """Transformed vector -> natural units array in REG order (NaN-safe)."""
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return np.full(len(REG), np.nan)
    sg = lambda a: 1 / (1 + np.exp(-a))  # noqa
    return np.array([sg(v[0]), np.exp(v[1]), np.exp(v[2]), np.expm1(v[3]), np.exp(v[4]), np.exp(v[5]), np.exp(v[6]),
                     sg(v[7]), sg(v[8]), sg(v[9]), np.exp(v[10]), np.exp(v[10] + v[11]), sg(v[12])])


def err(pred, true, k):
    if k in FRAC:
        return np.abs(pred - true)
    if k == 'starts_yr':
        return np.abs((pred + 1) / (true + 1) - 1)
    return np.abs(pred / true - 1)


def table(df, mode='ctx', kinds=('month', 'quarter', 'year', 'pooled'), techs=None):
    out = []
    for kind in kinds:
        d = df[df.kind == kind]
        if techs is not None:
            d = d[d.tech.isin(techs)]
        if not len(d):
            continue
        Y = np.array([natural(v) for v in d.y])
        Pm = np.array([natural(v) for v in d[f'm_{mode}']])
        acc = float((np.array([np.argmax(p) for p in d[f'p_{mode}']]) == d.C.values).mean())
        row = dict(kind=kind, mode=mode, n=len(d), units=d.unit.nunique(), tech_acc=acc)
        if mode == 'ctx' and 'rule' in d:
            pr = [np.asarray(p) * (np.asarray(r) if r is not None else 1) for p, r in zip(d['p_ctx'], d['rule'])]
            row['tech_acc_rule'] = float((np.array([np.argmax(p) for p in pr]) == d.C.values).mean())
        for j, k in enumerate(REG):
            m = np.isfinite(Y[:, j]) & np.isfinite(Pm[:, j])
            row[k] = float(np.median(err(Pm[m, j], Y[m, j], k))) if m.sum() else np.nan
            for b in ('b_stats', 'b_default', 'b_923', 'b_dumb'):
                if b not in d:
                    continue
                Bv = np.array([natural(v) for v in d[b]])
                mb = np.isfinite(Y[:, j]) & np.isfinite(Bv[:, j])
                row[f'{k}|{b}'] = float(np.median(err(Bv[mb, j], Y[mb, j], k))) if mb.sum() >= 3 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def coverage(df, mode='ctx', kind='year', factors=None):
    d = df[df.kind == kind]
    Y = np.array(list(d.y)); M = np.array(list(d[f'm_{mode}'])); S = np.array(list(d[f's_{mode}']))
    f = np.ones(len(REG)) if factors is None else np.array([factors.get(k, 1.0) for k in REG])
    res = {}
    for j, k in enumerate(REG):
        m = np.isfinite(Y[:, j])
        z = np.abs(M[m, j] - Y[m, j]) / (S[m, j] * f[j])
        res[k] = dict(n=int(m.sum()), c68=float(np.mean(z <= 1)), c95=float(np.mean(z <= 2)))
    return res


def fit_factors(df, mode='ctx', kinds=('month', 'quarter', 'year', 'pooled')):
    """Widen-only sigma factor per output so that >= 68.3 % of dev windows fall within 1 sigma."""
    d = df[df.kind.isin(kinds)]
    Y = np.array(list(d.y)); M = np.array(list(d[f'm_{mode}'])); S = np.array(list(d[f's_{mode}']))
    fac = {}
    for j, k in enumerate(REG):
        m = np.isfinite(Y[:, j])
        z = np.abs(M[m, j] - Y[m, j]) / S[m, j]
        fac[k] = float(max(1.0, np.quantile(z, 0.683))) if m.sum() >= 5 else 1.0
    return fac


def print_table(t, title):
    print('\n' + title)
    cols = ['kind', 'units', 'tech_acc'] + REG
    print(t[cols].round(3).to_string(index=False))
    bl = [c for c in t.columns if '|' in c]
    if bl:
        b = t[['kind'] + bl].copy()
        print(b.round(3).T.to_string())


if __name__ == '__main__':
    df = pd.read_pickle(sys.argv[1])
    fac = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    for mode in ('ctx', 'out'):
        print_table(table(df, mode), f'== mode {mode}: median error (fractions: abs; others: relative) ==')
    print('\ncoverage (year windows, ctx):', json.dumps(coverage(df, 'ctx', 'year', fac), indent=0)[:2000])
