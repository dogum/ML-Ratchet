"""The home energy-signature law's REAL exam. Run ONCE, after results/bldg_ratchet.json is final. Chooses nothing.

Exam: RECS 2020 single-family homes in BuildingData's exam split (annual site energy from EIA's billing-based
estimates / floor area; 2020 heating and cooling degree days, base 18 C).
Rows: the released law; every rung (after the fact); the degree-day energy signature a + b HDD + c CDD fitted on the
dev homes (textbook, tuned on dev); the same signature fitted on BuildingSim; BuildingData's style of baseline
(log energy on log area + HDD + CDD, fitted on dev); dumb: dev median EUI.
No network in the collection predicts annual home energy from these inputs (LoadCast is hourly BDG2; BuildingDetective
reads meters), so there is no network row.
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import lawratchet as lr
import law_bldg as L

ROOT = lr.ROOT
OUT = f'{ROOT}/results/exam_bldg.json'


def mets(y, p, sig=None):
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 3:
        return {'n': int(ok.sum())}
    y, p = y[ok], p[ok]
    m = {'n': int(ok.sum()), 'mdape': float(np.median(np.abs(p / y - 1)) * 100), 'mae': float(np.mean(np.abs(p - y))),
         'bias_median_pct': float(np.median(p / y - 1) * 100), 'rho': float(spearmanr(y, p)[0]), 'r2': lr.r2(y, p)}
    if sig is not None:
        m['cov68'] = float(np.mean(np.abs(p - y) <= sig))
    return m


def loglin_fit(df):
    X = np.c_[np.ones(len(df)), np.log(df.area_m2), df.hdd / 1e3, df.cdd / 1e3]
    return np.linalg.lstsq(X, np.log(df.eui * df.area_m2), rcond=None)[0]


def loglin(df, b):
    X = np.c_[np.ones(len(df)), np.log(df.area_m2), df.hdd / 1e3, df.cdd / 1e3]
    return np.exp(X @ b) / df.area_m2.values


def main(dry=False):
    global OUT
    if dry:
        OUT = OUT.replace('exam_', 'dryrun_')   # code check on DEV units only; never the exam
    if os.path.exists(OUT) and not dry:
        sys.exit(f'{OUT} exists: the exam runs once.')
    dry or open(f'{ROOT}/logs/exam_bldg.started', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    R = json.load(open(f'{ROOT}/results/bldg_ratchet.json'))
    rel = next(r for r in R['rungs'] if r['id'] == R['released'])
    dev = L.load_dev()
    c_dev = L.fit_signature(dev)
    tr = L.load_bank(L.VERSIONS[-1], 'train')
    c_sim = L.fit_signature(tr)
    b_ll = loglin_fit(dev)
    ex = L.recs_rows('dev' if dry else 'exam')
    X = L.X_of(ex); y = ex.eui.values
    kn = lr.knee(R)
    P = {'law': lr.law_eval(rel, X), 'sim_law': lr.law_eval(kn, X), 'signature_dev': L.signature(ex, c_dev), 'signature_sim': L.signature(ex, c_sim),
         'loglin_dev': loglin(ex, b_ll), 'dumb': np.full(len(y), float(np.median(dev.eui)))}
    names = {'law': f'LawFinder law ({rel["id"]})', 'sim_law': f'BuildingSim\'s law ({kn["id"]})',
             'signature_dev': f'Degree-day signature {c_dev[0]:.0f} + {c_dev[1]:.1f} HDD + {c_dev[2]:.1f} CDD (dev)',
             'signature_sim': f'Degree-day signature {c_sim[0]:.0f} + {c_sim[1]:.1f} HDD + {c_sim[2]:.1f} CDD (BuildingSim)',
             'loglin_dev': 'BuildingData-style: log energy ~ log area + HDD + CDD (dev)', 'dumb': 'Dev median EUI'}
    res = {'knee': kn['id'], 'released': rel['id'], 'expr': rel['expr'], 'sigma': R['bars']['sigma'], 'coef_dev': c_dev, 'coef_sim': c_sim,
           'rows': {k: {'name': names[k], **mets(y, p, R['bars']['sigma'] if k == 'law' else None)} for k, p in P.items()},
           'rungs': {r['id']: mets(y, lr.law_eval(r, X)) for r in R['rungs']}}
    res['by_fuel'] = {f: {k: mets(y[ex.heating_fuel == f], P[k][ex.heating_fuel == f]) for k in P}
                      for f in ex.heating_fuel.value_counts().index[:3]}
    for k, p in P.items():
        ex[f'pred_{k}'] = p
    dry or ex.to_parquet(f'{ROOT}/results/exam_bldg_points.parquet')
    json.dump(res, open(OUT, 'w'), indent=1, default=str)
    for k, v in res['rows'].items():
        if v['n'] < 3:
            continue
        print(f"{v['name']:70s} n={v['n']} MdAPE={v['mdape']:.1f}% bias={v['bias_median_pct']:+.1f}% rho={v['rho']:.3f}"
              + (f" cov68={v['cov68']:.2f}" if 'cov68' in v else ''))


if __name__ == '__main__':
    main(dry='--dry-dev' in sys.argv)
