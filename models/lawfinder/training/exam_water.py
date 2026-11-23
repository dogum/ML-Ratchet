"""The water-balance law's REAL exam. Run ONCE, after results/water_ratchet.json is final. Chooses nothing.

Exam: the 619 CAMELS-US exam basins (WaterData splits), validation decade WY1990-99. Truth: runoff ratio from gauged
Q (NWIS where present, else Caravan) over the decade, on days with Q. Inputs: the same decade's ERA5-Land forcing
(no Q) and the attribute-prior soil storage.

Rows: the released law; every rung of the ladder (reported after the fact); Budyko-Fu with B's w = 2.6; Fu with w
fitted on the dev basins (the textbook law tuned on the same dev data); Fu with w fitted on the simulated bank;
FlowCast v1.0 (the collection's network for this platform: ungauged daily flow, its decade mean / P);
the dumb baseline (dev median runoff ratio).
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import waterdata as wd
import lawratchet as lr
import law_water as L
from water_real import basin_record, VAL

ROOT = lr.ROOT
OUT = f'{ROOT}/results/exam_water.json'


def metrics(y, p, sig=None):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 3:
        return {'n': int(ok.sum())}
    y, p = y[ok], p[ok]
    m = {'n': int(ok.sum()), 'mae': float(np.mean(np.abs(p - y))), 'rmse': float(np.sqrt(np.mean((p - y) ** 2))),
         'bias': float(np.mean(p - y)), 'r2': lr.r2(y, p), 'rho': float(spearmanr(y, p)[0]),
         'median_abs': float(np.median(np.abs(p - y)))}
    if sig is not None:
        m['cov68'] = float(np.mean(np.abs(p - y) <= np.asarray(sig)[ok]))
    return m


def main(dry=False):
    global OUT
    if dry:
        OUT = OUT.replace('exam_', 'dryrun_')   # code check on DEV units only; never the exam
    if os.path.exists(OUT) and not dry:
        sys.exit(f'{OUT} exists: the exam runs once. Delete it only if the run crashed before scoring.')
    dry or open(f'{ROOT}/logs/exam_water.started', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    R = json.load(open(f'{ROOT}/results/water_ratchet.json'))
    rel = next(r for r in R['rungs'] if r['id'] == R['released'])
    sig = R['bars']['sigma']
    # textbook constants from sim (last physics version) and from dev only
    tr = L.load_bank(L.VERSIONS[-1], 'train')
    w_sim = L.fit_fu(tr['aridity'].values, tr['rr'].values)
    dev = L.load_dev()
    w_dev = L.fit_fu(dev['aridity'].values, dev['rr'].values)
    dev_med = float(np.median(dev['rr']))
    ids = wd.basins('camels', split='dev' if dry else 'exam')
    t0 = time.time()
    ex = pd.DataFrame([basin_record(b) for b in ids])
    ex = ex[np.isfinite(ex['rr'])].reset_index(drop=True)
    X = L.X_of(ex); y = ex['rr'].values
    kn = lr.knee(R)
    P = {'law': lr.law_eval(rel, X), 'sim_law': lr.law_eval(kn, X), 'fu_2.6': L.fu_rr(X[:, 0], 2.6), 'fu_dev': L.fu_rr(X[:, 0], w_dev),
         'fu_sim': L.fu_rr(X[:, 0], w_sim), 'dumb': np.full(len(y), dev_med)}
    from _paths import other_work
    fc = pd.read_parquet(f"{other_work('flowcast')}/results/r1b_exam_cal.parquet")   # FlowCast's exam output
    fc = fc[fc.period == 'val'].set_index('basin')['beta']
    P['flowcast'] = ex['basin'].map(fc).values * y          # KGE beta = mean(sim)/mean(obs) over the decade
    res = {'rows': {}, 'knee': kn['id'], 'released': rel['id'], 'released_expr': rel['expr'], 'w_dev': w_dev, 'w_sim': w_sim,
           'dev_median': dev_med, 'sigma': sig, 'n_exam': int(len(y)), 'period': VAL}
    names = {'law': f'LawFinder law ({rel["id"]})', 'sim_law': f'WaterSim\'s law ({kn["id"]}, simplest within 0.02 of best sim R2)', 'fu_2.6': 'Budyko-Fu, w = 2.6 (WaterData baseline)',
             'fu_dev': f'Budyko-Fu, w = {w_dev:.2f} fitted on dev', 'fu_sim': f'Budyko-Fu, w = {w_sim:.2f} fitted on WaterSim',
             'flowcast': 'FlowCast v1.0 (network, ungauged)', 'dumb': f'Dev median ({dev_med:.3f})'}
    for k, p in P.items():
        res['rows'][k] = {'name': names[k], **metrics(y, p, np.full(len(y), sig) if k == 'law' else None)}
    # strata
    strata = {'snow (f_s > 0.3)': ex['frac_snow'] > 0.3, 'arid (phi > 1)': ex['aridity'] > 1,
              'humid (phi < 0.6)': ex['aridity'] < 0.6}
    res['strata'] = {s: {k: metrics(y[m], P[k][m]) for k in P} for s, m in strata.items()}
    # every rung, after the fact
    res['rungs'] = {r['id']: metrics(y, lr.law_eval(r, X)) for r in R['rungs']}
    ex['pred_law'] = P['law']; ex['pred_fu_dev'] = P['fu_dev']; ex['pred_flowcast'] = P['flowcast']
    ex['pred_fu_2.6'] = P['fu_2.6']
    dry or ex.to_parquet(f'{ROOT}/results/exam_water_points.parquet')
    res['wall_s'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), indent=1)
    for k, v in res['rows'].items():
        if v['n'] < 3:
            continue
        print(f"{v['name']:45s} n={v['n']} MAE={v['mae']:.4f} R2={v['r2']:.3f} rho={v['rho']:.3f} bias={v['bias']:+.4f}"
              + (f" cov68={v['cov68']:.2f}" if 'cov68' in v else ''))


if __name__ == '__main__':
    main(dry='--dry-dev' in sys.argv)
