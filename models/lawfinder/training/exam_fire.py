"""The daily-growth law's REAL exam. Run ONCE, after results/fire_ratchet.json is final. Chooses nothing.

Exam: FireData's exam fire-days (fire_days_exam.parquet: FEDS PM perimeter -> the PM perimeter 24 h later, 2012-2024,
fires never in dev), with gridMET next-day wind and 100-h fuel moisture and the fire's LANDFIRE fuel group.
Score: log10 ratio of (predicted + 1 cell) / (observed + 1 cell) new area, 1 cell = 0.0081 km2 (FireData's convention).

Rows: the released law; every ladder rung (after the fact); Rothermel head ROS x perimeter x 24 h with k fitted on dev
(the textbook law tuned on dev) and with k fitted on FireSim; FireData's best baseline fl_prev (yesterday's growth
from the active fire line) and its clim (dumb: dev median growth by size bin) and persist rows; FireCast v1.0
(the collection's network; its thresholded new area).
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import lawratchet as lr
import law_fire as L

ROOT = lr.ROOT
OUT = f'{ROOT}/results/exam_fire.json'
from _paths import FIRE, other_work
B = f'{FIRE}/data/proc/baselines/spread_exam.jsonl'
FC = f"{other_work('firecast')}/out/exam/firecast_r2.jsonl"   # FireCast's exam output


def lmetrics(ylog, plog, sig=None):
    ok = np.isfinite(ylog) & np.isfinite(plog)
    if ok.sum() < 3:
        return {'n': int(ok.sum())}
    e = plog[ok] - ylog[ok]
    m = {'n': int(ok.sum()), 'mae_log10': float(np.mean(np.abs(e))), 'median_abs_log10': float(np.median(np.abs(e))),
         'bias_log10': float(np.median(e)), 'rho': float(spearmanr(ylog[ok], plog[ok])[0]),
         'within_x3': float(np.mean(np.abs(e) <= np.log10(3)))}
    if sig is not None:
        m['cov68'] = float(np.mean(np.abs(e) <= sig))
    return m


def baseline_rows(dry=False):
    rows = []
    with open(B.replace('_exam', '_dev') if dry else B) as f:
        for line in f:
            for r in json.loads(line)['rows']:
                if r['method'] in ('fl_prev', 'clim', 'persist'):
                    rows.append((r['fire_id'], r['k'], r['method'], r['new_pred_km2']))
    b = pd.DataFrame(rows, columns=['fire_id', 'k', 'method', 'pred']).pivot_table(index=['fire_id', 'k'], columns='method', values='pred')
    fc = []
    with open(FC) as f:
        for line in f:
            j = json.loads(line)
            for r in j.get('rows', [j]):
                if r.get('mode', 'F') == 'F':
                    fc.append((r['fire_id'], r['k'], r['new_pred_km2']))
    fc = pd.DataFrame(fc, columns=['fire_id', 'k', 'firecast']).drop_duplicates(['fire_id', 'k']).set_index(['fire_id', 'k'])
    return b.join(fc, how='outer')


def main(dry=False):
    global OUT
    if dry:
        OUT = OUT.replace('exam_', 'dryrun_')   # code check on DEV units only; never the exam
    if os.path.exists(OUT) and not dry:
        sys.exit(f'{OUT} exists: the exam runs once.')
    dry or open(f'{ROOT}/logs/exam_fire.started', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    R = json.load(open(f'{ROOT}/results/fire_ratchet.json'))
    rel = next(r for r in R['rungs'] if r['id'] == R['released'])
    sig = R['bars']['sigma']
    dev = L.load_dev()
    k_dev = L.fit_k(L.rothermel_growth(dev), L.target(dev))
    tr = L.load_bank(L.VERSIONS[-1], 'train')
    k_sim = L.fit_k(L.rothermel_growth(tr), L.target(tr))
    ex = L.real_rows('dev' if dry else 'exam')
    X = L.X_of(ex); y = L.target(ex)
    raw = L.rothermel_growth(ex)
    kn = lr.knee(R)
    P = {'law': lr.law_eval(rel, X), 'sim_law': lr.law_eval(kn, X), 'rothermel_dev': np.log10(k_dev * raw + L.CELL),
         'rothermel_sim': np.log10(k_sim * raw + L.CELL)}
    bl = baseline_rows(dry)
    j = ex.set_index(['fire_id', 'k']).join(bl, how='left')
    for c, name in (('fl_prev', 'fl_prev'), ('clim', 'clim'), ('persist', 'persist'), ('firecast', 'firecast')):
        P[name] = np.log10(j[c].values.astype(float) + L.CELL)
    names = {'law': f'LawFinder law ({rel["id"]})', 'sim_law': f'FireSim\'s law ({kn["id"]})', 'rothermel_dev': f'Rothermel ROS x perimeter, k = {k_dev:.3g} (dev)',
             'rothermel_sim': f'Rothermel ROS x perimeter, k = {k_sim:.3g} (FireSim)',
             'fl_prev': "FireData best baseline: yesterday's growth from the active line",
             'clim': 'FireData clim (dumb: dev median growth by size)', 'persist': 'Persistence (no growth)',
             'firecast': 'FireCast v1.0 (network, threshold area)'}
    res = {'knee': kn['id'], 'released': rel['id'], 'released_expr': rel['expr'], 'k_dev': k_dev, 'k_sim': k_sim, 'sigma': sig,
           'n_rows': int(len(y)), 'n_fires': int(ex.fire_id.nunique()), 'rows': {}, 'growth_days': {}}
    grow = ex['new_km2'].values > 0
    for k, p in P.items():
        res['rows'][k] = {'name': names[k], **lmetrics(y, p, sig if k == 'law' else None)}
        res['growth_days'][k] = {'name': names[k], **lmetrics(y[grow], p[grow], sig if k == 'law' else None)}
    # the common subset where every row exists
    ok = np.all([np.isfinite(p) for p in P.values()], axis=0)
    res['common'] = {k: {'name': names[k], **lmetrics(y[ok], p[ok])} for k, p in P.items()}
    res['strata'] = {}
    for s, m in {'small (<5 km2)': ex.area0_km2 < 5, 'medium (5-50)': (ex.area0_km2 >= 5) & (ex.area0_km2 < 50),
                 'large (>=50)': ex.area0_km2 >= 50, 'recent 2022-24': ex.year >= 2022}.items():
        m = m.values
        res['strata'][s] = {k: lmetrics(y[m], P[k][m]) for k in ('law', 'sim_law', 'rothermel_dev', 'fl_prev', 'firecast')}
    res['rungs'] = {r['id']: lmetrics(y, lr.law_eval(r, X)) for r in R['rungs']}
    out = ex.copy()
    for k, p in P.items():
        out[f'pred_{k}'] = p
    out['y'] = y
    dry or out.to_parquet(f'{ROOT}/results/exam_fire_points.parquet')
    json.dump(res, open(OUT, 'w'), indent=1)
    for k, v in res['rows'].items():
        if v['n'] < 3:
            continue
        g = res['growth_days'][k]
        print(f"{v['name']:58s} all n={v['n']} MAE={v['mae_log10']:.3f} bias={v['bias_log10']:+.2f} rho={v['rho']:.3f}"
              f" | growth MAE={g['mae_log10']:.3f} rho={g['rho']:.3f}" + (f" cov68={v['cov68']:.2f}" if 'cov68' in v else ''))


if __name__ == '__main__':
    main(dry='--dry-dev' in sys.argv)
