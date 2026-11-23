"""The grid laws' REAL exam (inertia and nadir). Run ONCE, after both grid ratchets are final. Chooses nothing.

Exam: GridData's 412 GB exam events (2018-2025, NESO 1 s frequency). Inertia law: the 410 with NESO outturn inertia
and demand. Nadir law: the events with a loss label (medium or low confidence) and NESO inertia; truth = f0 - nadir
(GridData catalogue, lowest 1 s value within 90 s).

Rows, inertia: the law; every rung (after the fact); k x demand with k fitted on the 4 dev events (GridPulse's best
baseline, 7.0 s); GridPulse v1.0 (the collection's network, 60 s of trace + demand); dumb: dev median inertia.
Rows, nadir: the law; every rung; the swing-equation ramp law k L^2/E with k fitted on dev; GridPulse's nadir
forecast from the first 5 s of the trace (more information than the law has: context); dumb: dev median drop.
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import lawratchet as lr
import law_grid_inertia as LI, law_grid_nadir as LN
from grid_real import events

ROOT = lr.ROOT
OUT = f'{ROOT}/results/exam_grid.json'
from _paths import other_work
GP = f"{other_work('gridpulse')}/logs/exam_gb_predictions.parquet"   # GridPulse's exam output


def mets(y, p, sig=None, rel_sig=False):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 3:
        return {'n': int(ok.sum())}
    y, p = y[ok], p[ok]
    m = {'n': int(ok.sum()), 'mae': float(np.mean(np.abs(p - y))), 'mdape': float(np.median(np.abs(p / y - 1)) * 100),
         'bias_median': float(np.median(p - y)), 'rho': float(spearmanr(y, p)[0]), 'r2': lr.r2(y, p)}
    if sig is not None:
        m['cov68'] = float(np.mean(np.abs(p - y) <= sig))
    return m


def main(dry=False):
    global OUT
    if dry:
        OUT = OUT.replace('exam_', 'dryrun_')   # code check on DEV units only; never the exam
    if os.path.exists(OUT) and not dry:
        sys.exit(f'{OUT} exists: the exam runs once.')
    dry or open(f'{ROOT}/logs/exam_grid.started', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    RI = json.load(open(f'{ROOT}/results/grid_inertia_ratchet.json'))
    RN = json.load(open(f'{ROOT}/results/grid_nadir_ratchet.json'))
    reli = next(r for r in RI['rungs'] if r['id'] == RI['released'])
    reln = next(r for r in RN['rungs'] if r['id'] == RN['released'])
    devi = LI.load_dev(); devn = LN.load_dev()
    k_dem = float(np.median(devi.inertia_GVAs / (devi.demand_MW / 1e3)))
    k_sw = float(np.median(devn.df_nadir_Hz / ((devn.loss_MW / 1e3) ** 2 / devn.inertia_GVAs)))
    ex = events('dev' if dry else 'exam')
    gp = pd.read_parquet(GP).set_index('event_id')
    ex = ex.join(gp[['E_60', 'nadir_5', 'f0']].rename(columns={'f0': 'gp_f0'}), on='event_id')
    res = {'inertia': {}, 'nadir': {}, 'k_demand_GVAs_per_GW': k_dem, 'k_swing': k_sw}
    # ---- inertia
    ei = ex[np.isfinite(ex.inertia_GVAs) & np.isfinite(ex.demand_MW)].reset_index(drop=True)
    Xi = LI.X_of(ei); yi = ei.inertia_GVAs.values
    kni, knn = lr.knee(RI), lr.knee(RN)
    Pi = {'law': lr.law_eval(reli, Xi), 'sim_law': lr.law_eval(kni, Xi), 'k_x_demand': k_dem * ei.demand_MW.values / 1e3, 'gridpulse': ei.E_60.values,
          'dumb': np.full(len(yi), float(np.median(devi.inertia_GVAs)))}
    ni = {'law': f'LawFinder law ({reli["id"]})', 'sim_law': f'GridSim\'s law ({kni["id"]})', 'k_x_demand': f'{k_dem:.2f} s x demand (dev fit)',
          'gridpulse': 'GridPulse v1.0 (network: 60 s trace + demand)', 'dumb': 'Dev median inertia'}
    res['inertia'] = {'knee': kni['id'], 'released': reli['id'], 'expr': reli['expr'], 'sigma': RI['bars']['sigma'],
                      'rows': {k: {'name': ni[k], **mets(yi, p, RI['bars']['sigma'] if k == 'law' else None)} for k, p in Pi.items()},
                      'by_year': {int(y): {k: mets(yi[ei.year == y], Pi[k][ei.year == y]) for k in Pi} for y in sorted(ei.year.unique())},
                      'rungs': {r['id']: mets(yi, lr.law_eval(r, Xi)) for r in RI['rungs']}}
    # ---- nadir
    en = ex[np.isfinite(ex.loss_MW) & np.isfinite(ex.inertia_GVAs) & np.isfinite(ex.demand_MW)].reset_index(drop=True)
    Xn = LN.X_of(en); yn = en.df_nadir_Hz.values
    Pn = {'law': lr.law_eval(reln, Xn), 'sim_law': lr.law_eval(knn, Xn), 'swing_ramp': LN.swing_ramp(en.loss_MW.values / 1e3, en.inertia_GVAs.values, k_sw),
          'gridpulse_5s': (en.gp_f0 - en.nadir_5).values,
          'dumb': np.full(len(yn), float(np.median(devn.df_nadir_Hz)))}
    nn = {'law': f'LawFinder law ({reln["id"]})', 'sim_law': f'GridSim\'s law ({knn["id"]})', 'swing_ramp': f'Swing equation + ramp response, k L^2/E (k = {k_sw:.3g}, dev)',
          'gridpulse_5s': 'GridPulse v1.0 nadir forecast after 5 s of trace (context: reads the trace)',
          'dumb': 'Dev median drop'}
    res['nadir'] = {'knee': knn['id'], 'released': reln['id'], 'expr': reln['expr'], 'sigma': RN['bars']['sigma'],
                    'rows': {k: {'name': nn[k], **mets(yn, p, RN['bars']['sigma'] if k == 'law' else None)} for k, p in Pn.items()},
                    'by_confidence': {c: {k: mets(yn[en.loss_confidence == c], Pn[k][en.loss_confidence == c]) for k in Pn}
                                      for c in ('medium', 'low')},
                    'rungs': {r['id']: mets(yn, lr.law_eval(r, Xn)) for r in RN['rungs']}}
    for k, p in Pi.items():
        ei[f'pred_{k}'] = p
    for k, p in Pn.items():
        en[f'pred_{k}'] = p
    keep = ['event_id', 'year', 'loss_MW', 'loss_confidence', 'inertia_GVAs', 'demand_MW', 'ibr_share', 'df_nadir_Hz']
    dry or ei[[c for c in keep + [f'pred_{k}' for k in Pi]]].to_parquet(f'{ROOT}/results/exam_grid_inertia_points.parquet')
    dry or en[[c for c in keep + [f'pred_{k}' for k in Pn]]].to_parquet(f'{ROOT}/results/exam_grid_nadir_points.parquet')
    json.dump(res, open(OUT, 'w'), indent=1)
    for part in ('inertia', 'nadir'):
        for k, v in res[part]['rows'].items():
            if v['n'] < 3:
                continue
            print(f"{part:8s} {v['name']:62s} n={v['n']} MAE={v['mae']:.4g} MdAPE={v['mdape']:.1f}% rho={v['rho']:.3f}"
                  + (f" cov68={v['cov68']:.2f}" if 'cov68' in v else ''))


if __name__ == '__main__':
    main(dry='--dry-dev' in sys.argv)
