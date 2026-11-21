"""Reproduction re-score of the released BasinDetective on the 619 CAMELS-US exam basins (added 18 Nov 2026, after
the import; seen: the exam was already taken once, nothing is chosen or changed from this).

It runs the shipped exam path unchanged: realfeats.build('exam') (features, truth, GR4J / Budyko baselines) and
exam.main(weights/basindetective_v1.npz), with two stand-ins for data that are not on disk, both documented here:
  - WaterData's daily GR4J simulations (proc/baselines/sims/gr4j_gauged.parquet) were not shipped. They are
    re-simulated exactly as data/scripts/run_gr4j.py writes them: waterdata.models.gr4j_snow with each basin's shipped
    calibrated parameters (proc/baselines/gr4j/<id>.json), 1980-10-01..2023-12-30, float32. Written under WORK only.
  - USGS NWIS annual peaks: the data root's proc/nwis/peaks.parquet (built by data/scripts/build_nwis.py) when it exists
    (fetched later on 18 Nov 2026); without it an empty stand-in, and the peak-code regulation row cannot be re-scored
    (its AUC comes out NaN). WORK/feats/real_exam.peaks records which one built the cached features; a mismatch
    rebuilds them.
Then a paired bootstrap over basins (10,000 resamples, numpy default_rng(20261118), the same resamples for every row)
of BasinDetective minus each baseline: Spearman rho and median |error| of the six changes and six levels, and the
regulation AUCs (also against chance, 0.5).

    ML_RATCHET_WATER=<data root with data/proc/caravan/camels> python rescore.py
Writes WORK/feats/real_exam*.{npz,parquet}, WORK/exam.json, WORK/feats/exam_pred.parquet, WORK/baselines/ (all
git-ignored) and, in this folder, exam_per_basin.csv (no time series) and exam_rescore.json.
"""
import common  # noqa: F401  (WORK, threads, paths)
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import D, DELTA_T, EPSQ
from _paths import SRC, MODEL, WATER_PROC
import waterdata as wd

WEIGHTS = f'{MODEL}/weights/basindetective_v1.npz'
SEED, NBOOT = 20261118, 10000
SIM0, SIM1 = '1980-10-01', '2023-12-30'
LEVEL = {'d_logq': 'q_mean', 'd_rr': 'runoff_ratio', 'd_bfi': 'bfi_lh', 'd_logq5': 'q5', 'd_logq95': 'q95',
         'd_hfd': 'hfd_mean'}
LOGT = {'d_logq', 'd_logq5', 'd_logq95'}


def stage_baselines():
    """WORK/baselines/{gr4j/*.json (copies), sims/gr4j_gauged.parquet (re-simulated as run_gr4j.py does)}."""
    from waterdata.models import gr4j_snow
    out = f'{D}/baselines'
    os.makedirs(f'{out}/sims', exist_ok=True)
    if os.path.exists(f'{out}/sims/gr4j_gauged.parquet'):
        return out
    shutil.copytree(f'{WATER_PROC}/baselines/gr4j', f'{out}/gr4j', dirs_exist_ok=True)
    ids = wd.basins('camels')
    sims = {}
    for b in ids:
        th = json.load(open(f'{out}/gr4j/{b}.json'))['params']
        f = wd.forcing(b, SIM0, SIM1)
        sims[b] = gr4j_snow(th, f.P.values, f.T_mean.values, f.PET.values).astype(np.float32)
    idx = wd.forcing(ids[0], SIM0, SIM1).index
    pd.DataFrame(sims, index=idx).to_parquet(f'{out}/sims/gr4j_gauged.parquet', compression='zstd')
    return out


def stage_peaks():
    """The real NWIS peaks when the data root has them, else an empty stand-in. Returns (proc root, label)."""
    if os.path.exists(f'{WATER_PROC}/nwis/peaks.parquet'):
        return WATER_PROC, 'nwis'
    p = f'{D}/proc_nopeaks/nwis'
    os.makedirs(p, exist_ok=True)
    pd.DataFrame({'gauge_id': pd.Series([], dtype=str), 'wy': pd.Series([], dtype='int64'),
                  'peak_cd': pd.Series([], dtype=str)}).to_parquet(f'{p}/peaks.parquet')
    return f'{D}/proc_nopeaks', 'empty stand-in'


# ---------------------------------------------------------------------------------------------- statistics
def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auc(y, s):
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def lvl(t, cal, d):
    return np.exp(np.log(cal + EPSQ) + d) - EPSQ if t in LOGT else cal + d


def boot(idx_all, fn):
    """Point value and 95 % percentile interval of fn over the shared resamples."""
    vals = np.array([fn(ix) for ix in idx_all])
    vals = vals[np.isfinite(vals)]
    return {'value': float(fn(np.arange(idx_all.shape[1]))), 'lo': float(np.percentile(vals, 2.5)),
            'hi': float(np.percentile(vals, 97.5)), 'n_boot': int(vals.size)}


def verdict_vs(diff, higher_better):
    if diff['lo'] > 0:
        return 'win' if higher_better else 'loss'
    if diff['hi'] < 0:
        return 'loss' if higher_better else 'win'
    return 'tie'


def row_verdict(vs):
    vv = [v['verdict'] for v in vs.values()]
    if 'loss' in vv:
        return 'loss'
    return 'win' if vv and all(v == 'win' for v in vv) else 'tie'


def compare(n, idx_all, y, model, baselines, higher_better_rho=True):
    """Spearman and median |error| of model vs each baseline on the basins where all three are finite."""
    out = {'rho': {}, 'mae': {}}
    for kind in ('rho', 'mae'):
        res = {'model': None, 'vs': {}}
        for name, x in baselines.items():
            if kind == 'rho' and x is None:
                continue
            xb = np.zeros(n) if x is None else x
            ok = np.isfinite(y) & np.isfinite(model) & np.isfinite(xb)
            yy, mm, bb = y[ok], model[ok], xb[ok]
            sub = idx_all[:, :ok.sum()] if ok.all() else None
            # resample within the finite basins (same generator stream, re-drawn per subset size)
            if sub is None:
                sub = np.random.default_rng(SEED).integers(0, ok.sum(), size=(NBOOT, ok.sum()))
            if kind == 'rho':
                f = lambda ix: spearman(mm[ix], yy[ix]) - spearman(bb[ix], yy[ix])
                fm = lambda ix: spearman(mm[ix], yy[ix])
                fb = lambda ix: spearman(bb[ix], yy[ix])
            else:
                f = lambda ix: np.median(np.abs(mm[ix] - yy[ix])) - np.median(np.abs(bb[ix] - yy[ix]))
                fm = lambda ix: np.median(np.abs(mm[ix] - yy[ix]))
                fb = lambda ix: np.median(np.abs(bb[ix] - yy[ix]))
            d = boot(sub, f)
            d['n'] = int(ok.sum())
            d['baseline'] = fb(np.arange(ok.sum()))
            d['model'] = fm(np.arange(ok.sum()))
            d['verdict'] = verdict_vs(d, kind == 'rho')
            res['vs'][name] = d
        res['verdict'] = row_verdict(res['vs'])
        out[kind] = res
    return out


def _sklearn_shim():
    """evaluate.py imports sklearn.metrics.roc_auc_score; neither venv here has scikit-learn. For a binary truth,
    roc_auc_score equals the Mann-Whitney statistic with ties counted half (the auc() above), so a stand-in module
    supplies exactly that when scikit-learn is missing."""
    try:
        import sklearn.metrics  # noqa: F401
        return False
    except ImportError:
        import types
        m = types.ModuleType('sklearn.metrics')
        m.roc_auc_score = lambda y, s: auc(np.asarray(y), np.asarray(s, float))
        pkg = types.ModuleType('sklearn')
        pkg.metrics = m
        sys.modules['sklearn'], sys.modules['sklearn.metrics'] = pkg, m
        return True


def main():
    shim = _sklearn_shim()
    import realfeats
    import exam
    realfeats.B = stage_baselines()
    realfeats.WATER_PROC, peaks_src = stage_peaks()
    ids = wd.basins('camels', 'exam')
    split = json.load(open(f'{WATER_PROC}/splits.json'))
    assert sorted(ids) == sorted(split['camels_us']['exam']) and len(ids) == 619
    mark = f'{D}/feats/real_exam.peaks'
    cached = open(mark).read().strip() if os.path.exists(mark) else 'empty stand-in'
    if not os.path.exists(f'{D}/feats/real_exam.npz') or cached != peaks_src:
        realfeats.build('exam')
        open(mark, 'w').write(peaks_src)
    exam.main(WEIGHTS)                      # -> WORK/exam.json, WORK/feats/exam_pred.parquet
    meta = pd.read_parquet(f'{D}/feats/real_exam_meta.parquet')
    pred = pd.read_parquet(f'{D}/feats/exam_pred.parquet')
    assert list(meta.basin) == list(pred.basin) and sorted(meta.basin) == sorted(ids)

    # ---- compare with the shipped exam.json
    new, old = json.load(open(f'{D}/exam.json')), json.load(open(f'{SRC}/exam.json'))
    diffs = []

    def walk(a, b, path):
        if isinstance(b, dict):
            for k in b:
                walk(a.get(k) if isinstance(a, dict) else None, b[k], path + [k])
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            av = a if isinstance(a, (int, float)) else np.nan
            if not (np.isfinite(av) and abs(av - b) <= 1e-9 * max(1, abs(b))):
                diffs.append({'path': '/'.join(path), 'shipped': b, 'rescored': None if not np.isfinite(av) else av})
    walk(new, old, [])

    # ---- per-basin file (no time series)
    pb = pd.DataFrame({'basin': meta.basin})
    for t in DELTA_T:
        pb[f'mu_{t}'] = pred[f'mu_{t}']
        pb[f'sd_{t}'] = pred[f'sd_{t}']
        pb[f'true_{t}'] = meta[f'true_{t}']
        pb[f'gr4j_{t}'] = meta[f'gr4j_{t}']
        lv = LEVEL[t]
        g, c = meta[f'gr4jlvl_{lv}'].values, meta[f'cal_{lv}'].values
        pb[f'gr4jown_{t}'] = (np.log(g + EPSQ) - np.log(c + EPSQ)) if t in LOGT else g - c
    for t in ('d_logq', 'd_rr'):
        pb[f'budyko_{t}'] = meta[f'budyko_{t}']
    for lv in LEVEL.values():
        pb[f'cal_{lv}'] = meta[f'cal_{lv}']
        pb[f'tgtobs_{lv}'] = meta[f'tgtobs_{lv}']
        pb[f'gr4jlvl_{lv}'] = meta[f'gr4jlvl_{lv}']
    pb['p_regulated'] = pred['p_reg']
    pb['mu_log_dor'], pb['sd_log_dor'] = pred['mu_log_dor'], pred['sd_log_dor']
    pb['dor'] = meta['dor']
    pb['peak_reg_frac'] = meta['peak_reg_frac']
    pb['reg_peak_code'] = (meta['peak_reg_frac'] >= 1 / 3).astype(int)
    pb['reg_dor_gt_0.1'] = (meta['dor'] > 0.1).astype(int)
    pb['gr4j_nse_cal'] = meta['gr4j_nse_cal']
    pb['seasdamp'] = meta['seasdamp']
    z = np.load(f'{D}/feats/real_exam.npz')
    names = [str(s) for s in z['names']]
    pb['c_fsnow'], pb['c_arid'] = z['X'][:, names.index('c_fsnow')], z['X'][:, names.index('c_arid')]
    pb.to_csv(f'{SRC}/exam_per_basin.csv', index=False, float_format='%.7g', lineterminator='\n')

    # ---- paired bootstrap
    n = len(pb)
    idx_all = np.random.default_rng(SEED).integers(0, n, size=(NBOOT, n))
    res = {'method': f'paired bootstrap over the {n} exam basins: {NBOOT} resamples with replacement, numpy '
                     f'default_rng({SEED}).integers, the same resamples for every row (a comparison restricted to the '
                     'basins where model, truth and baseline are all finite re-draws with the same seed); 95 % '
                     'percentile intervals of BasinDetective minus the baseline. Verdict per baseline: win or loss '
                     'only if the interval excludes 0; row: loss if any baseline wins, win only if it beats every '
                     'baseline, else tie.',
           'model': 'weights/basindetective_v1.npz (as released)', 'n_basins': n, 'changes': {}, 'levels': {},
           'regulation': {}, 'coverage': {}, 'exam_json_differences': diffs,
           'sklearn_shim': shim}
    for t in DELTA_T:
        y = pb[f'true_{t}'].values
        bl = {'GR4J model-implied': pb[f'gr4j_{t}'].values, 'GR4J vs own record': pb[f'gr4jown_{t}'].values,
              'no change': None}
        if t in ('d_logq', 'd_rr'):
            bl['Budyko-Fu'] = pb[f'budyko_{t}'].values
        res['changes'][t] = compare(n, idx_all, y, pb[f'mu_{t}'].values, bl)
        mu, sd = pb[f'mu_{t}'].values, pb[f'sd_{t}'].values
        for k, z in (('cov68', 1.0), ('cov95', 1.96)):
            inside = (np.abs(y - mu) <= z * sd).astype(float)
            res['coverage'].setdefault(t, {})[k] = boot(idx_all, lambda ix: inside[ix].mean())
    for t, lv in LEVEL.items():
        y = pb[f'tgtobs_{lv}'].values
        model = lvl(t, pb[f'cal_{lv}'].values, pb[f'mu_{t}'].values)
        bl = {'own 1999-2008 value': pb[f'cal_{lv}'].values, 'GR4J sim (gauged)': pb[f'gr4jlvl_{lv}'].values}
        res['levels'][lv] = compare(n, idx_all, y, model, bl)
    p, mis, sdx = pb['p_regulated'].values, 1 - pb['gr4j_nse_cal'].values, -pb['seasdamp'].values
    dor = pb['dor'].values
    truths = [('dor>0.1 (mid+high)', dor > 0.1), ('dor>0.5 (high)', dor > 0.5), ('dor>0 (any)', dor > 0)]
    if peaks_src == 'nwis':
        truths.append(('NWIS peak code 5/6 in >=1/3 of WY1990-2008 peaks', pb['peak_reg_frac'].values >= 1 / 3))
    for name, y in truths:
        r = {'n_pos': int(y.sum()), 'auc': boot(idx_all, lambda ix: auc(y[ix], p[ix])), 'vs': {}}
        for bn, s in (('GR4J misfit (1-NSE_cal)', mis), ('seasonal damping index', sdx)):
            d = boot(idx_all, lambda ix: auc(y[ix], p[ix]) - auc(y[ix], s[ix]))
            d['baseline'] = auc(y, s)
            d['verdict'] = verdict_vs(d, True)
            r['vs'][bn] = d
        d = boot(idx_all, lambda ix: auc(y[ix], p[ix]) - 0.5)
        d['baseline'] = 0.5
        d['verdict'] = verdict_vs(d, True)
        r['vs']['chance (AUC 0.5)'] = d
        r['verdict'] = row_verdict(r['vs'])
        r['at_chance'] = bool(d['lo'] <= 0 <= d['hi'])
        res['regulation'][name] = r
    if peaks_src != 'nwis':
        res['regulation']['NWIS peak code 5/6'] = 'not re-scored: USGS NWIS peaks were not fetched'
    res['peaks_source'] = peaks_src
    json.dump(res, open(f'{SRC}/exam_rescore.json', 'w', newline='\n'), indent=1, default=float)
    print(f'{len(diffs)} numbers differ from the shipped exam.json')
    for d in diffs[:60]:
        print('  ', d)


if __name__ == '__main__':
    if not os.environ.get('ML_RATCHET_WATER'):
        sys.exit('set ML_RATCHET_WATER to the data root with data/proc/caravan/camels')
    main()
