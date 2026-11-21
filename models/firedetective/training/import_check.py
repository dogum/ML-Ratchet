"""Import check (21 Nov 2026): what the shipped files say, re-derived without re-running anything.

    python training/import_check.py        # numpy + scipy; reads results/exam.json and examples/; ~1 min

1. The per-fire rows in results/exam.json (fires[], every scored exam fire, values rounded as shipped) re-derive the
   headline Spearman rho and MAE numbers of exam.json (rounding moves them in the third decimal at most).
2. The four demo fires in examples/ (raw observations) go through pipeline.py and the released weights and must
   give the exam rows' predictions to their rounding.
3. Intervals for the verdicts, computed from shipped numbers only (no model run, no new exam fire):
   - zou:    the difference of two Spearman rhos from exam.json's own 95 % bootstrap intervals (Zou 2007, treating
             the two correlations as independent: conservative, because they share the fires); the LFMC MAE
             differences combine exam.json's MAE intervals the same way;
   - paired: a paired bootstrap over the exam fires of the per-fire rows (10,000 resamples, seed 20261119), for the
             baselines the rows carry (crown-fire potential, ERC max, Rothermel back-calculation, dev medians).
             FWI, BI and the LFMC climatology are not in the rows, so they get the Zou interval only.
Writes WORK/import_check.json (training/import_check.json is a copy of the 21 Nov 2026 run).
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, rankdata

from _paths import MODEL, SRC, WORK

B, SEED = 10000, 20261119
E = json.load(open(f'{SRC}/results/exam.json', encoding='utf-8'))
cal = E['calib']
R = E['fires']


def col(k):
    return np.array([np.nan if r[k] is None else r[k] for r in R], float)


hs, hp, cr, ff = col('hs'), col('hs_pred'), col('crown'), col('forest')
w, wp, bc = col('wind'), col('wind_pred'), col('wind_bc')
lf, lp, cfp, erc, hsd = col('lfmc'), col('lfmc_pred'), col('cfp'), col('erc'), col('hs_sd')
per = np.array([r['period'] for r in R])
out = {'n_rows': len(R)}

# ---------------------------------------------------------------- 1. headline numbers from the per-fire rows
forest = (ff >= 0.5) & np.isfinite(hs)
allf = np.isfinite(hs)
okbc = np.isfinite(bc)
mb = cal['map_wind_bc']
mae = lambda p, y: float(np.mean(np.abs(p - y)))
rho = lambda x, y: float(spearmanr(x, y)[0])
rows = {
    'forest n': (int(forest.sum()), E['all']['forest']['n']),
    'forest rho network': (rho(cr[forest], hs[forest]), E['all']['forest']['rho']['network']['rho']),
    'forest rho cfp': (rho(cfp[forest], hs[forest]), E['all']['forest']['rho']['cfp']['rho']),
    'forest rho erc_max': (rho(erc[forest], hs[forest]), E['all']['forest']['rho']['erc_max']['rho']),
    'forest MAE network': (mae(hp[forest], hs[forest]), E['all']['forest']['mae']['network']['mae']),
    'forest MAE dumb': (mae(cal['dev_median_hs'], hs[forest]), E['all']['forest']['mae']['dumb']['mae']),
    'forest cov80 network': (float(np.mean(np.abs(hp[forest] - hs[forest]) < 1.2816 * hsd[forest])), E['all']['forest']['cov80_network']),
    'all fuels n': (int(allf.sum()), E['all']['all_fuels']['n']),
    'all fuels rho network': (rho(cr[allf], hs[allf]), E['all']['all_fuels']['rho']['network']['rho']),
    'all fuels rho cfp': (rho(cfp[allf], hs[allf]), E['all']['all_fuels']['rho']['cfp']['rho']),
    'wind rho network': (rho(wp, w), E['all']['wind']['rho']['network']['rho']),
    'wind rho back-calc (raw)': (rho(bc[okbc], w[okbc]), E['all']['wind']['rho']['rothermel_bc']['rho']),
    'wind MAE network': (mae(wp, w), E['all']['wind']['mae']['network']['mae']),
    'wind MAE dumb': (mae(cal['dev_median_wind'], w), E['all']['wind']['mae']['dumb']['mae']),
    'wind MAE back-calc (dev-mapped)': (mae(mb[0] + mb[1] * bc[okbc], w[okbc]), E['all']['wind']['mae']['rothermel_bc']['mae']),
    'recent wind rho network': (rho(wp[per == 'recent'], w[per == 'recent']), E['recent']['wind']['rho']['network']['rho']),
    'lfmc rho network': (rho(lp[np.isfinite(lf)], lf[np.isfinite(lf)]), E['all']['lfmc']['rho']['network']['rho']),
    'lfmc MAE network': (mae(lp[np.isfinite(lf)], lf[np.isfinite(lf)]), E['all']['lfmc']['mae']['network']['mae']),
}
out['rederived'] = {k: {'rows': a, 'exam_json': b, 'diff': a - b} for k, (a, b) in rows.items()}
out['notes'] = {
    'forest_n': 'The rows round forest share to 2 decimals, so one fire with share just below 0.5 counts as forest '
                'in the rows (358 vs 357).',
    'wind_bc_map_slope': mb[1],
    'wind_bc_capped_at_30': int(np.sum(bc == 30.0)),
    'wind_bc_missing': int(np.sum(~okbc)),
}
print('1. per-fire rows vs exam.json')
for k, v in out['rederived'].items():
    print(f"   {k:34s} rows {v['rows']:.4f}  exam.json {v['exam_json']:.4f}  diff {v['diff']:+.4f}")

# ---------------------------------------------------------------- 2. demo fires through the raw pipeline
sys.path.insert(0, MODEL)
import demo           # noqa: E402
import firedetective  # noqa: E402
import pipeline       # noqa: E402

model = firedetective.load()
byfid = {r['fid']: r for r in R}
worst = 0.0
for fn in sorted(os.listdir(f'{MODEL}/examples')):
    if not fn.endswith('.npz'):
        continue
    fire, meta = demo.load_fire(f'{MODEL}/examples/{fn}')
    o = pipeline.run([fire], model)
    r = byfid[meta['fid']]
    for k, kk, nd in (('hs', 'hs_pred', 3), ('hs_sd', 'hs_sd', 3), ('crown', 'crown', 3), ('wind', 'wind_pred', 2),
                      ('lfmc', 'lfmc_pred', 1), ('ros_adj', 'ros', 3), ('m1', 'm1', 2), ('p_spot', 'spot', 2)):
        d = abs(round(float(o[k][0]), nd) - r[kk])
        worst = max(worst, d / 10 ** -nd)
out['demo_vs_exam_rows_worst_in_last_digit'] = worst
print(f'2. demo fires vs exam rows: worst difference {worst:.2f} units of the last shipped digit')

# ---------------------------------------------------------------- 3. intervals
def zou(r1, ci1, r2, ci2):
    l1, u1 = ci1
    l2, u2 = ci2
    d = r1 - r2
    return [d - np.sqrt((r1 - l1) ** 2 + (u2 - r2) ** 2), d + np.sqrt((u1 - r1) ** 2 + (r2 - l2) ** 2)]


def fast_rho(x, y, idx):
    """Spearman rho for each bootstrap row of indices (average ranks, as scipy)."""
    xs, ys = x[idx], y[idx]
    rx = np.apply_along_axis(rankdata, 1, xs)
    ry = np.apply_along_axis(rankdata, 1, ys)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx * rx).sum(1) * (ry * ry).sum(1))


def paired_rho(a, b, y, sel):
    sel = sel & np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[sel], b[sel], y[sel]
    idx = np.random.default_rng(SEED).integers(0, len(y), (B, len(y)))
    d = fast_rho(a, y, idx) - fast_rho(b, y, idx)
    return {'n': int(len(y)), 'diff': rho(a, y) - rho(b, y), 'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def paired_mae(pa, pb, y, sel):
    sel = sel & np.isfinite(pa) & np.isfinite(pb) & np.isfinite(y)
    ea, eb = np.abs(pa[sel] - y[sel]), np.abs(pb[sel] - y[sel])
    idx = np.random.default_rng(SEED).integers(0, len(ea), (B, len(ea)))
    d = ea[idx].mean(1) - eb[idx].mean(1)
    return {'n': int(len(ea)), 'diff': float(ea.mean() - eb.mean()), 'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def zrow(sub, task, base):
    a, b = E[sub][task]['rho']['network'], E[sub][task]['rho'][base]
    return {'diff': a['rho'] - b['rho'], 'ci': zou(a['rho'], a['ci'], b['rho'], b['ci']), 'n': [a['n'], b['n']]}


def zmae(sub, base):
    """MAE difference from the two shipped 95 % intervals, combined as for Zou (independent: conservative)."""
    a, b = E[sub]['lfmc']['mae']['network'], E[sub]['lfmc']['mae'][base]
    return {'diff': a['mae'] - b['mae'], 'ci': zou(a['mae'], a['ci'], b['mae'], b['ci']), 'n': [a['n'], b['n']]}


ones = np.ones(len(R), bool)
full = lambda v: np.full(len(R), v)
I = {}
for sub, sel in (('all', ones), ('main', per == 'main'), ('recent', per == 'recent')):
    fsel, asel = sel & (ff >= 0.5), sel
    I[sub] = {
        'forest_rho': {'zou': {k: zrow(sub, 'forest', k) for k in ('fwi_max', 'erc_max', 'cfp', 'bi_max')},
                       'paired': {'erc_max': paired_rho(cr, erc, hs, fsel), 'cfp': paired_rho(cr, cfp, hs, fsel)}},
        'forest_mae': {'paired': {'dumb': paired_mae(hp, full(cal['dev_median_hs']), hs, fsel),
                                  'cfp': paired_mae(hp, cal['map_cfp'][0] + cal['map_cfp'][1] * cfp, hs, fsel),
                                  'erc_max': paired_mae(hp, cal['map_erc_max'][0] + cal['map_erc_max'][1] * erc, hs, fsel)}},
        'all_fuels_rho': {'zou': {k: zrow(sub, 'all_fuels', k) for k in ('cfp', 'erc_max', 'fwi_max', 'bi_max')},
                          'paired': {'cfp': paired_rho(cr, cfp, hs, asel), 'erc_max': paired_rho(cr, erc, hs, asel)}},
        'all_fuels_mae': {'paired': {'dumb': paired_mae(hp, full(cal['dev_median_hs']), hs, asel),
                                     'cfp': paired_mae(hp, cal['map_cfp'][0] + cal['map_cfp'][1] * cfp, hs, asel),
                                     'erc_max': paired_mae(hp, cal['map_erc_max'][0] + cal['map_erc_max'][1] * erc, hs, asel)}},
        'wind_rho': {'zou': {'rothermel_bc': zrow(sub, 'wind', 'rothermel_bc')},
                     'paired': {'rothermel_bc': paired_rho(wp, bc, w, sel)}},
        'wind_mae': {'paired': {'dumb': paired_mae(wp, full(cal['dev_median_wind']), w, sel),
                                'rothermel_bc': paired_mae(wp, mb[0] + mb[1] * bc, w, sel)}},
        'lfmc_rho': {'zou': {'clim': zrow(sub, 'lfmc', 'clim')}},
        'lfmc_mae': {'zou': {k: zmae(sub, k) for k in ('clim', 'dumb')}},
    }
out['intervals'] = I
out['method'] = {'bootstrap': B, 'seed': SEED, 'zou': 'Zou (2007), from exam.json 95 % bootstrap intervals'}
print('3. intervals (model minus baseline; rho: + = model better; MAE: - = model better)')
for sub in I:
    for task, d in I[sub].items():
        for kind, dd in d.items():
            for base, v in dd.items():
                print(f"   {sub:6s} {task:14s} {kind:6s} vs {base:13s} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
os.makedirs(WORK, exist_ok=True)
with open(f'{WORK}/import_check.json', 'w', encoding='utf-8', newline='\n') as f:
    json.dump(out, f, indent=1)
print('wrote', f'{WORK}/import_check.json')
