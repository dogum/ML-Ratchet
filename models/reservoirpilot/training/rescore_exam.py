"""Reproduction re-score of the released policy on WaterData's reservoir exam (18 Nov 2026, on import; the exam had
already been seen, so this is a reproduction, not a rung: nothing is chosen from it).

1. Checks the dams: the exam tasks built by evaluate.prep('exam') must be exactly splits.json reservoirs/exam (244
   scorable of 250) and the CDEC crosswalk's exam reservoirs (15).
2. Re-scores the shipped weights (../reservoirpilot_weights.npz) through the shipped exam path, final_exam.main, which
   imports WaterData's own doy_curve / cost / simulate / dp via task.py and evaluate.py. The sim-tuned rules it needs
   are rebuilt from their logged constants (ratchet.json: c1 f=0.75, tau=3, H=10 on g1; c2 f=0.6, tau=3, H=10 on g4).
3. Scores the dev-tuned rule (tune_rule_dev.py, frozen in dev_tuned_rule.json) once on the same tasks.
4. Compares re-scored baselines per dam with WaterData's shipped reservoir_scores.parquet / reservoir_recent_scores.parquet
   (does the data rebuild match?) and the pilot's medians with the shipped exam_results.json.
5. Paired bootstrap over dams (10,000 resamples, numpy default_rng(20261118)) of the difference in median between the
   pilot and each baseline, for J and the main components; verdict win / loss only if the 95 % interval excludes 0.

Writes: WORK/exam/* (final_exam.py's outputs), training/rescore_per_dam.csv (scores only, no time series) and
training/rescore_exam.json.
usage: ML_RATCHET_WATER=... python rescore_exam.py

27 Nov 2026: with RESERVOIRPILOT_TASK=causal (task.py) the same script is the owner-approved dated re-score of every arm
on the causal task (capacity from GRanD and the calibration half only; split before interpolating; split day in the
scored half only). Set ML_RATCHET_WORK to a separate folder so its caches do not mix with the exam definition's. It
then writes training/rescore_per_dam_causal.csv and training/rescore_exam_causal.json, compares the baselines with the
table WaterData's run_baselines_reservoir.py --causal wrote (RESERVOIRPILOT_TABLES, default WORK/baselines_causal) and
compares the medians with the shipped exam log, which is then the old board.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
import final_exam  # noqa: E402
from evaluate import prep  # noqa: E402
from policy import load_policy, rollout_np  # noqa: E402
from task import components, CAUSAL  # noqa: E402

WEIGHTS = os.path.join(_paths.MODEL, 'reservoirpilot_weights.npz')
SEED, NBOOT = 20261118, 10000
POL = ['pilot', 'sop', 'rule', 'historical', 'tuned_rule_g4', 'tuned_rule_g1', 'tuned_rule_dev', 'dp']
COLS = ['J', 'J_short', 'J_flood', 'shortage_ratio', 'shortage_days_frac', 'flood_days_per_yr', 'max_excess_ratio',
        'release_mean_over_D', 'S_mean_frac', 'S_min_frac', 'S_end_minus_start_frac', 'empty_days_frac', 'full_days_frac']
TAG = '_causal' if CAUSAL else ''
TABLES = os.environ.get('RESERVOIRPILOT_TABLES') or (f'{_paths.WORK}/baselines_causal' if CAUSAL
                                                     else f'{_paths.WATER_DATA}/proc/baselines')
MAIN = {'J': 'lower', 'shortage_ratio': 'lower', 'flood_days_per_yr': 'lower', 'S_mean_frac': 'higher',
        'S_end_minus_start_frac': 'higher'}


def check_dams(tasks):
    sp = json.load(open(f'{_paths.WATER_DATA}/proc/splits.json'))['reservoirs']
    ex = [t['res_id'] for t in tasks if t['split'] == 'exam']
    cd = sorted(t['cdec_id'] for t in tasks if t['split'] == 'cdec')
    xw = pd.read_csv(f'{_paths.WATER_DATA}/proc/reservoirs/cdec/crosswalk.csv')
    xw_exam = sorted(xw[xw.res_id.isin(sp['exam'])].cdec_id)
    rb = pd.read_parquet(f'{_paths.WATER_DATA}/proc/baselines/reservoir_recent_scores.parquet')
    want_cd = sorted(rb[rb.split == 'exam'].cdec_id.unique())            # WaterData's scored recent exam
    b = pd.read_parquet(f'{TABLES}/reservoir_scores.parquet')
    want_ex = sorted(b[b.split == 'exam'].res_id.unique())                # WaterData's 244 scored exam dams
    assert set(ex) <= set(sp['exam']) and len(ex) == len(set(ex)), 'exam task outside the frozen exam split'
    assert not set(ex) & set(sp['dev']) and not set(ex) & set(sp['aux'])
    return {'exam_split': len(sp['exam']), 'exam_scored': len(ex), 'exam_match_waterdata_244': sorted(ex) == want_ex,
            'exam_not_scorable': sorted(set(sp['exam']) - set(ex)), 'cdec_scored': cd,
            'cdec_match_waterdata_15': cd == want_cd, 'cdec_crosswalk_exam_not_scorable': sorted(set(xw_exam) - set(cd))}


def write_tuned_sop():
    for name, (f, tau, H) in {'tuned_sop_g1': (0.75, 3, 10), 'tuned_sop_g4': (0.6, 3, 10)}.items():
        out = f'{_paths.WORK}/runs/{name}'
        os.makedirs(out, exist_ok=True)
        np.savez(f'{out}/weights.npz', fset='tuned_sop', f=f, tau=tau, H=H)
    out = f'{_paths.WORK}/runs/tuned_sop_dev'                       # tune_rule_dev.py's frozen constants
    if not os.path.exists(f'{out}/weights.npz'):
        dv = json.load(open(os.path.join(_paths.SRC, 'dev_tuned_rule.json')))
        os.makedirs(out, exist_ok=True)
        np.savez(f'{out}/weights.npz', fset='tuned_sop', f=dv['f'], tau=dv['tau'], H=dv['H'])


def boot(df, rng_idx):
    """Paired bootstrap of median(pilot) - median(baseline) per component; df rows = dams, columns (policy, metric)."""
    out = {}
    for p in POL[1:]:
        out[p] = {}
        for m, better in MAIN.items():
            x, y = df[('pilot', m)].values, df[(p, m)].values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3:
                continue
            x, y = x[ok], y[ok]
            idx = rng_idx if ok.all() else np.random.default_rng(SEED).integers(0, len(x), (NBOOT, len(x)))
            d = np.median(x[idx], 1) - np.median(y[idx], 1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            pt = float(np.median(x) - np.median(y))
            if better == 'lower':
                v = 'win' if hi < 0 else 'loss' if lo > 0 else 'tie'
            else:
                v = 'pilot higher' if lo > 0 else 'pilot lower' if hi < 0 else 'tie'
            out[p][m] = {'n': int(len(x)), 'diff_of_medians': pt, 'ci95': [float(lo), float(hi)], 'verdict': v,
                         'pilot_median': float(np.median(x)), 'baseline_median': float(np.median(y))}
    return out


if __name__ == '__main__':
    tasks = prep('exam')
    dams = check_dams(tasks)
    print(dams, flush=True)
    write_tuned_sop()
    final_exam.main(WEIGHTS)                                      # the shipped exam path, unchanged
    per = pd.read_parquet(f'{_paths.WORK}/exam/exam_per_dam.parquet')
    dev = json.load(open(os.path.join(_paths.SRC, 'dev_tuned_rule.json')))
    R, S = rollout_np(load_policy(f'{_paths.WORK}/runs/tuned_sop_dev/weights.npz'), tasks)
    rows = [{'res_id': tk['res_id'], 'cdec_id': tk.get('cdec_id', ''), 'split': tk['split'], 'policy': 'tuned_rule_dev',
             'main_use': tk['main_use'], 'mb_corr': tk.get('mb_corr', np.nan), 'n_days': len(tk['I']),
             **components(r, s, tk['D'], tk['Qmax'], tk['cap'], tk['S0'])} for tk, r, s in zip(tasks, R, S)]
    per = pd.concat([per, pd.DataFrame(rows)], ignore_index=True)
    keep = ['res_id', 'cdec_id', 'split', 'main_use', 'mb_corr', 'n_days', 'policy'] + COLS
    out = per[keep].copy()
    for c in COLS + ['mb_corr']:
        out[c] = out[c].astype(float).map(lambda v: float(f'{v:.5g}') if np.isfinite(v) else v)
    out.sort_values(['split', 'res_id', 'policy']).to_csv(os.path.join(_paths.SRC, f'rescore_per_dam{TAG}.csv'), index=False)

    # 4a. baselines per dam vs WaterData's shipped tables
    b = pd.read_parquet(f'{TABLES}/reservoir_scores.parquet')
    b = b[b.split == 'exam'].set_index(['res_id', 'policy'])
    e = per[per.split == 'exam'].set_index(['res_id', 'policy'])
    cmp = {}
    for p in ('historical', 'sop', 'rule', 'dp'):
        ids = [i for i in b.xs(p, level='policy').index]
        mine = e.xs(p, level='policy').reindex(ids)
        theirs = b.xs(p, level='policy').reindex(ids)
        d = (mine.J - theirs.J).abs()
        cmp[p] = {'n_shipped': len(ids), 'n_rescored': int(mine.J.notna().sum()), 'max_abs_dJ': float(d.max()),
                  'n_dJ_gt_1e-6': int((d > 1e-6).sum()), 'median_J_shipped_table': float(theirs.J.median()),
                  'median_J_rescored': float(mine.J.median())}
    rb = pd.read_parquet(f'{_paths.WATER_DATA}/proc/baselines/reservoir_recent_scores.parquet')
    rb = rb[rb.split == 'exam'].set_index(['cdec_id', 'policy'])
    c = per[per.split == 'cdec'].set_index(['cdec_id', 'policy'])
    for p in ('historical', 'sop', 'rule', 'dp'):
        ids = list(rb.xs(p, level='policy').index)
        d = (c.xs(p, level='policy').reindex(ids).J - rb.xs(p, level='policy').reindex(ids).J).abs()
        cmp['cdec_' + p] = {'n': len(ids), 'max_abs_dJ': float(d.max()), 'n_dJ_gt_1e-6': int((d > 1e-6).sum())}

    # 4b. medians vs the shipped exam log
    shipped = json.load(open(os.path.join(_paths.SRC, 'exam_results.json')))
    new = json.load(open(f'{_paths.WORK}/exam/exam_results.json'))
    vs = {}
    for split in ('resopsus_exam', 'resopsus_exam_mb_ok', 'cdec_recent'):
        vs[split] = {p: {k: [shipped[split][p][k], new[split][p][k]] for k in shipped[split][p]
                         if abs(shipped[split][p][k] - new[split][p][k]) > 1e-5} for p in shipped[split]}
        vs[split + '_pilot_vs'] = {p: [shipped[split + '_pilot_vs'][p]['pilot_better_frac'], new[split + '_pilot_vs'][p]['pilot_better_frac']]
                                   for p in shipped[split + '_pilot_vs']}
        vs[split + '_median_J_90ci'] = {p: [shipped[split + '_median_J_90ci'][p], new[split + '_median_J_90ci'][p]]
                                        for p in shipped[split + '_median_J_90ci']}

    # 5. paired bootstrap over dams
    subsets = {'resopsus_exam': per[per.split == 'exam'],
               'resopsus_exam_mb_ok': per[(per.split == 'exam') & (per.mb_corr >= 0.9)],
               'cdec_recent': per[per.split == 'cdec']}
    res = {}
    for name, sub in subsets.items():
        key = 'cdec_id' if name == 'cdec_recent' else 'res_id'
        wide = sub.pivot_table(index=key, columns='policy', values=list(MAIN), aggfunc='first').swaplevel(0, 1, axis=1)
        n = len(wide)
        idx = np.random.default_rng(SEED).integers(0, n, (NBOOT, n))
        med = sub.groupby('policy')[COLS].median()
        better = {p: (wide[('pilot', 'J')] < wide[(p, 'J')] - 1e-9).mean() for p in POL[1:]}
        k = {p: int(round(better[p] * n)) for p in better}
        cp = {p: [float(v) for v in binomtest(k[p], n).proportion_ci(0.95, method='exact')] for p in k}
        res[name] = {'n': n, 'medians': med.round(6).to_dict(orient='index'),
                     'pilot_better_frac': {p: float(v) for p, v in better.items()}, 'pilot_better_cp95': cp,
                     'paired_bootstrap': boot(wide, idx)}
    # 6. the card's per-dam failure claims, now checkable
    ex = per[per.split == 'exam']
    dS = ex.pivot_table(index='res_id', columns='policy', values='S_end_minus_start_frac')
    Jx = ex.pivot_table(index='res_id', columns='policy', values='J')
    use = ex[ex.policy == 'pilot'].set_index('res_id').main_use
    ti = pd.DataFrame([{'res_id': t['res_id'], 'cap_years': t['cap'] / (t['cal_I_mean'] * 365.25 * 0.0864),
                        'scored_over_cal_inflow': float(np.mean(t['I']) / t['cal_I_mean'])}
                       for t in tasks if t['split'] == 'exam' and t['cal_I_mean'] > 0]).set_index('res_id')
    lose = Jx.index[Jx.pilot > Jx.sop + 1e-9]
    failures = {
        'end_minus_start_le_-0.2_share': {p: float((dS[p] <= -0.2).mean()) for p in POL},
        'pilot_loses_to_sop': {'n': int(len(lose)), 'main_use': use[lose].value_counts().to_dict(),
                               'cap_years_median': float(ti.cap_years.reindex(lose).median()),
                               'cap_years_median_all': float(ti.cap_years.median()),
                               'drier_scored_decade_share': float((ti.scored_over_cal_inflow.reindex(lose) < 1).mean()),
                               'drier_scored_decade_share_all': float((ti.scored_over_cal_inflow < 1).mean())}}
    by_use = {}                                  # info rows: J by main use, same paired bootstrap
    for u in ('fcon', 'irri', 'elec', 'supp'):
        sub = Jx[use.reindex(Jx.index) == u]
        n = len(sub); idx = np.random.default_rng(SEED).integers(0, n, (NBOOT, n)); by_use[u] = {'n': n, 'pilot': float(sub.pilot.median())}
        for p in POL[1:]:
            d = np.median(sub.pilot.values[idx], 1) - np.median(sub[p].values[idx], 1)
            by_use[u][p] = {'median': float(sub[p].median()), 'diff_of_medians': float(sub.pilot.median() - sub[p].median()),
                            'ci95': [float(v) for v in np.percentile(d, [2.5, 97.5])]}
    failures['by_main_use_J'] = by_use
    what = ('Dated supplementary re-score (27 Nov 2026) of every arm on the causal task definition (seen exam; '
            'nothing chosen); old board = the shipped exam log' if CAUSAL else
            'Reproduction re-score of the released v1.0 weights on WaterData exam (seen); not a rung')
    summary = {'what': what, 'task': 'causal' if CAUSAL else 'exam definition', 'baseline_tables': TABLES,
               'data_root': 'ML_RATCHET_WATER rebuilt; build.py --only verify passed (splits sha256 33f88ca0..., field ed8109e5...)',
               'bootstrap': f'paired over dams, {NBOOT} resamples, numpy default_rng({SEED}); 95 % percentile interval of median(pilot) - median(baseline)',
               'dams': dams, 'baselines_vs_shipped_tables': cmp, 'shipped_vs_rescored_differences': vs,
               'dev_tuned_rule': {k2: dev[k2] for k2 in ('f', 'tau', 'H', 'dev_median_J')}, 'failure_checks': failures, 'results': res}
    json.dump(summary, open(os.path.join(_paths.SRC, f'rescore_exam{TAG}.json'), 'w'), indent=1, default=float)
    print(json.dumps({k2: v for k2, v in summary.items() if k2 in ('dams', 'baselines_vs_shipped_tables')}, indent=1, default=float))
