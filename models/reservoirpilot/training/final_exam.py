"""The final exam (run ONCE, after the ratchet is closed). Scores the release model and the classical comparators on
B's 244 scored ResOpsUS exam dams and the 15-dam CDEC recent exam with B's cost J and every component.
usage: python3 final_exam.py weights.npz  -> exam/exam_results.json, exam/exam_per_dam.parquet"""
import sys, os, json, pickle
import numpy as np, pandas as pd
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from policy import load_policy, rollout_np
from task import components
ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)


def main(w):
    tasks = pickle.load(open(f'{ROOT}/data/real_exam.pkl', 'rb'))
    pols = {'pilot': w, 'tuned_rule_g4': f'{ROOT}/runs/tuned_sop_g4/weights.npz', 'tuned_rule_g1': f'{ROOT}/runs/tuned_sop_g1/weights.npz'}
    rows = []
    sims = {}
    for name, path in pols.items():
        R, S = rollout_np(load_policy(path), tasks); sims[name] = (R, S)
        for tk, r, s in zip(tasks, R, S):
            rows.append({'res_id': tk['res_id'], 'cdec_id': tk.get('cdec_id', ''), 'split': tk['split'], 'policy': name,
                         'main_use': tk['main_use'], 'mb_corr': tk.get('mb_corr', np.nan), 'n_days': len(tk['I']),
                         **components(r, s, tk['D'], tk['Qmax'], tk['cap'], tk['S0'])})
    for tk in tasks:
        for p in ('historical', 'sop', 'rule', 'dp'):
            rows.append({'res_id': tk['res_id'], 'cdec_id': tk.get('cdec_id', ''), 'split': tk['split'], 'policy': p,
                         'main_use': tk['main_use'], 'mb_corr': tk.get('mb_corr', np.nan), 'n_days': len(tk['I']), **tk['scores'][p]})
    df = pd.DataFrame(rows)
    # B's mb_corr for CDEC dams comes from ResOpsUS; fill from the exam rows
    os.makedirs(f'{ROOT}/exam', exist_ok=True); df.to_parquet(f'{ROOT}/exam/exam_per_dam.parquet')
    cols = ['J', 'J_short', 'J_flood', 'shortage_ratio', 'shortage_days_frac', 'flood_days_per_yr', 'max_excess_ratio',
            'release_mean_over_D', 'S_mean_frac', 'S_min_frac', 'S_end_minus_start_frac', 'empty_days_frac', 'full_days_frac']
    out = {}
    for split, sub in (('resopsus_exam', df[df.split == 'exam']), ('resopsus_exam_mb_ok', df[(df.split == 'exam') & (df.mb_corr >= 0.9)]),
                       ('cdec_recent', df[df.split == 'cdec'])):
        tab = sub.groupby('policy')[cols].median()
        tab['n'] = sub.groupby('policy').size()
        out[split] = tab.round(5).to_dict(orient='index')
        piv = sub.pivot_table(index='res_id', columns='policy', values='J')
        out[split + '_pilot_vs'] = {p: {'pilot_better_frac': float((piv.pilot < piv[p] - 1e-9).mean()),
                                         'pilot_worse_frac': float((piv.pilot > piv[p] + 1e-9).mean())}
                                     for p in piv.columns if p != 'pilot'}
        # bootstrap 90 % interval of the median J (dams resampled)
        rng = np.random.default_rng(0); ci = {}
        for p in ('pilot', 'sop', 'dp', 'tuned_rule_g4'):
            v = piv[p].dropna().values
            bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
            ci[p] = [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))]
        out[split + '_median_J_90ci'] = ci
        print(split); print(tab[['n', 'J', 'J_short', 'J_flood', 'shortage_ratio', 'flood_days_per_yr', 'S_mean_frac', 'S_end_minus_start_frac']].round(4).to_string())
        print(out[split + '_pilot_vs']); print(ci)
    # by main use
    ex = df[df.split == 'exam']
    out['by_main_use'] = ex.pivot_table(index='main_use', columns='policy', values='J', aggfunc='median').round(4).to_dict(orient='index')
    out['n_by_main_use'] = ex[ex.policy == 'pilot'].main_use.value_counts().to_dict()
    json.dump(out, open(f'{ROOT}/exam/exam_results.json', 'w'), indent=1)
    pickle.dump({k: v for k, v in sims.items()}, open(f'{ROOT}/exam/exam_sims.pkl', 'wb'))


if __name__ == '__main__':
    main(sys.argv[1])
