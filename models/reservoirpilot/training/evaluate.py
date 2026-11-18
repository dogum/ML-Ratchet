"""Score policies on B's real reservoir tasks with B's cost J and every component.

  python3 evaluate.py prep dev            build + cache the 15 dev tasks with B's baselines (historical, SOP, rule, DP)
  python3 evaluate.py prep exam           the same for the 244 scored exam dams + the CDEC recent exam (final only)
  python3 evaluate.py score dev W.npz     score a policy (weights file) on the cached dev tasks
Baselines are recomputed with B's own functions and checked against B's per-dam table (reservoir_scores.parquet).
"""
import os, sys, pickle, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from task import resops_task, cdec_task, simulate, dp, cost, components, wd, ROOT as DROOT  # noqa: E402
from policy import load_policy, rollout_np                                                  # noqa: E402

ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)


def baselines(tk):
    I, D, Q, cap, S0 = tk['I'], tk['D'], tk['Qmax'], tk['cap'], tk['S0']
    sc = {'historical': components(tk['R_obs'], tk['S_obs'], D, Q, cap, S0)}
    for pol in ('sop', 'rule'):
        R, S = simulate(pol, I, D, Q, cap, S0, tk['Ss'])
        sc[pol] = components(R, S, D, Q, cap, S0)
    Rd = dp(I, D, Q, cap, S0)
    # DP storage path (replay B's DP releases through the same mass balance)
    S = np.empty(len(I)); s = S0
    for t in range(len(I)):
        s = min(max(s + (I[t] - Rd[t]) * 0.0864, 0.0), cap); S[t] = s
    sc['dp'] = components(Rd, S, D, Q, cap, S0)
    return sc


def prep(split):
    out = f'{ROOT}/data/real_{split}.pkl'
    if os.path.exists(out):
        return pickle.load(open(out, 'rb'))
    sp = wd.splits()['reservoirs']; meta = wd.reservoir_meta().set_index('res_id')
    tasks = []; t0 = time.time()
    for rid in sp[split]:
        tk = resops_task(rid, meta)
        if tk is None:
            continue
        tk.pop('cal'); tk['split'] = split; tk['main_use'] = meta.loc[rid, 'main_use']
        tk['mb_corr'] = float(meta.loc[rid, 'mb_corr']); tk['scores'] = baselines(tk); tasks.append(tk)
        print(split, rid, len(tasks), round(time.time() - t0), {k: round(v['J'], 4) for k, v in tk['scores'].items()}, flush=True)
    if split == 'exam':
        xw = pd.read_csv(f'{DROOT}/proc/reservoirs/cdec/crosswalk.csv')
        for r in xw.itertuples():
            if r.res_id not in sp['exam']:
                continue
            tk = cdec_task(r.res_id, r.cdec_id, meta)
            if tk is None:
                continue
            tk['split'] = 'cdec'; tk['main_use'] = meta.loc[r.res_id, 'main_use']; tk['scores'] = baselines(tk)
            tasks.append(tk); print('cdec', r.cdec_id, {k: round(v['J'], 4) for k, v in tk['scores'].items()}, flush=True)
    # check against B's frozen per-dam baseline table
    b = pd.read_parquet(f'{DROOT}/proc/baselines/reservoir_scores.parquet')
    b = b.set_index(['res_id', 'policy']).J
    diffs = [abs(tk['scores'][p]['J'] - b[(tk['res_id'], p)]) for tk in tasks if tk['split'] != 'cdec'
             for p in ('historical', 'sop', 'rule', 'dp')]
    print(split, 'max |J - B| over dams and baselines:', max(diffs))
    pickle.dump(tasks, open(out, 'wb'))
    return tasks


def score(tasks, wpath):
    pol = load_policy(wpath)
    R, S = rollout_np(pol, tasks)
    rows = []
    for r, s, tk in zip(R, S, tasks):
        c = components(r, s, tk['D'], tk['Qmax'], tk['cap'], tk['S0'])
        rows.append({'res_id': tk['res_id'], 'split': tk['split'], **c})
    return pd.DataFrame(rows), R, S


def table(tasks, df=None):
    """Median of every component per policy."""
    rows = {}
    for p in ('historical', 'sop', 'rule', 'dp'):
        rows[p] = pd.DataFrame([tk['scores'][p] for tk in tasks]).median()
    if df is not None:
        rows['pilot'] = df.drop(columns=['res_id', 'split']).median()
    return pd.DataFrame(rows).T


if __name__ == '__main__':
    cmd, split = sys.argv[1], sys.argv[2]
    tasks = prep(split)
    if cmd == 'score':
        df, _, _ = score(tasks, sys.argv[3])
        print(table(tasks, df).round(4).to_string())
