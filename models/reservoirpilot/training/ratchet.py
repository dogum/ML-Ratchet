"""Ratchet bookkeeping. Scores a candidate on the frozen simulated set of the current generator version and on the
15 dev dams, applies the rule, and logs the rung (kept or not) in ratchet.json.

Primary sim metric:  S_J = 1 - median(J_policy) / median(J_SOP)  on the frozen set (B's J, B's mass balance,
                     full scored window from the task's start storage). Kept if S_J > incumbent + 0.01.
Dev gate:            median J on the 15 dev dams <= 1.05 x the incumbent's (first rung: <= SOP's dev median).

  python3 ratchet.py score RUNG_ID weights.npz "description"    (adds/updates the rung with its numbers)
  python3 ratchet.py rescore GEN                                  (re-score every rung on frozen set GEN)
"""
import sys, json, os, pickle
import numpy as np
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from policy import load_policy, rollout_np            # noqa: E402
from task import components                          # noqa: E402

ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)
RJ = f'{ROOT}/ratchet.json'
_cache = {}


def frozen(gen):
    if gen not in _cache:
        _cache[gen] = pickle.load(open(f'{ROOT}/data/val_{gen}.pkl', 'rb'))
    return _cache[gen]


def sim_score(wpath, gen):
    tasks = frozen(gen); pol = load_policy(wpath)
    R, S = rollout_np(pol, tasks)
    c = [components(r, s, t['D'], t['Qmax'], t['cap'], t['S0']) for r, s, t in zip(R, S, tasks)]
    J = np.array([x['J'] for x in c]); Js = np.array([t['scores']['sop']['J'] for t in tasks])
    Jd = np.array([t['scores']['dp']['J'] for t in tasks])
    gap = np.clip((Js - J) / np.maximum(Js - Jd, 1e-4), -2, 1.5)
    return {'S_J': float(1 - np.median(J) / np.median(Js)), 'J_median': float(np.median(J)),
            'J_sop_median': float(np.median(Js)), 'J_dp_median': float(np.median(Jd)),
            'gap_closed_median': float(np.median(gap[Js - Jd > 1e-3])),
            'dS_end_median': float(np.nanmedian([x['S_end_minus_start_frac'] for x in c])),
            'shortage_ratio_median': float(np.nanmedian([x['shortage_ratio'] for x in c])),
            'flood_days_median': float(np.nanmedian([x['flood_days_per_yr'] for x in c]))}


def dev_score(wpath):
    tasks = pickle.load(open(f'{ROOT}/data/real_dev.pkl', 'rb')); pol = load_policy(wpath)
    R, S = rollout_np(pol, tasks)
    c = [components(r, s, t['D'], t['Qmax'], t['cap'], t['S0']) for r, s, t in zip(R, S, tasks)]
    return {'J_median': float(np.nanmedian([x['J'] for x in c])),
            'J_sop_median': float(np.median([t['scores']['sop']['J'] for t in tasks])),
            'J_per_dam': {t['res_id']: round(x['J'], 5) for t, x in zip(tasks, c)},
            'shortage_ratio_median': float(np.nanmedian([x['shortage_ratio'] for x in c])),
            'flood_days_median': float(np.nanmedian([x['flood_days_per_yr'] for x in c])),
            'dS_end_median': float(np.nanmedian([x['S_end_minus_start_frac'] for x in c]))}


def load():
    return json.load(open(RJ)) if os.path.exists(RJ) else {'rule': __doc__.split('\n\n')[1], 'gen': 'g1', 'rungs': []}


def save(r):
    json.dump(r, open(RJ, 'w'), indent=1)


def decide(r, rung):
    """Compare with the incumbent (latest kept rung) on the current frozen set."""
    gen = r['gen']; inc = [x for x in r['rungs'] if x.get('kept')]
    s = rung['sim'][gen]['S_J']; d = rung['dev']['J_median']
    if not inc:
        ok_sim = s > 0.01; ok_dev = d <= rung['dev']['J_sop_median']
        ref = 'SOP'
    else:
        I = inc[-1]; ok_sim = s > I['sim'][gen]['S_J'] + 0.01; ok_dev = d <= 1.05 * I['dev']['J_median']; ref = I['id']
    rung['kept'] = bool(ok_sim and ok_dev)
    rung['decision'] = f"vs {ref} on frozen {gen}: sim {'pass' if ok_sim else 'FAIL'}, dev gate {'pass' if ok_dev else 'FAIL'}"
    return rung


if __name__ == '__main__':
    r = load()
    if sys.argv[1] == 'score':
        rid, w, desc = sys.argv[2], sys.argv[3], sys.argv[4]
        rung = {'id': rid, 'weights': w, 'desc': desc, 'sim': {}, 'dev': dev_score(w)}
        for g in ('g1', 'g2', 'g3', 'g4'):
            if os.path.exists(f'{ROOT}/data/val_{g}.pkl'):
                rung['sim'][g] = sim_score(w, g)
        r['rungs'] = [x for x in r['rungs'] if x['id'] != rid]
        rung = decide(r, rung); r['rungs'].append(rung); save(r)
        print(json.dumps({k: v for k, v in rung.items() if k != 'dev'}, indent=1))
        print('dev', {k: v for k, v in rung['dev'].items() if k != 'J_per_dam'})
    elif sys.argv[1] == 'rescore':
        # new frozen set (physics/generator changed): score every rung on it and re-take every decision in order
        g = sys.argv[2]; r['gen'] = g
        rungs = r['rungs']; r['rungs'] = []
        for x in rungs:
            if g not in x['sim']:
                x['sim'][g] = sim_score(x['weights'], g)
            x = decide(r, x); r['rungs'].append(x)
            print(x['id'], round(x['sim'][g]['S_J'], 4), x['decision'])
        r.setdefault('history', []).append(f'frozen set switched to {g}; all rungs re-scored and decisions re-taken in order')
        save(r)
