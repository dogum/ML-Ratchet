"""Simulator-vs-real fidelity of the planning world (dev fires here; the exam script reuses fidelity_case).

For each fire: start from the first PM FEDS detection, run the planning world's truth physics with NO hidden error
(f_ros = 1: the dev-calibrated replay physics as stored) for 72 h, and compare the simulated burned area with the
observed FEDS PM perimeters ~24, 48 and 72 h later: Dice on the total burned area, F1 on the newly burned cells
(firedata.metrics.nextday definitions, inside the 10 km score window), and log10 of simulated / observed new area.
Persistence (no growth) is the reference for Dice-total. Real fires were fought, so 'no action' is not the real
counterfactual; we also run the sim with two classical suppression plans to see how much simulated action would be
needed to match the observed growth.

  python3 fidelity.py dev [p1|p0]   -> results/fidelity_dev_<phys>.json
"""
import os
import sys
import json

import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
import numpy as np

from fp.scenarios import real_scenario, dev_ids
from fp.planner import Context, BASELINES
import firedata as fd

OUT = _paths.WORK + '/results'
os.makedirs(OUT, exist_ok=True)
ACTIONS = [('none', None, None), ('anchor_flank', 300.0, 'anchor_flank'), ('anchor_flank', 1000.0, 'anchor_flank'),
           ('indirect_2km', 1000.0, 'indirect_2km')]


def scores(b0, sim_mask, obs_mask, res):
    w = fd.metrics.score_window(b0, res_m=res)
    m = fd.metrics.nextday(b0, obs_mask, sim_mask.astype(float), res ** 2 / 1e6, window=w)
    new_s = (sim_mask & ~b0 & w).sum()
    new_o = (obs_mask & ~b0 & w).sum()
    return dict(dice_total=fd.metrics.dice(sim_mask, obs_mask),
                f1_new=m['f1'], log10_new_ratio=float(np.log10((new_s + 1) / (new_o + 1))),
                persist_dice_total=fd.metrics.dice(b0, obs_mask), new_obs_ha=float(new_o * res ** 2 / 1e4),
                new_sim_ha=float(new_s * res ** 2 / 1e4))


def fidelity_case(S):
    """S: a real Scenario with meta['obs'] (observed masks). Returns per-action, per-lead scores."""
    out = dict(fire_id=S.meta['fire_id'], a0_ha=S.a0(), edge_window=S.meta.get('edge_window'),
               n_obs=len(S.meta['obs']), leads=[o['h'] for o in S.meta['obs']], runs={})
    ctx = None
    for tag, rate, base in ACTIONS:
        if base is None:
            arr = S.none()['arr']
        else:
            S.rate = rate
            if ctx is None:
                ctx = Context.from_scenario(S)
            idx, tb, km, n, nov = S.lines_for(ctx.fr, BASELINES[base])
            arr = S.run('truth', (idx, tb))
        rows = []
        for o in S.meta['obs']:
            sim = np.isfinite(arr) & (arr <= o['h'])
            rows.append(dict(h=o['h'], **scores(S.burned0, sim, o['mask'], S.res)))
        out['runs'][f'{tag}@{int(rate) if rate else 0}'] = rows
    return out


def summarize(cases):
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cases:
        for k, rows in c['runs'].items():
            for r in rows:
                lead = int(round(r['h'] / 24.0)) * 24
                if lead not in (24, 48, 72):
                    continue
                for m in ('dice_total', 'f1_new', 'log10_new_ratio', 'persist_dice_total'):
                    agg[(k, lead)][m].append(r[m])
    out = {}
    for (k, lead), d in sorted(agg.items()):
        out[f'{k}|{lead}h'] = dict(n=len(d['f1_new']), **{f'{m}_median': float(np.median(v)) for m, v in d.items()},
                                   **{f'{m}_mean': float(np.mean(v)) for m, v in d.items()})
    return out


if __name__ == '__main__':
    which = sys.argv[1]
    phys = sys.argv[2] if len(sys.argv) > 2 else 'p1'
    assert which == 'dev'
    fn = os.path.join(OUT, f'fidelity_dev_{phys}.json')
    cases = []
    for fid in dev_ids():
        assert not fd.guard.is_exam(fid)
        try:
            S = real_scenario(fid, 300.0, dict(f_ros=1.0, f_ws=1.0, d_wd=0.0), physics=phys)
        except Exception as e:
            print('skip', fid, e, flush=True)
            continue
        c = fidelity_case(S)
        cases.append(c)
        r = c['runs']['none@0']
        print(fid, 'a0 %.0f' % c['a0_ha'], [(round(x['h']), round(x['dice_total'], 2), round(x['f1_new'], 2),
                                             round(x['log10_new_ratio'], 2)) for x in r], flush=True)
    json.dump(dict(what=__doc__, physics=phys, cases=cases, summary=summarize(cases)), open(fn, 'w'), indent=1)
    print(json.dumps(summarize(cases), indent=1))
