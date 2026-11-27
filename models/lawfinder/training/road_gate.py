"""Road law v1.1: the sign gate as a release rule (owner decision, 27 Nov 2026). No new search.

Rule added to the road ladder after the audit (docs/audits/lawfinder-v1.0.md), before any exam was run for this
decision: a rung may not be kept, and so not released, if its energy falls as net grade rises at any val_p2 row
(E(G + 0.25) < E(G - 0.25) - 1e-9 with the other inputs held; road_sign.py's check). Physics: tank and battery energy
rise with uphill grade for every powertrain. The released v1.0 law p1-c23 failed it on 21 % of val_p2 rows and on every
battery-electric row. Otherwise the rule is unchanged (frozen R^2 beats the incumbent by more than 0.01, dev error no
more than 5 % worse).

Reads results/road_ratchet.json (v1.0: 40 rungs, the walk on val_p1 then val_p2), results/road_sign.json and
results/road_search_p2s.json (the sign-checked search's 13 rungs, 27 Nov 2026), and the rebuilt p2 banks and dev set.
Appends the p2s rungs, re-walks every rung on val_p2 with the gate, adds the walk to the history, recomputes the error
bars for the new release as fit_law.py does, and writes results/road_ratchet.json (the v1.0 file is kept as
results/road_ratchet_v10.json). The exam is not read.
python road_gate.py
"""
import json, os, shutil
import numpy as np
import lawratchet as lr
import law_road as L
from road_sign import law_falls
from _paths import ROOT

DATE = '2026-11-27'
GATE = 'sign gate (adopted 27 Nov 2026 after the audit): energy may not fall as net grade rises at any val_p2 row'


def main():
    src = f'{ROOT}/results/road_ratchet.json'
    keep = f'{ROOT}/results/road_ratchet_v10.json'
    if not os.path.exists(keep):
        shutil.copy(src, keep)
    R = json.load(open(keep))
    assert R['released'] == 'p1-c23' and len(R['rungs']) == 40, 'expects the v1.0 ratchet'
    S = {f"p2s-c{f['complexity']}": f for f in json.load(open(f'{ROOT}/results/road_search_p2s.json'))['front']}
    va = L.load_bank('p2', 'val'); dev = L.load_dev()
    Xv, yv = L.X_of(va), L.target(va); Xd, yd = L.X_of(dev), L.target(dev)

    rungs = [dict(r) for r in R['rungs']]
    for k in sorted(S, key=lambda k: S[k]['complexity']):
        if S[k]['complexity'] == 1:
            continue
        rungs.append({'id': k, 'version': 'p2s', 'change': f"Physics p2, sign-checked search ({DATE}), complexity {S[k]['complexity']}",
                      **S[k]})
    for r in rungs:
        sim, devv = lr.score(r, Xv, yv, Xd, yd, dev_err=L.dev_error)
        r.setdefault('scores', {})['p2'] = {'sim': sim, 'dev': devv}
        r['sim'], r['dev'] = sim, devv
        r['dev_r2'] = lr.r2(yd, lr.law_eval(r, Xd))
        f = law_falls(r, Xv)
        r['sign'] = {'ok': bool(not f.any()), 'falls_share': float(f.mean()),
                     'falls_share_bev': float(f[Xv[:, L.VARS.index('e')] > 0.5].mean())}

    # the walk: lawratchet.walk's rule, with failing rungs rejected before the comparison
    inc, dec = None, []
    for r in rungs:
        if not r['sign']['ok']:
            d, why = 'rejected', (f"Energy falls as net grade rises on {100 * r['sign']['falls_share']:.0f} % of val_p2 rows "
                                  f"({100 * r['sign']['falls_share_bev']:.0f} % of battery-electric rows): sign gate.")
        else:
            d1, inc1 = lr.walk(([inc] if inc else []) + [r])
            d, why = d1[-1]['decision'], d1[-1]['why']
        dec.append({'id': r['id'], 'decision': d, 'why': why})
        if d == 'kept':
            inc = r
    R['rungs'] = rungs
    R['history'].append({'frozen_set': 'val_p2', 'date': DATE, 'rule': GATE, 'decisions': dec, 'released': inc['id']})
    R['released'] = inc['id']
    R['rule'] = R['rule'] + ' Since 27 Nov 2026 (road law v1.1): ' + GATE + '.'
    R['sign_gate'] = {'date': DATE, 'check': 'E(G + 0.25) >= E(G - 0.25) - 1e-9 at every val_p2 row, other inputs held',
                      'reason': 'tank and battery energy rise with uphill grade; v1.0 p1-c23 failed on 21 % of val_p2 rows '
                                'and every battery-electric row (audit, 25 Nov 2026)',
                      'adopted_before_exam': True, 'previous_release': 'p1-c23'}

    # error bars for the new release (fit_law.py's recipe)
    rel = next(r for r in rungs if r['id'] == inc['id'])
    res_v = yv - lr.law_eval(rel, Xv); res_d = yd - lr.law_eval(rel, Xd)
    sigma = float(np.std(res_v[np.isfinite(res_v)]))
    k = lr.widen_factor(res_d[np.isfinite(res_d)], sigma)
    for r in rungs:
        rv = yv - lr.law_eval(r, Xv)
        r['sigma_sim'] = float(np.std(rv[np.isfinite(rv)])) if np.isfinite(rv).mean() > 0.99 else None
    R['bars'] = {'sigma_sim': sigma, 'widen_dev': k, 'sigma': sigma * k,
                 'dev_cov68_raw': float(np.mean(np.abs(res_d) <= sigma)),
                 'dev_cov68_widened': float(np.mean(np.abs(res_d) <= sigma * k))}
    json.dump(R, open(src, 'w'), indent=1, default=float)
    for r, d in zip(rungs, dec):
        if d['decision'] == 'kept' or r['id'] in ('p1-c23', 'p1-c25'):
            print(f"{r['id']:8s} R2 {r['sim']:.4f} dev {r['dev']:.4f} {d['decision']:8s} {d['why']}")
    print('released', R['released'], 'bars', R['bars'])


if __name__ == '__main__':
    main()
