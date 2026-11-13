"""Final exam report: tables per stratum, paired bootstrap CIs (clustered by vehicle) of HybridPilot minus A-ECMS,
fixed-factor vs matched-SOC comparison, worst cases. Reads data/base_exam_*.pkl and data/pol_exam_*.pkl.
usage: exam_report.py POLNAME [POLNAME2 ...]  -> data/exam_report.json"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import evaluate as E
from hp_env import ROOT

STRATS = ['dp', 'rule', 'ecms', 'aecms', 'mpc', 'phev_cdcs', 'phev_blended', 'series_power_follower']


def boot_diff(base, pa, nb, key, ref='aecms', n=2000, seed=0):
    """mean over cases of (pa - ref) pct_DP, 95 % CI by resampling vehicles (cluster bootstrap)."""
    rows = [(b['_meta']['vehicle'], pa[c]['pct_dp'] - b[ref]['pct_dp']) for c, b in base.items()
            if '_fail' not in b and c in pa and ref in b and (key == 'ALL' or E.stratum(b['_meta']) == key or b['_meta'].get('group') == key)]
    if not rows:
        return None
    veh = sorted(set(r[0] for r in rows)); byv = {v: [r[1] for r in rows if r[0] == v] for v in veh}
    rng = np.random.default_rng(seed)
    m = np.mean([r[1] for r in rows])
    bs = []
    for _ in range(n):
        pick = rng.choice(len(veh), len(veh))
        x = np.concatenate([byv[veh[i]] for i in pick]); bs.append(x.mean())
    return dict(n=len(rows), n_veh=len(veh), mean=float(m), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def main(names):
    base = pickle.load(open(E.base_path('exam'), 'rb'))
    cases = {c['case_id']: c for c in __import__('sets').get('exam')}
    for cid, b in base.items():
        if cid in cases and '_meta' in b:
            b['_meta']['group'] = cases[cid].get('group')
    pols = {n: pickle.load(open(f'{ROOT}/data/pol_exam_{n}.pkl', 'rb')) for n in names}
    T = E.table(base, pols, names=STRATS + names)
    # by group (twin vs generated)
    G = {}
    for g in ('twin', 'generated'):
        bg = {c: b for c, b in base.items() if '_meta' in b and b['_meta'].get('group') == g}
        G[g] = E.table(bg, pols, names=STRATS + names).get('ALL')
    rep = dict(table=T, groups=G, skill={})
    for n in names:
        rep['skill'][n] = E.skill(base, pols[n])[0]
    for s in ['ecms', 'aecms', 'mpc']:
        rep['skill'][s] = E.skill(base, {c: b[s] for c, b in base.items() if s in b})[0]
    rep['boot_vs_aecms'] = {n: {k: boot_diff(base, pols[n], None, k) for k in list(T) + ['twin', 'generated']} for n in names}
    # fixed-factor correction vs matched: mean over HEV cases of (fixed-factor fuel ratio) - (matched pct), per strategy
    ff = {}
    for s in ['rule', 'ecms', 'aecms', 'mpc'] + names:
        d = []
        for c, b in base.items():
            if '_fail' in b or b['_meta']['phev']: continue
            m = pols[s].get(c) if s in pols else (b.get(s) if s != 'rule' else b.get(E.RULE[b['_meta']['arch']]))
            if m is None: continue
            dp_ff = b['dp']['fuel_corr_fixed_g']
            d.append(dict(ff=dp_ff / max(m['fuel_corr_fixed_g'], 1.0), mt=m['pct_dp'], dsoc=m['soc_end'] - b['dp']['soc_end']))
        if d:
            ff[s] = dict(fixed_factor_pct=float(np.mean([x['ff'] for x in d])), matched_pct=float(np.mean([x['mt'] for x in d])),
                         corr_with_dsoc=float(np.corrcoef([x['ff'] - x['mt'] for x in d], [x['dsoc'] for x in d])[0, 1]))
    rep['fixed_vs_matched'] = ff
    # worst cases for the first policy
    n0 = names[0]
    w = sorted(((pols[n0][c]['pct_dp'], c, base[c]['_meta']['kind'], E.stratum(base[c]['_meta']), base[c]['aecms']['pct_dp'] if 'aecms' in base[c] else None)
                for c in pols[n0] if c in base), key=lambda x: x[0])[:8]
    rep['worst'] = w
    json.dump(rep, open(f'{ROOT}/data/exam_report.json', 'w'), indent=1, default=float)
    for k in sorted(T):
        print(k.ljust(22), ' | '.join(f"{n} {100 * v['pct_dp_mean']:.2f}/{v['cs_rate']:.2f}/{v['starts_per_km']:.1f}" for n, v in T[k].items()))
    print(json.dumps(rep['skill'], indent=1)); print(json.dumps(rep['boot_vs_aecms'], indent=1)[:3000]); print(rep['fixed_vs_matched'])


if __name__ == '__main__':
    main(sys.argv[1:])
