"""(15 Nov 2026) explore decision-rule variants on a simulated set (probe / tune / frozen; never exam) before a rung is
logged: scores each variant with matched-SOC DP and prints mean % of DP / charge-holding rate per stratum.
usage: v21_explore.py SET CKPT '[{"name":..., params...}, ...]' [stratum substring filter]
Results: WORK/data/explore_<set><ptag>.json (appended)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import evaluate as E
import sets
from hp_env import ROOT, PTAG

if __name__ == '__main__':
    setname, ckpt, variants = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    filt = sys.argv[4] if len(sys.argv) > 4 else None
    base = pickle.load(open(E.base_path(setname), 'rb'))
    cases = sets.get(setname)
    if filt:
        cases = [c for c in cases if c['case_id'] in base and '_fail' not in base[c['case_id']]
                 and any(f in E.stratum(base[c['case_id']]['_meta']) for f in filt.split(','))]
    ids = {c['case_id'] for c in cases}
    b2 = {k: v for k, v in base.items() if k in ids}
    pols = {}
    for v in variants:
        v = dict(v); nm = v.pop('name')
        pols[nm] = E.score_policy(setname, ckpt, v, base=base, cases=cases)
    T = E.table(b2, pols, names=list(pols) + ['aecms', 'ecms', 'rule'])
    fn = f'{ROOT}/data/explore_{setname}{PTAG}.json'
    log = json.load(open(fn)) if os.path.exists(fn) else []
    log.append(dict(ckpt=ckpt, variants=variants, filt=filt, table=T,
                    skill={n: E.skill(b2, p)[0] for n, p in pols.items()}))
    json.dump(log, open(fn, 'w'), indent=1)
    for k in sorted(T):
        print(f'{k:24s} n {list(T[k].values())[0]["n"]:3d} ', ' | '.join(f"{n} {100*v['pct_dp_mean']:.2f}/{100*v['cs_rate']:.0f}" for n, v in T[k].items()))
    for n, p in pols.items():
        print('skill', n, round(E.skill(b2, p)[0], 4))
