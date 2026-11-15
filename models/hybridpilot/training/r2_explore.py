"""(15 Nov 2026, round 2) probes of decision-rule variants on a simulated set (probe / probe_mild / tune; never dev or
exam): like v21_explore.py, but every variant's per-trip results are kept (WORK/data/explore_r2/<set>_<name><ptag>.pkl,
reused if present) so a probe can be re-tabulated, and every probe is appended to WORK/data/explore_r2<ptag>.json.
usage: r2_explore.py SET CKPT '[{"name":..., params...}, ...]' [stratum substring filter]"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import evaluate as E
import sets
from hp_env import ROOT, PTAG

if __name__ == '__main__':
    setname, ckpt, variants = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    filt = sys.argv[4] if len(sys.argv) > 4 else None
    assert setname not in ('dev', 'exam', 'frozen'), 'probes run on training-distribution sets only'
    base = pickle.load(open(E.base_path(setname), 'rb'))
    cases = sets.get(setname)
    if filt:
        cases = [c for c in cases if c['case_id'] in base and '_fail' not in base[c['case_id']]
                 and any(f in E.stratum(base[c['case_id']]['_meta']) for f in filt.split(','))]
    ids = {c['case_id'] for c in cases}
    b2 = {k: v for k, v in base.items() if k in ids}
    os.makedirs(f'{ROOT}/data/explore_r2', exist_ok=True)
    pols = {}
    for v in variants:
        v = dict(v); nm = v.pop('name')
        fn = f'{ROOT}/data/explore_r2/{setname}_{nm}{PTAG}.pkl'
        old = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else None
        if old is not None and old['params'] == v and ids <= set(old['pol']):
            pols[nm] = {k: old['pol'][k] for k in ids if k in old['pol']}
            continue
        pol = E.score_policy(setname, ckpt, v, base=base, cases=cases)
        if old is not None and old['params'] == v:
            old['pol'].update(pol); pol = old['pol']
        pickle.dump(dict(params=v, ckpt=ckpt, pol=pol), open(fn, 'wb'))
        pols[nm] = {k: pol[k] for k in ids if k in pol}
    T = E.table(b2, pols, names=list(pols) + ['aecms', 'ecms', 'rule'])
    fn = f'{ROOT}/data/explore_r2{PTAG}.json'
    log = json.load(open(fn)) if os.path.exists(fn) else []
    log.append(dict(set=setname, ckpt=ckpt, variants=variants, filt=filt, table=T,
                    skill={n: E.skill(b2, p)[0] for n, p in pols.items()}))
    json.dump(log, open(fn, 'w'), indent=1)
    for k in sorted(T):
        print(f'{k:24s} n {list(T[k].values())[0]["n"]:3d} ', ' | '.join(f"{n} {100*v['pct_dp_mean']:.2f}/{100*v['cs_rate']:.0f}" for n, v in T[k].items()))
    for n, p in pols.items():
        print('skill', n, round(E.skill(b2, p)[0], 4))
