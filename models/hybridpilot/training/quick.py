"""quick look: score a ckpt on a set with given preview / nav / knobs, print per-stratum table vs baselines.
usage: quick.py SET CKPT PREVIEW_S ['json knobs' (nav, subset: hev|phev)] [tag]"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import evaluate as E
import sets

if __name__ == '__main__':
    setname, ckpt, prev = sys.argv[1], sys.argv[2], int(sys.argv[3])
    kw = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    subset = kw.pop('subset', None)
    kw['preview_s'] = prev
    cases = sets.get(setname)
    if subset == 'hev':
        cases = [c for c in cases if not c['spec']['architecture'].startswith('phev')]
    elif subset == 'phev':
        cases = [c for c in cases if c['spec']['architecture'].startswith('phev')]
    base = pickle.load(open(E.base_path(setname), 'rb'))
    pol = E.score_policy(setname, ckpt, kw, base=base, cases=cases)
    tag = os.path.basename(ckpt)[:-4] + f'_p{prev}' + (sys.argv[5] if len(sys.argv) > 5 else '')
    base = {c['case_id']: base[c['case_id']] for c in cases if c['case_id'] in base}
    pickle.dump(pol, open(f'{_P.WORK}/data/quick_{setname}_{tag}.pkl', 'wb'))
    T = E.table(base, {'HP': pol}, names=['HP', 'aecms', 'ecms', 'mpc', 'rule', 'dp', 'phev_cdcs', 'phev_blended'])
    for k in sorted(T):
        print(k, ' | '.join(f"{n} {100*v['pct_dp_mean']:.2f}/{v['cs_rate']:.2f}/{v['starts_per_km']:.1f}/{v['chatter_per_km']:.1f}" for n, v in T[k].items()))
    print('skill HP', E.skill(base, pol)[0], 'aecms', E.skill(base, {c: b['aecms'] for c, b in base.items() if 'aecms' in b})[0])
