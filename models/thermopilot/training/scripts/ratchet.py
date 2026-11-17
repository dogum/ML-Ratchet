"""Score a policy on the frozen val set and the dev gate; append the rung to ratchet.json.

usage: ratchet.py score TAG                     print val + dev scores of results/{val,dev}/pol_TAG.json
       ratchet.py rung TAG "what changed"       score, decide keep/reject vs the incumbent, log to ratchet.json

skill = sum_e (J_fixed - J_policy)/J_fixed / sum_e (J_fixed - J_oracle)/J_fixed   (0 = fixed thermostat, 1 = perfect foresight)
J = control-dependent bill (energy + demand + gas) + lambda x degree-hours outside the occupant band.
Keep: val skill > incumbent + 0.01 AND dev gate: dev skill > 0, dev skill >= incumbent dev skill - 0.03, and
mean degree-hours per dev episode <= fixed + 1 K h.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # training/: tp, _paths
from _paths import WORK as ROOT   # data/ and results/ live in WORK (was the sandbox's thermopilot/ folder)
MARGIN = 0.01


def load_base(name):
    out = {}
    for f in glob.glob(os.path.join(ROOT, 'results', name, 'base', '*.json')):
        d = json.load(open(f))
        if 'error' not in d:
            out[d['meta']['id']] = d
    return out


def score(name, tag, base=None):
    base = base or load_base(name)
    pol = json.load(open(os.path.join(ROOT, 'results', name, f'pol_{tag}.json')))
    ids = [i for i in base if i in pol and 'error' not in pol[i]]
    Jf = np.array([base[i]['fixed']['J'] for i in ids]); Jo = np.array([base[i]['oracle']['J'] for i in ids])
    Jp = np.array([pol[i]['J'] for i in ids]); Jm = np.array([base[i]['mpc_c']['J'] for i in ids])
    # relative savings per episode, so large commercial bills do not dominate
    gap = max(float(np.sum((Jf - Jo) / Jf)), 1e-9)
    res = dict(n=len(ids), skill=float(np.sum((Jf - Jp) / Jf) / gap), skill_mpc_c=float(np.sum((Jf - Jm) / Jf) / gap),
               bill_saving_vs_fixed=float(1 - sum(pol[i]['bill'] for i in ids) / sum(base[i]['fixed']['bill'] for i in ids)),
               DH_pol=float(np.mean([pol[i]['DH'] for i in ids])), DH_fixed=float(np.mean([base[i]['fixed']['DH'] for i in ids])),
               beats_fixed=float(np.mean(Jp <= Jf + 1e-6)), errors=sum('error' in v for v in pol.values()))
    return res


def main():
    cmd, tag = sys.argv[1], sys.argv[2]
    v = score('val', tag); d = score('dev', tag)
    print('val', json.dumps(v)); print('dev', json.dumps(d))
    if cmd != 'rung':
        return
    f = os.path.join(ROOT, 'ratchet.json')
    R = json.load(open(f)) if os.path.exists(f) else dict(rules=__doc__, rungs=[], incumbent=None)
    inc = next((r for r in R['rungs'] if r['tag'] == R['incumbent']), None)
    iv = inc['val']['skill'] if inc else 0.0
    idv = inc['dev']['skill'] if inc else 0.0
    gate = d['skill'] > 0 and d['skill'] >= idv - 0.03 and d['DH_pol'] <= d['DH_fixed'] + 1.0
    keep = v['skill'] > iv + MARGIN and gate
    why = (f"val skill {v['skill']:.3f} vs incumbent {iv:.3f} (margin {MARGIN}); dev skill {d['skill']:.3f} "
           f"(incumbent {idv:.3f}), dev DH {d['DH_pol']:.1f} vs fixed {d['DH_fixed']:.1f} K h -> "
           + ('KEPT' if keep else 'REJECTED' + ('' if gate else ' (dev gate failed)')))
    R['rungs'] = [r for r in R['rungs'] if r['tag'] != tag]
    R['rungs'].append(dict(tag=tag, change=sys.argv[3] if len(sys.argv) > 3 else '', val=v, dev=d, kept=keep, why=why))
    if keep:
        R['incumbent'] = tag
    json.dump(R, open(f, 'w'), indent=1)
    print(why)


if __name__ == '__main__':
    main()
