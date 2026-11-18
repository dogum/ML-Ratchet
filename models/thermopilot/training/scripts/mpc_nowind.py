"""Fairness check of the classical MPC (integrity pass, 17 Nov 2026). Dev episodes only.

usage: mpc_nowind.py [SET]      -> results/SET/mpc_nowind.json (default SET = dev)

The MPC's one-step zone maps (tp/mpc.py Model.coeffs) take outdoor temperature and solar from the noisy forecast but
the wind term of infiltration (CwU2) from the weather file, i.e. the true future wind. ThermoPilot sees no wind. This
re-runs the MPC with CwU2 replaced by its mean over the episode (no hourly wind), to measure what that leak is worth.
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT
import numpy as np
from evaluate import load_cases, episode, clean
from ratchet import load_base


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    cs = load_cases(name); base = load_base(name)
    res = {}
    for c in cs:
        ep = episode(c)
        ep.model.CwU2 = np.full_like(ep.model.CwU2, float(np.mean(ep.model.CwU2[:ep.n])))
        m, _ = ep.run(ep.mpc(False).act)
        res[c['id']] = clean(m)
        print(c['id'], 'J %.2f (with true wind %.2f)' % (m['J'], base[c['id']]['mpc_c']['J']), flush=True)
    ids = [i for i in base if i in res]
    Jf = np.array([base[i]['fixed']['J'] for i in ids]); Jo = np.array([base[i]['oracle']['J'] for i in ids])
    gap = np.sum((Jf - Jo) / Jf)
    sk = lambda J: float(np.sum((Jf - J) / Jf) / gap)
    out = dict(n=len(ids), skill_true_wind=sk(np.array([base[i]['mpc_c']['J'] for i in ids])),
               skill_mean_wind=sk(np.array([res[i]['J'] for i in ids])), results=res)
    json.dump(out, open(os.path.join(ROOT, 'results', name, 'mpc_nowind.json'), 'w'))
    print(name, 'MPC skill with true wind %.3f, with mean wind %.3f' % (out['skill_true_wind'], out['skill_mean_wind']))


if __name__ == '__main__':
    main()
