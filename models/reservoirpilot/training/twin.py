"""Sim-to-real gap checks on the 15 DEV dams (never exam).
1. Denoised twin: each dev dam's task with its inflow replaced by a 5-day centred mean (volume kept). If a policy
   scores much better on the twin than on the real record, day-to-day inflow noise (back-calculated inflow) is a gap.
2. Feature distributions: features seen along the policy's own trajectory on dev dams vs on simulated tasks
   (selection set of the given generator); KS distance per feature.
usage: python3 twin.py weights.npz gen  -> printed + runs/twin_<name>.json"""
import sys, json, pickle, os
import numpy as np
from scipy.stats import ks_2samp
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from policy import load_policy, rollout_np, features, task_arrays, EMA_TAUS
from task import components, simulate
ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)


def feats_along(pol, tasks, R_S):
    out = []
    for tk, S in zip(tasks, R_S[1]):
        n = len(tk['I']); cal = task_arrays(tk, n); Sb = np.r_[tk['S0'], S[:-1]]
        e = [pol.init_ema(tk)] * 3; E = np.zeros((n, 3))
        for t in range(n):
            e = [x + (tk['I'][t] - x) / tau for x, tau in zip(e, EMA_TAUS)]; E[t] = e
        f = np.stack(np.broadcast_arrays(*features(np, Sb, tk['I'], [E[:, 0], E[:, 1], E[:, 2]], cal['D'], cal['Dn30'], cal['Dn90'],
                              cal["Ic30"], cal["Ic90"], tk["Qmax"], tk["cap"], pol.fset)), -1)
        out.append(f[::5])
    f = np.concatenate(out); return f[np.isfinite(f).all(1)]


if __name__ == '__main__':
    w, gen = sys.argv[1], sys.argv[2]
    pol = load_policy(w)
    dev = pickle.load(open(f'{ROOT}/data/real_dev.pkl', 'rb'))
    rows = []
    twins = []
    for tk in dev:
        t2 = dict(tk); I = tk['I']; k = np.ones(5) / 5
        t2['I'] = np.convolve(np.r_[I[:2], I, I[-2:]], k, mode='valid'); twins.append(t2)
    Rr = rollout_np(pol, dev); Rt = rollout_np(pol, twins)
    for tk, t2, r, s, r2, s2 in zip(dev, twins, Rr[0], Rr[1], Rt[0], Rt[1]):
        cr = components(r, s, tk['D'], tk['Qmax'], tk['cap'], tk['S0'])['J']
        ct = components(r2, s2, t2['D'], t2['Qmax'], t2['cap'], t2['S0'])['J']
        Rs, Ss = simulate('sop', t2['I'], t2['D'], t2['Qmax'], t2['cap'], t2['S0'])
        js = components(Rs, Ss, t2['D'], t2['Qmax'], t2['cap'], t2['S0'])['J']
        rows.append({'res_id': tk['res_id'], 'J_real': cr, 'J_twin_denoised': ct, 'J_sop_real': tk['scores']['sop']['J'],
                     'J_sop_twin': js, 'neg_inflow_frac': float(np.mean(tk['I'] < 0))})
    sel = pickle.load(open(f'{ROOT}/data/sel_{gen}.pkl', 'rb'))
    fd = feats_along(pol, dev, Rr); fs = feats_along(pol, sel, rollout_np(pol, sel))
    ks = [float(ks_2samp(fd[:, j], fs[:, j]).statistic) for j in range(fd.shape[1])]
    names = ['S/cap', 'asinh I/Qmax', 'ema5', 'ema20', 'ema60', 'D/Qmax', 'Dn30/Qmax', 'log cap/Qmax-days',
             'headroom', 'SOP spill pressure', 'storage days of demand', 'Ic30', 'Ic90', 'expected overflow 90d',
             'expected storage 30d'][:fd.shape[1]]
    res = {'weights': w, 'gen': gen, 'dams': rows,
           'median_J_real': float(np.median([r['J_real'] for r in rows])),
           'median_J_twin_denoised': float(np.median([r['J_twin_denoised'] for r in rows])),
           'median_J_sop_real': float(np.median([r['J_sop_real'] for r in rows])),
           'median_J_sop_twin': float(np.median([r['J_sop_twin'] for r in rows])),
           'feature_ks': dict(zip(names, ks))}
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print({k: v for k, v in res.items() if k.startswith('median')})
    print('KS', {k: round(v, 3) for k, v in res['feature_ks'].items()})
    json.dump(res, open(f'{ROOT}/runs/twin_{os.path.basename(os.path.dirname(w))}_{gen}.json', 'w'), indent=1)
