"""PREREG section 6: does the five families' disagreement predict a dev basin's error? Dev only; no exam data.

usage: python disagree.py <tag> <member_preds.npz> x5      (evaluate.py --save-preds files: per basin (days, 7) =
                                                             obs, mean, q05..q95)
spread_b = mean over days of the sd across members' daily means / mean over days of the ensemble mean;
error_b  = 1 - NSE_b of the ensemble mean (equal weights) against observed Caravan Q.
Spearman rho over the 52 dev basins, 95 % percentile CI from 10,000 basin bootstraps (default_rng(20261118)).
Extras (no pass mark): error = |volume bias| of the ensemble mean; each member's own 1 - NSE.
Writes round2/results/disagreement_<tag>.json.
"""
import json
import os
import sys

import r2
import numpy as np
from scipy.stats import spearmanr


def nse(o, p):
    k = np.isfinite(o)
    o, p = o[k], p[k]
    return 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)


def boot_ci(x, y, n=10000, seed=20261118):
    rng = np.random.default_rng(seed)
    N = len(x); out = np.empty(n)
    for i in range(n):
        j = rng.integers(0, N, N)
        out[i] = spearmanr(x[j], y[j])[0]
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def main(tag, files):
    Z = [np.load(f) for f in files]
    basins = sorted(set(Z[0].files).intersection(*[set(z.files) for z in Z[1:]]))
    rows = []
    for b in basins:
        P = [z[b].astype(np.float64) for z in Z]
        obs = P[0][:, 0]
        M = np.stack([p[:, 1] for p in P], 0)
        ens = M.mean(0)
        spread = float(np.mean(M.std(0)) / max(np.mean(ens), 1e-9))
        k = np.isfinite(obs)
        rows.append({'basin': b, 'spread': spread, 'err': float(1 - nse(obs, ens)),
                     'abs_vol_bias': float(abs(ens[k].sum() / obs[k].sum() - 1)),
                     'member_err': [float(1 - nse(obs, m)) for m in M]})
    s = np.array([r['spread'] for r in rows]); e = np.array([r['err'] for r in rows])
    v = np.array([r['abs_vol_bias'] for r in rows])
    rho = float(spearmanr(s, e)[0]); ci = boot_ci(s, e)
    rho_v = float(spearmanr(s, v)[0]); ci_v = boot_ci(s, v)
    res = {'tag': tag, 'members': files, 'n_basins': len(rows),
           'primary': {'rho': rho, 'ci95': ci, 'passes': bool(rho > 0 and ci[0] > 0),
                       'definition': 'Spearman(relative spread, 1 - NSE of the equal-weight ensemble mean)'},
           'extra_abs_volume_bias': {'rho': rho_v, 'ci95': ci_v},
           'extra_member_err': [{'member': i, 'rho': float(spearmanr(s, [r['member_err'][i] for r in rows])[0])}
                                for i in range(len(files))],
           'ensemble_dev_median_nse': float(np.median(1 - e)),
           'spread_quartiles': np.percentile(s, [25, 50, 75]).tolist(), 'per_basin': rows}
    os.makedirs(r2.RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(r2.RESULTS, f'disagreement_{tag}.json'), 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('primary', 'extra_abs_volume_bias', 'extra_member_err',
                                          'ensemble_dev_median_nse')}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
