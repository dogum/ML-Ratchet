"""(15 Nov 2026) per-class prior of the equivalence factor from the TRAINING teacher: the median DP costate s* over the
labelled decisions of each vehicle group in a DP-teacher bank (gen_bank ACT=teacher; the cases are re-drawn from the
bank's seed, so every label is matched to its vehicle). Training data only.
usage: v21_prior.py BANK N SEED [fleet]   -> WORK/data/prior_<bank>.json"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hp_env import *

if __name__ == '__main__':
    bank, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fleet = train_fleet(sys.argv[4] if len(sys.argv) > 4 else 'gen')
    tr_v, _ = train_noise_partition()
    ved = ved_routes('train_noise', vehicles=tr_v)
    cases = train_cases(n, seed, fleet=fleet, ved=ved)
    key = lambda sp: 'mild' if is_mild(sp) else ('bus_' + sp['architecture'] if sp['class'] == 'bus' else sp['architecture'])
    S = {}
    for f in sorted(glob.glob(f'{ROOT}/data/bank_{bank}/s{seed}_*.npz')):
        z = np.load(f)
        for ci in np.unique(z['case']):
            m = z['case'] == ci
            S.setdefault(key(cases[int(ci)]['spec']), []).append(z['s'][m])
    out = {}
    for k, v in sorted(S.items()):
        a = np.concatenate(v)
        trip_med = [float(np.median(x)) for x in v]
        out[k] = dict(n_trips=len(v), n_labels=int(len(a)), median=float(np.median(a)), p25=float(np.percentile(a, 25)),
                      p75=float(np.percentile(a, 75)), trip_median_median=float(np.median(trip_med)))
        print(f"{k:22s} trips {len(v):4d} labels {len(a):7d} median s* {out[k]['median']:.3f} [{out[k]['p25']:.2f}, {out[k]['p75']:.2f}] trip-median median {out[k]['trip_median_median']:.3f}")
    json.dump(dict(bank=bank, seed=seed, physics=PHYS, groups=out), open(f'{ROOT}/data/prior_{bank}.json', 'w'), indent=1)
