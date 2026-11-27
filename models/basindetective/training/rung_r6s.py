"""Score the symmetric-prior rung R6s against the released R5 (27 Nov 2026, audit follow-up; sim and dev only).

R6s is R5's recipe (5 seeds of the R1 network, banks v3 + v4 shards 0-39, 60 epochs) on banks v3s + v4s: the same
basins draw for draw, with the decadal precipitation-drift prior delta ~ N(0, 0.038) instead of N(0.029, 0.038)
(simbank.py). The physics changed, so a new frozen set val-v4s (shards 900-903) was built and the incumbent is
re-scored on it, as every earlier physics change was. Rule (ratchet.yaml): keep if the sim skill on val-v4s beats R5's
by more than 0.01 AND the dev composite drops by no more than 0.02. Also reported, as the audit asked: the
forward-plus-reverse sums on paired dev periods (symmetry.py) and both rungs on R5's own frozen set val-v4.

python rung_r6s.py   -> WORK/models/r6se_v34s.npz (5-member ensemble), training/rung_r6s.json
"""
import common  # noqa: F401
import json
import os

import numpy as np

from common import D
from _paths import SRC, MODEL
from net import load_bank, Detective, pack_ensemble
import evaluate as E
import symmetry

R5 = f'{MODEL}/weights/basindetective_v1.npz'
VAL = [900, 901, 902, 903]


def main():
    members = [f'{D}/models/r6s_s{s}/model.npz' for s in range(5)]
    ens = f'{D}/models/r6se_v34s.npz'
    pack_ensemble(members, ens)
    out = {'what': 'R6s: R5 recipe on banks v3s + v4s (symmetric drift prior); 27 Nov 2026', 'members': members}
    for name, path in (('R5', R5), ('R6s', ens)):
        m = Detective(path)
        r = {}
        for ph in ('v4s', 'v4'):
            X, lab, _ = load_bank(ph, VAL)
            s = E.sim_scores(m, X, lab)
            r[f'sim_val_{ph}'] = s['_skill']
            r[f'sim_val_{ph}_detail'] = {k: v for k, v in s.items() if k != '_skill'}
        dev, _ = E.dev_gate(path)
        r['dev'] = dev
        r['dev_composite'] = dev['_composite']
        r['symmetry'] = symmetry.check(path)
        out[name] = r
    d_sim = out['R6s']['sim_val_v4s'] - out['R5']['sim_val_v4s']
    d_dev = out['R6s']['dev_composite'] - out['R5']['dev_composite']
    keep = d_sim > 0.01 and d_dev >= -0.02
    out['decision'] = 'kept' if keep else 'rejected'
    out['reason'] = (f"sim skill {out['R6s']['sim_val_v4s']:.4f} vs R5 {out['R5']['sim_val_v4s']:.4f} on val-v4s "
                     f"(delta {d_sim:+.4f}, need > 0.01); dev composite {out['R6s']['dev_composite']:.4f} vs "
                     f"{out['R5']['dev_composite']:.4f} (delta {d_dev:+.4f}, need >= -0.02)")
    json.dump(out, open(os.path.join(SRC, 'rung_r6s.json'), 'w'), indent=1, default=float)
    print(out['decision'], out['reason'])
    for name in ('R5', 'R6s'):
        r = out[name]
        print(name, 'val-v4s', round(r['sim_val_v4s'], 4), 'val-v4', round(r['sim_val_v4'], 4), 'dev', round(r['dev_composite'], 4),
              'sym d_logq median sum', round(r['symmetry']['d_logq']['median_sum'], 4),
              {t: round(r['dev'][t]['rho'], 3) for t in ('d_logq', 'd_rr', 'd_bfi', 'd_logq5', 'd_logq95', 'd_hfd')},
              'reg auc', round(r['dev']['reg']['auc'], 3))


if __name__ == '__main__':
    main()
