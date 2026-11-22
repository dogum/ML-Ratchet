"""Re-run the exam with the RELEASED weights (weights/firedetective_v1.npz), to check the shipped numbers reproduce.

The sandbox's exam loaded WORK/models/R6.ens.json and its members' .pt checkpoints, which were not shipped. export.py
wrote the same ensemble's state_dict into the npz (members m0..m2, float32) with the input normalisation (seq_mu,
seq_sd, st_mu, st_sd). This wrapper rebuilds the torch ensemble from the npz and swaps it in for ratchet.load_net,
then runs exam.py unchanged:

  python rescore_release.py build | score       (outputs under WORK, as exam.py writes them)

Nothing is chosen: the incumbent rung (R6), its dev calibration and every exam rule are the shipped ones. Added on
import, 22 Nov 2026.
"""
import os
import sys

import numpy as np
import torch

import model as M
import ratchet

NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights', 'firedetective_v1.npz')


def _load_net(rung):
    if rung != 'R6':
        raise SystemExit(f'the release is rung R6, not {rung}')
    z = np.load(NPZ)
    members = []
    for i in range(3):
        sd = {k[len(f'm{i}.'):]: torch.from_numpy(z[k].copy()) for k in z.files if k.startswith(f'm{i}.')}
        n_seq, n_stat = sd['emb.weight'].shape[1], sd['st.weight'].shape[1]
        h, c, d = sd['emb.weight'].shape[0], sd['c1.weight'].shape[0], sd['f1.weight'].shape[0]
        net = M.Net(n_seq, n_stat, h, c, d)
        net.load_state_dict(sd)
        members.append(net.eval())
    ns = {k: z[k].astype(np.float32).copy() for k in ('seq_mu', 'seq_sd', 'st_mu', 'st_sd')}
    return M.Ens(members).eval(), ns


ratchet.load_net = _load_net

if __name__ == '__main__':
    import exam
    {'build': exam.build, 'score': exam.score}[sys.argv[1]]()
