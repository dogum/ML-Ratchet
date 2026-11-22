"""Re-run the exam with the RELEASED weights (firecast_weights.npz, numpy), to check the shipped numbers reproduce.

The sandbox's exam loaded the torch checkpoint WORK/ckpt/r2.pt, which was not shipped. The release has the same
network with BatchNorm folded into the convs (numpy == torch to 2.5e-6, see model.yaml). This wrapper swaps
evaluate.load_net / logits_full for firecast_np.FireCastNP and then runs exam.py unchanged:

  python rescore_np.py run r2 | summary r2 | examples r2       (outputs under WORK/out/exam, as exam.py writes them)

Nothing is chosen: the rung, the dev calibration (results/cal_r2_F.json, cal_r2_L.json) and every exam rule are the
shipped ones. Added on import, 22 Nov 2026.
"""
import os
import sys

import evaluate as EV
from firecast_np import FireCastNP

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'firecast_weights.npz')


def _load_net(rung):
    if rung != 'r2':
        raise SystemExit(f'the release is rung r2, not {rung}')
    return FireCastNP(WEIGHTS)


EV.load_net = _load_net
EV.logits_full = lambda net, X, s: net.forward(X, s)

if __name__ == '__main__':
    import exam
    {'run': exam.run, 'summary': exam.summary, 'examples': exam.examples}[sys.argv[1]](sys.argv[2])
