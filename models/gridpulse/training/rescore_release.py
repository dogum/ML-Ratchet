"""Re-score the v1.0 final exam from the SHIPPED weights, with exam.py as run (added in the repository, 14 Nov 2026).

The torch checkpoints exam.py loads (ckpt/r5.pt, ckpt/ercot_head_p1.npz) were not part of the release. So:
  - EV.predict is pointed at the release network gridpulse_weights.npz (numpy forward of gridpulse_np.py, with the
    stored widening set to 1: exam.py multiplies by widen.json's factors itself, as it did on the checkpoint);
  - the release ERCOT head ercot_head.npz is copied to WORK/ckpt/ercot_head_p1.npz;
  - widen.json is the shipped training/checks/widen.json, copied to WORK/.
Everything else (event list, onset, demand lookup, baselines, metrics) is exam.py's own code on shared/grid/data.
Writes WORK/exam.json and WORK/logs/exam_*; prints the headline numbers next to the released ones.
Usage: python rescore_release.py   (needs torch only because evaluate.py and ercot_head.py import it)"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: E402,F401  (paths: WORK, GRID on sys.path)
from _paths import MODEL, SRC  # noqa: E402

os.makedirs(f'{ROOT}/ckpt', exist_ok=True); os.makedirs(f'{ROOT}/logs', exist_ok=True)
shutil.copy(f'{SRC}/checks/widen.json', f'{ROOT}/widen.json')
shutil.copy(f'{MODEL}/ercot_head.npz', f'{ROOT}/ckpt/ercot_head_p1.npz')

import evaluate as EV  # noqa: E402
import gridpulse_np as GPN  # noqa: E402
import exam  # noqa: E402

REL = dict(np.load(f'{MODEL}/gridpulse_weights.npz'))
REL['widen'] = np.ones_like(REL['widen'])


def predict_release(rung, w, N, is_gb, f0, demand=None):
    return GPN.forward(REL, w, np.asarray(N), is_gb, f0, demand)


EV.predict = predict_release

if __name__ == '__main__':
    R = dict(rung='r5 (release weights)', ercot_head='p1 (release ercot_head.npz)', widen=exam.W,
             GB=exam.gb('r5'), ERCOT=exam.ercot('p1'))
    json.dump(R, open(f'{ROOT}/exam.json', 'w'), indent=1, default=float)
    old = json.load(open(f'{SRC}/checks/exam.json'))
    def g(d, *k):
        for x in k:
            d = d[x]
        return d
    rows = [('GB inertia MdAPE N=60', ('GB', 'inertia', 'gridpulse_N60', 'MdAPE')),
            ('GB inertia MdAPE N=10', ('GB', 'inertia', 'gridpulse_N10', 'MdAPE')),
            ('GB inertia coverage 1 sigma', ('GB', 'inertia', 'gridpulse_N60', 'cov1')),
            ('GB inertia coverage 2 sigma', ('GB', 'inertia', 'gridpulse_N60', 'cov2')),
            ('GB inertia trace only N=60', ('GB', 'inertia', 'trace_only_N60', 'MdAPE')),
            ('GB inertia constant 230', ('GB', 'inertia', 'baseline_constant_230', 'MdAPE')),
            ('GB inertia k x demand', ('GB', 'inertia', 'baseline_k_x_demand', 'MdAPE')),
            ('GB MW lost medium N=60', ('GB', 'loss', 'medium', 'gridpulse_N60')),
            ('GB MW lost medium RoCoF E230', ('GB', 'loss', 'medium', 'rocof_assumed_E230')),
            ('GB MW lost medium RoCoF NESO E', ('GB', 'loss', 'medium', 'rocof_true_E')),
            ('GB MW lost medium cov1', ('GB', 'loss', 'medium', 'cov1')),
            ('GB MW lost low N=60', ('GB', 'loss', 'low', 'gridpulse_N60')),
            ('GB nadir N5 (nadir later) MAE mHz', ('GB', 'forecast', 'N5_nadir_after_N', 'nadir_MAE_mHz')),
            ('GB nadir N10 (nadir later) MAE mHz', ('GB', 'forecast', 'N10_nadir_after_N', 'nadir_MAE_mHz')),
            ('GB settle from N10 MAE mHz', ('GB', 'forecast', 'N10_all', 'settle_MAE_mHz')),
            ('ERCOT MW lost MdAPE', ('ERCOT', 'loss', 'MdAPE')),
            ('ERCOT MW lost k x delta f', ('ERCOT', 'loss', 'baseline_k_delta_f')),
            ('ERCOT delta f forecast MdAPE', ('ERCOT', 'dfnad', 'MdAPE')),
            ('ERCOT inertia MdAPE', ('ERCOT', 'E', 'MdAPE'))]
    print(f"{'metric':38s} {'released':>10s} {'re-scored':>10s}")
    for name, k in rows:
        print(f'{name:38s} {g(old, *k):10.4f} {g(R, *k):10.4f}')
    print('n GB inertia events', R['GB']['inertia']['n'], '| n ERCOT', R['ERCOT']['n'])
