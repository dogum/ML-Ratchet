"""Round 2: freeze a b2 rung into a round-2 arm folder (round2/arms/<arm_id>/). ratchet.json and card.md are written
separately (with the rung numbers).

usage: python release2.py seq <run> <arm_id> <label>      weights from work/cand/<run>/weights.npz; predict.py uses
                                                         the frozen round-1 pipeline arms_seq/common/seqlib.py
       python release2.py gbm <rung_dir> <arm_id>          gbm_trees.npz from <rung_dir>; predict.py, feats.py and
                                                         treeexport.py are byte copies of arms_tab/tab_gbm's
"""
import os
import shutil
import sys

import r2

SEQ_PREDICT = '''"""River Bake-off round-2 arm {arm_id}: {label}. Ungauged (forcing + attributes only). numpy only.

The shared pipeline and the numpy network are round 1's frozen arms_seq/common/seqlib.py (hashed with this arm via
extra_files); this file only points it at this arm's weights (weights.npz, trained on bank b2). See card.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', '..', 'arms_seq', 'common')))
import seqlib  # noqa: E402

_M = seqlib.Model(os.path.join(HERE, 'weights.npz'))


def predict(batch):
    return _M.predict(batch)


def predict_many(batches):
    return _M.predict_many(batches)
'''


def main():
    kind = sys.argv[1]
    if kind == 'seq':
        run, arm_id, label = sys.argv[2:5]
        dst = os.path.join(r2.ROUND2, 'arms', arm_id)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(r2.WORK, 'cand', run, 'weights.npz'), os.path.join(dst, 'weights.npz'))
        open(os.path.join(dst, 'predict.py'), 'w').write(SEQ_PREDICT.format(arm_id=arm_id, label=label))
    elif kind == 'gbm':
        rung, arm_id = sys.argv[2:4]
        dst = os.path.join(r2.ROUND2, 'arms', arm_id)
        os.makedirs(dst, exist_ok=True)
        src = os.path.join(r2.BAKEOFF, 'arms_tab', 'tab_gbm')
        for f in ('predict.py', 'feats.py', 'treeexport.py'):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        shutil.copy(os.path.join(rung, 'gbm_trees.npz'), os.path.join(dst, 'gbm_trees.npz'))
    print('released', sys.argv[2:])


if __name__ == '__main__':
    main()
