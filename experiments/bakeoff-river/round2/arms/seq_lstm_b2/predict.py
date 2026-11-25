"""River Bake-off round-2 arm seq_lstm_b2: LSTM (FlowCast's network) retrained on bank b2. Ungauged (forcing + attributes only). numpy only.

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
