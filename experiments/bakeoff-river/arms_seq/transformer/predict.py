"""River Bake-off arm seq_transformer: Transformer (causal encoder over daily tokens). Ungauged (forcing + attributes only). numpy only.

The shared pipeline and the numpy network live in ../common/seqlib.py (hashed with this arm via extra_files);
this file only points it at this arm's weights (weights.npz, released rung r3). See card.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
import seqlib  # noqa: E402

_M = seqlib.Model(os.path.join(HERE, 'weights.npz'))


def predict(batch):
    return _M.predict(batch)


def predict_many(batches):
    return _M.predict_many(batches)
