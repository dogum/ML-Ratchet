"""Transformer on the trip trace (20-s patch tokens) (`trace_transformer`): Vehicle Bake-off trace track, arms_trace (agent B).
numpy-only inference: the shared trace pipeline + numpy core in ../common/tracelib.py, weights in weights.npz."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
import tracelib  # noqa: E402

_M = tracelib.Model(os.path.join(HERE, 'weights.npz'))


def predict(batch):
    return _M.predict(batch)


def predict_many(batches):
    return _M.predict_many(batches)
