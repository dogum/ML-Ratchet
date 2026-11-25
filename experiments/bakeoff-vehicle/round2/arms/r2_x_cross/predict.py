"""Round-2 cross-family ensemble (PREREG section 3): equal weights over the round-2 entries of the transformer, GRU,
MLP and GP families; mean log-quantile per level, pred = exp(mean log q50). A trace-track arm: the table members'
inputs are rebuilt from the trace batch (../common/xlib.py table_from_trace). numpy only."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
import xlib  # noqa: E402

_MEM = json.load(open(os.path.join(HERE, 'members.json')))
_M = xlib.CrossFamily([(n, os.path.normpath(os.path.join(HERE, p)), k) for n, p, k in _MEM])


def predict(batch):
    return _M.predict(batch)


def predict_many(batches):
    return _M.predict_many(batches)


def member_outputs(batch):
    return _M.member_outputs(batch)
