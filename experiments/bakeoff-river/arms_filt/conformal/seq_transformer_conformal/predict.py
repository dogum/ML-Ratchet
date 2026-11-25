"""Conformal wrapper (agent C, River Bake-off): seq_transformer with Mondrian split-conformal bands.

The wrapped arm's predict() runs unchanged; its 5-95 % and 25-75 % band endpoints are shifted in log space per
basin stratum (snow / arid / humid / other) and protocol/lead, with shifts fitted on seq_transformer's dev residuals only
(calib.json; method in ../lib/cf_mondrian.py). mean and q50 are untouched. numpy only (plus the wrapped arm).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, 'calib.json')))
_LIB = os.path.normpath(os.path.join(HERE, '..', 'lib'))
_HARNESS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'harness'))
for _p in (_LIB, _HARNESS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import cf_mondrian as CF  # noqa: E402
import armlib  # noqa: E402

BASE = os.path.normpath(os.path.join(HERE, CAL['base_rel']))
_base_predict, _base_mod = armlib.load_predict(BASE)


def _wrap(b, o):
    sh = CAL['jobs'][CF.job_key(b)][CF.stratum(b['classes'])]
    return CF.apply(o, sh, CF.offset(b['static']))


def predict(batch):
    return _wrap(batch, _base_predict(batch))


def predict_many(batches):
    if hasattr(_base_mod, 'predict_many'):
        outs = _base_mod.predict_many(batches)
    else:
        outs = [_base_predict(b) for b in batches]
    return [_wrap(b, o) for b, o in zip(batches, outs)]
