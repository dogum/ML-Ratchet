"""Round-2 trace ensemble (Vehicle Bake-off round 2): 5 members of one v1.0 trace architecture, numpy inference.
Shared pipeline + cores: ../../../arms_trace/common/tracelib.py (v1.0, frozen); ensemble rule: ../common/enslib.py
(mean log-quantile per level; pred = exp(mean log q50))."""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_d = os.path.dirname(HERE)            # released arms: round2/arms/<arm>/ -> ../common; candidates under round2/work/
while not os.path.exists(os.path.join(_d, 'common', 'enslib.py')) and os.path.dirname(_d) != _d:
    _d = os.path.join(_d, 'arms') if os.path.exists(os.path.join(_d, 'arms', 'common', 'enslib.py')) else os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, 'common'))
import enslib  # noqa: E402

_M = enslib.TraceEnsemble(sorted(glob.glob(os.path.join(HERE, 'weights', 'm*.npz'))))


def predict(batch):
    return _M.predict(batch)


def predict_many(batches):
    return _M.predict_many(batches)
