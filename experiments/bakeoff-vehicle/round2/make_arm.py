"""Round 2: write a candidate / released arm folder from trained members.

python make_arm.py trace <arm_dir> <members_dir>      members_dir/m*/weights.npz -> arm_dir/weights/m*.npz + predict.py
python make_arm.py mlp <arm_dir> <members_dir>        members_dir/s*.npz -> arm_dir/weights/ + predict.py, model_np.py
python make_arm.py x <arm_dir> <members.json>         cross-family ensemble (members: [[name, predict.py rel path, kind]])
ratchet.json and card.md are written by the caller (finalize.py) once the rung decision is known; a candidate gets a
minimal ratchet.json so harness/evaluate.py can load it.
"""
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BAKEOFF = os.path.dirname(HERE)

TRACE_PREDICT = '''"""Round-2 trace ensemble (Vehicle Bake-off round 2): {n} members of one v1.0 trace architecture, numpy inference.
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
'''

X_PREDICT = '''"""Round-2 cross-family ensemble (PREREG section 3): equal weights over the round-2 entries of the transformer, GRU,
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
'''


def minimal_meta(arm_dir, kind):
    arm_id = os.path.basename(arm_dir.rstrip('/\\'))
    tracks = ['table'] if kind == 'mlp' else ['trace']
    return {'arm_id': arm_id, 'label': arm_id, 'family': 'ensemble', 'group': 'round2', 'tracks': tracks,
            'numpy_only': True, 'numpy_only_reason': '', 'real_labels_in_training': False,
            'compute': {'train_cpu_hours': 0.0}, 'feature_engineering': 'see card.md',
            'rungs': [{'id': 'cand', 'change': 'candidate', 'decision': 'released', 'why': 'candidate'}], 'released': 'cand'}


def main():
    kind, arm_dir, src = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
    os.makedirs(os.path.join(arm_dir, 'weights'), exist_ok=True)
    if kind == 'trace':
        ms = sorted(glob.glob(os.path.join(src, 'm*', 'weights.npz')))
        for p in ms:
            shutil.copy(p, os.path.join(arm_dir, 'weights', os.path.basename(os.path.dirname(p)) + '.npz'))
        open(os.path.join(arm_dir, 'predict.py'), 'w').write(TRACE_PREDICT.format(n=len(ms)))
    elif kind == 'mlp':
        ms = sorted(glob.glob(os.path.join(src, 's*.npz')))
        for p in ms:
            shutil.copy(p, os.path.join(arm_dir, 'weights', os.path.basename(p)))
        v1 = os.path.join(BAKEOFF, 'arms_table', 'tab_mlp')
        shutil.copy(os.path.join(v1, 'model_np.py'), os.path.join(arm_dir, 'model_np.py'))
        shutil.copy(os.path.join(v1, 'predict.py'), os.path.join(arm_dir, 'predict.py'))
    elif kind == 'x':
        shutil.rmtree(os.path.join(arm_dir, 'weights'))
        shutil.copy(src, os.path.join(arm_dir, 'members.json'))
        open(os.path.join(arm_dir, 'predict.py'), 'w').write(X_PREDICT)
    p = os.path.join(arm_dir, 'ratchet.json')
    if not os.path.exists(p):
        json.dump(minimal_meta(arm_dir, kind), open(p, 'w'), indent=1)
    if not os.path.exists(os.path.join(arm_dir, 'card.md')):
        open(os.path.join(arm_dir, 'card.md'), 'w').write('# candidate\n\nRound-2 candidate arm; card written at release.\n')
    print('wrote', arm_dir, len(os.listdir(os.path.join(arm_dir, 'weights'))) if kind != 'x' else '')


if __name__ == '__main__':
    main()
