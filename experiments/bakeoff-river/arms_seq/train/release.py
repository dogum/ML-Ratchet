"""Freeze the released rung of an arm into its arm folder (predict.py + weights.npz); ratchet.json/card.md are written
separately. usage: python3 release.py <arm> <rung>
Also re-checks numpy (released predict.py) against torch (the rung's checkpoint) on 16 frozen sim val windows and
3 dev basins in mm/day, and writes <arm>/work/export_check.json."""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.dirname(HERE)
sys.path.insert(0, '/home/claude/bakeoff/river/harness')
sys.path.insert(0, os.path.join(SEQ, 'common'))
import bake  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(1)
import nets  # noqa: E402
import seqlib  # noqa: E402

PREDICT = '''"""River Bake-off arm {arm_id}: {label}. Ungauged (forcing + attributes only). numpy only.

The shared pipeline and the numpy network live in ../common/seqlib.py (hashed with this arm via extra_files);
this file only points it at this arm's weights (weights.npz, released rung {rung}). See card.md.
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
'''


def torch_predict(net, b, dtype=np.float32):
    X, blocks = seqlib.blocks_of(b)
    st = np.repeat(np.asarray(b['static'], np.float32)[None], len(blocks), 0)
    with torch.no_grad():
        m, q = net(torch.from_numpy(X.astype(dtype)), torch.from_numpy(st.astype(dtype)))
    pm = float(np.exp(np.float64(np.asarray(b['static'], np.float32)[11])))
    m = np.concatenate([m[i, 365:365 + bb - aa].numpy() for i, (aa, bb) in enumerate(blocks)]) * pm
    q = np.concatenate([q[i, 365:365 + bb - aa].numpy() for i, (aa, bb) in enumerate(blocks)]) * pm
    return m, q


def main(arm, rung, arm_id, label):
    cand = os.path.join(SEQ, arm, 'work', f'cand_{rung}')
    dst = os.path.join(SEQ, arm)
    shutil.copy(os.path.join(cand, 'weights.npz'), os.path.join(dst, 'weights.npz'))
    open(os.path.join(dst, 'predict.py'), 'w').write(PREDICT.format(arm_id=arm_id, label=label, rung=rung))
    ck = torch.load(os.path.join(SEQ, 'ckpt', f'{arm}_{rung}.pt'), weights_only=False)
    net = nets.build(ck['cfg']); net.load_state_dict(ck['state']); net.eval()
    M = seqlib.Model(os.path.join(dst, 'weights.npz'))
    B = bake.load_bank('val', ['F', 'qo', 'attrs', 'st', 'lat', 'lon', 'start'])
    bs = [bake.sim_window(B, i, a) for i in range(8) for a in bake.SIMVAL_OFFSETS]
    bs += [bake.dev_batch(x) for x in bake.dev_basins()[:3]]
    net64 = nets.build(ck['cfg']); net64.load_state_dict(ck['state']); net64.eval(); net64.double()
    if hasattr(net64, 'pe'):
        net64.pe = net64.pe.double()
    worst = {'f32': [0.0, 0.0], 'f64': [0.0, 0.0]}
    for b in bs:
        o = M.predict(b)
        qn = np.stack([o[k] for k in seqlib.QKEYS], 1)
        scale = max(float(np.mean(o['mean'])), 1e-3)
        for key, n_, dt in (('f32', net, np.float32), ('f64', net64, np.float64)):
            mt, qt = torch_predict(n_, b, dt)
            worst[key][0] = max(worst[key][0], float(np.max(np.abs(o['mean'] - mt))) / scale)
            worst[key][1] = max(worst[key][1], float(np.max(np.abs(qn - np.maximum(qt, 0)))) / scale)
    rec = {'arm': arm, 'rung': rung, 'cases': len(bs),
           'max_abs_diff_mean_rel_to_mean_flow': worst['f32'][0], 'max_abs_diff_quantiles_rel_to_mean_flow': worst['f32'][1],
           'torch_float64': {'max_abs_diff_mean_rel': worst['f64'][0], 'max_abs_diff_quantiles_rel': worst['f64'][1]},
           'note': 'numpy float64 (released predict.py) vs torch float32 (training checkpoint), mm/day, 16 sim val '
                   'windows + 3 dev basins (14 years each)'}
    json.dump(rec, open(os.path.join(dst, 'work', 'export_check.json'), 'w'), indent=1)
    print(json.dumps(rec))


if __name__ == '__main__':
    main(*sys.argv[1:5])
