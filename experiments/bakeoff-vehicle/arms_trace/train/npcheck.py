"""Release check: numpy inference (the arm's predict.py -> common/tracelib.py) vs the torch checkpoint of the released
rung, row by row (each trip unpadded, chunked exactly as tracelib does for > 1,440 blocks), in float64 and float32.
Cases: 16 frozen sim val trips + the first 24 dev records of VED and EPA + every HD dev record (5 ZTBus legs, Mersin,
ICCT; the longest ZTBus legs exercise the chunking). Writes <arm>/work/npcheck.json.
usage: python3 npcheck.py <arm> <rung>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRC = os.path.dirname(HERE)
sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
sys.path.insert(0, os.path.join(TRC, 'common'))
import bake  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(1)
import nets  # noqa: E402
import tracelib as TL  # noqa: E402


def torch_predict(net, M, batch, dtype):
    n = batch['n']
    enc = [(TL.dyn_features(t),) + TL.trip_dist_dur(t) for t in batch['traces']]
    dist = np.array([e[1] for e in enc]); dur = np.array([e[2] for e in enc])
    A0, Eph, anc = M.anchors(batch, dist)
    st = TL.statics(M.W, M.cfg, batch['spec'], batch['rl'], dist, dur, A0, Eph)
    out = []
    with torch.no_grad():
        for i in range(n):
            dyn = enc[i][0]
            s = torch.tensor(st[i:i + 1], dtype=dtype)
            P, wsum = 0.0, 0
            for a in range(0, len(dyn), TL.CHUNK):
                ch = dyn[a:a + TL.CHUNK]
                T = len(ch)
                p = net.encode(torch.tensor(ch[None], dtype=dtype), s, torch.ones(1, T, dtype=dtype),
                               torch.tensor([T]))
                P = P + p * T; wsum += T
            q, c = net.head(P / wsum, s)
            out.append(anc[i] * np.exp(q.numpy()[0].astype(np.float64)))
    return np.array(out), np.array([len(e[0]) for e in enc])


def main(arm, rung, wpath=None):
    cfg = json.load(open(os.path.join(HERE, 'configs', f'{arm}_{rung}.json')))
    sd = torch.load(os.path.join(TRC, 'ckpt', f'{arm}_{rung}.pt'), weights_only=False)['state']
    M = TL.Model(wpath or os.path.join(TRC, arm, 'weights.npz'))
    batches = [('simval', next(bake.simval_batches('trace', chunk=16, nmax=16))[0])]
    for name in ('ved', 'epa', 'hd'):
        for nm, s, b in bake._real_batches('dev', 'trace', sets=(name,), chunk=24 if name != 'hd' else 400):
            batches.append((name, b))
            break
    res = {}
    for prec, dtype in (('float64', torch.float64), ('float32', torch.float32)):
        net = nets.build(cfg); net.load_state_dict(sd); net.eval()
        net = net.to(dtype)
        worst, nrows, maxlen = 0.0, 0, 0
        for name, b in batches:
            o = M.predict(b)
            Qn = np.stack([o[k] for k in TL.QKEYS], 1)
            Qt, nbk = torch_predict(net, M, b, dtype)
            d = float(np.max(np.abs(np.log(Qn / Qt))))
            res.setdefault(prec, {})[name] = {'rows': b['n'], 'max_abs_log_diff': d, 'max_blocks': int(nbk.max())}
            worst = max(worst, d); nrows += b['n']; maxlen = max(maxlen, int(nbk.max()))
        res[prec]['all'] = {'rows': nrows, 'max_abs_log_diff': worst, 'max_blocks': maxlen}
        print(prec, res[prec]['all'], flush=True)
    json.dump(res, open(os.path.join(TRC, arm, 'work', 'npcheck.json' if wpath is None else 'npcheck_test.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
