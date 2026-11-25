"""Speed / export check of the four nets on frozen val trips (no training data written)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train as TT  # noqa: E402
import nets  # noqa: E402
import bake  # noqa: E402
import tracelib as TL  # noqa: E402

cfgs = [json.load(open(os.path.join(TT.HERE, 'configs', f'{a}.json'))) for a in sys.argv[1:]]
vrows = bake.simval_rows()
V = TT.Bank('val', idx=vrows[:600])
W = TT.norm_stats(V, cfgs[0])
EV = V.epoch_data(bake.val_tiers(), W, cfgs[0])
for cfg in cfgs:
    torch.manual_seed(0)
    net = nets.build(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    bl = TT.make_batches(V.nb, np.random.default_rng(0), cfg['tokens'], cfg['max_batch'])
    c0 = time.process_time(); nblk = 0
    for k, bt in enumerate(bl[:12]):
        X, S, M, Ln, t, ct, w = V.batch(bt, EV)
        q, c = net(X, S, M, Ln)
        loss, _, _ = nets.loss_fn(q, c, t, ct, w, 0.2)
        opt.zero_grad(); loss.backward(); opt.step()
        nblk += int(M.sum())
    dt = time.process_time() - c0
    per_epoch = dt / nblk * 6.64e6
    d = '/tmp/claude-0/-home-claude/706c0d77-bfd2-594c-b22e-e532edd10b67/scratchpad/bench_' + cfg['arch']
    os.makedirs(d, exist_ok=True)
    nets.export(net, cfg, W, os.path.join(d, 'weights.npz'))
    rows = np.argsort(V.nb)[[0, 50, 300, 599]]
    chk = TT.np_check(net, d, V, EV, rows)
    # numpy predict on a sim batch through the Model (chunking path with a long synthetic trace)
    M_ = TL.Model(os.path.join(d, 'weights.npz'))
    b, y = next(bake.simval_batches('trace', chunk=16, nmax=16))
    t1 = time.process_time(); o = M_.predict(b); tp = (time.process_time() - t1) / 16
    print(json.dumps({'arch': cfg['arch'], 'params': nets.n_params(net), 'cpu_s_per_epoch_est': round(per_epoch),
                      'batches_12_cpu': round(dt, 1), 'np_check': chk, 'np_predict_s_per_trip': round(tp, 4),
                      'pred_ok': bool(np.all(o['pred'] > 0))}), flush=True)
