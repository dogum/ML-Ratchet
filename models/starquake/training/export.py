"""Export a StarQuake checkpoint to numpy weights (every BatchNorm folded into the convolution before it) and check
that the numpy forward pass equals torch."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import snet as SN

def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64); b = conv.bias.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps); sh = (-1,) + (1,) * (w.ndim - 1)
    return (w * g.reshape(sh)).astype(np.float32), ((b - bn.running_mean.numpy()) * g + bn.bias.detach().numpy()).astype(np.float32)

def export(ckpt, out, extra=None):
    net = SN.load_net(ckpt); W = {}
    W['s1_w'], W['s1_b'] = fold(net.s1[0], net.s1[1]); W['s2_w'], W['s2_b'] = fold(net.s2[0], net.s2[1])
    for nm, body in (('r1', net.r1), ('r2', net.r2)):
        for i, blk in enumerate(body):
            for a, c, b in (('c1', blk.c1, blk.b1), ('c2', blk.c2, blk.b2), ('sk', blk.sk, blk.bs)): W[f'{nm}_{i}_{a}_w'], W[f'{nm}_{i}_{a}_b'] = fold(c, b)
    lin = [m for m in net.head if hasattr(m, 'weight')]
    for j, m in enumerate(lin): W[f'h{j}_w'] = m.weight.detach().numpy(); W[f'h{j}_b'] = m.bias.detach().numpy()
    meta = dict(n1=len(net.r1), n2=len(net.r2), n_head=len(lin), mu_y=net.mu_y.tolist(), sd_y=net.sd_y.tolist(), ckpt=ckpt.split('/')[-1], **(extra or {}))
    np.savez_compressed(out, meta=json.dumps(meta), **W); print('wrote', out, sum(v.size for v in W.values()), 'numbers'); return net

if __name__ == '__main__':
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    net = export(sys.argv[1], sys.argv[2], extra)
    sys.path.insert(0, _P.WORK + '/release'); import importlib, starquake as R; importlib.reload(R)
    z = np.load(_P.WORK + '/data/real.npz'); i = np.arange(0, len(z['x1']), 97)[:20]
    P = SN.predict(net, z['x1'][i], z['x2'][i], z['s'][i]); m = R.Net(sys.argv[2])
    Q = [m.forward(z['x1'][j].astype(np.float32), z['x2'][j].astype(np.float32), z['s'][j]) for j in i]
    d = max(np.max(np.abs(np.array([q[k] for q in Q]) - P[k])) for k in ('ln_numax', 'ln_dnu', 'sig_numax', 'sig_dnu', 'p_det')); print('max |numpy - torch| =', d)
