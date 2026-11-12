"""Export a QuakeNet checkpoint to numpy weights for the release, with every BatchNorm folded into its convolution."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import qnet as Q

def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64); b = conv.bias.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps); return (w * g[:, None, None]).astype(np.float32), ((b - bn.running_mean.numpy()) * g + bn.bias.detach().numpy()).astype(np.float32)

def export(ckpt, out, extra=None):
    net = Q.load_net(ckpt); W = {}
    for i, blk in enumerate(net.body):
        W[f'b{i}_c1_w'], W[f'b{i}_c1_b'] = fold(blk.c1, blk.b1); W[f'b{i}_c2_w'], W[f'b{i}_c2_b'] = fold(blk.c2, blk.b2); W[f'b{i}_sk_w'], W[f'b{i}_sk_b'] = fold(blk.sk, blk.bs)
    lin = [m for m in net.head if hasattr(m, 'weight')]
    for j, m in enumerate(lin): W[f'h{j}_w'] = m.weight.detach().numpy(); W[f'h{j}_b'] = m.bias.detach().numpy()
    meta = dict(n_blocks=len(net.body), n_head=len(lin), mu_y=net.mu_y.tolist(), sd_y=net.sd_y.tolist(), outputs=['mw', 'dist', 'logz'], ckpt=ckpt.split('/')[-1], **(extra or {}))
    np.savez_compressed(out, meta=json.dumps(meta), **W); print('wrote', out, sum(v.size for v in W.values()), 'numbers')

if __name__ == '__main__': export(sys.argv[1], sys.argv[2])
