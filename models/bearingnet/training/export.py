"""Export a BearingNet checkpoint to numpy weights (every BatchNorm folded into its convolution) and check numpy == torch."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import bnet as BN, model as M

def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64); b = conv.bias.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    return (w * g[:, None, None]).astype(np.float32), ((b - bn.running_mean.numpy()) * g + bn.bias.detach().numpy()).astype(np.float32)

def trunk(W, p, stem, bst, body):
    W[p + 'stem_w'], W[p + 'stem_b'] = fold(stem, bst)
    for i, blk in enumerate(body):
        W[f'{p}b{i}_c1_w'], W[f'{p}b{i}_c1_b'] = fold(blk.c1, blk.b1); W[f'{p}b{i}_c2_w'], W[f'{p}b{i}_c2_b'] = fold(blk.c2, blk.b2); W[f'{p}b{i}_sk_w'], W[f'{p}b{i}_sk_b'] = fold(blk.sk, blk.bs)

def mlp(W, p, seq):
    lin = [m for m in seq if hasattr(m, 'weight')]
    for j, m in enumerate(lin): W[f'{p}{j}_w'] = m.weight.detach().numpy(); W[f'{p}{j}_b'] = m.bias.detach().numpy()
    return len(lin)

def export(ckpt, out, extra=None):
    net = BN.load_net(ckpt); W = {}
    if isinstance(net, M.HypNet):
        trunk(W, '', net.stem, net.bst, net.body); trunk(W, 'r', net.rstem, net.rbst, net.rbody); W['emb'] = net.emb.detach().numpy()
        meta = dict(arch='hyp', n_blocks=len(net.body), n_rblocks=len(net.rbody), n_fault=mlp(W, 'f', net.fault), n_rest=mlp(W, 'h', net.rest), nb=net.nb, nh=net.nh)
    else:
        trunk(W, '', net.stem, net.bst, net.body); meta = dict(arch='base', n_blocks=len(net.body), n_head=mlp(W, 'h', net.head))
    meta.update(sp_scale=M.SP_SCALE, sz_mu=M.SZ_MU, sz_sd=M.SZ_SD, classes=list(BN.CLS), ckpt=ckpt.split('/')[-1], **(extra or {}))
    np.savez_compressed(out, meta=json.dumps(meta), **W); print('wrote', out, sum(v.size for v in W.values()), 'numbers'); return net

if __name__ == '__main__':
    net = export(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None)
    sys.path.insert(0, _P.WORK + '/release'); import bearingnet as RB
    Z = np.load(BN.RD.D + 'real_dev.npz'); X, S = Z['x'][:64].astype(np.float32), Z['s'][:64]
    a = BN.raw(net, X, S); b = RB.BearingNet(sys.argv[2]).forward(X, S); print('numpy vs torch max abs diff', float(np.abs(a - b).max()), 'max |out|', float(np.abs(a).max()))
