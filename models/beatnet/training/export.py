"""Export a BeatNet checkpoint to numpy weights for the release, every BatchNorm folded into its convolution."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import bnet as BN, train as TR

def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64); b = conv.bias.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps); return (w * g[:, None, None]).astype(np.float32), ((b - bn.running_mean.numpy()) * g + bn.bias.detach().numpy()).astype(np.float32)

def export(ckpt, out, extra=None):
    net = BN.load_net(ckpt); W = {}
    for i, blk in enumerate(net.body):
        W[f'b{i}_c1_w'], W[f'b{i}_c1_b'] = fold(blk.c1, blk.b1); W[f'b{i}_c2_w'], W[f'b{i}_c2_b'] = fold(blk.c2, blk.b2); W[f'b{i}_sk_w'], W[f'b{i}_sk_b'] = fold(blk.sk, blk.bs)
    lin = [m for m in net.head if hasattr(m, 'weight')]
    for j, m in enumerate(lin): W[f'h{j}_w'] = m.weight.detach().numpy(); W[f'h{j}_b'] = m.bias.detach().numpy()
    meta = dict(n_blocks=len(net.body), strides=[blk.c1.stride[0] for blk in net.body], n_head=len(lin), cin=net.cin, ctr=net.ctr, qmu=TR.QMU, qsd=TR.QSD,
                classes=['N', 'S', 'V', 'F'], ckpt=ckpt.split('/')[-1], **(extra or {}))
    np.savez_compressed(out, meta=json.dumps(meta), **W); print('wrote', out, sum(v.size for v in W.values()), 'numbers'); return meta

if __name__ == '__main__': export(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None)
