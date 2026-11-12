"""Export a SwellNet checkpoint to numpy weights (BatchNorm folded into every convolution) and check numpy == torch."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import model as M, evaluate as EVL
ROOT = EVL.ROOT

def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64); b = conv.bias.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    return (w * g[:, None, None, None]).astype(np.float32), ((b - bn.running_mean.numpy()) * g + bn.bias.detach().numpy()).astype(np.float32)

def export(ck, out, extra=None):
    net = EVL.load_net(ck); W = {}
    W['stem_w'], W['stem_b'] = fold(net.stem, net.bstem)
    for i, blk in enumerate(net.body):
        W[f'b{i}_c1_w'], W[f'b{i}_c1_b'] = fold(blk.c1, blk.b1); W[f'b{i}_c2_w'], W[f'b{i}_c2_b'] = fold(blk.c2, blk.b2); W[f'b{i}_sk_w'], W[f'b{i}_sk_b'] = fold(blk.sk, blk.bs)
    W['red_w'], W['red_b'] = fold(net.red, net.bred)
    lin = [m for m in net.head if hasattr(m, 'weight')]
    for j, m in enumerate(lin): W[f'h{j}_w'] = m.weight.detach().numpy(); W[f'h{j}_b'] = m.bias.detach().numpy()
    meta = dict(n_blocks=len(net.body), n_head=len(lin), mu_y=net.mu_y.tolist(), sd_y=net.sd_y.tolist(), outputs=M.OUT, ckpt=os.path.basename(ck), **(extra or {}))
    np.savez_compressed(out, meta=json.dumps(meta), **W); print('wrote', out, sum(v.size for v in W.values()), 'numbers')
    return net

if __name__ == '__main__':
    ck, out = sys.argv[1], sys.argv[2]; extra = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    net = export(ck, out, extra)
    sys.path.insert(0, os.path.dirname(out)); import swellcast as SC
    X = EVL.real_windows('dev', True)[1][:6]; f = M.features(X)
    with torch.no_grad(): ot = net(torch.from_numpy(f)).numpy()
    on = SC.SwellCast(out).forward(f); print('max |numpy - torch| =', float(np.abs(on - ot).max()), 'max |torch| =', float(np.abs(ot).max()))
