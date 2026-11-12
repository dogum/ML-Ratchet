"""Fuse BatchNorm and export CellSense to portable numpy weights; check the numpy forward against torch; pack for the browser."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
import realexam as RX

def fuse(conv, bn):
    w = conv.weight.detach().numpy(); s = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    return (w * s[:, None, None]).astype(np.float32), ((conv.bias.detach().numpy() if conv.bias is not None else 0) * s + bn.bias.detach().numpy() - bn.running_mean.numpy() * s).astype(np.float32)

def export(ckpt, out, js_out=None):
    net = RX.load_net(ckpt); A = {}; meta = {'blocks': [], 'labels': net.labels, 'mu_y': net.mu_y.tolist(), 'sd_y': net.sd_y.tolist(), 'extra': int(net.extra), 'smooth': float(getattr(net, 'smooth', 0) or 0)}
    A['stem_w'], A['stem_b'] = fuse(net.stem[0], net.stem[1])
    for i, b in enumerate(net.blocks):
        for nm, (c, bn) in zip(('c1', 'c2', 'sk'), ((b.c1, b.b1), (b.c2, b.b2), (b.sk, b.bs))): A[f'b{i}{nm}_w'], A[f'b{i}{nm}_b'] = fuse(c, bn)
        meta['blocks'].append([b.c1.stride[0], b.c1.padding[0], b.c2.padding[0]])
    A['h0_w'] = net.head[0].weight.detach().numpy(); A['h0_b'] = net.head[0].bias.detach().numpy(); A['h1_w'] = net.head[2].weight.detach().numpy(); A['h1_b'] = net.head[2].bias.detach().numpy()
    np.savez_compressed(out, meta=json.dumps(meta), **{k: v.astype(np.float32) for k, v in A.items()})
    import cellsense
    cs = cellsense.CellSenseNet(out); X = np.random.default_rng(0).standard_normal((3, 8, 250)).astype(np.float32); S = np.random.default_rng(1).random((3, max(1, net.extra))).astype(np.float32)
    with torch.no_grad(): ref = net(torch.from_numpy(X), torch.from_numpy(S) if net.extra else None).numpy()
    got = cs.forward(X, S if net.extra else None); print('max |numpy - torch| %.2e' % np.abs(got - ref).max(), 'params', sum(v.size for v in A.values()))
    if js_out:
        parts, man, off = [], [], 0
        for k in sorted(A):
            a = np.ascontiguousarray(A[k], np.float32); parts.append(a.ravel()); man.append({'k': k, 'shape': list(a.shape), 'off': off}); off += a.size
        json.dump({'man': {'tensors': man, 'n': off, 'meta': meta}, 'b64': base64.b64encode(np.concatenate(parts).astype('<f4').tobytes()).decode()}, open(js_out, 'w'))
        print('browser weights MB %.2f' % (off * 4 * 4 / 3 / 1e6))

if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
