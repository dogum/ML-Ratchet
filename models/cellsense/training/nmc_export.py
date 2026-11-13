"""Fuse BatchNorm and export the CellSense NMC811 network to numpy weights (cellsense_nmc811_weights.npz next to
cellsense.py), with the dev-fitted widen factors in its meta; check numpy against torch.

    python nmc_export.py <ckpt> <exam json with the widen factors> [out.npz]
"""
import _paths as _P
import os, sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import nmc_ratchet as RT


def fuse(conv, bn):
    """fold a BatchNorm into the convolution before it (as export.py)"""
    w = conv.weight.detach().numpy(); s = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    return (w * s[:, None, None]).astype(np.float32), ((conv.bias.detach().numpy() if conv.bias is not None else 0) * s + bn.bias.detach().numpy() - bn.running_mean.numpy() * s).astype(np.float32)


def export(ckpt, factors, out, note=''):
    net = RT.load_net(ckpt); A = {}
    meta = {'blocks': [], 'labels': net.labels, 'mu_y': net.mu_y.tolist(), 'sd_y': net.sd_y.tolist(), 'extra': 0, 'smooth': float(getattr(net, 'smooth', 0) or 0),
            'chemistry': 'nmc811', 'cell': 'LG INR21700-M50T', 'vgrid': [4.15, 2.50, 250], 'widen': factors, 'note': note, 'target': getattr(net, 'target', 'total'), 'endrule': bool(getattr(net, 'endrule', False))}
    A['stem_w'], A['stem_b'] = fuse(net.stem[0], net.stem[1])
    for i, b in enumerate(net.blocks):
        for nm, (c, bn) in zip(('c1', 'c2', 'sk'), ((b.c1, b.b1), (b.c2, b.b2), (b.sk, b.bs))): A[f'b{i}{nm}_w'], A[f'b{i}{nm}_b'] = fuse(c, bn)
        meta['blocks'].append([b.c1.stride[0], b.c1.padding[0], b.c2.padding[0]])
    A['h0_w'] = net.head[0].weight.detach().numpy(); A['h0_b'] = net.head[0].bias.detach().numpy(); A['h1_w'] = net.head[2].weight.detach().numpy(); A['h1_b'] = net.head[2].bias.detach().numpy()
    np.savez_compressed(out, meta=json.dumps(meta), **{k: v.astype(np.float32) for k, v in A.items()})
    import cellsense
    X = np.random.default_rng(0).standard_normal((4, 6, 250)).astype(np.float32)
    with torch.no_grad(): ref = net(torch.from_numpy(X)).numpy()
    got = cellsense.CellSenseNet(out).forward(X); err = float(np.abs(got - ref).max()); print('max |numpy - torch| %.2e' % err, 'params', sum(v.size for v in A.values()))
    return err


if __name__ == '__main__':
    ex = json.load(open(sys.argv[2])); out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_P.MODEL, 'cellsense_nmc811_weights.npz')
    export(sys.argv[1], ex['factors'], out, note=f"rung checkpoint {os.path.basename(sys.argv[1])}")
