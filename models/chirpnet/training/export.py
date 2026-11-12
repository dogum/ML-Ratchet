"""Fuse BatchNorm into convolutions and export portable numpy weights (+ a JS-friendly binary)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
from evaluate import load

def fuse(conv, bn):
    w = conv.weight.detach().numpy(); g = bn.weight.detach().numpy(); b = bn.bias.detach().numpy()
    m = bn.running_mean.numpy(); v = bn.running_var.numpy(); s = g / np.sqrt(v + bn.eps)
    return (w * s[:, None, None]).astype(np.float32), (b - m * s).astype(np.float32)

def export(ckpt, out, threshold, background, live_s, extra_meta=None):
    net = load(ckpt)
    arr, conv = {}, []
    st = net.stem
    w, b = fuse(st[0], st[1]); arr['w0'], arr['b0'] = w, b; conv.append((st[0].stride[0], st[0].padding[0]))
    i = 1
    for blk in net.blocks:
        for c, bn in ((blk.c1, blk.b1), (blk.c2, blk.b2), (blk.sk, blk.bs)):
            w, b = fuse(c, bn); arr[f'w{i}'], arr[f'b{i}'] = w, b; conv.append((c.stride[0], c.padding[0])); i += 1
    arch = 'cnn'
    if net.__class__.__name__ == 'ChirpNet2S':
        arch = '2s'
        w, b = fuse(net.mix[0], net.mix[1]); arr['mixw'], arr['mixb'] = w, b
        for nm, (c, bn) in zip(('p1', 'p2', 'ps'), ((net.post.c1, net.post.b1), (net.post.c2, net.post.b2), (net.post.sk, net.post.bs))):
            w, b = fuse(c, bn); arr[nm + 'w'], arr[nm + 'b'] = w, b
        arr['auxw'] = net.aux.weight.detach().numpy(); arr['auxb'] = net.aux.bias.detach().numpy()
    if getattr(net, 'pos', False):
        arr['rw'] = net.red.weight.detach().numpy().astype(np.float32); arr['rb'] = net.red.bias.detach().numpy().astype(np.float32)
    arr['lw0'] = net.head[0].weight.detach().numpy(); arr['lb0'] = net.head[0].bias.detach().numpy()
    arr['lw1'] = net.head[2].weight.detach().numpy(); arr['lb1'] = net.head[2].bias.detach().numpy()
    meta = {'arch': arch, 'conv': conv, 'threshold': float(threshold), 'background_livetime_s': float(live_s), 'fs': 2048, 'window': 4096,
            'outputs': ['score', 'log_chirp_mass', 'log_sigma_log_chirp_mass', 'mass_ratio', 'log_sigma_q', 't_merger_s', 'log_sigma_t']}
    meta.update(extra_meta or {})
    np.savez_compressed(out, meta=json.dumps(meta), background=np.sort(np.asarray(background, np.float32)), **arr)
    # verify against torch
    import chirpnet
    cn = chirpnet.ChirpNet(out)
    x = np.random.default_rng(0).standard_normal((4, 2, 4096)).astype(np.float32)
    with torch.no_grad(): ref = net(torch.from_numpy(x)).numpy()
    got = cn.forward(x); ref = ref[:, :got.shape[1]]
    print('max |numpy - torch| =', float(np.abs(ref - got).max()), 'params', sum(v.size for k, v in arr.items()))
    return arr, meta

if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2], 0.0, [0.0], 1.0)
