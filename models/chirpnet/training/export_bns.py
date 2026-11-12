"""Export the neutron-star model to portable numpy weights (BatchNorm fused), with its real-noise background."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
from bns_train import load_bns, NFFT, HOP, NFR

def fuse(conv, bn):
    w = conv.weight.detach().numpy(); s = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    return (w * s[:, None, None, None]).astype(np.float32), (bn.bias.detach().numpy() - bn.running_mean.numpy() * s).astype(np.float32)

def export(ckpt, out, threshold, background, live_s, extra=None):
    net = load_bns(ckpt); arr, conv = {}, []
    w, b = fuse(net.stem[0], net.stem[1]); arr['w0'], arr['b0'] = w, b; conv.append((net.stem[0].stride, net.stem[0].padding)); i = 1
    for blk in net.blocks:
        for c, bn in ((blk.c1, blk.b1), (blk.c2, blk.b2), (blk.sk, blk.bs)):
            w, b = fuse(c, bn); arr[f'w{i}'], arr[f'b{i}'] = w, b; conv.append((c.stride, c.padding)); i += 1
    arr['lw0'] = net.head[0].weight.detach().numpy(); arr['lb0'] = net.head[0].bias.detach().numpy()
    arr['lw1'] = net.head[2].weight.detach().numpy(); arr['lb1'] = net.head[2].bias.detach().numpy()
    arr['band_matrix'] = net.M.numpy()
    meta = {'arch': 'bns_tf', 'conv': [[list(s), list(p)] for s, p in conv], 'nfft': NFFT, 'hop': HOP, 'frames': NFR, 'pnorm': float(net.pnorm),
            'threshold': float(threshold), 'background_livetime_s': float(live_s), 'fs': 1024, 'window': 16384, 'band_hz': [25, 500],
            'outputs': ['score', 'log_chirp_mass', 'log_sigma_log_chirp_mass', 'mass_ratio', 'log_sigma_q', 't_merger_s', 'log_sigma_t']}
    meta.update(extra or {})
    np.savez_compressed(out, meta=json.dumps(meta), background=np.sort(np.asarray(background, np.float32)), **arr)
    import chirpnet
    cn = chirpnet.ChirpNetBNS(out); x = np.random.default_rng(0).standard_normal((3, 2, 16384)).astype(np.float32)
    with torch.no_grad(): ref = net(torch.from_numpy(x)).numpy()
    got = cn.forward(x); print('max |numpy - torch| =', float(np.abs(ref - got).max()), 'params', sum(v.size for k, v in arr.items() if k != 'band_matrix'))
    return arr, meta

if __name__ == '__main__':
    name = sys.argv[1]; bg = np.load(f'{_P.WORK}/ckpt/{name}_bg.npy'); real = json.load(open(f'{_P.WORK}/ckpt/{name}_real.json'))
    thr = float(np.ceil((bg.max() + 0.5) * 2) / 2)
    export(f'{_P.WORK}/ckpt/{name}.pt', _P.WORK + '/release/chirpnet_bns_weights.npz', thr, bg, real['bg_live_s'], {'release': name})
    print('threshold', thr)
