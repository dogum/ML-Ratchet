"""Export ChirpNet-PE (trunk + context MLP + RealNVP flow) to numpy arrays and verify against torch."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
from flow import ChirpPE, MU, SD, PARAMS
from export import fuse

def export(ckpt, out):
    net = ChirpPE(); net.load_state_dict(torch.load(ckpt, weights_only=False)['ema']); net.eval()
    arr, conv = {}, []
    st = net.cnn.stem; w, b = fuse(st[0], st[1]); arr['w0'], arr['b0'] = w, b; conv.append((st[0].stride[0], st[0].padding[0]))
    i = 1
    for blk in net.cnn.blocks:
        for c, bn in ((blk.c1, blk.b1), (blk.c2, blk.b2), (blk.sk, blk.bs)):
            w, b = fuse(c, bn); arr[f'w{i}'], arr[f'b{i}'] = w, b; conv.append((c.stride[0], c.padding[0])); i += 1
    for j, k in enumerate((0, 2)):
        arr[f'cw{j}'] = net.ctx[k].weight.detach().numpy(); arr[f'cb{j}'] = net.ctx[k].bias.detach().numpy()
    for li, L in enumerate(net.flow.layers):
        arr[f'm{li}'] = L.mask.numpy()
        for j, k in enumerate((0, 2, 4)):
            arr[f'f{li}w{j}'] = L.net[k].weight.detach().numpy(); arr[f'f{li}b{j}'] = L.net[k].bias.detach().numpy()
    meta = {'conv': conv, 'n_layers': len(net.flow.layers), 'mu': MU.tolist(), 'sd': SD.tolist(), 'params': PARAMS,
            'asd_freqs': [25, 35, 50, 70, 100, 140, 200, 280, 400, 560]}
    np.savez_compressed(out, meta=json.dumps(meta), **{k: v.astype(np.float32) for k, v in arr.items()})
    # verify: same latent -> same sample
    import chirpnet
    pe = chirpnet.ChirpPE(out)
    x = np.random.default_rng(0).standard_normal((1, 2, 4096)).astype(np.float32); a = np.zeros((1, 20), np.float32)
    z = np.random.default_rng(1).standard_normal((5, 6)).astype(np.float32)
    with torch.no_grad():
        c = net.context(torch.from_numpy(x), torch.from_numpy(a)).expand(5, -1); zz = torch.from_numpy(z)
        for L in reversed(net.flow.layers): zz = L.inverse(zz, c)
        ref = (zz * SD + MU).numpy()
    got = pe.sample(x[0], a[0], z=z)
    print('max |numpy - torch| =', float(np.abs(ref - got).max()))

if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2])
