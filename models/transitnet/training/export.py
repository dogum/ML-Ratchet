"""Fuse BatchNorm into the convolutions and export TransitNet as plain arrays: an .npz for the numpy release and a base64
JSON pack for the browser. Checks the numpy forward against torch."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release'); import tnet as TN

def fuse(conv, bn):
    w = conv.weight.detach().numpy(); s = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    return (w * s[:, None, None]).astype(np.float32), (conv.bias.detach().numpy() * s + bn.bias.detach().numpy() - bn.running_mean.numpy() * s).astype(np.float32)

def export(ckpt, out_npz, out_js=None):
    net = TN.load_net(ckpt); A = {}; meta = {'branches': {}, 'mu_y': net.mu_y.tolist(), 'sd_y': net.sd_y.tolist(), 'residual': int(getattr(net, 'residual', 0))}
    for name in ('loc', 'sec', 'glo', 'oe') + (('cen',) if net.use_cent else ()):
        seq = list(getattr(net, name)); ops = []; i = 0; c = 0
        while i < len(seq):
            m = seq[i]
            if isinstance(m, torch.nn.Conv1d):
                w, b = fuse(m, seq[i + 1]); A[f'{name}{c}_w'], A[f'{name}{c}_b'] = w, b; ops.append(['conv', f'{name}{c}']); c += 1; i += 3   # conv, bn, gelu
            elif isinstance(m, torch.nn.MaxPool1d): ops.append(['pool']); i += 1
            else: i += 1
        meta['branches'][name] = ops
    lin = [m for m in net.head if isinstance(m, torch.nn.Linear)]
    for j, m in enumerate(lin): A[f'h{j}_w'], A[f'h{j}_b'] = m.weight.detach().numpy().astype(np.float32), m.bias.detach().numpy().astype(np.float32)
    meta['n_head'] = len(lin); np.savez_compressed(out_npz, meta=json.dumps(meta), **A)
    import transitnet as TR
    rn = TR.Net(out_npz); rng = np.random.default_rng(0); Z = {'local': rng.standard_normal((3, 2, 201)), 'secondary': rng.standard_normal((3, 2, 101)), 'glob': rng.standard_normal((3, 2, 401)), 'oddeven': rng.standard_normal((3, 4, 101)), 's': rng.standard_normal((3, 10)), 'cent': rng.standard_normal((3, 4, 101)), 'sc': rng.standard_normal(3)}
    with torch.no_grad(): ref = net(*[torch.from_numpy(Z[k].astype(np.float32)) for k in ('local', 'secondary', 'glob', 'oddeven', 's')], *([torch.from_numpy(Z['cent'].astype(np.float32)), torch.from_numpy(Z['sc'].astype(np.float32))] if net.use_cent else [])).numpy()
    got = rn.forward(Z['local'], Z['secondary'], Z['glob'], Z['oddeven'], Z['s'], Z['cent'], Z['sc']); print('max |numpy - torch| %.2e' % np.abs(got - ref).max(), 'params', sum(v.size for v in A.values()))
    if out_js:
        parts, man, off = [], [], 0
        for k in sorted(A):
            a = np.ascontiguousarray(A[k], np.float32); parts.append(a.ravel()); man.append({'k': k, 'shape': list(a.shape), 'off': off}); off += a.size
        json.dump({'man': {'tensors': man, 'n': off, 'meta': meta}, 'b64': base64.b64encode(np.concatenate(parts).astype('<f4').tobytes()).decode()}, open(out_js, 'w'))
        print('browser weights MB %.2f' % (off * 4 * 4 / 3 / 1e6))

if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
