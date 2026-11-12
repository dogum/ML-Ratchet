"""Fuse BatchNorm and export SunDoctor to portable numpy weights; verify the numpy forward against torch."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
import evaluate as E

def fuse(conv, bn):
    w = conv.weight.detach().numpy(); s = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    shape = (-1,) + (1,) * (w.ndim - 1)
    return (w * s.reshape(shape)).astype(np.float32), (bn.bias.detach().numpy() - bn.running_mean.numpy() * s).astype(np.float32)

def export(ckpt, out, extra=None):
    net = E.load(ckpt); A = {}; meta = {'conv2d': [], 'tconv': []}
    w, b = fuse(net.stem[0], net.stem[1]); A['s_w'], A['s_b'] = w, b; meta['stem'] = [list(net.stem[0].stride), list(net.stem[0].padding)]
    for i, blk in enumerate(net.blocks):
        for nm, (c, bn) in zip(('c1', 'c2', 'sk'), ((blk.c1, blk.b1), (blk.c2, blk.b2), (blk.sk, blk.bs))):
            w, b = fuse(c, bn); A[f'b{i}{nm}_w'], A[f'b{i}{nm}_b'] = w, b
        meta['conv2d'].append([[list(c.stride), list(c.padding)] for c in (blk.c1, blk.c2, blk.sk)])
    w, b = fuse(net.proj[0], net.proj[1]); A['p_w'], A['p_b'] = w, b
    for i, tb in enumerate(net.tblocks):
        w, b = fuse(tb.c1, tb.b1); A[f't{i}c1_w'], A[f't{i}c1_b'] = w, b; w, b = fuse(tb.c2, tb.b2); A[f't{i}c2_w'], A[f't{i}c2_b'] = w, b
        meta['tconv'].append(int(tb.c1.dilation[0]))
    A['d_w'] = net.day.weight.detach().numpy(); A['d_b'] = net.day.bias.detach().numpy()
    A['g0_w'] = net.glob[0].weight.detach().numpy(); A['g0_b'] = net.glob[0].bias.detach().numpy(); A['g1_w'] = net.glob[2].weight.detach().numpy(); A['g1_b'] = net.glob[2].bias.detach().numpy()
    from model import REG, SCALE, GLOB, GSCALE
    meta.update({'reg': REG, 'scale': SCALE, 'glob': GLOB, 'gscale': GSCALE, 'days': int(net.days)}); meta.update(extra or {})
    np.savez_compressed(out, meta=json.dumps(meta), **{k: v.astype(np.float32) for k, v in A.items()})
    import sundoctor
    sd = sundoctor.SunDoctorNet(out); X = np.random.default_rng(0).standard_normal((2, 10, net.days, 96)).astype(np.float32)
    with torch.no_grad(): d, g = net(torch.from_numpy(X))
    dn, gn = sd.forward(X)
    print('max |numpy - torch| day %.2e glob %.2e' % (np.abs(dn - d.numpy()).max(), np.abs(gn - g.numpy()).max()), 'params', sum(v.size for v in A.values()))

if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2])
