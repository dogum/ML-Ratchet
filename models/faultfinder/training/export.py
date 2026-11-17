"""Export a trained rung to numpy weights (BatchNorm folded into the convolutions) and check numpy == torch.
Usage: python3 export.py <model_name> <out.npz> [temperature]"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
from common import ROOT  # noqa: E402
from ratchet import load_model  # noqa: E402


def fold(conv, bn):
    W = conv.weight.detach().numpy().astype(np.float64)
    b = conv.bias.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    return (W * g[:, None, None]).astype(np.float32), ((b - bn.running_mean.numpy()) * g + bn.bias.detach().numpy()).astype(np.float32)


def export(name, out, T=1.0):
    m = load_model(name)
    d = {}
    for i, blk in enumerate([m.c1, m.c2, m.c3, m.c4], 1):
        d[f'c{i}_W'], d[f'c{i}_b'] = fold(blk[0], blk[1])
    d['fc_W'], d['fc_b'] = m.fc[0].weight.detach().numpy(), m.fc[0].bias.detach().numpy()
    for h in ('cls', 'par', 'waste'):
        lin = getattr(m, h)
        d[f'{h}_W'], d[f'{h}_b'] = lin.weight.detach().numpy(), lin.bias.detach().numpy()
    d['temperature'] = np.array(T, np.float32)
    np.savez(out, **d)
    # check
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # models/faultfinder (faultfinder.py)
    import faultfinder as ffr
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, m.c1[0].in_channels, 96)).astype(np.float32)
    with torch.no_grad():
        lo, pa, wa = m(torch.as_tensor(x))
    W = dict(np.load(out))
    lo2, pa2, wa2 = ffr.forward(W, x)
    err = max(np.abs(lo.numpy() - lo2).max(), np.abs(pa.numpy() - pa2).max(), np.abs(wa.numpy() - wa2).max())
    print('numpy vs torch max abs diff', err)
    return float(err)


if __name__ == '__main__':
    e = export(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
    json.dump(dict(model=sys.argv[1], max_abs_diff=e), open(sys.argv[2] + '.check.json', 'w'))
