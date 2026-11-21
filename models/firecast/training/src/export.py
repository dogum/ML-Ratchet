"""Export a trained rung to numpy (BatchNorm folded), verify numpy == torch, write weights npz (+ calibration).
Usage: python3 export.py <rung> <out.npz> [cal_json]"""
from common import *  # noqa
import torch
from evaluate import load_net
from firecast_np import FireCastNP

torch.set_num_threads(1)


def fold(seq):
    conv, bn = seq[0], seq[1]
    w = conv.weight.detach().numpy()
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    b = bn.bias.detach().numpy() - bn.running_mean.numpy() * g
    return (w * g[:, None, None, None]).astype(np.float32), b.astype(np.float32)


def export(rung, fn, cal=None):
    net = load_net(rung)
    P = {}
    for name, mod in [('stem', net.stem), ('e1a', net.e1[0]), ('e1b', net.e1[1]), ('e2a', net.e2[1]), ('e2b', net.e2[2]),
                      ('e3a', net.e3[1]), ('e3b', net.e3[2]), ('e4a', net.e4[1]), ('e4b', net.e4[2]), ('d3', net.d3),
                      ('d2', net.d2), ('d1', net.d1), ('d0', net.d0)]:
        P[name + '.w'], P[name + '.b'] = fold(mod)
    P['head.w'] = net.head.weight.detach().numpy()[0, :, 0, 0]
    P['head.b'] = net.head.bias.detach().numpy()
    P['mlp0.w'], P['mlp0.b'] = net.mlp[0].weight.detach().numpy(), net.mlp[0].bias.detach().numpy()
    P['mlp1.w'], P['mlp1.b'] = net.mlp[2].weight.detach().numpy(), net.mlp[2].bias.detach().numpy()
    for i, f in enumerate(net.film):
        P[f'film{i}.w'], P[f'film{i}.b'] = f.weight.detach().numpy(), f.bias.detach().numpy()
    P['head_s.w'] = net.head_s.weight.detach().numpy()[0]
    P['head_s.b'] = net.head_s.bias.detach().numpy()
    if cal:
        P['cal_T'] = np.array(cal['T'], np.float32)
        P['cal_b'] = np.array(cal['b'], np.float32)
        P['thr'] = np.array(cal.get('thr', 0.5), np.float32)
    np.savez(fn, **{k: np.asarray(v, np.float32) for k, v in P.items()})
    # verify on random + real-like inputs
    npn = FireCastNP(fn)
    rng = np.random.default_rng(0)
    errs = []
    for H, W in ((128, 128), (101, 137), (240, 200)):
        X = rng.random((22, H, W)).astype(np.float32)
        s = rng.standard_normal(22).astype(np.float32)
        with torch.no_grad():
            Hp, Wp = int(np.ceil(H / 16) * 16), int(np.ceil(W / 16) * 16)
            Xp = np.zeros((22, Hp, Wp), np.float32)
            Xp[:, :H, :W] = X
            t = net(torch.from_numpy(Xp)[None], torch.from_numpy(s)[None])[0].numpy()[:H, :W]
        n = npn.forward(X, s)
        errs.append(float(np.abs(t - n).max()))
    print('max |numpy - torch| logits:', errs)
    return errs


if __name__ == '__main__':
    cal = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    export(sys.argv[1], sys.argv[2], cal)
