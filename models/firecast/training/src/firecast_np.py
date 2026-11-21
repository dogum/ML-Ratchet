"""FireCast numpy inference (no torch). Weights: firecast_weights.npz (BatchNorm folded into the convs).
    net = FireCastNP('firecast_weights.npz'); logits = net.forward(X, s)   # X (22, H, W), s (22,)
    prob = 1 / (1 + exp(-(logits / T + b)))   # T, b: dev calibration stored in the npz (cal_T, cal_b)
"""
import numpy as np

LAYERS = ['stem', 'e1a', 'e1b', 'e2a', 'e2b', 'e3a', 'e3b', 'e4a', 'e4b', 'd3', 'd2', 'd1', 'd0']


def conv3x3(x, W, b, stride=1):
    C, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    Ho, Wo = (H + 2 - 3) // stride + 1, (Wd + 2 - 3) // stride + 1
    out = np.zeros((W.shape[0], Ho * Wo), np.float32)
    for dy in range(3):
        for dx in range(3):
            xs = xp[:, dy:dy + stride * (Ho - 1) + 1:stride, dx:dx + stride * (Wo - 1) + 1:stride]
            out += W[:, :, dy, dx] @ xs.reshape(C, -1)
    out += b[:, None]
    return out.reshape(W.shape[0], Ho, Wo)


def relu(x):
    return np.maximum(x, 0, out=x)


def pool2(x):
    C, H, W = x.shape
    return x[:, :H // 2 * 2, :W // 2 * 2].reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))


def up2(x):
    return x.repeat(2, axis=1).repeat(2, axis=2)


class FireCastNP:
    def __init__(self, path):
        z = np.load(path)
        self.p = {k: z[k].astype(np.float32) for k in z.files}
        self.cal_T = float(self.p.get('cal_T', np.array(1.0)))
        self.cal_b = float(self.p.get('cal_b', np.array(0.0)))
        self.thr = float(self.p.get('thr', np.array(0.5)))

    def c(self, name, x, stride=1):
        return relu(conv3x3(x, self.p[name + '.w'], self.p[name + '.b'], stride))

    def forward(self, X, s):
        p = self.p
        X = np.asarray(X, np.float32)
        H, W = X.shape[1:]
        Hp, Wp = int(np.ceil(H / 16) * 16), int(np.ceil(W / 16) * 16)
        Xp = np.zeros((X.shape[0], Hp, Wp), np.float32)
        Xp[:, :H, :W] = X
        h = np.maximum(p['mlp0.w'] @ s + p['mlp0.b'], 0)
        h = np.maximum(p['mlp1.w'] @ h + p['mlp1.b'], 0)
        fb = [(p[f'film{i}.w'] @ h + p[f'film{i}.b'])[:, None, None] for i in range(9)]
        x0 = self.c('stem', Xp) + fb[0]
        x1 = self.c('e1b', self.c('e1a', x0, 2)) + fb[1]
        x2 = self.c('e2b', self.c('e2a', pool2(x1))) + fb[2]
        x3 = self.c('e3b', self.c('e3a', pool2(x2))) + fb[3]
        x4 = self.c('e4b', self.c('e4a', pool2(x3))) + fb[4]
        y = self.c('d3', np.concatenate([up2(x4), x3])) + fb[5]
        y = self.c('d2', np.concatenate([up2(y), x2])) + fb[6]
        y = self.c('d1', np.concatenate([up2(y), x1])) + fb[7]
        y = self.c('d0', np.concatenate([up2(y), x0])) + fb[8]
        o = np.tensordot(p['head.w'], y, axes=(0, 0)) + p['head.b'] + float(p['head_s.w'] @ h + p['head_s.b'][0])
        return o[:H, :W]

    def prob(self, X, s):
        return 1.0 / (1.0 + np.exp(-(self.forward(X, s) / self.cal_T + self.cal_b)))
