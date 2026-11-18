"""r8: r6 (released numpy weights) + the 47 reference input rows, zero-initialised; only those new first-layer weights
train (NOTEBOOK 2026-11-17 r8 plan). Usage: python train8.py <seed>
Writes WORK/models/r8_s<seed>.npz (faultfinder.forward format, c1_W [64, 82, 5]) and a step-0 / end check."""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
from common import ROOT, NC  # noqa: E402
from context import context_batch, NF1, NF7, MAX_REF  # noqa: E402
from train import NO_PAR  # noqa: E402
import r7  # noqa: E402

EPOCHS, LR, BS = 3, 1e-3, 256


class Folded(nn.Module):
    """faultfinder.forward in torch: BatchNorm-folded convolutions, frozen; c1's new input rows trainable."""
    def __init__(self, W, nf=NF7):
        super().__init__()
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32))
        self.W = {k: t(v) for k, v in W.items() if k != 'temperature'}
        for v in self.W.values():
            v.requires_grad_(False)
        self.c1_new = nn.Parameter(torch.zeros(W['c1_W'].shape[0], nf - W['c1_W'].shape[1], W['c1_W'].shape[2]))
        self.drop = nn.Dropout(0.2)

    def c1_full(self):
        return torch.cat([self.W['c1_W'], self.c1_new], 1)

    def forward(self, x):
        W = self.W
        conv = lambda h, w, b: Fnn.conv1d(h, w, b, padding=w.shape[2] // 2)
        h = torch.relu(conv(x, self.c1_full(), W['c1_b']))
        h = Fnn.max_pool1d(torch.relu(conv(h, W['c2_W'], W['c2_b'])), 2)
        h = Fnn.max_pool1d(torch.relu(conv(h, W['c3_W'], W['c3_b'])), 2)
        h = torch.relu(conv(h, W['c4_W'], W['c4_b']))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        h = self.drop(torch.relu(h @ W['fc_W'].T + W['fc_b']))
        return h @ W['cls_W'].T + W['cls_b'], h @ W['par_W'].T + W['par_b'], (h @ W['waste_W'].T + W['waste_b'])[:, 0]

    def export(self, W0, out):
        d = {k: np.asarray(v) for k, v in W0.items()}
        d['c1_W'] = self.c1_full().detach().numpy().astype(np.float32)
        np.savez(out, **d)


def check(model, W, F, Fn=None):
    """max |torch(F) - faultfinder.forward(W, Fn or F)|, all three heads."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import faultfinder as ffr
    model.eval()
    with torch.no_grad():
        a = [v.numpy() for v in model(torch.as_tensor(F))]
    b = ffr.forward(W, F if Fn is None else Fn)
    model.train()
    return float(max(np.abs(x - y).max() for x, y in zip(a, b)))


def main(seed):
    name = f'r8_s{seed}'
    W0 = dict(np.load(r7.WEIGHTS))
    torch.manual_seed(seed)
    model = Folded(W0)
    val = r7.build_frozen()
    Fv = val['F'][:2000].astype(np.float32)
    rep = dict(name=name, seed=seed, epochs=EPOCHS, lr=LR, bs=BS, trainable=int(model.c1_new.numel()))
    rep['step0_torch82_vs_r6_numpy35'] = check(model, W0, Fv, Fv[:, :NF1])
    W82 = dict(W0, c1_W=model.c1_full().detach().numpy())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import faultfinder as ffr
    a = ffr.forward(W82, Fv)[0]
    b = ffr.forward(W0, Fv[:, :NF1])[0]
    rep['step0_82rows_vs_r6_35rows'] = float(np.abs(a - b).max())
    print(name, rep, flush=True)
    b_ = r7.load_bank('train_v5')
    ix = r7.Index(b_)
    keep = b_['sample'] & ((b_['cls'] == 0) | (b_['eff'] >= 1.0))
    S = np.where(keep)[0]
    n = len(S)
    nz = r7.noise5()
    opt = torch.optim.AdamW([model.c1_new], lr=LR, weight_decay=0.0)
    steps = EPOCHS * ((n + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    cw = torch.ones(NC)
    has_par = torch.ones(NC, dtype=torch.bool)
    has_par[NO_PAR] = False
    cls_all = b_['cls'].astype(np.int64)
    rows_all = np.arange(len(cls_all))
    model.train()
    for ep in range(EPOCHS):
        t0 = time.time()
        rng = np.random.default_rng([seed, 800 + ep])
        Fd, _ = r7.featurize_rows(b_, rows_all, nz, rng)
        pos = {r: k for k, r in enumerate(Fd)}
        F = np.stack(list(Fd.values()))
        del Fd
        si = np.array([pos[int(i)] for i in S])
        R = -np.ones((n, MAX_REF), np.int64)
        for k, i in enumerate(S):
            rr = ix.refs(int(i), rng)
            R[k, :len(rr)] = [pos[r] for r in rr]
        nref = (R >= 0).sum(1)
        tf = time.time() - t0
        perm = rng.permutation(n)
        tot = 0.0
        for i0 in range(0, n, BS):
            j = perm[i0:i0 + BS]
            xb = torch.as_tensor(context_batch(F[si[j]].astype(np.float32), F[np.maximum(R[j], 0)].astype(np.float32),
                                               nref[j]))
            y = torch.as_tensor(cls_all[S[j]])
            lo, pa, wa = model(xb)
            loss = Fnn.cross_entropy(lo, y, label_smoothing=0.02)
            sel = has_par[y]
            if sel.any():
                pt = torch.as_tensor(b_['par'][S[j]])
                loss = loss + 0.5 * Fnn.smooth_l1_loss(pa[sel, y[sel]], pt[sel], beta=0.1)
            wt = torch.as_tensor(np.clip(b_['waste'][S[j]], -1, 3))
            loss = loss + 0.5 * Fnn.smooth_l1_loss(wa, wt, beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(j)
        print(f'epoch {ep} loss {tot / n:.4f} with-ref share {(nref > 0).mean():.2f} feat {tf:.0f}s '
              f'total {time.time() - t0:.0f}s |new W| mean {model.c1_new.detach().abs().mean():.4f}', flush=True)
    out = f'{ROOT}/models/{name}.npz'
    model.export(W0, out)
    Wn = dict(np.load(out))
    rep['end_torch_vs_numpy'] = check(model, Wn, Fv)
    Fs = Fv.copy()
    Fs[:, NF1:] = 0
    rep['end_single_day_vs_r6'] = float(np.abs(ffr.forward(Wn, Fs)[0] - ffr.forward(W0, Fv[:, :NF1])[0]).max())
    rep['old_weights_unchanged'] = bool(all(np.array_equal(Wn[k], W0[k]) for k in W0 if k != 'c1_W') and
                                        np.array_equal(Wn['c1_W'][:, :NF1], W0['c1_W']))
    json.dump(rep, open(f'{ROOT}/models/{name}.check.json', 'w'), indent=1)
    print('saved', out, rep, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
