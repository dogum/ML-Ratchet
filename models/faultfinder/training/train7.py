"""Train r7_context (or its single-day control) on bank v5 with noise v5. Usage: python train7.py configs/<rung>.json
Config keys as train.py, plus bank (one v5 bank name) and ctx (true: 82-row input with references; false: today's 35
rows only). Each epoch redraws every window's noise state, every day's noise and every sample's references (30 %
none, else 1-3 earlier days of the same run within 7 days). Resumable (checkpoint every epoch)."""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
from common import ROOT, NC  # noqa: E402
from context import context_batch, NF1, NF7, MAX_REF  # noqa: E402
from model import FFNet  # noqa: E402
from train import NO_PAR  # noqa: E402
import r7  # noqa: E402


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    name = cfg['name']
    ctx = bool(cfg.get('ctx', True))
    os.makedirs(f'{ROOT}/models', exist_ok=True)
    ck = f'{ROOT}/models/{name}.ckpt'
    seed = cfg.get('seed', 0)
    torch.manual_seed(seed)
    b = r7.load_bank(cfg['bank'])
    ix = r7.Index(b)
    keep = b['sample'] & ((b['cls'] == 0) | (b['eff'] >= cfg.get('eff_min', 1.0)))
    S = np.where(keep)[0]
    if len(S) > cfg.get('n_max', 60000):
        S = np.sort(np.random.default_rng(seed).choice(S, cfg['n_max'], replace=False))
    n = len(S)
    print(name, 'samples', n, 'faulty share %.2f' % (b['cls'][S] > 0).mean(), 'bank rows', len(b['cls']),
          'ctx', ctx, flush=True)
    nz = r7.noise5()
    model = FFNet(nf=NF7 if ctx else NF1, w=cfg.get('w', 48))
    print('params', sum(p.numel() for p in model.parameters()), flush=True)
    E = cfg.get('epochs', 12)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 2e-3), weight_decay=1e-4)
    bs = cfg.get('bs', 256)
    steps = E * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get('lr', 2e-3), total_steps=steps, pct_start=0.15)
    e0 = 0
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        e0 = st['epoch'] + 1
        print('resumed at epoch', e0, flush=True)
    cw = torch.ones(NC)
    cw[0] = cfg.get('w_none', 1.0)
    has_par = torch.ones(NC, dtype=torch.bool)
    has_par[NO_PAR] = False
    cls_all = b['cls'].astype(np.int64)
    rows_all = np.arange(len(cls_all)) if ctx else S
    for ep in range(e0, E):
        t0 = time.time()
        rng = np.random.default_rng([seed, ep])
        Fd, _ = r7.featurize_rows(b, rows_all, nz, rng)
        pos = {r: k for k, r in enumerate(Fd)}
        F = np.stack(list(Fd.values()))
        del Fd
        si = np.array([pos[int(i)] for i in S])
        R = -np.ones((n, MAX_REF), np.int64)
        if ctx:
            for k, i in enumerate(S):
                rr = ix.refs(int(i), rng)
                R[k, :len(rr)] = [pos[r] for r in rr]
        nref = (R >= 0).sum(1)
        tf = time.time() - t0
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i0 in range(0, n, bs):
            j = perm[i0:i0 + bs]
            ft = F[si[j]].astype(np.float32)
            if ctx:
                fr = F[np.maximum(R[j], 0)].astype(np.float32)
                xb = context_batch(ft, fr, nref[j])
            else:
                xb = ft
            xb = torch.as_tensor(xb)
            y = torch.as_tensor(cls_all[S[j]])
            lo, pa, wa = model(xb)
            loss = Fnn.cross_entropy(lo, y, weight=cw, label_smoothing=cfg.get('smooth', 0.02))
            sel = has_par[y]
            if sel.any():
                pt = torch.as_tensor(b['par'][S[j]])
                loss = loss + cfg.get('w_par', 0.5) * Fnn.smooth_l1_loss(pa[sel, y[sel]], pt[sel], beta=0.1)
            wt = torch.as_tensor(np.clip(b['waste'][S[j]], -1, 3))
            loss = loss + cfg.get('w_waste', 0.5) * Fnn.smooth_l1_loss(wa, wt, beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(j)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep), ck)
        print(f'epoch {ep} loss {tot / n:.4f} with-ref share {(nref > 0).mean():.2f} feat {tf:.0f}s '
              f'total {time.time() - t0:.0f}s', flush=True)
    torch.save(model.state_dict(), f'{ROOT}/models/{name}.pt')
    print('saved', f'{ROOT}/models/{name}.pt', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
