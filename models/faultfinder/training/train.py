"""Train one rung. Usage: python3 train.py configs/<rung>.json
Config keys: name, banks (list of bank names in data/banks), noise ('v1' Gaussian | 'v2' real-noise model),
eff_min, epochs, w, lr, bs, seed, n_max. Resumable (checkpoint every epoch)."""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
from common import ROOT, CI, NC  # noqa: E402
from features import featurize  # noqa: E402
from model import FFNet  # noqa: E402
import evaluate as ev  # noqa: E402

NO_PAR = [CI['none'], CI['economizer'], CI['schedule'], CI['plant_efficiency']]


def load_bank(names, eff_min, n_max, seed):
    parts = [dict(np.load(f'{ROOT}/data/banks/{n}.npz', allow_pickle=True)) for n in names]
    b = {k: np.concatenate([p[k] for p in parts]) for k in ('x', 'cls', 'par', 'waste', 'eff', 'system')}
    keep = (b['cls'] == 0) | (b['eff'] >= eff_min)
    idx = np.where(keep)[0]
    rng = np.random.default_rng(seed)
    if len(idx) > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    return {k: v[idx] for k, v in b.items()}


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    name = cfg['name']
    os.makedirs(f'{ROOT}/models', exist_ok=True)
    ck = f'{ROOT}/models/{name}.ckpt'
    torch.manual_seed(cfg.get('seed', 0))
    b = load_bank(cfg['banks'], cfg.get('eff_min', 1.0), cfg.get('n_max', 60000), cfg.get('seed', 0))
    n = len(b['cls'])
    print(name, 'samples', n, 'faulty share %.2f' % (b['cls'] > 0).mean(), flush=True)
    nz = ev.noise_for(cfg.get('noise', 'v2'))
    model = FFNet(w=cfg.get('w', 48))
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
    for ep in range(e0, E):
        t0 = time.time()
        rng = np.random.default_rng([cfg.get('seed', 0), ep])
        F = np.empty((n,) + featurize(b['x'][0].astype(np.float32)).shape, np.float16)   # float16 storage (RAM)
        for i in range(n):
            F[i] = featurize(nz.apply(b['x'][i].astype(np.float32), rng, str(b['system'][i])))
        tf = time.time() - t0
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            xb = torch.as_tensor(F[j].astype(np.float32))
            y = torch.as_tensor(b['cls'][j].astype(np.int64))
            lo, pa, wa = model(xb)
            loss = Fnn.cross_entropy(lo, y, weight=cw, label_smoothing=cfg.get('smooth', 0.02))
            sel = has_par[y]
            if sel.any():
                pt = torch.as_tensor(b['par'][j])
                loss = loss + cfg.get('w_par', 0.5) * Fnn.smooth_l1_loss(pa[sel, y[sel]], pt[sel], beta=0.1)
            wt = torch.as_tensor(np.clip(b['waste'][j], -1, 3))
            loss = loss + cfg.get('w_waste', 0.5) * Fnn.smooth_l1_loss(wa, wt, beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(j)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep), ck)
        print(f'epoch {ep} loss {tot / n:.4f} feat {tf:.0f}s total {time.time() - t0:.0f}s', flush=True)
    torch.save(model.state_dict(), f'{ROOT}/models/{name}.pt')
    val = ev.load_val(cfg.get('val', cfg.get('noise', 'v2')))
    res = dict(cfg=cfg, val_own=ev.skill(model, val), **ev.dev_eval(model))
    json.dump(res, open(f'{ROOT}/models/{name}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'dev_p'}, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
