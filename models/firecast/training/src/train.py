"""Train one FireCast rung (resumable). Config: configs/<rung>.json
  {"bank": "p0", "steps": 1500, "bs": 12, "crop": 128, "lr": 2e-3, "land_p": 0.5, "ch": [16,32,48,64,64],
   "init": null, "seed": 0, "max_fires": null}
Usage: python3 train.py <rung>
"""
import time
from common import *  # noqa
import torch
import torch.nn.functional as F
import features as FT
import wxfeat as WX
from dataset import Bank
from model import FireCastNet, n_params

torch.set_num_threads(1)


def build(cfg):
    return FireCastNet(FT.N_SPATIAL, WX.N_WX + FT.N_FIRE, tuple(cfg.get('ch', (16, 32, 48, 64, 64))))


def main(rung):
    cfg = json.load(open(f'{CONFIGS}/{rung}.json'))
    ck = f'{ROOT}/ckpt/{rung}.pt'
    log = open(f'{ROOT}/logs/train_{rung}.log', 'a')
    bank = Bank(cfg['bank'], cfg.get('bank_split', 'train'), cfg.get('max_fires'))
    print(f'bank {cfg["bank"]}: {len(bank.blobs)} fires, {len(bank)} fire-days, mem {mem_mb():.0f} MB', file=log, flush=True)
    torch.manual_seed(cfg.get('seed', 0))
    net = build(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg.get('wd', 0.0))
    steps = cfg['steps']
    step = 0
    rng = np.random.default_rng(cfg.get('seed', 0))
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        net.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        step = st['step']
        rng.bit_generator.state = st['rng']
        print(f'resumed at {step}', file=log, flush=True)
    elif cfg.get('init'):
        st = torch.load(f'{ROOT}/ckpt/{cfg["init"]}.pt', weights_only=False)
        net.load_state_dict(st['model'])
        print(f'init from {cfg["init"]}', file=log, flush=True)
    print(f'params {n_params(net)}', file=log, flush=True)
    sched = lambda s: cfg['lr'] * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(s / steps, 1.0))))
    t0 = time.time()
    run = []
    pos_w = cfg.get('pos_weight', 1.0)
    while step < steps:
        for g in opt.param_groups:
            g['lr'] = sched(step)
        idx = rng.integers(len(bank), size=cfg['bs'])
        items = [bank.train_item(int(i), rng, cfg['crop'], cfg.get('land_p', 0.5)) for i in idx]
        X = torch.from_numpy(np.stack([a[0] for a in items]))
        S = torch.from_numpy(np.stack([a[1] for a in items]))
        Y = torch.from_numpy(np.stack([a[2] for a in items]))
        M = torch.from_numpy(np.stack([a[3] for a in items]))
        net.train()
        out = net(X, S)
        w = M * (1 + (pos_w - 1) * Y)
        loss = (F.binary_cross_entropy_with_logits(out, Y, reduction='none') * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        step += 1
        run.append(float(loss.detach()))
        if step % 50 == 0:
            print(f'step {step} loss {np.mean(run[-50:]):.4f} lr {sched(step):.2e} {time.time() - t0:.0f}s mem {mem_mb():.0f}',
                  file=log, flush=True)
        if step % 200 == 0 or step == steps:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), step=step, rng=rng.bit_generator.state,
                            cfg=cfg), ck + '.tmp')
            os.replace(ck + '.tmp', ck)
    print('DONE', file=log, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
