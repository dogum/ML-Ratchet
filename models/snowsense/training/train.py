"""Train SnowSense on a simulated bank (resumable).

python3 train.py <run> <bank1[,bank2..]> [epochs] [init_ckpt]
Writes ckpt/<run>.pt (final), ckpt/<run>.resume (every 500 steps), logs to stdout.
Loss: Gaussian NLL in z = sqrt(SWE)/10 on the water-year days of each window (warm-up days excluded).
"""
import sys
import time
import numpy as np
import torch
from common import *
from model import SnowNet, nll, pinball, to_swe

torch.set_num_threads(1)
run, banks = sys.argv[1], sys.argv[2].split(',')
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
INIT = sys.argv[4] if len(sys.argv) > 4 else None
HID = int(os.environ.get('SS_HID', '64'))
BS = int(os.environ.get("SS_BS", "128"))
LR = float(os.environ.get('SS_LR', '2e-3'))

def _bank(b):
    """Memory-mapped bank (banks/<b>_X.npy) when present, else the npz loaded into RAM."""
    if os.path.exists(f'{ROOT}/banks/{b}_X.npy'):
        return np.load(f'{ROOT}/banks/{b}_X.npy', mmap_mode='r'), np.load(f'{ROOT}/banks/{b}_Y.npy', mmap_mode='r')
    d = np.load(f'{ROOT}/banks/{b}.npz')
    return d['X'], d['Y']


parts = [_bank(b) for b in banks]
offs = np.cumsum([0] + [len(p[1]) for p in parts])


class Cat:
    """Row-gather across several (possibly memory-mapped) arrays."""
    def __init__(self, arrs):
        self.a = arrs
    def __len__(self):
        return int(offs[-1])
    def __getitem__(self, idx):
        idx = np.sort(np.asarray(idx))
        out = []
        for j, a in enumerate(self.a):
            m = (idx >= offs[j]) & (idx < offs[j + 1])
            if m.any():
                out.append(np.asarray(a[idx[m] - offs[j]]))
        return np.concatenate(out)
    @property
    def shape(self):
        return (len(self),) + self.a[0].shape[1:]


X = Cat([p[0] for p in parts])
Y = Cat([p[1] for p in parts])
n = len(X)
print(run, 'bank', X.shape, flush=True)
dv = dict(np.load(f'{ROOT}/banks/val_quick.npz')) if os.path.exists(f'{ROOT}/banks/val_quick.npz') else None

w = np.zeros(L, np.float32)
w[WARM:WARM + 365] = 1.0
W = torch.tensor(w)

HEAD = os.environ.get('SS_HEAD', 'gauss')
net = SnowNet(hidden=HID, head=HEAD)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = n // BS
total = EPOCHS * steps_per_epoch
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.05)
step = 0
ck = f'{ROOT}/ckpt/{run}.resume'
if os.path.exists(ck):
    s = torch.load(ck, weights_only=False)
    net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); step = s['step']
    print('resumed at', step, flush=True)
elif INIT:
    sd = torch.load(f'{ROOT}/ckpt/{INIT}.pt', weights_only=False)['net']
    own = net.state_dict()
    sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    net.load_state_dict(sd, strict=False)
    print('init keys', len(sd), 'of', len(own), flush=True)
    print('init from', INIT, flush=True)
print('params', sum(p.numel() for p in net.parameters()), flush=True)


def quick_val():
    if dv is None:
        return float('nan')
    net.eval()
    with torch.no_grad():
        s = to_swe(net.quantiles(torch.tensor(dv['X'][:512].astype(np.float32)))[..., 2]).numpy()
    net.train()
    y = dv['Y'][:512].astype(np.float32)
    m = np.zeros_like(y, bool)
    m[:, WARM:WARM + 365] = True
    m &= np.isfinite(y)
    return float(1 - ((s - y) ** 2)[m].sum() / ((y - y[m].mean()) ** 2)[m].sum())


t0 = time.time()
rng = np.random.default_rng(SEED + step)
net.train()
lsum = 0.0
while step < total:
    idx = rng.integers(0, n, BS)
    xb = torch.tensor(X[idx].astype(np.float32))
    yb = torch.tensor(np.nan_to_num(Y[idx].astype(np.float32)))   # X[idx], Y[idx] use the same sorted order
    if HEAD == 'gauss':
        mu, sg = net(xb)
        loss = nll(mu, sg, yb, W.expand_as(yb))
    else:
        loss = pinball(net(xb), yb, W.expand_as(yb))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    sched.step()
    step += 1
    lsum += float(loss.detach())
    if step % 250 == 0:
        print(f'step {step}/{total} loss {lsum / 250:.4f} valNSE {quick_val():.4f} {time.time() - t0:.0f}s', flush=True)
        lsum = 0.0
    if step % 500 == 0 or step == total:
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step}, ck + '.tmp')
        os.replace(ck + '.tmp', ck)
torch.save({'net': net.state_dict(), 'hidden': HID, 'head': HEAD, 'banks': banks, 'steps': total}, f'{ROOT}/ckpt/{run}.pt')
print('done', run, time.time() - t0, flush=True)
