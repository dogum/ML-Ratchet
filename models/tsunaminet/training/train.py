"""Train TsunamiNet on simulated banks only. Resumable (checkpoint every 500 steps).
usage: python train.py <rung> <bank1[,bank2,...]> <steps> [init_ckpt] [--width small|base]"""
import os, sys, json, time, numpy as np, torch
from _paths import BANK, LOGS
torch.set_num_threads(1)
from model import TsunamiNet
rung, banks, steps = sys.argv[1], sys.argv[2].split(','), int(sys.argv[3])
init = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None
ANCHOR = '--anchor' in sys.argv; LR = 3e-4 if '--polish' in sys.argv else 1e-3
CK = os.path.join(BANK, 'ckpt'); os.makedirs(CK, exist_ok=True)
ck = os.path.join(CK, f'{rung}.pt')
X, S, M, TH = [], [], [], []
for b in banks:
    name, _, n = b.partition(':'); n = int(n) if n else None     # "train_p0:20000" = the first 20,000 examples
    z = np.load(os.path.join(BANK, f'bank_{name}.npz'))
    X.append(z['X'][:n].copy()); S.append(z['S'][:n]); M.append(z['M'][:n]); TH.append(z['TH'][:n]); del z
X = np.concatenate(X); S = np.concatenate(S); M = np.concatenate(M); TH = np.concatenate(TH)
N = len(X); print('bank', N, flush=True)
EX = True if ANCHOR else None   # NNLS anchor features are computed per batch, on the stations kept after dropout

torch.manual_seed(0); rng = np.random.default_rng(0)
net = TsunamiNet(extra=23 if ANCHOR else 0); print('params', sum(p.numel() for p in net.parameters()), flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
step = 0
if os.path.exists(ck):
    st = torch.load(ck); net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); step = st['step']
    rng = np.random.default_rng(step); print('resumed at', step, flush=True)
elif init:
    net.load_state_dict(torch.load(os.path.join(CK, init + '.pt'))['net'], strict=False); print('init from', init, flush=True)
BS = 256; t0 = time.time(); run = 0.0
while step < steps:
    idx = rng.integers(0, N, BS)
    x = torch.from_numpy(X[idx].astype(np.float32)); s = torch.from_numpy(S[idx]); m = torch.from_numpy(M[idx].copy()); th = torch.from_numpy(TH[idx])
    # station dropout: keep a random non-empty subset, so 1-3 stations all train
    keep = torch.rand(m.shape) > 0.25; keep[:, 0] |= ~(keep & m).any(1); m = m & keep
    x = x * m[:, :, None, None]; s = s * m[:, :, None]
    e = None
    if EX is not None:
        import features as F
        e = torch.from_numpy(np.stack([F.nnls_features(xi, mi) for xi, mi in zip(x.numpy(), m.numpy())]))
    loss = -net.log_prob(th, x, s, m, e).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step(); step += 1
    run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 100 == 0: print(step, 'loss %.3f' % run, '%.0fs' % (time.time() - t0), flush=True)
    if step % 500 == 0 or step == steps:
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step), ck + '.tmp'); os.replace(ck + '.tmp', ck)
print('done', rung, flush=True)
