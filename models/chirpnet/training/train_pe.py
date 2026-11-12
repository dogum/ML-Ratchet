import _paths as _P  # portable paths (see _paths.py)
import sys, time, json, math, argparse, copy, numpy as np, torch
sys.path.insert(0, _P.SRC)
from flow import ChirpPE
from pe_sampler import PESampler

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--trunk', default=None); ap.add_argument('--init', default=None)
ap.add_argument('--steps', type=int, default=4000); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--warm', type=int, default=200)
a = ap.parse_args()
torch.set_num_threads(2); torch.manual_seed(a.seed)
net = ChirpPE()
if a.init:
    net.load_state_dict(torch.load(a.init, weights_only=False)['ema'])
elif a.trunk:
    src = torch.load(a.trunk, weights_only=False)['ema']; own = net.cnn.state_dict()
    ok = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape and not k.startswith('head')}
    own.update(ok); net.cnn.load_state_dict(own); print('trunk tensors', len(ok), flush=True)
ema = copy.deepcopy(net).eval()
S = PESampler(np.load(_P.WORK + '/pbank_train.npy'), np.load(_P.WORK + '/ppar_train.npy'), np.load(_P.WORK + '/ppid_train.npy'),
              np.load(_P.WORK + '/sky_train.npy'), np.load(_P.WORK + '/asd_feats.npy'), seed=500 + a.seed)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
t0 = time.time(); acc = 0; n = 0; log = []
for step in range(1, a.steps + 1):
    lr = a.lr * (step / a.warm if step < a.warm else 0.5 * (1 + math.cos(math.pi * (step - a.warm) / max(1, a.steps - a.warm))) * 0.98 + 0.02)
    for g in opt.param_groups: g['lr'] = lr
    x, th, asd, snr = S.batch(a.bs)
    net.train(); lp = net.log_prob(torch.from_numpy(th), torch.from_numpy(x), torch.from_numpy(asd))
    loss = -lp.mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.999).add_(pn, alpha=0.001)
        for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
    acc += loss.item(); n += 1
    if step % 100 == 0:
        rec = {'step': step, 'nll': acc / n, 'lr': lr, 't': round(time.time() - t0)}; log.append(rec); print(json.dumps(rec), flush=True); acc = 0; n = 0
torch.save({'ema': ema.state_dict(), 'args': vars(a), 'log': log}, f'{_P.WORK}/ckpt/{a.name}.pt'); print('saved', flush=True)
