import _paths as _P  # portable paths (see _paths.py)
import sys, time, json, math, argparse, copy, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, _P.SRC)
from model import ChirpNet, ChirpNet2S
from sampler import Sampler

def nll(mu, ls, t):
    ls = ls.clamp(-7, 3)
    return 0.5 * ((t - mu) * torch.exp(-ls)) ** 2 + ls

def loss_fn(out, y, tgt, w_reg=1.0):
    bce = F.binary_cross_entropy_with_logits(out[:, 0], y)
    pos = y > 0
    if pos.any():
        o, t = out[pos], tgt[pos]
        reg = nll(o[:, 1], o[:, 2], t[:, 0]).mean() + 0.5 * nll(o[:, 3], o[:, 4], t[:, 1]).mean() + 0.5 * nll(o[:, 5], o[:, 6], t[:, 2]).mean()
    else:
        reg = out.sum() * 0
    return bce + w_reg * 0.1 * reg, bce.item(), reg.item()

def snr_range(sched, frac):
    if sched == 'curriculum':     # start loud, end at the detection threshold
        lo = 12 - 7 * min(1, frac / 0.6); hi = 35 - 13 * min(1, frac / 0.6)
        return lo, hi
    lo, hi = map(float, sched.split(','))
    return lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--init', default=None)
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--snr', default='6,30')
    ap.add_argument('--glitch', type=float, default=0.0); ap.add_argument('--widths', default='24,32,48,64,96,128')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--warm', type=int, default=200); ap.add_argument('--pos', type=int, default=0); ap.add_argument('--wreg', type=float, default=1.0); ap.add_argument('--trunk', default=None); ap.add_argument('--single', type=float, default=0.0); ap.add_argument('--single_snr', default='6,25'); ap.add_argument('--arch', default='cnn'); ap.add_argument('--hidden', type=int, default=128)
    a = ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(a.seed)
    widths = tuple(int(v) for v in a.widths.split(','))
    if a.init:
        ck = torch.load(a.init, weights_only=False); widths = tuple(ck['widths']); a.pos = ck.get('pos', 0); a.hidden = ck.get('hidden', 128); a.arch = ck.get('arch', 'cnn')
        net = ChirpNet2S(widths=widths) if a.arch == '2s' else ChirpNet(widths=widths, pos=bool(a.pos), hidden=a.hidden); net.load_state_dict(ck['ema'])
    elif a.arch == '2s':
        net = ChirpNet2S(widths=widths)
    else:
        net = ChirpNet(widths=widths, pos=bool(a.pos), hidden=a.hidden)
    if a.trunk:   # transplant matching layers (conv trunk) from an earlier rung
        src = dict(torch.load(a.trunk, weights_only=False)['ema']); own = net.state_dict()
        if a.arch == '2s' and src['stem.0.weight'].shape[1] == 2: src['stem.0.weight'] = src['stem.0.weight'].mean(1, keepdim=True)
        ok = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape and not k.startswith('head.0')}
        own.update(ok); net.load_state_dict(own); print('transplanted', len(ok), 'tensors', flush=True)
    if not a.init:
        with torch.no_grad():
            b = net.head[-1].bias; b.zero_(); b[1] = 2.8; b[3] = 0.6; b[5] = 1.25; b[2] = -1; b[4] = -1.5; b[6] = -2
    ema = copy.deepcopy(net).eval()
    bank = np.load(_P.WORK + '/bank_train.npy', mmap_mode='r'); par = np.load(_P.WORK + '/par_train.npy'); sky = np.load(_P.WORK + '/sky_train.npy')
    S = Sampler(np.asarray(bank), par, sky, seed=1000 + a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    t0 = time.time(); log = []
    acc = {'loss': 0, 'bce': 0, 'reg': 0, 'n': 0}
    for step in range(1, a.steps + 1):
        frac = step / a.steps
        lr = a.lr * (step / a.warm if step < a.warm else 0.5 * (1 + math.cos(math.pi * (step - a.warm) / max(1, a.steps - a.warm))) * 0.97 + 0.03)
        for g in opt.param_groups: g['lr'] = lr
        lo, hi = snr_range(a.snr, frac)
        x, y, tgt = S.batch(a.bs, snr_lo=lo, snr_hi=hi, p_glitch=a.glitch, p_single=a.single, single_snr=tuple(float(v) for v in a.single_snr.split(',')))
        x, y, tgt = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(tgt)
        net.train(); out = net(x)
        loss, bce, reg = loss_fn(out, y, tgt, w_reg=a.wreg)
        if a.arch == '2s':
            loss = loss + 0.3 * (F.binary_cross_entropy_with_logits(out[:, 7], (tgt[:, 4] > 4).float()) + F.binary_cross_entropy_with_logits(out[:, 8], (tgt[:, 5] > 4).float()))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(a.ema).add_(pn, alpha=1 - a.ema)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        acc['loss'] += loss.item(); acc['bce'] += bce; acc['reg'] += reg; acc['n'] += 1
        if step % 100 == 0 or step == a.steps:
            n = acc['n']; rec = {'step': step, 'loss': acc['loss'] / n, 'bce': acc['bce'] / n, 'reg': acc['reg'] / n, 'lr': lr, 'snr': [round(lo, 1), round(hi, 1)], 't': round(time.time() - t0)}
            log.append(rec); print(json.dumps(rec), flush=True); acc = {k: 0 for k in acc}
    torch.save({'ema': ema.state_dict(), 'widths': widths, 'pos': int(a.pos), 'hidden': a.hidden, 'arch': a.arch, 'args': vars(a), 'log': log}, f'{_P.WORK}/ckpt/{a.name}.pt')
    print('saved', a.name, flush=True)

if __name__ == '__main__':
    main()
