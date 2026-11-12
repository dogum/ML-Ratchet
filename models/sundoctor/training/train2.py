"""Train SunDoctor on freshly simulated windows (never the same window twice)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, time, json, math, argparse, copy, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, _P.SRC)
from model2 import SunDoctor, REG, SCALE, GLOB, GSCALE
import simbatch2 as SB

def add_reference(rng, X, G, p_ref=0.65):
    """the system's own healthy performance level, as the release estimates it from history (or absent)"""
    B = len(X); has = rng.random(B) < p_ref
    ref = np.where(has, G[:, 0] * rng.uniform(0.97, 1.0, B) * (1 + 0.015 * rng.standard_normal(B)), 0.0)
    R = np.broadcast_to(ref[:, None, None, None], (B, 1) + X.shape[2:]); Fl = np.broadcast_to(has[:, None, None, None].astype(np.float32), (B, 1) + X.shape[2:])
    return np.concatenate([X, R.astype(np.float32), Fl], 1)

def batch(rng, B, D, p_ref=0.65, home=None):
    X, Y, G, meta = SB.simulate_batch(rng, B, D, home=home)
    return add_reference(rng, X, G, p_ref), Y, G, meta

SC = torch.tensor(SCALE)[None, :, None]; GSC = torch.tensor(GSCALE)[None]
IDX = [SB.DAY_LABELS.index(k) for k in REG]; STEP = SB.DAY_LABELS.index('step')

BETA = 0.0
def nll(mu, ls, y):
    """Gaussian NLL; with BETA > 0 it is beta-NLL (Seitzer et al. 2022): each point's loss is weighted by sigma^(2 beta),
    detached, so the mean cannot hide behind a large error bar"""
    ls = ls.clamp(-6, 3); l = 0.5 * ((y - mu) * torch.exp(-ls)) ** 2 + ls
    return l * torch.exp(2 * BETA * ls).detach() if BETA > 0 else l

def loss_fn(day, g, Y, Gt, w_step=1.0):
    n = len(IDX); y = Y[:, IDX] / SC; mu = day[:, :n]; ls = day[:, n:2 * n]
    l_reg = nll(mu, ls, y).mean()
    l_step = F.binary_cross_entropy_with_logits(day[:, 2 * n], Y[:, STEP], pos_weight=torch.tensor(8.0))
    l_g = nll(g[:, :2], g[:, 2:], Gt / GSC).mean()
    return l_reg + w_step * l_step + 0.5 * l_g, l_reg.item(), l_step.item(), l_g.item()

OLD_CH = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11]          # where v1's 10 input channels sit among v2's 12
def transplant(net, path):
    """start v2 from a v1 checkpoint: copy every matching tensor; map the stem's input channels and the day head's rows"""
    src = torch.load(path, weights_only=False)['ema']; own = net.state_dict(); n_ok = 0
    for k, v in src.items():
        if k in own and own[k].shape == v.shape: own[k] = v.clone(); n_ok += 1
    w = torch.zeros_like(own['stem.0.weight']); w[:, OLD_CH] = src['stem.0.weight']; own['stem.0.weight'] = w
    dw = torch.zeros_like(own['day.weight']); db = torch.zeros_like(own['day.bias'])
    for i in range(7): dw[i] = src['day.weight'][i]; db[i] = src['day.bias'][i]; dw[9 + i] = src['day.weight'][7 + i]; db[9 + i] = src['day.bias'][7 + i]
    dw[18] = src['day.weight'][14]; db[18] = src['day.bias'][14]; own['day.weight'] = dw; own['day.bias'] = db
    net.load_state_dict(own); print('transplanted', n_ok, 'tensors + stem / head maps', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--init', default=None); ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--warm', type=int, default=150)
    ap.add_argument('--days', type=int, default=14); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ema', type=float, default=0.998)
    ap.add_argument('--widths', default='24,32,48,64,96'); ap.add_argument('--tw', type=int, default=128); ap.add_argument('--p_ref', type=float, default=0.65); ap.add_argument('--beta', type=float, default=0.0); ap.add_argument('--realism', type=int, default=0); ap.add_argument('--transplant', default=None); ap.add_argument('--fronts', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(2); torch.manual_seed(a.seed); SB.FRONTS = bool(a.fronts)
    global BETA; BETA = a.beta
    widths = tuple(int(v) for v in a.widths.split(','))
    net = SunDoctor(widths, a.tw)
    if a.init: net.load_state_dict(torch.load(a.init, weights_only=False)['ema'])
    if a.transplant: transplant(net, a.transplant)
    elif not a.init:
        with torch.no_grad():
            b = net.day.bias; b.zero_(); b[0] = 0.8 / SCALE[0]; b[2] = 0.9 / SCALE[2]
            net.glob[-1].bias.zero_(); net.glob[-1].bias[0] = 0.91 / GSCALE[0]; net.glob[-1].bias[1] = 0.85 / GSCALE[1]
    ema = copy.deepcopy(net).eval(); rng = np.random.default_rng(1000 + a.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); t0 = time.time(); log = []; acc = np.zeros(5)
    for step in range(1, a.steps + 1):
        lr = a.lr * (step / a.warm if step < a.warm else 0.5 * (1 + math.cos(math.pi * (step - a.warm) / max(1, a.steps - a.warm))) * 0.97 + 0.03)
        for gr in opt.param_groups: gr['lr'] = lr
        X, Y, G, _ = batch(rng, a.bs, a.days, a.p_ref)
        net.train(); day, g = net(torch.from_numpy(X))
        loss, lr_, ls_, lg_ = loss_fn(day, g, torch.from_numpy(Y), torch.from_numpy(G))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(a.ema).add_(pn, alpha=1 - a.ema)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        acc += (loss.item(), lr_, ls_, lg_, 1)
        if step % 100 == 0 or step == a.steps:
            rec = {'step': step, 'loss': acc[0] / acc[4], 'reg': acc[1] / acc[4], 'step_bce': acc[2] / acc[4], 'glob': acc[3] / acc[4], 'lr': lr, 't': round(time.time() - t0)}
            log.append(rec); print(json.dumps(rec), flush=True); acc[:] = 0
    torch.save({'ema': ema.state_dict(), 'widths': widths, 'tw': a.tw, 'days': a.days, 'args': vars(a), 'log': log, 'version': 2}, f'{_P.WORK}/ckpt/{a.name}.pt')
    print('saved', a.name, flush=True)

if __name__ == '__main__':
    main()
