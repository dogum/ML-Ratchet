"""Train TripDetective on a simulation bank. Resumable (checkpoint every 500 steps).
usage: python3 train.py <bank_tag> <ckpt_name> [steps=6000] [opts as k=v: width=..., chdrop=0.1, kmax=8, seed=0]"""
from common import *
import time, torch, torch.nn.functional as Fn
torch.set_num_threads(1)
import simbank, features as F
from model import TDNet, reg_targets, LED_SCALE, REG


class BankSampler:
    def __init__(s, bank, seed=0, chdrop=0.1, kmax=8):
        s.b = bank; s.rng = np.random.default_rng(seed); s.chdrop = chdrop; s.kmax = kmax; s.feat = 1
        s.meta = {m['vid']: m for m in bank['meta']}
        vids = bank['veh']
        s.cars = sorted(s.meta)
        s.trips = {v: np.where(vids == v)[0] for v in s.cars}
        s.cars = [v for v in s.cars if len(s.trips[v]) > 0]
        s.arch = {v: ARCHS.index(s.meta[v]['arch']) for v in s.cars}
        s.strat = {v: STRATS.index(s.meta[v]['strategy']) for v in s.cars}
        s.reg = {v: reg_targets(s.meta[v]) for v in s.cars}

    def trip_input(s, j, drop=None):
        o = s.b['offs']; raw = s.b['raw'][o[j]:o[j + 1]].T.astype(np.float32)
        fl = s.b['flags'][j].copy()
        if drop is not None:
            raw = raw.copy()
            if drop[0] and fl[0]: fl[0] = 0; fl[1] = 0; raw[5] = 0
            if drop[1] and fl[2]: fl[2] = 0; raw[6] = 0
            if drop[2] and fl[3]: fl[3] = 0; fl[4] = 0; raw[7] = 0; raw[8] = 0; raw[9] = 0
        return F.net_input(raw, fl, feat=s.feat)

    def batch(s, ncar=24, k=None):
        cars = s.rng.choice(s.cars, ncar, replace=False)
        X, M, SC, grp, led, ledm = [], [], [], [], [], []
        for gi, v in enumerate(cars):
            tr = s.trips[v]
            kk = k or (1 if s.rng.random() < 0.35 else int(s.rng.integers(2, s.kmax + 1)))
            sel = s.rng.choice(tr, min(kk, len(tr)), replace=False)
            drop = s.rng.random(3) < s.chdrop
            for j in sel:
                x, m, sc = s.trip_input(j, drop); X.append(x); M.append(m); SC.append(sc); grp.append(gi)
                led.append(s.b['ledger'][j] / LED_SCALE); ledm.append(s.b['km'][j] > 0.3)
        T = int(max(m.sum() for m in M)); T = min(L, int(np.ceil(T / 32) * 32))
        X = torch.tensor(np.stack([x[:, :T] for x in X])); M = torch.tensor(np.stack([m[:T] for m in M]))
        return dict(x=X, mask=M, sc=torch.tensor(np.stack(SC)), group=torch.tensor(grp), n=len(cars),
                    arch=torch.tensor([s.arch[v] for v in cars]), strat=torch.tensor([s.strat[v] for v in cars]),
                    reg=torch.tensor(np.stack([s.reg[v] for v in cars])),
                    led=torch.tensor(np.stack(led)), ledm=torch.tensor(np.array(ledm)))


def nll(mu, ls, y):
    ok = torch.isfinite(y)
    yy = torch.where(ok, y, torch.zeros_like(y))
    l = 0.5 * ((yy - mu) * torch.exp(-ls)) ** 2 + ls
    return (l * ok).sum(0) / ok.sum(0).clamp(min=1)


def loss_fn(out, b, w_led=0.5, w_strat=0.3):
    la = Fn.cross_entropy(out['arch'], b['arch'])
    ls_ = Fn.cross_entropy(out['strat'], b['strat'])
    lr = nll(out['mu'], out['ls'], b['reg']).sum()
    y = torch.where(b['ledm'][:, None], b['led'], torch.full_like(b['led'], float('nan')))
    ll = nll(out['led_mu'], out['led_ls'], y).sum()
    return la + w_strat * ls_ + lr + w_led * ll, dict(arch=la.item(), strat=ls_.item(), reg=lr.item(), led=ll.item())


def make_net(opts):
    width = tuple(int(x) for x in opts.get('width', '32,48,64,96,128,160').split(','))
    return TDNet(width=width, feat=int(opts.get('feat', 1)))


def train(bank_tag, name, steps=6000, **opts):
    torch.manual_seed(int(opts.get('seed', 0)))
    bank = simbank.load_bank(bank_tag, int(opts['chunks']) if 'chunks' in opts else None)
    S = BankSampler(bank, seed=int(opts.get('seed', 0)), chdrop=float(opts.get('chdrop', 0.1)), kmax=int(opts.get('kmax', 8)))
    net = make_net(opts)
    S.feat = int(opts.get('feat', 1))
    print(name, 'params', sum(p.numel() for p in net.parameters()), 'cars', len(S.cars), 'trips', len(bank['veh']), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=float(opts.get('lr', 2e-3)), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(opts.get('lr', 2e-3)), total_steps=steps, pct_start=0.05)
    ck = f'{ROOT}/ckpt/{name}.pt'; part = ck + '.part'; step = 0
    if os.path.exists(part):
        st = torch.load(part, weights_only=False); net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        step = st['step']; S.rng = np.random.default_rng([int(opts.get('seed', 0)), step]); print('resume', step)
    net.train(); t0 = time.time(); run = {}
    while step < steps:
        b = S.batch(ncar=int(opts.get('ncar', 24)))
        out = net(b['x'], b['mask'], b['group'], b['n'], b['sc'])
        loss, parts = loss_fn(out, b, w_led=float(opts.get('w_led', 0.5)))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step(); step += 1
        for k, v in parts.items(): run[k] = 0.98 * run.get(k, v) + 0.02 * v
        if step % 100 == 0:
            print(f'{step} {time.time() - t0:.0f}s ' + ' '.join(f'{k} {v:.3f}' for k, v in run.items()), flush=True)
        if step % 500 == 0 or step == steps:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step), part)
    torch.save(dict(net=net.state_dict(), opts=opts, bank=bank_tag, steps=steps), ck)
    os.remove(part)
    print('saved', ck, flush=True)


if __name__ == '__main__':
    a = sys.argv
    opts = dict(x.split('=', 1) for x in a[4:])
    train(a[1], a[2], int(a[3]) if len(a) > 3 else 6000, **opts)
