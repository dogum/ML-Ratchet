"""Round 2: one member of a trace-arm ensemble, trained on several banks, on the GPU (training only).

python train_ens.py <config> <member> [--device cuda]
  config  round2/trace/configs/<name>.json: {"name", "arm", "base": [v1.0 config names, one per phase],
          "banks": ["train101", ...], "val": "val", "members": 5, "seed_offset_phase": 100}
  member  0..members-1

Same recipe as v1.0's arms_trace/train/train.py (imported, not edited): the shared 5-s block pipeline
(common/tracelib.py), the same nets (train/nets.py), the same loss, batching (token budget, <= max_batch trips of
similar length), one-cycle cosine schedule, fresh tiers every epoch, last step used. Differences: every bank in
`banks` is in the stream (tiers drawn per bank per epoch), the network runs on the GPU (torch default device), and the
member's seed m replaces the config's seed (phase k > 0 uses data seed m + 100 k, as v1.0's r2 used seed 1 after r1's 0).
Normalisation statistics (spec / rl means and sds, dumb rates) come from train101, as in v1.0.
Writes round2/work/trace/<name>/m<member>/weights.npz (numpy inference: tracelib.Model) + train.json; resumable.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths as P  # noqa: E402
P.setup()
TRC = os.path.join(P.BAKEOFF, 'arms_trace')
sys.path.insert(0, os.path.join(TRC, 'common'))
sys.path.insert(0, os.path.join(TRC, 'train'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import bake  # noqa: E402
import nets  # noqa: E402
import tracelib as TL  # noqa: E402
import dyncache  # noqa: E402

V1CFG = os.path.join(TRC, 'train', 'configs')


def load_bank(name):
    d = P.bank_dir(name)
    B = dict(np.load(os.path.join(d, 'rows.npz')))
    B['_split'] = 'train'; B['_bank'] = name
    return B


class Multi:
    """the dyn block features of every trip of several banks (cached on disk) + per-epoch targets / statics."""
    def __init__(self, names):
        self.names = names
        self.B = [load_bank(n) for n in names]
        self.dyn, self.nb, self.dist, self.dur, self.off = [], [], [], [], []
        base = 0
        for n in names:
            c = dyncache.load(n)
            self.dyn.append(c['dyn']); self.nb.append(c['nb']); self.dist.append(c['dist']); self.dur.append(c['dur'])
        self.nbank = [len(x) for x in self.nb]
        self.nb_all = np.concatenate(self.nb)
        self.boff = np.concatenate([[0], np.cumsum(self.nbank)])
        self.toff = [np.concatenate([[0], np.cumsum(x)]) for x in self.nb]
        self.n = int(self.boff[-1])
        print(f'banks {names}: {self.n} trips, {sum(len(d) for d in self.dyn)} blocks', flush=True)

    def epoch_data(self, rng, W, cfg):
        parts = []
        for k, B in enumerate(self.B):
            T = bake.draw_tiers(B, rng)
            D = bake.sim_table(B, T); S = bake.sim_spec(B, T)
            y = D['y']; Eph = np.maximum(D['E_phys'], 1e-3)
            A0 = TL.anchor_dumb(W, D['arch'], D['cls'], self.dist[k])
            anc = Eph if cfg['anchor'] == 'phys' else A0
            st = TL.statics(W, cfg, S['spec'], S['rl'], self.dist[k], self.dur[k], A0, Eph).astype(np.float32)
            t = np.log(np.maximum(y, 0.02) / anc).astype(np.float32)
            w = (y > bake.Y_MIN).astype(np.float32)
            comp = np.asarray(B['comp'][:, :5], np.float64)
            ct = np.log(np.clip(comp, 1e-3 * anc[:, None], None) / anc[:, None]).astype(np.float32)
            parts.append(dict(st=st, t=t, w=w, ct=ct))
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def batch(self, rows, E, dev):
        L = int(self.nb_all[rows].max())
        X = np.zeros((len(rows), L, TL.N_DYN), np.float32); M = np.zeros((len(rows), L), np.float32)
        for j, r in enumerate(rows):
            k = int(np.searchsorted(self.boff, r, side='right') - 1); i = r - self.boff[k]
            a, b = self.toff[k][i], self.toff[k][i + 1]
            X[j, :b - a] = self.dyn[k][a:b]; M[j, :b - a] = 1.0
        f = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
        return (f(X), f(E['st'][rows]), f(M), f(self.nb_all[rows].astype(np.int64)), f(E['t'][rows]), f(E['ct'][rows]),
                f(E['w'][rows]))


def norm_stats(B101):
    T0 = bake.draw_tiers(B101, np.random.default_rng(12345))
    S0 = bake.sim_spec(B101, T0)
    dr = bake.dumb_rates()
    rate = np.full((7, 7), np.median(list(dr.values())))
    for k, v in dr.items():
        rate[k // 10, k % 10] = v
    rlf = TL.rl_feats(S0['rl'])
    return {'sp_mu': S0['spec'].astype(np.float64).mean(0), 'sp_sd': S0['spec'].astype(np.float64).std(0) + 1e-3,
            'rl_mu': rlf.mean(0), 'rl_sd': rlf.std(0) + 1e-3, 'dumb_rate': rate}


def make_batches(nb, rng, tokens, max_batch):
    order = np.argsort(nb + rng.uniform(0, 30, len(nb)), kind='stable')
    out, cur, mx = [], [], 0
    for r in order:
        m = max(mx, nb[r])
        if cur and (m * (len(cur) + 1) > tokens or len(cur) >= max_batch):
            out.append(np.array(cur)); cur, m = [], nb[r]
        cur.append(r); mx = m
    if cur:
        out.append(np.array(cur))
    rng.shuffle(out)
    return out


def run_phase(net, Tr, W, cfg, seed, tag, odir, dev, log):
    epochs, lr0 = cfg['epochs'], cfg['lr']
    nb_est = len(make_batches(Tr.nb_all, np.random.default_rng(0), cfg['tokens'], cfg['max_batch']))
    total = epochs * nb_est
    opt = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=cfg.get('wd', 1e-4))
    rp = os.path.join(odir, f'{tag}.resume.pt')
    ep, k0, step, wall = 0, 0, 0, 0.0
    if os.path.exists(rp):
        r = torch.load(rp, weights_only=False, map_location=dev)
        net.load_state_dict(r['state']); opt.load_state_dict(r['opt'])
        ep, k0, step, wall = r['epoch'], r['k'], r['step'], r['wall']
        log(f'{tag}: resumed at epoch {ep} batch {k0} step {step}')
    t0 = time.time()

    def save(ep_, k_):
        torch.save({'state': net.state_dict(), 'opt': opt.state_dict(), 'epoch': ep_, 'k': k_, 'step': step,
                    'wall': wall + time.time() - t0}, rp + '.tmp')
        os.replace(rp + '.tmp', rp)

    net.train()
    taus = None
    while ep < epochs:
        rng = np.random.default_rng([seed, ep])
        E = Tr.epoch_data(rng, W, cfg)
        bl = make_batches(Tr.nb_all, rng, cfg['tokens'], cfg['max_batch'])
        run = []
        for k in range(k0, len(bl)):
            frac = min(step / total, 1.0)
            wu = cfg.get('warmup', 0.03)
            lr = lr0 * frac / wu if frac < wu else 1e-5 + 0.5 * (lr0 - 1e-5) * (1 + math.cos(math.pi * (frac - wu) / (1 - wu)))
            for g in opt.param_groups:
                g['lr'] = lr
            X, S, M, Ln, t, ct, w = Tr.batch(bl[k], E, dev)
            q, c = net(X, S, M, Ln)
            loss, pin, cm = nets.loss_fn(q, c, t, ct, w, cfg.get('wc', 0.2))
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); step += 1
            run.append(loss.detach())
            if step % 500 == 0:
                a = float(torch.stack(run).mean()); run = []
                log(f'{tag} ep {ep} step {step}/{total} loss {a:.4f} lr {lr:.2e} wall {wall + time.time() - t0:.0f}s')
            if step % 1000 == 0:
                save(ep, k + 1)
        k0 = 0; ep += 1
        save(ep, 0)
    return wall + time.time() - t0, step


def main(cfgname, member, device):
    C2 = json.load(open(os.path.join(HERE, 'configs', cfgname + '.json')))
    odir = os.path.join(P.WORK, 'trace', C2['name'], f'm{member}')
    os.makedirs(odir, exist_ok=True)
    if os.path.exists(os.path.join(odir, 'train.json')):
        print('done already', odir); return
    logf = open(os.path.join(odir, 'train.log'), 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    dev = torch.device(device)
    torch.set_default_device(dev)       # nets.py builds index tensors without a device argument
    torch.set_num_threads(1)
    W = norm_stats(load_bank('train101'))
    Tr = Multi(C2['banks'])
    phases = [json.load(open(os.path.join(V1CFG, b + '.json'))) for b in C2['base']]
    torch.manual_seed(member)
    net = nets.build(phases[0]).to(dev)
    log(f'{C2["name"]} m{member}: {phases[0]["arch"]} {nets.n_params(net)} params, banks {C2["banks"]}, device {dev}')
    hist = []; wall = 0.0; c0 = time.process_time()
    for k, cfg in enumerate(phases):
        seed = member + 100 * k
        pth = os.path.join(odir, f'phase{k}.pt')
        if os.path.exists(pth):
            net.load_state_dict(torch.load(pth, weights_only=False, map_location=dev)); log(f'phase {k} loaded'); continue
        w, steps = run_phase(net, Tr, W, cfg, seed, f'phase{k}', odir, dev, log)
        wall += w
        torch.save(net.state_dict(), pth)
        hist.append({'phase': k, 'base': C2['base'][k], 'epochs': cfg['epochs'], 'lr': cfg['lr'], 'seed': seed,
                     'steps': steps, 'wall_s': round(w, 1)})
        log(f'phase {k} done: {steps} steps, {w:.0f}s')
    cfg_out = dict(phases[-1]); cfg_out.pop('init_from', None)
    cfg_out.update(rung=C2['name'], member=member, banks=C2['banks'], seed=member, note=C2.get('note', ''))
    net = net.to('cpu'); torch.set_default_device('cpu')
    nets.export(net, cfg_out, W, os.path.join(odir, 'weights.npz'))
    rec = {'name': C2['name'], 'member': member, 'n_params': nets.n_params(net), 'phases': hist,
           'gpu_wall_s': round(wall, 1), 'cpu_s': round(time.process_time() - c0, 1), 'device': str(dev),
           'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
    json.dump(rec, open(os.path.join(odir, 'train.json'), 'w'), indent=1)
    log('done ' + json.dumps(rec))


if __name__ == '__main__':
    dv = 'cuda'
    if '--device' in sys.argv:
        dv = sys.argv[sys.argv.index('--device') + 1]
    main(sys.argv[1], int(sys.argv[2]), dv)
