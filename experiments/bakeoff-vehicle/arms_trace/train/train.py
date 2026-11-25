"""Train one rung of one trace arm on the harness's simulated train bank (the same recipe for every arm).
usage: python3 train.py <config name>        (configs/<name>.json)

Data: the 28,800 train trips, every trip whole (5-s blocks, <= 1,440 blocks), batches of trips of similar length
(token budget cfg['tokens'] blocks, <= cfg['max_batch'] trips). Fresh tiers every epoch (EnergyNet's draw_tiers),
target = bake.target (EnergyNet v2.1's version), trips with y <= 0.02 MJ masked (EnergyNet). Loss: pinball on 5
log-quantiles of log(y / anchor) + 0.2 x MSE of the log component ratios (EnergyNet's wc). AdamW, one-cycle cosine,
grad clip 1.0, last step used. Same data stream for every arm with the same seed (tiers and batch order depend on
the seed and the epoch only). Resumable (ckpt/<name>.resume.pt every 200 steps and every epoch end).
At the end: exports <arm>/work/cand_<rung>/ (a candidate arm folder for harness/evaluate.py) and checks numpy
(common/tracelib.py, unpadded) against torch (padded batch) on frozen sim val trips.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRC = os.path.dirname(HERE)
sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
sys.path.insert(0, os.path.join(TRC, 'common'))
import bake  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(1)
import nets  # noqa: E402
import tracelib as TL  # noqa: E402

CKPT = os.path.join(TRC, 'ckpt')


def release_traces(B):
    """drop the memory-mapped trace pages this process touched (they are read once; keeps RSS ~0.5 GB)."""
    import mmap
    for k in ('trace_v', 'trace_x'):
        m = getattr(B[k], '_mmap', None)
        if m is not None:
            try:
                m.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError, ValueError):
                pass
        f = os.path.join(bake.C.BANK, B['_split'], k + '.npy')
        try:
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (AttributeError, OSError):
            pass


class Bank:
    """dyn block features of every trip (tier-independent, computed once) + per-epoch tiers / targets / statics."""
    def __init__(self, split, W=None, idx=None):
        self.B = bake.load_bank(split)
        n = len(self.B['arch'])
        self.idx = np.arange(n) if idx is None else np.asarray(idx)
        t0 = time.time()
        L = np.diff(self.B['trace_off'].astype(np.int64))[self.idx]
        self.nb = -(-L // TL.BLOCK)
        self.off = np.concatenate([[0], np.cumsum(self.nb)])
        self.dyn = np.empty((self.off[-1], TL.N_DYN), np.float32)
        self.dist = np.zeros(len(self.idx)); self.dur = np.zeros(len(self.idx))
        for j, i in enumerate(self.idx):
            tr = bake.sim_trace(self.B, int(i))
            self.dyn[self.off[j]:self.off[j + 1]] = TL.dyn_features(tr)
            self.dist[j], self.dur[j] = TL.trip_dist_dur(tr)
            if j % 2000 == 1999:
                release_traces(self.B)
        release_traces(self.B)
        self.n = len(self.idx)
        print(f'{split}: {self.n} trips, {len(self.dyn)} blocks, dyn {time.time() - t0:.0f}s', flush=True)

    def epoch_data(self, T, W, cfg):
        """targets and statics for this tier draw (rows = self.idx)."""
        D = bake.sim_table(self.B, T, self.idx)
        S = bake.sim_spec(self.B, T, self.idx)
        y = D['y']; Eph = np.maximum(D['E_phys'], 1e-3)
        A0 = TL.anchor_dumb(W, D['arch'], D['cls'], self.dist)
        anc = Eph if cfg['anchor'] == 'phys' else A0
        st = TL.statics(W, cfg, S['spec'], S['rl'], self.dist, self.dur, A0, Eph).astype(np.float32)
        t = np.log(np.maximum(y, 0.02) / anc).astype(np.float32)
        w = (y > bake.Y_MIN).astype(np.float32)
        comp = np.asarray(self.B['comp'][self.idx, :5], np.float64)
        ct = np.log(np.clip(comp, 1e-3 * anc[:, None], None) / anc[:, None]).astype(np.float32)
        return dict(st=st, t=t, w=w, ct=ct, y=y, anc=anc)

    def batch(self, rows, E):
        L = int(self.nb[rows].max())
        X = np.zeros((len(rows), L, TL.N_DYN), np.float32); M = np.zeros((len(rows), L), np.float32)
        for j, r in enumerate(rows):
            a, b = self.off[r], self.off[r + 1]
            X[j, :b - a] = self.dyn[a:b]; M[j, :b - a] = 1.0
        return (torch.from_numpy(X), torch.from_numpy(E['st'][rows]), torch.from_numpy(M),
                torch.from_numpy(self.nb[rows].astype(np.int64)), torch.from_numpy(E['t'][rows]),
                torch.from_numpy(E['ct'][rows]), torch.from_numpy(E['w'][rows]))


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


def norm_stats(Tr, cfg):
    B = Tr.B
    T0 = bake.draw_tiers(B, np.random.default_rng(12345))
    S0 = bake.sim_spec(B, T0)
    dr = bake.dumb_rates()
    rate = np.full((7, 7), np.median(list(dr.values())))
    for k, v in dr.items():
        rate[k // 10, k % 10] = v
    rlf = TL.rl_feats(S0['rl'])
    return {'sp_mu': S0['spec'].astype(np.float64).mean(0), 'sp_sd': S0['spec'].astype(np.float64).std(0) + 1e-3,
            'rl_mu': rlf.mean(0), 'rl_sd': rlf.std(0) + 1e-3, 'dumb_rate': rate}


def quick_val(net, V, E, nmax=800):
    """monitoring only (decisions use harness/evaluate.py on all 3,985 trips): median |log error| on nmax val trips."""
    net.eval(); errs = []
    rows = np.arange(min(nmax, V.n))
    with torch.no_grad():
        for bt in make_batches(V.nb[rows], np.random.default_rng(0), 16384, 128):
            X, S, M, Ln, t, ct, w = V.batch(bt, E)
            q, c = net(X, S, M, Ln)
            ok = w.numpy() > 0
            errs.append(np.abs(q[:, 2].numpy() - t.numpy())[ok])
    net.train()
    return float(np.median(np.concatenate(errs)))


CAND_PREDICT = '''"""Candidate rung of an arms_trace arm (evaluation only; not an exam arm: it lives under work/)."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
d = HERE
while not os.path.exists(os.path.join(d, 'common', 'tracelib.py')):
    d = os.path.dirname(d)
sys.path.insert(0, os.path.join(d, 'common'))
import tracelib  # noqa: E402
_M = tracelib.Model(os.path.join(HERE, 'weights.npz'))


def predict(batch):
    return _M.predict(batch)


def predict_many(batches):
    return _M.predict_many(batches)
'''


def make_candidate(cfg, net, W, cpu_h):
    d = os.path.join(TRC, cfg['arm'], 'work', f'cand_{cfg["rung"]}')
    os.makedirs(d, exist_ok=True)
    nets.export(net, cfg, W, os.path.join(d, 'weights.npz'))
    open(os.path.join(d, 'predict.py'), 'w').write(CAND_PREDICT)
    meta = {'arm_id': f'trace_{cfg["arm"]}_cand_{cfg["rung"]}', 'family': 'x', 'group': 'trace', 'tracks': ['trace'],
            'numpy_only': True, 'compute': {'train_cpu_hours': cpu_h}, 'feature_engineering': 'see arms_trace/common',
            'rungs': [{'id': cfg['rung'], 'change': cfg.get('note', ''), 'decision': 'released', 'why': 'candidate'}],
            'released': cfg['rung'], 'cfg': cfg}
    json.dump(meta, open(os.path.join(d, 'ratchet.json'), 'w'), indent=1)
    return d


def np_check(net, d, V, E, rows):
    """numpy (float64, tracelib, each trip unpadded) vs torch (float32, one padded batch) in log space."""
    M = TL.Model(os.path.join(d, 'weights.npz'))
    net.eval()
    with torch.no_grad():
        X, S, Mk, Ln, t, ct, w = V.batch(rows, E)
        qt, ctt = net(X, S, Mk, Ln)
    qn = []
    for r in rows:
        dyn = V.dyn[V.off[r]:V.off[r + 1]].astype(np.float64)[None]
        st = E['st'][r:r + 1].astype(np.float64)
        q, c = TL.head(M.W, TL.encode(M.W, M.cfg, dyn, st), st)
        qn.append(q[0])
    qn = np.array(qn)
    return {'max_abs_diff_logq': float(np.max(np.abs(qn - qt.numpy()))), 'trips': len(rows),
            'lengths': [int(V.nb[r]) for r in rows]}


def main(name):
    cfg = json.load(open(os.path.join(HERE, 'configs', f'{name}.json')))
    os.makedirs(CKPT, exist_ok=True)
    Tr = Bank('train')
    W = norm_stats(Tr, cfg)
    vrows = bake.simval_rows()
    V = Bank('val', idx=vrows[:800])
    EV = V.epoch_data(bake.val_tiers(), W, cfg)          # frozen val tiers (full 4,000; sim_table subsets to V.idx)
    torch.manual_seed(cfg.get('seed', 0))
    net = nets.build(cfg)
    if cfg.get('init_from'):
        sd = torch.load(os.path.join(CKPT, f'{cfg["init_from"]}.pt'), weights_only=False)['state']
        net.load_state_dict(sd)
        print('init from', cfg['init_from'], flush=True)
    print(f'{name}: {cfg["arch"]} {nets.n_params(net)} params', flush=True)
    epochs, lr0 = cfg['epochs'], cfg['lr']
    nb_est = len(make_batches(Tr.nb, np.random.default_rng(0), cfg['tokens'], cfg['max_batch']))
    total = epochs * nb_est
    opt = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=cfg.get('wd', 1e-4))
    rp = os.path.join(CKPT, f'{name}.resume.pt')
    ep, k0, step, hist, cpu = 0, 0, 0, [], 0.0
    if os.path.exists(rp):
        r = torch.load(rp, weights_only=False)
        net.load_state_dict(r['state']); opt.load_state_dict(r['opt'])
        ep, k0, step, hist, cpu = r['epoch'], r['k'], r['step'], r['hist'], r['cpu']
        print(f'resumed at epoch {ep} batch {k0} step {step}', flush=True)
    c0 = time.process_time(); t0 = time.time()

    def save(ep_, k_):
        torch.save({'state': net.state_dict(), 'opt': opt.state_dict(), 'epoch': ep_, 'k': k_, 'step': step,
                    'hist': hist, 'cpu': cpu + time.process_time() - c0}, rp + '.tmp')
        os.replace(rp + '.tmp', rp)

    net.train()
    while ep < epochs:
        rng = np.random.default_rng([cfg.get('seed', 0), ep])
        T = bake.draw_tiers(Tr.B, rng)
        E = Tr.epoch_data(T, W, cfg)
        bl = make_batches(Tr.nb, rng, cfg['tokens'], cfg['max_batch'])
        run = []
        for k in range(k0, len(bl)):
            frac = min(step / total, 1.0)
            wu = cfg.get('warmup', 0.03)
            lr = lr0 * frac / wu if frac < wu else 1e-5 + 0.5 * (lr0 - 1e-5) * (1 + math.cos(math.pi * (frac - wu) / (1 - wu)))
            for g in opt.param_groups:
                g['lr'] = lr
            X, S, M, Ln, t, ct, w = Tr.batch(bl[k], E)
            q, c = net(X, S, M, Ln)
            loss, pin, cm = nets.loss_fn(q, c, t, ct, w, cfg.get('wc', 0.2))
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); step += 1
            run.append([loss.item(), pin.item(), cm.item()])
            if step % 100 == 0:
                a = np.mean(run, 0); run = []
                print(f'ep {ep} step {step}/{total} loss {a[0]:.4f} pin {a[1]:.4f} comp {a[2]:.4f} lr {lr:.2e} '
                      f'cpu {cpu + time.process_time() - c0:.0f}s wall {time.time() - t0:.0f}s', flush=True)
            if step % 200 == 0:
                save(ep, k + 1)
        k0 = 0; ep += 1
        hist.append({'epoch': ep, 'step': step, 'quick_val_medabslog_800': quick_val(net, V, EV)})
        print('VAL', hist[-1], flush=True)
        save(ep, 0)
    cpu = cpu + time.process_time() - c0
    torch.save({'state': net.state_dict(), 'cfg': cfg, 'hist': hist, 'cpu_s': cpu, 'n_params': nets.n_params(net)},
               os.path.join(CKPT, f'{name}.pt'))
    d = make_candidate(cfg, net, W, round(cpu / 3600, 3))
    rows = np.argsort(V.nb)[[0, 100, 400, 700, 799]]
    chk = np_check(net, d, V, EV, rows)
    rec = {'name': name, 'cfg': cfg, 'n_params': nets.n_params(net), 'train_cpu_s': round(cpu, 1),
           'train_cpu_h': round(cpu / 3600, 3), 'steps': step, 'hist': hist, 'export_check': chk, 'candidate': d,
           'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
    json.dump(rec, open(os.path.join(TRC, cfg['arm'], 'work', f'train_{cfg["rung"]}.json'), 'w'), indent=1)
    print('done', json.dumps({k: rec[k] for k in ('train_cpu_h', 'export_check', 'n_params')}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
