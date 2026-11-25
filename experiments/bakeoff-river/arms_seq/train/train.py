"""Train one rung of one sequence arm on the harness bank (FlowCast's recipe for every arm).
usage: python3 train.py <config name>        (configs/<name>.json)
Resumable (ckpt/<name>.resume.pt every 250 steps). The last step is used (no early stopping), as FlowCast.
Same data stream for every arm: rng seed -> identical (run, offset) draws in identical order.
At the end: exports weights to <arm>/work/cand_<rung>/ (a candidate arm folder for harness/evaluate.py) and checks
numpy (common/seqlib.py) against torch on 8 frozen sim val windows.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.path.dirname(HERE)
sys.path.insert(0, '/home/claude/bakeoff/river/harness')
sys.path.insert(0, os.path.join(SEQ, 'common'))
import bake  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(1)
import nets  # noqa: E402
import seqlib  # noqa: E402

WIN, WARM = 730, 365
CKPT = os.path.join(SEQ, 'ckpt')


def fu_runoff_ratio(log_aridity, w=2.6):
    phi = np.exp(log_aridity)
    return np.clip((1 + phi ** w) ** (1 / w) - phi, 0.0, 1.0)


class Bank:
    """Memory-mapped bank + FlowCast's window recipe (target in units of p_mean = exp(st[11]); budyko loss scale)."""
    def __init__(self, split):
        self.B = bake.load_bank(split, ['F', 'qo', 'st', 'doy0'])
        self.st = np.asarray(self.B['st'], np.float32)
        self.pm = np.exp(self.st[:, 11]).astype(np.float32)
        self.sig = fu_runoff_ratio(self.st[:, 13]).astype(np.float32)
        self.n = len(self.st)
        self.maxo = self.B['F'].shape[1] - WIN - 1

    def batch(self, idx, offs):
        X, Y = [], []
        for i, o in zip(idx, offs):
            F = np.asarray(self.B['F'][i, o:o + WIN + 1], np.float32)
            doy = (int(self.B['doy0'][i]) - 1 + o + np.arange(WIN)) % 365 + 1
            X.append(seqlib.dyn_features(F[:WIN], doy))
            Y.append(np.asarray(self.B['qo'][i, o:o + WIN], np.float32) / self.pm[i])
        return np.stack(X), np.stack(Y), self.st[idx], self.sig[idx]


def quick_val(net, V, nmax=200):
    """Monitoring only (decisions use harness/evaluate.py on all 800 runs): median NSE on the first nmax val runs."""
    net.eval(); nses = []
    with torch.no_grad():
        for a in range(0, nmax, 25):
            idx = np.arange(a, min(a + 25, nmax))
            P, Yy = [], []
            for o in (0, 731):
                X, Y, S, _ = V.batch(idx, [o] * len(idx))
                m, _ = net(torch.from_numpy(X), torch.from_numpy(S))
                P.append(m[:, WARM:].numpy() * V.pm[idx, None]); Yy.append(Y[:, WARM:] * V.pm[idx, None])
            P = np.concatenate(P, 1); Yy = np.concatenate(Yy, 1)
            for p, y in zip(P, Yy):
                v = np.var(y)
                nses.append(1 - np.mean((p - y) ** 2) / v if v > 0 else np.nan)
    net.train()
    return float(np.nanmedian(nses))


CAND_PREDICT = '''"""Candidate rung of an arms_seq arm (evaluation only; not an exam arm: it lives under work/)."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
d = HERE
while not os.path.exists(os.path.join(d, 'common', 'seqlib.py')):
    d = os.path.dirname(d)
sys.path.insert(0, os.path.join(d, 'common'))
import seqlib  # noqa: E402
_M = seqlib.Model(os.path.join(HERE, 'weights.npz'))


def predict(batch):
    return _M.predict(batch)


def predict_many(batches):
    return _M.predict_many(batches)
'''


def make_candidate(cfg, net, cpu_h):
    d = os.path.join(SEQ, cfg['arm'], 'work', f'cand_{cfg["rung"]}')
    os.makedirs(d, exist_ok=True)
    nets.export(net, cfg, os.path.join(d, 'weights.npz'))
    open(os.path.join(d, 'predict.py'), 'w').write(CAND_PREDICT)
    meta = {'arm_id': f'seq_{cfg["arm"]}_cand_{cfg["rung"]}', 'family': 'x', 'group': 'seq', 'protocols': ['ungauged'],
            'numpy_only': True, 'compute': {'train_cpu_hours': cpu_h}, 'feature_engineering': 'see arms_seq/common',
            'rungs': [{'id': cfg['rung'], 'change': cfg.get('note', ''), 'decision': 'released', 'why': 'candidate'}],
            'released': cfg['rung'], 'cfg': cfg}
    json.dump(meta, open(os.path.join(d, 'ratchet.json'), 'w'), indent=1)
    return d


def np_check(net, cfg, d, V):
    """numpy (float64, seqlib) vs torch (float32) on 8 frozen sim val windows, in mm/day relative to mean flow."""
    M = seqlib.Model(os.path.join(d, 'weights.npz'))
    idx = np.arange(8); X, _, S, _ = V.batch(idx, [0] * 8)
    net.eval()
    with torch.no_grad():
        mt, qt = net(torch.from_numpy(X), torch.from_numpy(S))
    mn, qn = seqlib.forward(M.W, M.cfg, X.astype(np.float64), S.astype(np.float64))
    rel_m = float(np.max(np.abs(mn - mt.numpy().astype(np.float64))) / np.mean(np.abs(mn)))
    rel_q = float(np.max(np.abs(qn - qt.numpy().astype(np.float64))) / np.mean(np.abs(qn)))
    return {'max_abs_diff_rel_mean': rel_m, 'max_abs_diff_rel_q': rel_q, 'windows': 8}


def main(name):
    cfg = json.load(open(os.path.join(HERE, 'configs', f'{name}.json')))
    os.makedirs(CKPT, exist_ok=True)
    Tr = Bank('train'); V = Bank('val')
    torch.manual_seed(cfg.get('seed', 0))
    net = nets.build(cfg)
    net.st_mu.copy_(torch.from_numpy(Tr.st.mean(0)))
    net.st_sd.copy_(torch.from_numpy(Tr.st.std(0) + 1e-3))
    if cfg.get('init_from'):
        sd = torch.load(os.path.join(CKPT, f'{cfg["init_from"]}.pt'), weights_only=False)['state']
        net.load_state_dict(sd)
        print('init from', cfg['init_from'], flush=True)
    print(f'{name}: {cfg["arch"]} {nets.n_params(net)} params; train {Tr.n} val {V.n}', flush=True)
    steps, bs, lr0 = cfg['steps'], cfg['batch'], cfg['lr']
    warm = cfg.get('warmup', 0)
    opt = torch.optim.Adam(net.parameters(), lr=lr0, weight_decay=cfg.get('wd', 0.0))
    rp = os.path.join(CKPT, f'{name}.resume.pt')
    step, hist, cpu = 0, [], 0.0
    rng = np.random.default_rng(cfg.get('seed', 0))
    if os.path.exists(rp):
        r = torch.load(rp, weights_only=False)
        net.load_state_dict(r['state']); opt.load_state_dict(r['opt']); step = r['step']; hist = r['hist']
        cpu = r['cpu']
        rng = np.random.default_rng([cfg.get('seed', 0), step])
        print('resumed at', step, flush=True)
    c0 = time.process_time(); t0 = time.time(); run = []
    net.train()
    while step < steps:
        lr = 1e-4 + 0.5 * (lr0 - 1e-4) * (1 + math.cos(math.pi * step / steps))
        if warm and step < warm:
            lr = lr * (step + 1) / warm
        for g in opt.param_groups:
            g['lr'] = lr
        idx = rng.integers(0, Tr.n, bs); offs = rng.integers(0, Tr.maxo + 1, bs)
        X, Y, S, SIG = Tr.batch(idx, offs)
        m, q = net(torch.from_numpy(X), torch.from_numpy(S))
        loss, l1, l2 = nets.loss_fn(m[:, WARM:], q[:, WARM:], torch.from_numpy(Y[:, WARM:]), torch.from_numpy(SIG))
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss at step {step}')
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); step += 1
        run.append([loss.item(), l1.item(), l2.item()])
        if step % 100 == 0:
            a = np.mean(run, 0); run = []
            print(f'step {step} loss {a[0]:.4f} nse* {a[1]:.4f} pin {a[2]:.4f} lr {lr:.2e} '
                  f'cpu {cpu + time.process_time() - c0:.0f}s wall {time.time() - t0:.0f}s', flush=True)
        if step % 1000 == 0 or step == steps:
            hist.append({'step': step, 'quick_val_nse_200': quick_val(net, V)})
            print('VAL', hist[-1], flush=True)
        if step % 250 == 0 or step == steps:
            torch.save({'state': net.state_dict(), 'opt': opt.state_dict(), 'step': step, 'hist': hist,
                        'cpu': cpu + time.process_time() - c0}, rp + '.tmp')
            os.replace(rp + '.tmp', rp)
    cpu = cpu + time.process_time() - c0
    torch.save({'state': net.state_dict(), 'cfg': cfg, 'hist': hist, 'cpu_s': cpu, 'n_params': nets.n_params(net)},
               os.path.join(CKPT, f'{name}.pt'))
    d = make_candidate(cfg, net, round(cpu / 3600, 3))
    chk = np_check(net, cfg, d, V)
    rec = {'name': name, 'cfg': cfg, 'n_params': nets.n_params(net), 'train_cpu_s': round(cpu, 1),
           'train_cpu_h': round(cpu / 3600, 3), 'hist': hist, 'export_check': chk, 'candidate': d,
           'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}
    json.dump(rec, open(os.path.join(SEQ, cfg['arm'], 'work', f'train_{cfg["rung"]}.json'), 'w'), indent=1)
    print('done', json.dumps({k: rec[k] for k in ('train_cpu_h', 'export_check', 'n_params')}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
