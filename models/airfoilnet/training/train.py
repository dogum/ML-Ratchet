"""Train one AirfoilNet rung on simulated polars only, export numpy weights, score frozen sim set + real dev.

usage: python train.py <rung_id>     (rung configs in RUNGS below; outputs in work/runs/<rung_id>/)
"""
import os, sys, json, glob, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
from _paths import WORK   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'release')); sys.path.insert(2, os.path.abspath(ROOT))   # ROOT: airfoilnet.py in the repository layout
import airfoilnet as an  # noqa
import realeval as rev  # noqa
torch.set_num_threads(1)

RUNGS = {
    'r1': dict(snapshot='A', arch='mlp', anchor=False, loose=True, banks=['p1'], width=128, depth=3),
    'r2': dict(snapshot='A', arch='deeponet', anchor=False, loose=True, banks=['p1'], width=128, depth=3, P=32),
    'r3': dict(snapshot='A', arch='deeponet', anchor=True, loose=True, banks=['p1'], width=128, depth=3, P=32),
    'r4': dict(snapshot='A', arch='deeponet', anchor=True, loose=False, banks=['p1'], width=128, depth=3, P=32),
    'r3b': dict(snapshot='A', arch='mlp', anchor=True, loose=True, banks=['p1'], width=128, depth=3),
    'r4b': dict(snapshot='A', arch='mlp', anchor=True, loose=False, banks=['p1'], width=128, depth=3),
    'r5': dict(snapshot='A', arch='mlp', anchor=False, loose=True, banks=['p1'], width=128, depth=3, calib=True),
    'r6': dict(snapshot='B', arch='mlp', anchor=False, loose=True, banks=['p1'], width=128, depth=3),
    'r7': dict(snapshot='B', arch='mlp', anchor=False, loose=True, banks=['p1'], width=128, depth=3, calib=True),
}
CALIB = json.load(open(os.path.join(WORK, 'calib_sim.json'))) if os.path.exists(
    os.path.join(WORK, 'calib_sim.json')) else None


def apply_calib(D):
    """physics p1c: dev-measured drag mismatch added to the simulated ln Cd labels"""
    D = dict(D); Y = D['Y'].copy()
    Y[:, 1] += CALIB['a'] + CALIB['b'] * (np.log10(D['RE']) - 5)
    D['Y'] = Y
    return D
FROZEN = 'p1_val'          # frozen simulated validation set (val1 = physics p1)


SNAP_N = {'A': 151, 'B': 256}      # training sweeps per snapshot, from rung_r1_result.json / rung_r6_result.json


def load_bank(tags, sets, loose=True, max_sweeps=None, snapshot=None):
    recs = []
    for t in tags:
        for st in sets:
            for f in sorted(glob.glob(os.path.join(WORK, 'bank', f'{t}_{st}_w*.jsonl'))):
                for line in open(f):
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    recs.sort(key=lambda r: r['id'])
    if snapshot:
        sp = os.path.join(WORK, f'snap_{snapshot}.json')
        if os.path.exists(sp):
            keep = set(json.load(open(sp)))
            recs = [r for r in recs if r['id'] in keep]
        else:   # the frozen ID lists were not shipped (NOTEBOOK, 22 Nov 2026): first N sweeps by id, N as recorded
            n = SNAP_N[snapshot]
            print(f'warning: {sp} missing; using the first {n} sweeps by id (the release used a frozen list not shipped)', flush=True)
            recs = recs[:n]
    if max_sweeps:
        recs = recs[:max_sweeps]
    F, A, Y, ANC, ANC0, G, RE = [], [], [], [], [], [], []
    for gi, r in enumerate(recs):
        pts = [q for q in r['pts'] if (loose or not q[4]) and q[2] > 0]
        if not pts:
            continue
        p = np.array(r['p']); q = np.array(pts)
        f = an.features(p, r['Re'])
        F.append(np.repeat(f[None], len(q), 0)); A.append(q[:, 0])
        Y.append(np.stack([q[:, 1], np.log(q[:, 2]), q[:, 3]], 1))
        ANC.append(an.anchors(p, q[:, 0])); ANC0.append(an.anchors_none(q[:, 0])); G.append(np.full(len(q), gi)); RE.append(np.full(len(q), r['Re']))
    cat = lambda L: np.concatenate(L) if L else np.zeros((0,))
    return dict(F=cat(F), A=cat(A), Y=cat(Y), ANC=cat(ANC), ANC0=cat(ANC0), G=cat(G), RE=cat(RE), n_sweeps=len(recs))


def mlp(sizes):
    L = []
    for i in range(len(sizes) - 1):
        L.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(torch.nn.SiLU())
    return torch.nn.Sequential(*L)


class Net(torch.nn.Module):
    def __init__(self, cfg, nin=50):
        super().__init__()
        self.cfg = cfg; w, d = cfg['width'], cfg['depth']
        if cfg['arch'] == 'deeponet':
            P = cfg['P']
            self.branch = mlp([nin] + [w] * d + [P * 6])
            self.trunk = mlp([3, 64, 64, P])
            self.bias = torch.nn.Parameter(torch.zeros(6))
        else:
            self.net = mlp([nin + 3] + [w] * d + [6])

    def forward(self, F, T):
        if self.cfg['arch'] == 'deeponet':
            b = self.branch(F).view(len(F), self.cfg['P'], 6)
            t = self.trunk(T)
            return torch.einsum('npk,np->nk', b, t) + self.bias
        return self.net(torch.cat([F, T], 1))


def export(nets, cfg, path, widen=(1, 1, 1)):
    arrs = {}
    for m, net in enumerate(nets):
        sd = net.state_dict()
        if cfg['arch'] == 'deeponet':
            for part in ('branch', 'trunk'):
                lin = [k for k in sd if k.startswith(part + '.') and k.endswith('.weight')]
                for i, k in enumerate(sorted(lin, key=lambda s: int(s.split('.')[1]))):
                    arrs[f'm{m}/{part}.{i}.w'] = sd[k].numpy().astype(np.float32)
                    arrs[f'm{m}/{part}.{i}.b'] = sd[k.replace('weight', 'bias')].numpy().astype(np.float32)
            arrs[f'm{m}/bias'] = sd['bias'].numpy().astype(np.float32)
        else:
            lin = [k for k in sd if k.endswith('.weight')]
            for i, k in enumerate(sorted(lin, key=lambda s: int(s.split('.')[1]))):
                arrs[f'm{m}/net.{i}.w'] = sd[k].numpy().astype(np.float32)
                arrs[f'm{m}/net.{i}.b'] = sd[k.replace('weight', 'bias')].numpy().astype(np.float32)
    meta = dict(arch=cfg['arch'], anchor=cfg['anchor'], p_latent=cfg.get('P', 0), n_members=len(nets),
                widen=list(map(float, widen)), cfg=cfg)
    np.savez(path, meta=np.array(json.dumps(meta)), **arrs)


def train_member(cfg, D, seed, epochs, log):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(cfg)
    anc = D['ANC'] if cfg['anchor'] else D['ANC0']
    F = torch.tensor(D['F'], dtype=torch.float32)
    T = torch.tensor(an.trunk_features(D['A']), dtype=torch.float32)
    Yn = torch.tensor((D['Y'] - anc) / an.OUT_SCALE, dtype=torch.float32)
    n = len(F); bs = min(512, max(32, n // 8))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = max(epochs, 6000 // max(1, n // bs))      # at least ~6000 optimiser steps
    steps = epochs * max(1, n // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    beta = 0.5; t0 = time.time(); k = 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            o = net(F[idx], T[idx]); mu, ls = o[:, :3], o[:, 3:].clamp(-7, 3)
            var = torch.exp(2 * ls)
            nll = 0.5 * (Yn[idx] - mu) ** 2 / var + ls
            loss = (nll * var.detach() ** beta).mean()         # beta-NLL (Seitzer et al. 2022)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); k += 1
        if ep % 50 == 0 or ep == epochs - 1:
            log(f'  seed {seed} epoch {ep} loss {loss.item():.4f} {time.time() - t0:.0f}s')
    return net


def sim_skill(model, D, dumb):
    """skill on the frozen sim set: mean over Cl, ln Cd, Cm of 1 - MAE/MAE(dumb constant = train mean)"""
    mu = []
    for i in range(0, len(D['F']), 4096):
        sl = slice(i, i + 4096)
        m, _ = model._forward(D['F'][sl], D['A'][sl], (model.anchors_arr(D, sl)))
        mu.append(m)
    mu = np.concatenate(mu)
    mae = np.mean(np.abs(mu - D['Y']), 0); mae_d = np.mean(np.abs(dumb[None, :] - D['Y']), 0)
    return float(np.mean(1 - mae / mae_d)), mae.tolist(), mae_d.tolist()


class Loaded(an.AirfoilNet):
    def anchors_arr(self, D, sl):
        return D['ANC'][sl] if self.meta.get('anchor', True) else D['ANC0'][sl]


def dev_metrics(model, split='dev'):
    units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
    splits = json.load(open(os.path.join(ROOT, 'splits.json')))
    z = np.load(os.path.join(WORK, 'shapes.npz'))
    pmap = {k: p for k, p in zip(z['tested_keys'], z['tested_p'])}
    def f(p, Re, al):
        r = model.predict(p, Re, al)
        return dict(cl=r['cl'], cd=r['cd'], cm=r['cm'], cl_sd=r['cl_sd'], lncd_sd=r['lncd_sd'], cm_sd=r['cm_sd'])
    return rev.evaluate(f, units, splits[split], pmap)


def main():
    rid = sys.argv[1]; cfg = RUNGS[rid]
    epochs = int(os.environ.get('EPOCHS', 300)); members = int(os.environ.get('MEMBERS', 3))
    od = os.path.join(WORK, 'runs', rid); os.makedirs(od, exist_ok=True)
    logf = open(os.path.join(od, 'train.log'), 'a')
    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    D = load_bank(cfg['banks'], ['train'], loose=cfg['loose'], max_sweeps=cfg.get('max_sweeps'), snapshot=cfg.get('snapshot'))
    if cfg.get('calib'):
        D = apply_calib(D)
    log(f'{rid} {cfg} train sweeps {D["n_sweeps"]} points {len(D["F"])}')
    nets = [train_member(cfg, D, 1000 + m, epochs, log) for m in range(members)]
    path = os.path.join(od, 'airfoilnet_weights.npz'); export(nets, cfg, path)
    model = Loaded(path)
    V = load_bank(['p1'], ['val'], loose=True)
    dumb = D['Y'].mean(0)
    sk, mae, mae_d = sim_skill(model, V, V['Y'].mean(0))
    Vc = apply_calib(V) if CALIB else None
    skc = sim_skill(model, Vc, Vc['Y'].mean(0))[0] if CALIB else None
    summ, per = dev_metrics(model)
    res = dict(rung=rid, cfg=cfg, n_train_sweeps=D['n_sweeps'], n_train_points=len(D['F']),
               frozen=FROZEN, n_val_sweeps=V['n_sweeps'], sim_skill=sk, sim_skill_p1c=skc, sim_mae=mae, sim_mae_dumb=mae_d,
               dev=summ, dev_per_airfoil=per, time=time.strftime('%Y-%m-%d %H:%M'))
    json.dump(res, open(os.path.join(od, 'result.json'), 'w'), indent=1, default=float)
    log(f'{rid} sim skill {sk:.4f} mae {np.round(mae, 4)} | dev ' +
        ' '.join(f'{k}={summ[k]:.4f}' for k in rev.KEYS) + f' cov {summ["coverage"]}')


if __name__ == '__main__':
    main()
