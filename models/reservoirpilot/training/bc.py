"""Behaviour cloning of the perfect-foresight DP (B's dp) on simulated tasks.
Labels: DP release along the DP's own storage path, on bank tasks (never real dams).
  python3 bc.py label g1 500         -> data/bc_g1.npz
  python3 bc.py fit NAME g1 base     -> runs/NAME/weights.npz
"""
import os, sys, time, json
import numpy as np, torch
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from task import dp                                          # noqa: E402
from policy import features, FEATS, torch_policy, export_weights, EMA_TAUS  # noqa: E402

torch.set_num_threads(1)
ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)
K = 0.0864


def label(gen, n):
    z = np.load(f'{ROOT}/data/bank_{gen}.npz'); rng = np.random.default_rng(77)
    idx = rng.choice(len(z['q']), n, replace=False)
    keep = {k: [] for k in ('S', 'I', 'e0', 'e1', 'e2', 'D', 'Dn30', 'Dn90', 'Ic30', 'Ic90', 'q', 'cap', 'R')}
    t0 = time.time()
    for j, i in enumerate(idx):
        I = z['I'][i].astype(float); doy = z['doy'][i].astype(int) - 1; C = z['C'][i].astype(float)
        cal = {k: C[m][doy] for m, k in enumerate(['D', 'Dn30', 'Dn90', 'Ic30', 'Ic90'])}
        q, cap, S0 = float(z['q'][i]), float(z['cap'][i]), float(z['S0'][i])
        R = dp(I, cal['D'], q, cap, S0)
        S = np.empty(len(I)); s = S0
        for t in range(len(I)):                          # state BEFORE day t's release
            S[t] = s; s = min(max(s + (I[t] - R[t]) * K, 0.0), cap)
        emas = []; e = [I[0]] * 3; E = np.zeros((len(I), 3))
        for t in range(len(I)):
            e = [x + (I[t] - x) / tau for x, tau in zip(e, EMA_TAUS)]; E[t] = e
        sel = np.arange(rng.integers(3), len(I), 3)
        for k, v in (('S', S), ('I', I), ('e0', E[:, 0]), ('e1', E[:, 1]), ('e2', E[:, 2]), ('R', R)):
            keep[k].append(v[sel])
        for k in ('D', 'Dn30', 'Dn90', 'Ic30', 'Ic90'):
            keep[k].append(cal[k][sel])
        keep['q'].append(np.full(len(sel), q)); keep['cap'].append(np.full(len(sel), cap))
        if j % 50 == 0:
            print('label', j, round(time.time() - t0), flush=True)
    np.savez(f'{ROOT}/data/bc_{gen}.npz', **{k: np.concatenate(v) for k, v in keep.items()})


def fit(name, gen, fset, epochs=6, bs=2048):
    z = np.load(f'{ROOT}/data/bc_{gen}.npz'); d = {k: torch.tensor(z[k]) for k in z.files}
    f = torch.stack(features(torch, d['S'], d['I'], [d['e0'], d['e1'], d['e2']], d['D'], d['Dn30'], d['Dn90'],
                             d['Ic30'], d['Ic90'], d['q'], d['cap'], fset), -1)
    ok = torch.isfinite(f).all(1); f = f[ok]; D, q, R = d['D'][ok], d['q'][ok], d['R'][ok]
    mu, sd = f.mean(0), f.std(0).clamp_min(1e-3)
    net = torch_policy(FEATS[fset]).double(); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    n = len(f); g = torch.Generator().manual_seed(0); out = f'{ROOT}/runs/{name}'; os.makedirs(out, exist_ok=True)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g); tot = 0
        for b in range(0, n, bs):
            i = perm[b:b + bs]
            o = net((f[i] - mu) / sd); h = torch.sigmoid(o[:, 0]); e = torch.sigmoid(o[:, 1])
            Rp = h * D[i] + 1.25 * e * torch.relu(q[i] - D[i])
            loss = (((Rp - R[i]) / torch.maximum(D[i], 0.05 * q[i])) ** 2).clamp_max(25).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(i)
        print('epoch', ep, tot / n, flush=True)
    w = export_weights(net, mu, sd); np.savez(f'{out}/weights.npz', **w, fset=fset)
    json.dump({'method': 'bc_dp', 'gen': gen, 'fset': fset, 'n': int(n)}, open(f'{out}/args.json', 'w'))


if __name__ == '__main__':
    if sys.argv[1] == 'label':
        label(sys.argv[2], int(sys.argv[3]))
    else:
        fit(sys.argv[2], sys.argv[3], sys.argv[4])
