"""PlantDetective network: a small MLP on the feature vector.
Heads: technology (4-way softmax) and, for each regression target in pdcore.REG (transformed), a Gaussian mean and
log-sigma (heteroscedastic NLL; masked where the target is undefined).  Torch for training, numpy for inference.

usage: python model.py train <set_version> <run_name> [epochs]
"""
from common import *
import json, time
import numpy as np

NREG = 13
# v1.1 count skips: (head j in REG, feature index, gate) -- the mean of head j is predicted as a correction to the
# statistic counted in the window itself (pdcore.features), where the window has one: gate -1 = always,
# -2 = feature != -1 (sentinel for "no complete run"), else the index of a 0/1 flag feature.
COUNT_SKIPS = ((3, 87, -1),     # starts_yr: log1p(starts counted in the window, per year)
               (4, 71, 72),     # min_up: log 5th percentile of complete runs (>= 3 runs)
               (5, 70, -2),     # med_up: log median complete run (>= 1 run)
               (6, 85, 86))     # min_dn: log 5th percentile of complete off spells (>= 3)


def count_offsets(X, skips):
    """(N, NREG) additive offsets on the mean heads from the counted statistics."""
    X = np.asarray(X)
    off = np.zeros((len(X), NREG), np.float32)
    for j, fi, g in np.asarray(skips, int).reshape(-1, 3):
        gate = np.ones(len(X)) if g == -1 else (X[:, fi] != -1.0).astype(float) if g == -2 else X[:, g]
        off[:, j] += gate * X[:, fi]
    return off


def np_forward(W, X):
    """numpy inference. W: dict of arrays (mu, sd, l0w, l0b, ...). Returns (tech_prob, mean, sigma)."""
    h = (X - W['mu']) / W['sd']
    i = 0
    while f'l{i}w' in W:
        h = h @ W[f'l{i}w'].T + W[f'l{i}b']
        if f'l{i + 1}w' in W:
            h = np.maximum(h, 0) + 0.01 * np.minimum(h, 0)   # leaky ReLU
        i += 1
    if 'cskip' in W:                      # count skips (v1.1): mean += counted statistic of the window
        h = h.copy(); h[:, 4:4 + NREG] += count_offsets(X, W['cskip'])
    if 'skip' in W:                       # heat-rate skip: mean log HR_full += [EIA-923 present] * log(EIA-923 HR)
        ih, ihas = int(W['skip'][0]), int(W['skip'][1])
        h = h.copy(); h[:, 4 + 10] += X[:, ihas] * (X[:, ih] + np.log(10.0))
    logit = h[:, :4]
    p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    mean = h[:, 4:4 + NREG]
    sig = np.exp(np.clip(h[:, 4 + NREG:4 + 2 * NREG], -6, 4))
    return p, mean, sig


def train(set_version, name, epochs=40, hidden=(256, 256, 128), lr=2e-3, seed=0, extra=None, modes=None, skip=False,
          cskip=False):
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed)
    z = np.load(f'{DATA}/set_{set_version}.npz')
    X, Y, C, V = z['X'], z['Y'], z['C'], z['V']
    if extra is not None:                      # optional extra training rows (never validation)
        X = np.r_[X, extra['X']]; Y = np.r_[Y, extra['Y']]; C = np.r_[C, extra['C']]; V = np.r_[V, np.zeros(len(extra['X']), bool)]
    tr = ~V
    if modes is not None:                      # train only on some unit modes (e.g. GridSim MILP units: R0)
        tr &= np.isin(z['G'], modes)
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-3
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(np.nan_to_num(Y), dtype=torch.float32); Mt = torch.tensor(np.isfinite(Y), dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.long)
    nf = X.shape[1]; ih, ihas = nf - 18 + 13, nf - 18 + 14      # context block starts 18 from the end (pdcore.CTX_KEYS)
    SK = torch.tensor(X[:, ihas] * (X[:, ih] + np.log(10.0)) if skip else np.zeros(len(X)), dtype=torch.float32)
    OFF = np.zeros((len(X), 4 + 2 * NREG), np.float32)
    if cskip:
        OFF[:, 4:4 + NREG] = count_offsets(X, COUNT_SKIPS)
    OFF = torch.tensor(OFF)
    layers = []; d = X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LeakyReLU(0.01), nn.Dropout(0.1)]; d = h
    layers += [nn.Linear(d, 4 + 2 * NREG)]
    net = nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (int(tr.sum()) // 256 + 1))
    idx_tr = np.flatnonzero(tr); idx_va = np.flatnonzero(V)

    def loss_fn(out, y, m, c):
        ce = nn.functional.cross_entropy(out[:, :4], c)
        mean = out[:, 4:4 + NREG]; ls = out[:, 4 + NREG:].clamp(-6, 4)
        nll = (0.5 * ((y - mean) / ls.exp()) ** 2 + ls) * m
        return ce + nll.sum() / m.sum().clamp(min=1) * NREG / 4

    best = (1e9, None)
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx_tr); tl = 0
        for b in range(0, len(idx_tr), 256):
            j = idx_tr[b:b + 256]
            out = net(Xt[j]); out = out + torch.nn.functional.pad(SK[j, None], (14, out.shape[1] - 15)) + OFF[j]
            l = loss_fn(out, Yt[j], Mt[j], Ct[j])
            opt.zero_grad(); l.backward(); opt.step(); sched.step(); tl += l.item() * len(j)
        net.eval()
        with torch.no_grad():
            ov = net(Xt[idx_va]); ov = ov + torch.nn.functional.pad(SK[idx_va, None], (14, ov.shape[1] - 15)) + OFF[idx_va]
            lv = loss_fn(ov, Yt[idx_va], Mt[idx_va], Ct[idx_va]).item() if len(idx_va) else 0
        if lv < best[0]:
            best = (lv, {k: v.clone() for k, v in net.state_dict().items()})
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'ep {ep} train {tl / len(idx_tr):.3f} val {lv:.3f}', flush=True)
    net.load_state_dict(best[1])
    W = dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    if skip:
        W['skip'] = np.array([ih, ihas])
    if cskip:
        W['cskip'] = np.array(COUNT_SKIPS, np.int64)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    for i, m in enumerate(lin):
        W[f'l{i}w'] = m.weight.detach().numpy().astype(np.float32); W[f'l{i}b'] = m.bias.detach().numpy().astype(np.float32)
    os.makedirs(f'{DATA}/models', exist_ok=True)
    np.savez(f'{DATA}/models/{name}.npz', **W)
    # numpy == torch check
    with torch.no_grad():
        o = (net(Xt[idx_va[:200]]) + torch.nn.functional.pad(SK[idx_va[:200], None], (14, 4 + 2 * NREG - 15))
             + OFF[idx_va[:200]]).numpy()
    p, m_, s_ = np_forward(W, X[idx_va[:200]])
    print('numpy vs torch max diff', float(np.abs(m_ - o[:, 4:4 + NREG]).max()))
    return W


def load(name):
    return dict(np.load(f'{DATA}/models/{name}.npz'))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 40,
              modes=sys.argv[5].split(',') if len(sys.argv) > 5 and sys.argv[5] != '-' else None,
              skip='skip' in sys.argv[6:6 + 1], seed=int(sys.argv[7]) if len(sys.argv) > 7 else 0,
              cskip='cskip' in sys.argv[8:])


def ens_forward(Ws, X):
    """Ensemble: mean of member probabilities and means; variance = mean member variance + spread of member means."""
    outs = [np_forward(W, X) for W in Ws]
    p = np.mean([o[0] for o in outs], 0)
    m = np.mean([o[1] for o in outs], 0)
    v = np.mean([o[2] ** 2 for o in outs], 0) + np.var([o[1] for o in outs], 0)
    return p, m, np.sqrt(v)


def load_any(name):
    """A single model 'name' or an ensemble 'name' saved as name_s0, name_s1, ... (returns callable forward)."""
    import glob
    fs = sorted(glob.glob(f'{DATA}/models/{name}_s[0-9].npz'))
    if fs:
        Ws = [dict(np.load(f)) for f in fs]
        return lambda X: ens_forward(Ws, X)
    W = load(name)
    return lambda X: np_forward(W, X)
