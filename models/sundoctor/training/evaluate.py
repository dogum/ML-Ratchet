"""Frozen validation set (simulated, seed 777, never trained on) and the metrics the ratchet judges by."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np, torch
sys.path.insert(0, _P.SRC)
from model import SunDoctor, REG, SCALE, GLOB, GSCALE
import simbatch as SB, train as T
VAL = _P.WORK + '/val_set.npz'

def val_set(n=2000, D=14):
    if os.path.exists(VAL):
        z = np.load(VAL); return z['X'], z['Y'], z['G'], {k[2:]: z[k] for k in z.files if k.startswith('m_')}
    rng = np.random.default_rng(777); Xs, Ys, Gs, Ms = [], [], [], []
    for _ in range(n // 100):
        X, Y, G, m = T.batch(rng, 100, D); Xs.append(X.astype(np.float16)); Ys.append(Y); Gs.append(G); Ms.append(m)
    M = {k: np.concatenate([m[k] for m in Ms]) for k in ('n_strings', 'kind', 'rate', 'sat')}
    np.savez_compressed(VAL, X=np.concatenate(Xs), Y=np.concatenate(Ys), G=np.concatenate(Gs), **{'m_' + k: v for k, v in M.items()})
    return val_set(n, D)

def load(path):
    ck = torch.load(path, weights_only=False); net = SunDoctor(tuple(ck['widths']), ck['tw']); net.load_state_dict(ck['ema']); net.eval(); net.days = ck.get('days', 14); return net

@torch.no_grad()
def predict(net, X, bs=100):
    D, G = [], []
    for i in range(0, len(X), bs):
        d, g = net(torch.from_numpy(np.asarray(X[i:i + bs], np.float32))); D.append(d.numpy()); G.append(g.numpy())
    D = np.concatenate(D); G = np.concatenate(G); sc = np.array(SCALE)[None, :, None]
    out = {k: D[:, i] * SCALE[i] for i, k in enumerate(REG)}
    out.update({k + '_sig': np.exp(np.clip(D[:, 7 + i], -6, 3)) * SCALE[i] for i, k in enumerate(REG)})
    out['step_p'] = 1 / (1 + np.exp(-D[:, 14]))
    for i, k in enumerate(GLOB): out[k] = G[:, i] * GSCALE[i]; out[k + '_sig'] = np.exp(np.clip(G[:, 2 + i], -6, 3)) * GSCALE[i]
    return out

def auc(score, y):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))

def evaluate(net):
    X, Y, G, M = val_set(); P = predict(net, X); res = {}
    valid_day = X[:, 7].astype(np.float32).mean(-1) > 0.2           # days that actually have data
    skills = []
    for i, k in enumerate(SB.DAY_LABELS):
        if k not in REG: continue
        y = Y[:, i][valid_day]; p = P[k][valid_day]; s = P[k + '_sig'][valid_day]
        mae = float(np.mean(np.abs(p - y))); base = float(np.mean(np.abs(y - np.median(y))))
        res[k] = {'mae': mae, 'baseline_mae': base, 'skill': 1 - mae / base, 'cov68': float(np.mean(np.abs(p - y) < s))}
        if k != 'dfrac': skills.append(1 - mae / base)
    st = Y[:, SB.DAY_LABELS.index('step')][valid_day]; res['step_auc'] = auc(P['step_p'][valid_day], st)
    # string counting: how many of the N strings are dead, on days with known N > 1
    N = M['n_strings'][:, None].repeat(Y.shape[2], 1)[valid_day]; fr = Y[:, SB.DAY_LABELS.index('frac')][valid_day]
    m = N > 1; k_true = np.round(N[m] * (1 - fr[m])); k_pred = np.round(N[m] * (1 - np.clip(P['frac'][valid_day][m], 0, 1)))
    res['count_acc'] = float(np.mean(k_true == k_pred)); res['count_acc_dead'] = float(np.mean((k_true == k_pred)[k_true > 0]))
    res['dead_auc'] = auc(-P['frac'][valid_day], (fr < 0.999).astype(float))
    for i, k in enumerate(GLOB):
        res[k] = {'mae': float(np.mean(np.abs(P[k] - G[:, i]))), 'cov68': float(np.mean(np.abs(P[k] - G[:, i]) < P[k + '_sig']))}
    res['objective'] = float(np.mean(skills) + 0.5 * res['step_auc'] + 0.5 * res['count_acc'])
    return res

if __name__ == '__main__':
    r = evaluate(load(sys.argv[1])); print(json.dumps(r, indent=1))
