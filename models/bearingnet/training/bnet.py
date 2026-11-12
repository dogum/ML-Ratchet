"""Evaluation helpers: frozen simulated validation objective, and real dev / exam reading-level metrics."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M, bn_simbank as SB, realdata as RD
CLS = ('N', 'OR', 'IR', 'B'); VAL = 'val1'

def load_net(path):
    ck = torch.load(path, weights_only=False); n = M.ARCH[ck['args'].get('arch', 'base')](); n.load_state_dict(ck['state']); n.eval(); return n

@torch.no_grad()
def raw(net, X, S, bs=512):
    return np.concatenate([net(torch.from_numpy(X[i:i + bs].astype(np.float32)), torch.from_numpy(S[i:i + bs])).numpy() for i in range(0, len(X), bs)])

def decode(o):
    lp = o[:, :4] - np.log(np.exp(o[:, :4] - o[:, :4].max(1, keepdims=True)).sum(1, keepdims=True)) - o[:, :4].max(1, keepdims=True)
    return dict(logp=lp, sp=o[:, 4] * M.SP_SCALE, sp_sig=np.exp(np.clip(o[:, 5], -5, 3)) * M.SP_SCALE, lsize=o[:, 6] * M.SZ_SD + M.SZ_MU, lsize_sig=np.exp(np.clip(o[:, 7], -5, 3)) * M.SZ_SD)

def bal_acc(y, p, k=4):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(k) if np.any(y == c)]))

def val(net, name=VAL):
    D = SB.load(name); d = decode(raw(net, D['x'], D['s'])); y = D['cls']; p = d['logp'].argmax(1)
    ba = bal_acc(y, p); sp_mae = np.mean(np.abs(d['sp'] - D['lsp'])); sp_skill = max(0, 1 - sp_mae / np.mean(np.abs(D['lsp'] - np.median(D['lsp']))))
    f = y > 0; sz_mae = np.mean(np.abs(d['lsize'][f] - D['lsize'][f])); sz_skill = max(0, 1 - sz_mae / np.mean(np.abs(D['lsize'][f] - np.median(D['lsize'][f]))))
    snr = D['snr']; by_snr = {}
    for lo, hi in (((-30, -18), (-18, -10), (-10, 1)) if name == 'val1' else ((-18, -6), (-6, 6), (6, 27))):
        m = f & (snr >= lo) & (snr < hi); by_snr['%d..%d dB' % (lo, hi)] = float(np.mean(p[m] == y[m]))
    return dict(objective=0.8 * ba + 0.1 * sp_skill + 0.1 * sz_skill, bal_acc=ba, acc_N=float(np.mean(p[y == 0] == 0)), sp_skill=sp_skill, sz_skill=sz_skill,
                sp_mae_pct=float(sp_mae * np.log(2) * 100), sz_mae_dex=float(sz_mae), fault_acc_by_snr=by_snr,
                conf=np.array([[int(np.sum((y == i) & (p == j))) for j in range(4)] for i in range(4)]).tolist())

def readings(net, part, F=None):
    """reading-level predictions: window log-probs averaged; speed and size: precision-weighted means of windows"""
    F = F or {}; Z = np.load(RD.D + 'real_%s.npz' % part); meta = json.load(open(RD.D + 'real_%s.json' % part)); o = raw(net, Z['x'], Z['s']); o[:, :4] /= F.get('temp', 1.0)
    d = decode(o); out = []
    for r in meta:
        sl = slice(r['win0'], r['win0'] + r['n_win']); lp = d['logp'][sl].mean(0); lp[0] += F.get('nbias', 0.0); lp = lp - np.log(np.exp(lp - lp.max()).sum()) - lp.max()
        w = 1 / d['sp_sig'][sl] ** 2; sp = float((d['sp'][sl] * w).sum() / w.sum()); sps = float(np.sqrt(1 / w.sum() * r['n_win']) * F.get('speed', 1))
        w2 = 1 / d['lsize_sig'][sl] ** 2; lz = float((d['lsize'][sl] * w2).sum() / w2.sum()); lzs = float(np.sqrt(1 / w2.sum() * r['n_win']) * F.get('size', 1))
        out.append(dict(r, p=np.exp(lp).tolist(), pred=CLS[int(lp.argmax())], f_pred=r['f_nom'] * 2 ** sp, f_sig_rel=sps * np.log(2), lsize_pred=lz, lsize_sig=lzs))
    return out

def summary(R, sel=lambda r: r['kind'] == 'main' and r['cls'] in CLS):
    R = [r for r in R if sel(r)]
    if not R: return {}
    y = np.array([CLS.index(r['cls']) for r in R]); p = np.array([CLS.index(r['pred']) for r in R])
    sp_err = np.array([abs(r['f_pred'] / r['f_true'] - 1) for r in R]); nom_err = np.array([abs(r['f_nom'] / r['f_true'] - 1) for r in R])
    return dict(n=len(R), acc=float(np.mean(y == p)), bal_acc=bal_acc(y, p), speed_mae_pct=float(100 * sp_err.mean()), speed_nominal_mae_pct=float(100 * nom_err.mean()),
                nll=float(-np.mean([np.log(max(r['p'][CLS.index(r['cls'])], 1e-6)) for r in R])))

def dev_score(net):
    R = readings(net, 'dev'); return dict(all=summary(R), cwru_near=summary(R, lambda r: r['kind'] == 'main' and r['src'] == 'CWRU' and r['loc'] == 'near'),
                                          pb=summary(R, lambda r: r['src'] == 'PB'))
