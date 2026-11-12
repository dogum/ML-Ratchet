"""Load BeatNet, predict, and score: frozen simulated set, real dev records, real exams (DS2, DS1 remainder, SVDB)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, glob, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M, train as TR, common as C
D = _P.WORK + '/data/'; B = _P.WORK + '/bank/'
import os
VAL = B + os.environ.get('BEAT_VAL', 'v1val') + '_*.npz'      # frozen simulated set, versioned with the physics (v1val, v2val, v3val: seeds 900000-900399)

def load_net(path):
    ck = torch.load(path, weights_only=False); net = M.BeatNet(cin=ck['cin'], nout=ck.get('nout', 6)); net.load_state_dict(ck['state']); net.eval(); net.cin = ck['cin']; return net

def predict(net, Dd, idx=None, bs=512):
    idx = np.arange(len(Dd['y'])) if idx is None else idx; out = []
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            x, r = TR.inputs(Dd, idx[k:k + bs], net.cin); out.append(net(torch.from_numpy(x), torch.from_numpy(r)).numpy())
    o = np.concatenate(out); return o

def probs(o, logprior=None):
    z = o[:, :4] + (0 if logprior is None else np.asarray(logprior)); z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)

def scores(y, pred, ncls=4):
    m = y < ncls; y, pred = y[m], pred[m]; cm = np.zeros((ncls, ncls), int); np.add.at(cm, (y, pred), 1)
    se = np.diag(cm) / np.maximum(cm.sum(1), 1); ppv = np.diag(cm) / np.maximum(cm.sum(0), 1); f1 = 2 * se * ppv / np.maximum(se + ppv, 1e-9)
    return dict(n=int(m.sum()), cm=cm.tolist(), se=se.tolist(), ppv=ppv.tolist(), f1=f1.tolist(), acc=float(np.trace(cm) / max(cm.sum(), 1)),
                macro_f1=float(np.mean(f1[cm.sum(1) > 0])), sv_f1=float(np.mean(f1[1:3])))

def af_scores(t, logit, thr=0.0):
    """t: 1 = beat inside atrial fibrillation, 0 = not (flutter beats, t = 2, are left out); logit: AF output"""
    m = t != 2; t, z = t[m] == 1, logit[m]; p = z > thr; tp = (p & t).sum()
    r = np.argsort(z); rk = np.empty(len(z)); rk[r] = np.arange(1, len(z) + 1); npos = t.sum(); auc = (rk[t].sum() - npos * (npos + 1) / 2) / max(npos * (len(t) - npos), 1)
    return dict(n=int(len(t)), n_af=int(npos), se=float(tp / max(npos, 1)), ppv=float(tp / max(p.sum(), 1)), spec=float((~p & ~t).sum() / max((~t).sum(), 1)), auc=float(auc))

def val(net):
    Z = TR.load(VAL); o = predict(net, Z); s = scores(Z['y'], o[:, :4].argmax(1)); s['objective'] = s['macro_f1']
    if o.shape[1] > 6: s['af'] = af_scores(Z['info'][:, 1].astype(int), o[:, 6])
    q = np.exp(o[:, 4] * TR.QSD + TR.QMU); s['qrs_mae_ms'] = float(np.mean(np.abs(q - Z['qrs']))); return s

_R = {}
def real_bank(db='mitdb'):
    if db not in _R:
        z = np.load(D + f'{db}_beats.npz'); _R[db] = {k: z[k] for k in z.files}; _R[db]['qrs'] = np.full(len(z['y']), 100.0, np.float32)
    return _R[db]

def split_idx(which):
    if which == 'svdb': return 'svdb', np.arange(len(real_bank('svdb')['y']))
    z = real_bank('mitdb'); dev = C.dev_records()
    recs = {'dev': dev, 'ds1rest': [r for r in C.DS1 if r not in dev], 'ds2': C.DS2}[which]; return 'mitdb', np.nonzero(np.isin(z['rec'], recs))[0]

def rhythm(db):
    if 'rh_' + db not in _R: _R['rh_' + db] = np.load(D + f'{db}_rhythm.npy').astype(int)
    return _R['rh_' + db]

def real(net, which, logprior=None):
    db, idx = split_idx(which); Z = real_bank(db); o = predict(net, Z, idx); p = probs(o, logprior); s = scores(Z['y'][idx], p.argmax(1))
    if o.shape[1] > 6: s['af'] = af_scores(rhythm(db)[idx], o[:, 6])
    return s, idx, o
