"""Frozen simulated validation sets, skill score, dev gate and run-level prediction helpers."""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, CLASSES, NC, load_real_runs  # noqa: E402
from features import featurize_batch  # noqa: E402
from noise import Noise  # noqa: E402

FROZEN = f'{ROOT}/data/frozen'
EFF_MIN = 1.0


def noise_for(version):
    """Simulator+noise versions. v1: BuildingSim + Gaussian sensor noise + source channel patterns.
    v2: + real-noise/quirk model (noise_v1.npz). v3: v2 on the dev-calibrated bank (bank suffix).
    v4: v3 + installed damper characteristic (logged command vs OA fraction) + larger static sensor offsets."""
    if version == 'v1':
        return Noise(f'{ROOT}/data/noise_v1.npz', residual=False, offsets=False, fanoff=False)
    if version in ('v4', 'v4n'):
        return Noise(f'{ROOT}/data/noise_v1.npz', damper_char=True, big_offsets=True)
    return Noise(f'{ROOT}/data/noise_v1.npz')


def bank_for(version):
    return {'v1': 'val_v1', 'v2': 'val_v1', 'v3': 'val_v3', 'v4': 'val_v3', 'tst': 'tst'}[version]


def load_val(version, n_max=12000):
    os.makedirs(FROZEN, exist_ok=True)
    fn = f'{FROZEN}/val_{version}.npz'
    if os.path.exists(fn):
        return dict(np.load(fn, allow_pickle=True))
    b = dict(np.load(f'{ROOT}/data/banks/{bank_for(version)}.npz', allow_pickle=True))
    keep = (b['cls'] == 0) | (b['eff'] >= EFF_MIN)
    idx = np.where(keep)[0]
    rng = np.random.default_rng(777)
    if len(idx) > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    nz = noise_for(version)
    xs = [nz.apply(b['x'][i].astype(np.float32), rng, str(b['system'][i])) for i in idx]
    F = featurize_batch(xs).astype(np.float16)
    out = dict(F=F, cls=b['cls'][idx], par=b['par'][idx], waste=b['waste'][idx], eff=b['eff'][idx],
               system=b['system'][idx], month=b['month'][idx], ftype=b['ftype'][idx])
    np.savez_compressed(fn, **out)
    return out


@torch.no_grad()
def predict(model, F, bs=512):
    model.eval()
    P, S, W = [], [], []
    for i in range(0, len(F), bs):
        lo, pa, wa = model(torch.as_tensor(np.asarray(F[i:i + bs], np.float32)))
        P.append(lo.numpy())
        S.append(pa.numpy())
        W.append(wa.numpy())
    return np.concatenate(P), np.concatenate(S), np.concatenate(W)


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def det_logit(logits):
    """log odds of 'any fault' = logsumexp(fault logits) - logit(none)."""
    f = logits[:, 1:]
    m = f.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(f - m).sum(1))) - logits[:, 0]


def auc(t, s):
    t = np.asarray(t, bool)
    s = np.asarray(s, float)
    pos, neg = s[t], s[~t]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.r_[pos, neg], kind='mergesort')
    ranks = np.empty(len(order))
    allv = np.r_[pos, neg][order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(t, s, fpr=0.02):
    t = np.asarray(t, bool)
    neg = np.sort(s[~t])
    if len(neg) == 0 or t.sum() == 0:
        return float('nan')
    k = int(np.floor(fpr * len(neg)))          # allow at most k false alarms
    thr = neg[len(neg) - k - 1] if k < len(neg) else -np.inf
    return float((s[t] > thr).mean())


def skill(model, val):
    lo, pa, wa = predict(model, val['F'])
    d = det_logit(lo)
    t = val['cls'] > 0
    A = auc(t, d)
    f = t & (val['eff'] >= EFF_MIN)
    diag = float((lo[f, 1:].argmax(1) + 1 == val['cls'][f]).mean())
    return dict(skill=0.5 * A + 0.5 * diag, auc=A, diag_acc=diag, recall_2pct=recall_at_fpr(t, d),
                waste_mae=float(np.mean(np.abs(np.clip(wa, -1, 3) - np.clip(val['waste'], -1, 3)))))


_DEV = None


def dev_runs():
    global _DEV
    if _DEV is None:
        _DEV = load_real_runs('dev')
    return _DEV


def run_predict(model, runs, T=1.0):
    from features import featurize
    F = np.stack([featurize(x) for _, x in runs])
    lo, pa, wa = predict(model, F)
    return lo, pa, wa


def dev_eval(model):
    runs = dev_runs()
    lo, pa, wa = run_predict(model, runs)
    d = det_logit(lo)
    t = np.array([r.fault_type != 'none' for r, _ in runs])
    p = 1 / (1 + np.exp(-d))
    return dict(dev_auc=auc(t, d), dev_fpr_at_0p5=float((p[~t] > 0.5).mean()), dev_tpr_at_0p5=float((p[t] > 0.5).mean()),
                dev_p={r.run_id: round(float(q), 3) for (r, _), q in zip(runs, p)})
