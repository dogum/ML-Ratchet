"""Shared constants: AAMI EC57 class mapping, de Chazal 2004 inter-patient split, record I/O."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, wfdb
D = _P.WORK + '/data/'
CLASSES = ['N', 'S', 'V', 'F']
AAMI = {**{s: 0 for s in 'NLRej'}, **{s: 1 for s in 'aSAJ'}, **{s: 2 for s in 'VE'}, 'F': 3, **{s: 4 for s in '/fQ'}}  # 4 = Q (unknown / paced)
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
PACED = [102, 104, 107, 217]

def read(db, rec, ch=None):
    r = wfdb.rdrecord(D + f'{db}/{rec}', channels=ch); a = wfdb.rdann(D + f'{db}/{rec}', 'atr')
    return r.p_signal.astype(np.float32), r.fs, r.sig_name, np.asarray(a.sample), np.asarray(a.symbol)

def beats(sample, symbol):
    """keep beat annotations only (AAMI classes 0-4) -> positions, class ids, original symbols"""
    k = np.array([s in AAMI for s in symbol]); return sample[k], np.array([AAMI[s] for s in symbol[k]]), symbol[k]

def dev_records(seed=2026):
    """small real dev set, drawn by a fixed seed within strata of DS1: 2 records rich in S beats (>= 50), 2 rich in V beats
    (>= 100), 2 others. Never part of any exam."""
    rng = np.random.default_rng(seed); s_rich = [118, 201, 207, 209, 220, 223]; v_rich = [106, 116, 119, 203, 208, 215]
    a = list(rng.choice(s_rich, 2, replace=False)); b = list(rng.choice([r for r in v_rich if r not in a], 2, replace=False))
    rest = [r for r in DS1 if r not in a + b and r not in s_rich + v_rich]; c = list(rng.choice(rest, 2, replace=False))
    return sorted(int(r) for r in a + b + c)
