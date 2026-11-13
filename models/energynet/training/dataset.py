"""Bank / real records -> network inputs, with spec masking ("tiers").

A record holds what could be known (route vector R, physics integrals PH for 4 road-load parameter sets, vehicle
fields, labels, start SOC, thermal state). A tier says what IS known for this example. Training draws tiers at random
every epoch, from EPA-like full specs (exact mass + road-load ABC + engine + transmission) down to class + architecture.
Real exams use the tier of what the data actually provides.
"""
import _paths as _P  # portable paths (see _paths.py)
import glob
import numpy as np
import features as F

TIER_KEYS = ['m_level', 'rl', 'disp', 'power', 'trans', 'ngears', 'cap', 'labf', 'labe', 'soc', 'thermal', 'maf', 'lab_noise']


def load_bank(name):
    fs = sorted(glob.glob(f'{_P.WORK}/data/bank_{name}/chunk_*.npz'))
    parts = [dict(np.load(f)) for f in fs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def draw_tiers(B, rng):
    n = len(B['arch'])
    arch = B['arch']; cls = B['cls']
    hd = np.isin(cls, [4, 5, 6]); bev = arch == 6; conv = arch == 0; plug = np.isin(arch, [4, 5])
    u = rng.random(n)
    T = {}
    T['m_level'] = np.where(u < 0.35, 2, np.where(u < 0.80, 1, 0))
    T['rl'] = (T['m_level'] == 2) & (rng.random(n) < 0.6)
    T['disp'] = (~bev) & (rng.random(n) < 0.75)
    T['power'] = rng.random(n) < 0.5
    T['trans'] = rng.random(n) < 0.6
    T['ngears'] = T['trans'] & (rng.random(n) < 0.4)
    T['cap'] = (~conv) & (rng.random(n) < 0.5)
    haslab = ~hd
    T['labf'] = haslab & ~bev & (rng.random(n) < 0.65)
    T['labe'] = haslab & (bev | plug) & (rng.random(n) < 0.65)
    T['soc'] = (~conv) & (rng.random(n) < 0.6)
    T['thermal'] = rng.random(n) < 0.6
    T['maf'] = (~bev) & (B['dyno'] < 0.5) & (rng.random(n) < 0.5)
    T['lab_noise'] = rng.uniform(0.02, 0.12, n) * rng.normal(0, 1, n)
    return {k: np.asarray(v) for k, v in T.items()}


def target(B, T, version):
    tg = B['tg']
    maf = T['maf'].astype(bool)
    if version == 'v1': return tg[:, 0]
    if version == 'v2': return np.where(maf, tg[:, 2], tg[:, 0])
    if version == 'v3': return np.where(maf, tg[:, 3], tg[:, 1])
    raise ValueError(version)


def assemble_one(B, i, T):
    arch = F.ARCHS[int(B['arch'][i])]; cls = F.CLASSES[int(B['cls'][i])]
    hd = cls in F.HD
    mc, Cc, Rc, cvc, rot = F.CLS[cls]
    ml = int(T['m_level'][i]); rl = bool(T['rl'][i]) and ml == 2
    if ml == 2: m = float(B['m_true'][i]); var = 0 if rl else 1
    elif ml == 1: m = float(B['m_rep'][i]); var = 2
    else: m = mc; var = 3
    if not np.isfinite(m): m, ml, var, rl = mc, 0, 3, False
    ph = B['PH'][i, var].astype(float)
    soc = float(B['soc'][i]) if (T['soc'][i] and np.isfinite(B['soc'][i])) else None
    cap = float(B['cap'][i]) if (T['cap'][i] and np.isfinite(B['cap'][i])) else None
    dur = float(B['dur'][i]); dist = float(B['dist'][i])
    E, Eh, Eb, f = F.e_phys(ph, arch, m, dur, soc, cap, hd)
    if rl: cda, crr = float(B['cda'][i]), float(B['crr'][i])
    else: cda, crr = Cc * (m / mc) ** 0.5, Rc
    ln = float(np.exp(T['lab_noise'][i]))
    lab = B['lab'][i]
    labf = (lab[0] * ln, lab[1] * ln) if (T['labf'][i] and np.isfinite(lab[0])) else None
    labe = (lab[2] * ln, lab[3] * ln) if (T['labe'][i] and np.isfinite(lab[2])) else None
    th = bool(T['thermal'][i])
    tr = int(B['trans'][i])
    V = F.vehicle_vec(arch, cls, m, ml, cda, crr, rl,
                      float(B['disp'][i]) if (T['disp'][i] and np.isfinite(B['disp'][i])) else None,
                      float(B['power'][i]) if (T['power'][i] and np.isfinite(B['power'][i])) else None,
                      B['diesel'][i] > 0.5, F.TRANS[tr] if (T['trans'][i] and tr >= 0) else None,
                      float(B['ngears'][i]) if (T['ngears'][i] and np.isfinite(B['ngears'][i])) else None,
                      cap, labf, labe, soc, th and B['warm'][i] > 0.5, th and B['cold'][i] > 0.5,
                      bool(T['maf'][i]), B['dyno'][i] > 0.5, (E, dist / 1000.0), dist / max(dur, 1.0))
    Fv = F.phys_vec(ph, E, Eh, Eb, f, m, dist)
    return np.concatenate([B['R'][i].astype(float), Fv, V]), E


def assemble(B, T):
    n = len(B['arch'])
    X = np.zeros((n, F.NR + F.NF + F.NV), np.float32); E = np.zeros(n)
    for i in range(n):
        X[i], E[i] = assemble_one(B, i, T)
    return X, E
