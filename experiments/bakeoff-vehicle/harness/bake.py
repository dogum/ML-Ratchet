"""Vehicle Bake-off common data API: identical inputs and targets for every arm, two tracks.

    import sys; sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness'); import bake

TRACKS
  'table'  EnergyNet v2.1's own 151 inputs (models/energynet/features.py, assembled by training/dataset.py): route
           vector R (90), physics vector (12, incl. log E_phys), vehicle vector V (49) with the tier's unknowns masked.
  'trace'  the observed trip at 1 Hz, 8 channels config.TRACE_CH = v [m/s], grade [-], elev - elev[0] [m], T_amb [C],
           wind along the heading [m/s, + = headwind], crosswind [m/s], rho_air [kg/m3], surface code [0..4]
           (features.prep's values: the same numbers EnergyNet's features are computed from), plus
           spec (49): EnergyNet's vehicle vector V, except the two window-sticker label entries 35 and 38, which are
                log(label energy per km, MJ/km) instead of log(label energy over the trip / E_phys);
           rl (n, 5): the road-load numbers the physics estimate uses at this tier: mass kg, CdA m2, Crr, Crr speed
                coefficient, rotating-mass fraction (class priors where the spec is unknown).
           E_phys is NOT given to the trace track; bake.e_phys_from_trace(batch) recomputes it from these inputs
           (allowed feature engineering; say so in your card if you use it).

SIMULATED BANK (frozen; build_bank.py: EnergyNet v2.1's recipe on RoadSim physics v2)
    B = bake.load_bank('train')            # 28,800 trips (2,400 vehicles x 12); 'val' = the frozen 4,000 (400 x 10)
    T = bake.draw_tiers(B, rng)            # EnergyNet's spec masking; draw new tiers every epoch (EnergyNet does)
    T = bake.val_tiers()                   # the frozen val tiers (seed 777, EnergyNet's)
    D = bake.sim_table(B, T)               # X (n,151) float32, E_phys, y (target MJ), arch, cls, ...
    S = bake.sim_spec(B, T)                # spec (n,49), rl (n,5) for the trace track
    tr = bake.sim_trace(B, i)              # (L, 8) float32 trace of trip i (memory-mapped bank)
    for b, y in bake.simval_batches('table'|'trace'): ...    # the frozen sim-val protocol (3,985 trips, y > 0.02 MJ)
    labels: B['tg'] (8 target versions), B['comp'] (9 energy terms, MJ), B['m_true'], B['cda'], ... ; bake.fleet(B)

REAL DATA (dev only; the exam is guarded)
    for b in bake.dev_batches('table'|'trace'): ...           # VED / EPA / HD dev rows, in record order, <= 256 a batch
    bake.dev_frame()                                           # the dev records' meta (for scoring only)

A batch (dict) is what predict(batch) receives:
    uid        (n,) opaque row ids (stable; cache on them, never look anything up with them)
    trace_key  (n,) opaque id of the trace content (rows with the same trace share it: EPA tests repeat 5 cycles)
    source     'sim' | 'real';  period 'simval' | 'simtrain' | 'dev' | 'exam';  track 'table' | 'trace'
    n          number of rows;  arch, cls  (n,) int indices into features.ARCHS / CLASSES (also inside X / spec)
    table: X (n, 151) float32, E_phys (n,) MJ
    trace: traces (list of n float32 arrays (L_i, 8)), spec (n, 49) float32, rl (n, 5) float32
predict(batch) returns {'pred', 'q05', 'q25', 'q50', 'q75', 'q95'}: (n,) trip energy in MJ (fuel chemical energy at
0.042 MJ/g for conventional / hybrid, battery DC net discharge for BEV, the sum for plug-ins), finite and > 0.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import features as F  # noqa: E402   (vendored EnergyNet features.py, via config's sys.path)
import dataset as DS  # noqa: E402   (vendored EnergyNet training/dataset.py)

TARGET_VERSION = 'e'      # dataset.V2COLS['e'] on the h banks (EnergyNet v2.1's frozen-set version h)
Y_MIN = 0.02              # MJ; EnergyNet scores sim trips with y > 0.02 MJ
_BANK = {}


# ============================================================================ simulated bank
def bank_info(split):
    with open(os.path.join(C.BANK, split, 'MANIFEST.json')) as f:
        return json.load(f)


def load_bank(split):
    """split 'train' or 'val'. Row fields (EnergyNet's bank layout + extras) are loaded into memory (~25 MB for
    train); traces stay memory-mapped (trace_v float32, trace_x float16)."""
    if split not in ('train', 'val'):
        raise ValueError(split)
    if split in _BANK:
        return _BANK[split]
    d = os.path.join(C.BANK, split)
    if not os.path.exists(os.path.join(d, 'MANIFEST.json')):
        raise FileNotFoundError(f'bank/{split} is not finalized yet (see harness/STATUS.md)')
    B = dict(np.load(os.path.join(d, 'rows.npz')))
    B['trace_v'] = np.load(os.path.join(d, 'trace_v.npy'), mmap_mode='r')
    B['trace_x'] = np.load(os.path.join(d, 'trace_x.npy'), mmap_mode='r')
    B['_split'] = split
    _BANK[split] = B
    return B


def fleet(B):
    """The bank's vehicles: list of {'spec', 'labels', 'model_year', 'idle_off', ['spec_sim']}; row i is vehicle
    B['vid'][i] -> index (vid // 100) * 50 + vid % 100. spec_sim (if present) is what RoadSim drove (idle-stop off)."""
    with open(os.path.join(C.BANK, B['_split'], 'fleet.json')) as f:
        return json.load(f)


def vehicle_index(B):
    v = B['vid'].astype(int)
    return (v // 100) * 50 + v % 100


def draw_tiers(B, rng):
    """EnergyNet's tier draw (training/dataset.py draw_tiers): what is known about the vehicle for each trip."""
    return DS.draw_tiers(_rows(B), rng)


def val_tiers():
    return draw_tiers(load_bank('val'), np.random.default_rng(C.VAL_TIER_SEED))


def _rows(B):
    return {k: v for k, v in B.items() if not k.startswith(('trace_', '_'))}


def target(B, T):
    """Training / scoring target: trip energy in MJ, EnergyNet v2.1's version (OBD MAF label model where the tier says
    the fuel is a MAF estimate; v1's plug-in hardware patch for plug-ins)."""
    return DS.target(B, T, TARGET_VERSION)


def sim_table(B, T, idx=None):
    """-> dict X (n,151) float32, E_phys (n,), y (n,) MJ, arch, cls, dist, dur."""
    R = _rows(B)
    if idx is not None:
        idx = np.asarray(idx)
        R = {k: v[idx] for k, v in R.items()}
        T = {k: np.asarray(v)[idx] for k, v in T.items()}
    X, E = DS.assemble(R, T)
    return dict(X=X, E_phys=E, y=DS.target(R, T, TARGET_VERSION).astype(np.float64), arch=R['arch'].astype(int),
                cls=R['cls'].astype(int), dist=R['dist'], dur=R['dur'])


def _rl_used(R, T):
    """road-load numbers dataset.assemble_one uses for E_phys at this tier: (m, CdA, Crr, cv, rot) and the variant."""
    n = len(R['arch'])
    out = np.zeros((n, 5)); var = np.zeros(n, int)
    for i in range(n):
        cls = F.CLASSES[int(R['cls'][i])]
        mc, Cc, Rc, cvc, rot = F.CLS[cls]
        ml = int(T['m_level'][i]); rl = bool(T['rl'][i]) and ml == 2
        if ml == 2: m = float(R['m_true'][i]); v = 0 if rl else 1
        elif ml == 1: m = float(R['m_rep'][i]); v = 2
        else: m = mc; v = 3
        if not np.isfinite(m): m, v, rl = mc, 3, False
        if rl:
            out[i] = (m, float(R['cda'][i]), float(R['crr'][i]), float(R['cv'][i]), rot)
        else:
            out[i] = (m, Cc * (m / mc) ** 0.5, Rc, cvc, rot)
        var[i] = v
    return out, var


def spec_from_X(X, E_phys, dist):
    """trace-track spec vector from the table's V block: label entries made absolute (log MJ/km)."""
    V = np.array(X[:, F.NR + F.NF:], np.float64)
    adj = np.log(np.maximum(E_phys, 1e-9) / np.maximum(dist / 1000.0, 1e-6))
    for j in (35, 38):
        has = V[:, j + 2] > 0.5
        V[has, j] = V[has, j] + adj[has]
    return V.astype(np.float32)


def sim_spec(B, T, idx=None):
    """-> dict spec (n,49), rl (n,5), arch, cls (the trace track's non-trace inputs)."""
    D = sim_table(B, T, idx)
    R = _rows(B)
    if idx is not None:
        R = {k: v[np.asarray(idx)] for k, v in R.items()}
        T = {k: np.asarray(v)[np.asarray(idx)] for k, v in T.items()}
    rl, _ = _rl_used(R, T)
    return dict(spec=spec_from_X(D['X'], D['E_phys'], D['dist']), rl=rl.astype(np.float32), arch=D['arch'], cls=D['cls'])


def sim_trace(B, i):
    a, b = int(B['trace_off'][i]), int(B['trace_off'][i + 1])
    return np.concatenate([np.asarray(B['trace_v'][a:b], np.float32)[:, None],
                           np.asarray(B['trace_x'][a:b], np.float32)], 1)


def _uid(*parts):
    return hashlib.sha1(('vehicle-bakeoff:' + ':'.join(map(str, parts))).encode()).hexdigest()[:12]


def sim_batch(B, T, idx, track, period=None):
    idx = np.asarray(idx)
    split = B['_split']
    b = {'uid': np.array([_uid('sim', split, i) for i in idx]), 'trace_key': np.array([_uid('simtr', split, i) for i in idx]),
         'source': 'sim', 'period': period or ('simval' if split == 'val' else 'simtrain'), 'track': track, 'n': len(idx)}
    D = sim_table(B, T, idx)
    b['arch'] = D['arch']; b['cls'] = D['cls']
    if track == 'table':
        b['X'] = D['X']; b['E_phys'] = D['E_phys']
    elif track == 'trace':
        R = {k: v[idx] for k, v in _rows(B).items()}
        Ti = {k: np.asarray(v)[idx] for k, v in T.items()}
        rl, _ = _rl_used(R, Ti)
        b['spec'] = spec_from_X(D['X'], D['E_phys'], D['dist']); b['rl'] = rl.astype(np.float32)
        b['traces'] = [sim_trace(B, int(i)) for i in idx]
    else:
        raise ValueError(track)
    return b, D['y']


def simval_rows():
    """indices of the frozen sim-val rows (target > 0.02 MJ under the frozen tiers): 3,985 of 4,000."""
    B = load_bank('val'); T = val_tiers()
    return np.where(target(B, T) > Y_MIN)[0]


def simval_batches(track, chunk=None, nmax=None):
    """The frozen sim-val protocol: yields (batch, y MJ) over the 3,985 scored val trips with the frozen tiers."""
    B = load_bank('val'); T = val_tiers()
    idx = simval_rows()
    if nmax:
        idx = idx[:nmax]
    chunk = chunk or (1000 if track == 'table' else 128)
    for a in range(0, len(idx), chunk):
        yield sim_batch(B, T, idx[a:a + chunk], track)


def simval_meta():
    """per scored val trip: arch, cls, src, dyno, dist, dur, cycle, T (for breakdowns)."""
    B = load_bank('val'); idx = simval_rows()
    return pd.DataFrame({'row': idx, 'arch': B['arch'][idx].astype(int), 'cls': B['cls'][idx].astype(int),
                         'src': B['src'][idx].astype(int), 'dyno': B['dyno'][idx], 'dist': B['dist'][idx],
                         'dur': B['dur'][idx], 'T': B['R'][idx, 76] * 20.0})


def dumb_rates():
    """EnergyNet's dumb baseline (evaluate.dumb_rates2): median target energy per km per (arch, class) on the TRAIN
    bank with tiers drawn by default_rng(1)."""
    p = os.path.join(C.BANK, 'dumb_rates.json')
    if os.path.exists(p):
        return {int(k): v for k, v in json.load(open(p)).items()}
    B = load_bank('train'); T = draw_tiers(B, np.random.default_rng(1))
    y = target(B, T); k = B['arch'] * 10 + B['cls']
    ok = (y > Y_MIN) & (B['dist'] > 100)
    r = {int(g): float(np.median(y[ok][k[ok] == g] / B['dist'][ok][k[ok] == g])) for g in np.unique(k[ok])}
    json.dump(r, open(p, 'w'), indent=1)
    return r


def e_phys_from_trace(batch):
    """EnergyNet's generic physics estimate E_phys (MJ) recomputed from a trace-track batch (trace + spec + rl):
    the same road-load integrals and constants as the table track (features.phys_ints / e_phys). Allowed feature
    engineering for trace arms (declare it). Equals the table's E_phys to float16-trace precision (~1e-3)."""
    out = np.zeros(batch['n'])
    for i, (tr, sp, rl) in enumerate(zip(batch['traces'], batch['spec'], batch['rl'])):
        P = prep_trace(tr)
        arch = F.ARCHS[int(batch['arch'][i])]; cls = F.CLASSES[int(batch['cls'][i])]
        ph = F.phys_ints(P, float(rl[0]), float(rl[1]), float(rl[2]), float(rl[3]), float(rl[4]), cls in F.HD)
        soc = float(sp[41]) if sp[42] > 0.5 else None
        cap = float(np.exp(sp[33]) * 10.0) if sp[34] > 0.5 else None
        out[i] = F.e_phys(ph, arch, float(rl[0]), float(P['N']), soc, cap, cls in F.HD)[0]
    return out


def prep_trace(tr):
    """trace (L, 8) -> the dict features.prep returns (v, a, grade, T, wh, wc, ws, rho, surf, N)."""
    tr = np.asarray(tr, np.float64)
    v = tr[:, 0]; N = len(v)
    return dict(v=v, a=np.gradient(v) if N > 1 else np.zeros(N), grade=tr[:, 1], T=tr[:, 3], wh=tr[:, 4], wc=tr[:, 5],
                ws=np.hypot(tr[:, 4], tr[:, 5]), rho=tr[:, 6], surf=np.clip(np.rint(tr[:, 7]), 0, 4).astype(int), N=N)


# ============================================================================ real data
def _splits():
    h = hashlib.sha256(open(C.SPLITS, 'rb').read()).hexdigest()
    if h != C.SPLITS_SHA256:
        raise RuntimeError(f'splits.json hash {h} != frozen {C.SPLITS_SHA256}')
    return json.load(open(C.SPLITS))


SETS = ('ved', 'epa', 'hd')
_REAL = {}


def _load_real(split, name):
    key = (split, name)
    if split == 'exam':
        _guard()
    if key not in _REAL:
        p = os.path.join(C.real_dir(split), name)
        B = dict(np.load(p + '.npz')); n = len(B['arch'])
        T = {k: B['T_' + k] if ('T_' + k) in B else np.zeros(n, bool) for k in DS.TIER_KEYS}
        M = pd.read_parquet(p + '_meta.parquet')
        M['split'] = split
        _REAL[key] = (B, T, M)
    return _REAL[key]


def _real_trace_raw(split, M, j):
    """the observed trip dict of real record j (meta row M.iloc[j])."""
    import realdata as rd
    import routes
    from bank import TR_KEYS, _fill
    src = M.trace_src.iloc[j]
    if src.startswith('VED_') or src.startswith('ZTBus_'):
        _, arr = rd.load_trip(src)
        return {k: arr[k] for k in TR_KEYS}
    if src.startswith('mersin:'):
        z = _mersin_traces(split)
        v = z[src[7:]]; N = len(v)
        tr = _fill({'v': v.astype(np.float32)}); tr['T_amb'] = np.full(N, float(M['T_test'].iloc[j]), np.float32)
        tr['rho_air'] = routes.rho_air(tr['T_amb'], np.zeros(N)).astype(np.float32)
        return tr
    name = src[6:] if src.startswith('cycle:') else src
    cyc = _cycles()
    tr = _fill({k: cyc[name][1][k] for k in TR_KEYS if k in cyc[name][1]}); N = len(tr['v'])
    T = float(M['T_test'].iloc[j])
    tr.update(T_amb=np.full(N, T, np.float32), wind_speed=np.zeros(N, np.float32))
    if not src.startswith('cycle:'):
        tr['grade'] = np.zeros(N, np.float32)
    tr['rho_air'] = routes.rho_air(tr['T_amb'], np.zeros(N)).astype(np.float32)
    return tr


_CYC = None
_MERSIN = {}


def _cycles():
    global _CYC
    if _CYC is None:
        import realdata as rd
        _CYC = rd.cycles()
    return _CYC


def _mersin_traces(split):
    if split not in _MERSIN:
        _MERSIN[split] = dict(np.load(os.path.join(C.real_dir(split), 'hd_traces.npz')))
    return _MERSIN[split]


def real_trace(split, name, j):
    """(L, 8) float32 trace of real record j, processed exactly as the bank's (build_bank.trace_of, no length cap)."""
    if split == 'exam':
        _guard()
    from build_bank import trace_of
    B, T, M = _load_real(split, name)
    obs = _real_trace_raw(split, M, j)
    v, X = trace_of(obs, cap=None)
    return np.concatenate([v[:, None], X.astype(np.float32)], 1)


def _real_batches(split, track, sets=SETS, chunk=None):
    chunk = chunk or (2048 if track == 'table' else 256)
    for name in sets:
        B, T, M = _load_real(split, name)
        n = len(B['arch'])
        X, E = DS.assemble(B, T)
        rl, _ = _rl_used(B, T)
        spec = spec_from_X(X, E, B['dist'])
        keys = np.array([f'{a}|{float(t):.2f}' for a, t in zip(M.trace_src.tolist(), M['T_test'].tolist())], dtype=object)
        for a in range(0, n, chunk):
            s = slice(a, min(n, a + chunk))
            b = {'uid': np.array([_uid('real', split, name, j) for j in range(s.start, s.stop)]),
                 'trace_key': np.array([_uid('realtr', k) for k in keys[s]]), 'source': 'real', 'period': split,
                 'track': track, 'n': s.stop - s.start, 'arch': B['arch'][s].astype(int), 'cls': B['cls'][s].astype(int)}
            if track == 'table':
                b['X'] = X[s]; b['E_phys'] = E[s]
            else:
                b['spec'] = spec[s]; b['rl'] = rl[s].astype(np.float32)
                b['traces'] = []
                cache = {}
                for j in range(s.start, s.stop):
                    k = keys[j]
                    if k not in cache:
                        cache[k] = real_trace(split, name, j)
                    b['traces'].append(cache[k])
            yield name, s, b


def dev_batches(track, sets=SETS, chunk=None):
    """Dev gate batches (VED 16 cars / 1,498 trips, EPA dev families / 10,345 test-legs, HD: 5 ZTBus legs, 224 Mersin
    route draws, 3 ICCT cycles). Yields batches only (use dev_frame() for the truth when scoring)."""
    for name, s, b in _real_batches('dev', track, sets, chunk):
        yield b


def real_frame(split, sets=SETS):
    """meta of the real records + y (label, MJ) + vr (valid-energy share; VED only) + E_phys, in batch order."""
    if split == 'exam':
        _guard()
    out = []
    for name in sets:
        B, T, M = _load_real(split, name)
        X, E = DS.assemble(B, T)
        d = M.copy(); d['y'] = B['y']; d['vr'] = B['vr']; d['E_phys'] = E; d['dist_m'] = B['dist']; d['dur'] = B['dur']
        d['recset'] = name; d['rec'] = np.arange(len(d))
        out.append(d)
    return pd.concat(out, ignore_index=True)


def dev_frame(sets=SETS):
    return real_frame('dev', sets)


# ============================================================================ exam (guarded)
LOCK = os.path.join(C.RESULTS, 'EXAM_RUNNING.json')
DONE = os.path.join(C.RESULTS, 'EXAM_DONE.json')
_UNLOCKED = None


def _guard():
    """Exam data may be read only inside exam.py's run: the lock file must exist and name this process's run id."""
    rid = _UNLOCKED or os.environ.get('BAKEOFF_EXAM_RUN')
    if not rid or not os.path.exists(LOCK):
        raise PermissionError('exam data is guarded: only exam.py (run once, after every arm is registered) reads it')
    lk = json.load(open(LOCK))
    if lk.get('run_id') != rid:
        raise PermissionError('exam lock names a different run')
    with open(os.path.join(C.RESULTS, 'exam_access.log'), 'a') as f:
        f.write(f'{pd.Timestamp.utcnow().isoformat()} pid {os.getpid()} run {rid}\n')
    return rid


def _require_exam(token=None):
    _guard()


def _unlock(run_id):
    global _UNLOCKED
    _UNLOCKED = run_id
    _guard()


def _relock():
    global _UNLOCKED
    _UNLOCKED = None
    os.environ.pop('BAKEOFF_EXAM_RUN', None)


def exam_batches(track, sets=SETS, chunk=None):
    _guard()
    for name, s, b in _real_batches('exam', track, sets, chunk):
        _guard()
        yield b


def exam_frame(sets=SETS):
    _guard()
    return real_frame('exam', sets)


def real_batches(split, track, sets=SETS, chunk=None):
    """dev: free; exam: guarded."""
    if split != 'dev':
        _guard()
    return _real_batches(split, track, sets, chunk)


# ============================================================================ shared trace resampling (optional)
def block_means(tr, k=5):
    """the shared resampling rule of the trace track (HARNESS.md section 4): (L, 8) trace -> (ceil(L / k), 11) blocks of
    k seconds: the mean of the 8 channels, then mean v^2, mean v^3 and the summed positive kinetic-energy increment per
    unit mass, sum(max(diff(v^2) / 2, 0)) [m2/s2], over the block (the last block may be shorter)."""
    tr = np.asarray(tr, np.float64)
    L = len(tr); nb = -(-L // k)
    pad = nb * k - L
    v = tr[:, 0]
    dke = np.maximum(np.diff(v * v, prepend=v[0] * v[0]) / 2.0, 0.0)
    X = np.concatenate([tr, (v * v)[:, None], (v ** 3)[:, None], dke[:, None]], 1)
    if pad:
        X = np.concatenate([X, np.full((pad, X.shape[1]), np.nan)], 0)
    X = X.reshape(nb, k, -1)
    out = np.nanmean(X[:, :, :10], 1)
    return np.concatenate([out, np.nansum(X[:, :, 10:], 1)], 1).astype(np.float32)
