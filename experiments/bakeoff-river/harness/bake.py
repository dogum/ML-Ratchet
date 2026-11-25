"""River Bake-off common data API: identical inputs and targets for every arm.

    import sys; sys.path.insert(0, '/home/claude/bakeoff/river/harness'); import bake

Simulated bank (frozen; built by build_bank.py, physics p1, FlowCast's released recipe):
    B = bake.load_bank('train')                 # dict of memory-mapped arrays (see BANK_FIELDS); 'val' = frozen val set
    bake.bank_info()                            # bank/MANIFEST.json (column names, theta_cols, hashes, build time)
    b = bake.sim_window(B, i, a, n_target)      # a standard batch: rows a .. a+365+n_target (+1) of run i
    T = bake.train_arrays(B, idx, offs)         # vectorised training windows (forcing, doy, target, static, ...)
    y = bake.sim_target(B, i, b)                # its target (qo, mm/day) over the batch's target days
    for b, y in bake.simval_batches(): ...      # the FROZEN sim-val protocol (800 runs x 2 windows of 365 scored days)
    bake.bank_spec('val', i)                    # the WaterSim spec + settings of run i (to re-run WaterSim)
    bake.bank_true_forcing('val', i)            # the forcing WaterSim really got (ERA5-Land x pmult, spin-up included)
    bake.bank_rerun('val', i)                   # re-runs WaterSim for run i (reproduces q exactly)

Real data (CAMELS-US via WaterData; dev only, the exam is guarded):
    bake.dev_basins()                           # the 52 dev basins
    for b in bake.dev_batches('ungauged'): ...  # 52 batches, WY2001-2014, warm-up from 1999-10-01
    for b in bake.dev_batches('gauged', lead=3): ...
    obs = bake.dev_obs(b)                       # observed mm/day over b's target days (Caravan Q); for scoring only
    bake.exam_batches(...)                      # PermissionError unless exam.py is running

A batch (dict) is what every arm's predict(batch) receives:
    uid        opaque id of the basin-window (stable; use it to cache, never to look anything up)
    source     'sim' | 'real';  period 'simval' | 'simtrain' | 'dev' | 'main' | 'recent'
    protocol   'ungauged' | 'gauged';  lead  None (ungauged) or days (gauged: 1, 3 or 7)
    forcing    (T, 8) float32, columns config.DYN = P, PET, T_mean, T_min, T_max, SRAD (NET shortwave), vp, wind;
               T = n_warm + n_target + 1 (one extra day after the last target day, as FlowCast)
    dates      (T,) datetime64[D] (sim: the real dates of the forcing donor's free-window segment)
    doy        (T,) day of year
    n_warm     365 warm-up days before the first target day;  n_target  number of target days
    attrs      (11,) float32 config.ATTR_NAMES (area km2, elevation mean/range m, slope m/km, forest, porosity,
               ksat cm/h, carbonate, reservoir storage / annual flow, irrigation, urban)
    static     (18,) float32 FlowCast's static vector (attributes + 7 climate indices of `forcing`;
               sim: of the run's 4 stored years)
    lat, lon   degrees (sim: the forcing donor's)
    classes    list from config.CLASSES (snow / arid / humid / regulated / large) or ['other']
    qobs       gauged only: (T,) float32 observed mm/day over every row (NaN = missing). The prediction for target
               row t may use qobs[:t - lead + 1] only (checked by register.py with a perturbation test).
predict(batch) returns {'mean': (n_target,), 'q05', 'q25', 'q50', 'q75', 'q95'} in mm/day.
"""
import glob
import gzip
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from features import (climate_stats, spec_attrs, static_vector, forcing_array, attr_vector,  # noqa: E402
                      classes_from)

BANK_FIELDS = ['F', 'qo', 'q', 'states', 'attrs', 'st', 'theta', 'doy0', 'model', 'snow', 'routing', 'prior', 'reg',
               'pmult', 'lat', 'lon', 'start', 'fbid']
_WD = None


def _wd():
    global _WD
    if _WD is None:
        import waterdata as wd
        _WD = wd
    return _WD


# ============================================================================ simulated bank
def bank_info():
    with open(os.path.join(C.BANK, 'MANIFEST.json')) as f:
        return json.load(f)


def load_bank(split, fields=None, mmap=True):
    """split 'train' (12,000 runs) or 'val' (800, frozen). Arrays are read-only memory maps unless mmap=False.
    F is float16 (cast windows to float32). Large: F 0.28 GB, states 0.35 GB for train; load only what you need."""
    if split not in ('train', 'val'):
        raise ValueError(split)
    d = os.path.join(C.BANK, split)
    if not os.path.exists(os.path.join(C.BANK, 'MANIFEST.json')):
        raise FileNotFoundError('bank not finalized yet (see harness/STATUS.md)')
    out = {}
    for k in (fields or BANK_FIELDS):
        a = np.load(os.path.join(d, f'{k}.npy'), mmap_mode='r' if mmap else None)
        out[k] = a
    out['_split'] = split
    return out


def _sim_classes(B, i):
    st = B['st'][i]
    at = B['attrs'][i]
    return classes_from(float(st[14]), float(np.exp(st[13])), float(at[8]), float(at[0]))


def sim_window(B, i, a, n_target=365, protocol='ungauged', lead=None, n_warm=C.WARM, period=None):
    """Batch over rows [a, a + n_warm + n_target] of run i (one extra row at the end, as for real basins).
    a + n_warm + n_target + 1 must be <= 1462."""
    T = n_warm + n_target + 1
    if a < 0 or a + T > C.NDAYS:
        raise ValueError(f'window {a}+{T} outside 0..{C.NDAYS}')
    start = np.datetime64(str(B['start'][i]))
    dates = start + np.arange(a, a + T).astype('timedelta64[D]')
    doy = pd.DatetimeIndex(dates).dayofyear.values
    split = B.get('_split', 'train')
    b = {'uid': f'sim:{split}:{i:05d}:{a}:{n_target}', 'source': 'sim',
         'period': period or ('simval' if split == 'val' else 'simtrain'), 'protocol': protocol,
         'lead': lead if protocol == 'gauged' else None,
         'forcing': np.asarray(B['F'][i, a:a + T], np.float32), 'dates': dates, 'doy': doy,
         'n_warm': n_warm, 'n_target': n_target, 'attrs': np.asarray(B['attrs'][i], np.float32),
         'static': np.asarray(B['st'][i], np.float32), 'lat': float(B['lat'][i]), 'lon': float(B['lon'][i]),
         'classes': _sim_classes(B, i)}
    if protocol == 'gauged':
        if lead is None:
            raise ValueError('gauged batches need a lead')
        b['qobs'] = np.asarray(B['qo'][i, a:a + T], np.float32).copy()
    return b


def sim_target(B, i, b):
    """Gauge-like target (qo, mm/day) over the batch's target days."""
    a = int(b['uid'].split(':')[3])
    return np.asarray(B['qo'][i, a + b['n_warm']:a + b['n_warm'] + b['n_target']], np.float64)


def train_arrays(B, idx, offs, n_target=365, n_warm=C.WARM):
    """Vectorised training windows (same rows as sim_window): returns dict with
    forcing (b, n_warm + n_target + 1, 8) float32, doy (b, T) int, target (b, n_target) float32 (qo, mm/day),
    static (b, 18), attrs (b, 11), states (b, n_target, 5) if B has 'states', theta (b, K) if B has 'theta'.
    idx: run indices; offs: first row of each window (0 <= off <= 1462 - T)."""
    T = n_warm + n_target + 1
    idx = np.asarray(idx); offs = np.asarray(offs)
    if (offs < 0).any() or (offs + T > C.NDAYS).any():
        raise ValueError('window outside the stored 1462 days')
    F = np.stack([np.asarray(B['F'][i, o:o + T], np.float32) for i, o in zip(idx, offs)])
    Y = np.stack([np.asarray(B['qo'][i, o + n_warm:o + n_warm + n_target], np.float32) for i, o in zip(idx, offs)])
    start = np.array([np.datetime64(str(B['start'][i])) for i in idx])
    doy = np.stack([pd.DatetimeIndex(s + np.arange(o, o + T).astype('timedelta64[D]')).dayofyear.values
                    for s, o in zip(start, offs)])
    out = {'forcing': F, 'doy': doy, 'target': Y, 'static': np.asarray(B['st'][idx], np.float32),
           'attrs': np.asarray(B['attrs'][idx], np.float32), 'n_warm': n_warm, 'n_target': n_target}
    if 'states' in B:
        out['states'] = np.stack([np.asarray(B['states'][i, o + n_warm:o + n_warm + n_target]) for i, o in zip(idx, offs)])
    if 'theta' in B:
        out['theta'] = np.asarray(B['theta'][idx], np.float32)
    return out


SIMVAL_OFFSETS = (0, 731)   # FlowCast's frozen-set protocol: per run two windows, the last 365 days of each scored


def simval_batches(protocol='ungauged', lead=None, nmax=None, B=None):
    """The frozen sim-val protocol: yields (batch, target) for 800 runs x 2 windows (rows 0-730 and 731-1461; days
    365-729 and 1096-1460 scored). Score per run on the 730 concatenated days (score.simval_scores)."""
    B = B or load_bank('val', ['F', 'qo', 'attrs', 'st', 'lat', 'lon', 'start'])
    n = len(B['qo']) if nmax is None else min(nmax, len(B['qo']))
    for i in range(n):
        for a in SIMVAL_OFFSETS:
            b = sim_window(B, i, a, 365, protocol, lead)
            yield b, sim_target(B, i, b)


def bank_spec(split, i):
    k = i // C.SHARD
    with gzip.open(os.path.join(C.BANK, split, f'specs_{k:03d}.jsonl.gz'), 'rt') as f:
        for j, line in enumerate(f):
            if j == i - k * C.SHARD:
                return json.loads(line)
    raise IndexError(i)


def bank_true_forcing(split, i):
    """(DataFrame, meta): the real ERA5-Land segment of run i with P x pmult, i.e. the forcing WaterSim simulated
    with (365 spin-up days + the 1462 stored days). The network-visible forcing (F) is this before x pmult, after the
    forcing-error model."""
    r = bank_spec(split, i)
    wd = _wd()
    df = wd.forcing(r['fbid'], '1951-01-01', '2015-09-30').loc[r['seg0']:r['end']].copy()
    df = df.interpolate(limit_direction='both')
    df['P'] = df['P'] * r['pmult']
    return df, wd.forcing_meta(r['fbid'])


def bank_rerun(split, i, **kw):
    """Re-run WaterSim for run i with its true spec, model, snow routine and routing (and the same noise seed).
    Returns the simulate() output; out['ts']['Q'] equals bank q[i] (float32) and states equal bank states[i]."""
    import watersim as ws
    r = bank_spec(split, i)
    df, meta = bank_true_forcing(split, i)
    args = dict(model=r['model'], snow=r['snow'], routing=r['routing'], labels=False, rng=r['sim_seed'],
                obs_noise=r['obs_noise'])
    args.update(kw)
    return ws.simulate(r['spec'], (df, meta), r['start'], r['end'], **args)


# ============================================================================ real data
def _splits():
    p = C.SPLITS
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    if h != C.SPLITS_SHA256:
        raise RuntimeError(f'splits.json hash {h} != frozen {C.SPLITS_SHA256}')
    return json.load(open(p))


def dev_basins():
    ids = list(_wd().basins('camels', split='dev'))
    assert len(ids) == 52
    return ids


def _exam_ids():
    ids = list(_wd().basins('camels', split='exam'))
    sp = _splits()['camels_us']['exam']
    want = sp if isinstance(sp, list) else (sp.get('basins') or sp.get('ids'))
    assert sorted(ids) == sorted(want) and len(ids) == 619, 'exam basins differ from splits.json'
    return ids


_ATTR = None


def basin_classes(bid):
    """Official class labels (FlowCast's strata, WaterData attributes): frac_snow > 0.3, aridity_FAO_PM > 1 (arid),
    < 0.6 (humid), reg_class mid/high (regulated), area > 2000 km2 (large)."""
    global _ATTR
    if _ATTR is None:
        _ATTR = _wd().attributes('camels')
    a = _ATTR.loc[bid]
    c = []
    if a.frac_snow > 0.3:
        c.append('snow')
    if a.aridity_FAO_PM > 1:
        c.append('arid')
    if a.aridity_FAO_PM < 0.6:
        c.append('humid')
    if a.reg_class in ('mid', 'high'):
        c.append('regulated')
    if a['area'] > 2000:
        c.append('large')
    return c or ['other']


def _uid(bid, period):
    return hashlib.sha1(f'river-bakeoff:{bid}:{period}'.encode()).hexdigest()[:12]


def _real_batch(bid, period, protocol='ungauged', lead=None, obs=None):
    (s0, s1), src = C.PERIODS[period]
    wd = _wd()
    w0 = pd.Timestamp(s0) - pd.Timedelta(days=C.WARM)
    e1 = pd.Timestamp(s1) + pd.Timedelta(days=1)
    days = pd.date_range(w0, e1)
    df = wd.forcing(bid, str(w0.date()), str(e1.date()))
    df = df.reindex(days)
    F = forcing_array(df)
    if len(F) != len(days) or not np.isfinite(F).all():
        raise ValueError(f'forcing of {bid} does not cover {w0.date()}..{e1.date()}')
    clim = climate_stats(F[:, 0], F[:, 1], F[:, 2])
    sp = wd.spec(bid)
    at = spec_attrs(sp)
    n_target = len(pd.date_range(s0, s1))
    b = {'uid': _uid(bid, period), 'source': 'real', 'period': period, 'protocol': protocol,
         'lead': lead if protocol == 'gauged' else None, 'forcing': F,
         'dates': days.values.astype('datetime64[D]'), 'doy': days.dayofyear.values, 'n_warm': C.WARM,
         'n_target': n_target, 'attrs': attr_vector(at), 'static': static_vector(at, clim),
         'lat': float(sp['lat']), 'lon': float(sp['lon']), 'classes': basin_classes(bid)}
    if protocol == 'gauged':
        if lead is None:
            raise ValueError('gauged batches need a lead')
        if obs is None:
            obs = wd.streamflow(bid, src).Q_mm
        b['qobs'] = obs.reindex(days).values.astype(np.float32)
    return b


def _obs(bid, period):
    (s0, s1), src = C.PERIODS[period]
    try:
        o = _wd().streamflow(bid, src).Q_mm
    except FileNotFoundError:
        return None
    return o.reindex(pd.date_range(s0, s1))


def scored(bid, period, obs=None):
    o = _obs(bid, period) if obs is None else obs
    return o is not None and int(o.notna().sum()) >= C.MIN_OBS_DAYS


def dev_batches(protocol='ungauged', lead=None, basins=None):
    """Dev gate batches: 52 dev basins, WY2001-2014 (2000-10-01..2014-09-30), warm-up from 1999-10-01."""
    for bid in (basins or dev_basins()):
        if bid not in dev_basins():
            raise PermissionError(f'{bid} is not a dev basin')
        yield _real_batch(bid, 'dev', protocol, lead)


def dev_batch(bid, protocol='ungauged', lead=None):
    if bid not in dev_basins():
        raise PermissionError(f'{bid} is not a dev basin')
    return _real_batch(bid, 'dev', protocol, lead)


_DEV_UID = None


def dev_obs(batch_or_bid):
    """Observed Caravan Q (mm/day, float64, NaN gaps) over the dev target days of a dev batch (or basin id)."""
    global _DEV_UID
    if _DEV_UID is None:
        _DEV_UID = {_uid(b, 'dev'): b for b in dev_basins()}
    bid = batch_or_bid if isinstance(batch_or_bid, str) else _DEV_UID.get(batch_or_bid['uid'])
    if bid not in _DEV_UID.values():
        raise PermissionError('dev_obs is for dev batches only')
    return _obs(bid, 'dev').values.astype(np.float64)


def dev_basin_of(batch):
    global _DEV_UID
    if _DEV_UID is None:
        _DEV_UID = {_uid(b, 'dev'): b for b in dev_basins()}
    return _DEV_UID.get(batch['uid'])


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


def _unlock(run_id):
    """Called by exam.py only."""
    global _UNLOCKED
    _UNLOCKED = run_id
    _guard()


def _relock():
    global _UNLOCKED
    _UNLOCKED = None
    os.environ.pop('BAKEOFF_EXAM_RUN', None)


def exam_basins():
    _guard()
    return _exam_ids()


def exam_batch(bid, period, protocol='ungauged', lead=None, obs=None):
    _guard()
    if period not in ('main', 'recent'):
        raise ValueError(period)
    return _real_batch(bid, period, protocol, lead, obs)


def exam_obs(bid, period):
    _guard()
    return _obs(bid, period)


def exam_batches(period, protocol='ungauged', lead=None):
    _guard()
    for bid in _exam_ids():
        o = _obs(bid, period)
        if o is None or int(o.notna().sum()) < C.MIN_OBS_DAYS:
            continue
        yield bid, _real_batch(bid, period, protocol, lead)


def real_batch(bid, period, protocol='ungauged', lead=None):
    """Any CAMELS-US basin: dev period on dev basins; exam basins or windows only under the exam guard."""
    if period == 'dev' and bid in dev_basins():
        return _real_batch(bid, period, protocol, lead)
    _guard()
    return _real_batch(bid, period, protocol, lead)
