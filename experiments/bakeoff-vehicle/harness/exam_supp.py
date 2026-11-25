"""Dated supplementary exams for the Vehicle Bake-off. Each runs ONCE and is reported beside the main board, never in it.

usage: python exam_supp.py <supp> --check            preflight only (no exam data read)
       python exam_supp.py <supp> --run [--workers N] the supplementary exam, once
       python exam_supp.py <supp> --resume [--workers N]
       python exam_supp.py <supp> --rehearsal [--arm ID ...]   the same pipeline on DEV records (no exam data, no lock)

supps (added 24 Nov 2026, in the repository, after the main exam 20261123T204251):
  tabpfn   The two arms excluded before the main exam for compute only (tab_tabpfn, tab_tabpfn_conformal;
           registry/excluded.json), on the main exam's records, rebuilt here by the same build_real code. Every
           other table-track arm and the reference rows are scored again beside them: their per-unit scores must
           equal the main exam's (a check that the rebuilt records are the same records).
  ztbfull  Every examined arm, plus the two TabPFN arms, on EnergyNet's full ZTBus exam (handoff item 21): the 5,905
           legs of the 1,115 exam missions in proc/hd/ztbus_full (5,847 labelled legs are scored as legs) plus one
           standstill layover record per mission. Legs are built exactly as build_real.hd builds the v1 ZTBus legs.
           Missions are scored two ways: the bake-off's (sum of the labelled legs) and EnergyNet's (every leg and the
           layover against the mission's metered kWh; evaluate.ztb_missions).

How it differs from exam.py (whose code it reuses, unchanged): results go to results/supp_<supp>/ (its own
EXAM_RUNNING.json / EXAM_DONE.json lock; the main exam's lock is never touched), records to work/real_supp_<supp>/,
workers run in parallel (--workers, default 6), and arm folders are read from this repository (the registry holds
the cloud paths). The TabPFN arms are checked file by file against their registry records; the only accepted
differences are the platform build of safetensors (a Windows .pyd for the Linux .so) and pip's install metadata
(*.dist-info) in pylib/. The TabPFN code, the weights, the context set and the predict code must be identical.
The exam chooses nothing: every listed arm is scored on every record of its tracks.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SUPPS = ('tabpfn', 'ztbfull')
SUPP = next((a for a in sys.argv[1:] if a in SUPPS), None)
if SUPP is None:
    raise SystemExit(__doc__)

import config as C  # noqa: E402

# TabPFN's own dependencies that the cloud machine had system-wide (pydantic, einops, huggingface-hub, tqdm, ...):
# installed with pip --target into work/tabpfn_deps (git-ignored) and APPENDED to sys.path, so they never shadow the
# environment's own packages. Only the two TabPFN arms import them.
sys.path.append(os.path.join(C.WORK, 'tabpfn_deps'))
C.RESULTS = os.path.join(C.BAKEOFF, 'results', f'supp_{SUPP}')
C.REAL = os.path.join(C.WORK, f'real_supp_{SUPP}')
os.makedirs(C.RESULTS, exist_ok=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import bake  # noqa: E402  (its LOCK / DONE now live in results/supp_<supp>/)
import register as REG  # noqa: E402
import armlib  # noqa: E402
import exam as EX  # noqa: E402
import score as S  # noqa: E402

CLOUD = '/home/claude/bakeoff/vehicle/'
TABPFN = ('tab_tabpfn', 'tab_tabpfn_conformal')
PLATFORM_NOTE = ('accepted differences (TabPFN arms only): the platform build of safetensors and pip install metadata '
                 'in pylib/; TabPFN code, weights, context and predict code identical to registration')


# ---------------------------------------------------------------- registry: repo paths, TabPFN included
def _local(r):
    r = dict(r)
    d = r['arm_dir']
    if d.startswith(CLOUD):
        r['arm_dir'] = os.path.join(C.BAKEOFF, d[len(CLOUD):]).replace(os.sep, '/')
    return r


_registered = REG.registered


def registered(include_excluded=True):
    """every registry record (the two excluded TabPFN arms included), arm folders mapped into this repository."""
    return {k: _local(r) for k, r in _registered(include_excluded=True).items()}


def _platform_file(rel):
    rel = rel.replace('\\', '/')
    return '/pylib/' in '/' + rel and ('.dist-info/' in rel or '/safetensors/' in rel)


_verify = REG.verify


def verify(rec):
    ok, h, ch = _verify(rec)
    if not ok and rec['arm_id'] in TABPFN and ch and all(_platform_file(c) for c in ch):
        return True, h, ch            # accepted (see PLATFORM_NOTE); the changed files are still recorded
    return ok, h, ch


REG.registered = registered
REG.verify = verify


# ---------------------------------------------------------------- ztbfull: records, sets, traces
ZTB_SETS = ('ztb',)


def build_ztb(split='exam'):
    """legs of the full-ZTBus missions of a split (exam: 1,115) + one layover record per mission -> work/real_supp_ztbfull/<split>/ztb."""
    import build_real as BR
    rd, F, RF, routes = BR.rd, BR.F, BR.RF, BR.routes
    A = BR.Acc()
    sp = rd.ztbus_full(table='splits')['missions']
    M = rd.ztbus_full(table='missions').set_index('name')
    seen = {}
    for meta, arr in rd.load_trips(split, source='ZTBus_full'):
        ms = meta['mission']
        assert sp[ms] == split, ms
        P = F.prep({k: arr[k] for k in BR.TR_KEYS})
        y = meta['elec_kWh'] * 3.6
        lab = bool(meta.get('q_label_ok', True)) and y > 0
        rec = dict(arch=6, cls=4, trans=F.TRANS.index('single'), dyno=0.0, warm=1.0, cold=0.0, y=y, vr=1.0)
        A.add(P, 'bus', rec, dict(m_level=1, trans=True, thermal=True), m_rep=RF.ZTB_CURB_KG,
              meta=dict(set='ZTBus_leg' if lab else 'ZTBus_leg_nolabel', kind='leg', split=split, trip_id=meta['trip_id'],
                        mission=ms, vehicle_id=meta['vehicle_id'], dist_km=meta['distance_m'] / 1000,
                        dur_s=float(len(P['v'])), T=meta['T_amb_mean'], y=y, trace_src=meta['trip_id'], n_s=0,
                        T_test=np.nan, arch0='bev', m_level0=0, group='BUS', arch='bev', cls='bus'))
        seen[ms] = seen.get(ms, 0.0) + float(meta['duration_s'])
    for ms, r in M.iterrows():
        if sp.get(ms) != split:
            continue
        gap = float(r.n_seconds) - seen.get(ms, 0.0)
        N = int(max(gap, 60))
        T = float(r.T_amb_mean_C) if np.isfinite(r.T_amb_mean_C) else 15.0
        P = F.prep(_layover_trace(N, T))
        rec = dict(arch=6, cls=4, trans=F.TRANS.index('single'), dyno=0.0, warm=1.0, cold=0.0, y=np.nan, vr=1.0)
        A.add(P, 'bus', rec, dict(m_level=1, trans=True, thermal=True), m_rep=RF.ZTB_CURB_KG,
              meta=dict(set='ZTBus_layover', kind='layover', split=split, trip_id=f'{ms}_layover', mission=ms,
                        vehicle_id=str(ms).split('_')[1] if '_' in str(ms) else '', dist_km=0.0, dur_s=float(max(gap, 0.0)),
                        T=T, y=np.nan, trace_src=f'layover:{ms}', n_s=N, T_test=T, arch0='bev', m_level0=0, group='BUS',
                        arch='bev', cls='bus'))
    d = os.path.join(C.real_dir(split))
    os.makedirs(d, exist_ok=True)
    A.save_to(os.path.join(d, 'ztb'))
    json.dump({'split': split, 'supp': 'ztbfull', 'built': pd.Timestamp.utcnow().isoformat()},
              open(os.path.join(d, 'BUILT.json'), 'w'))


def _layover_trace(N, T):
    """a standstill trace of N seconds at T C and 410 m (EnergyNet realfeat.ztb)."""
    import build_real as BR
    tr = BR._fill({'v': np.zeros(N, np.float32)})
    tr['T_amb'] = np.full(N, T, np.float32)
    tr['rho_air'] = BR.routes.rho_air(tr['T_amb'], np.full(N, 410.0)).astype(np.float32)
    return tr


def _patch_ztb():
    bake.SETS = ZTB_SETS
    f0, b0, raw0 = bake.real_frame, bake._real_batches, bake._real_trace_raw

    def real_frame(split, sets=ZTB_SETS):
        return f0(split, sets)

    def _real_batches(split, track, sets=ZTB_SETS, chunk=None):
        return b0(split, track, sets, chunk)

    def _real_trace_raw(split, M, j):
        src = M.trace_src.iloc[j]
        if src.startswith('ZTBusFull_'):
            import realdata as rd
            from bank import TR_KEYS
            _, arr = rd.load_trip(src)
            return {k: arr[k] for k in TR_KEYS}
        if src.startswith('layover:'):
            return _layover_trace(int(M.n_s.iloc[j]), float(M['T'].iloc[j]))
        return raw0(split, M, j)

    bake.real_frame, bake._real_batches, bake._real_trace_raw = real_frame, _real_batches, _real_trace_raw

    def build(split, token=None):
        bake._require_exam(token)
        build_ztb()
    import build_real
    build_real.build = build


if SUPP == 'ztbfull':
    _patch_ztb()


# ---------------------------------------------------------------- which arms
def arms_for():
    regs = registered()
    if SUPP == 'tabpfn':
        return {k: r for k, r in regs.items() if 'table' in r['tracks']}
    return regs


def preflight():
    notes, reasons = [], []
    if os.path.exists(bake.DONE):
        raise EX.Refused(f'supplement {SUPP} already ran ({json.load(open(bake.DONE)).get("run_id")}); it runs once')
    bake._splits()
    notes.append(f'splits.json sha256 OK ({C.SPLITS_SHA256[:8]})')
    arms = arms_for()
    for k in ('enet_v21', 'climatology') + TABPFN:
        if k not in arms:
            reasons.append(f'{k} missing')
    for k, r in arms.items():
        if not os.path.isdir(r['arm_dir']):
            reasons.append(f'{k}: arm folder {r["arm_dir"]} is missing')
            continue
        ok, h, ch = verify(r)
        if not ok:
            reasons.append(f'{k}: artefact differs from registration in {len(ch)} files: {ch[:6]}')
        elif ch:
            notes.append(f'{k}: {len(ch)} platform/metadata files differ ({PLATFORM_NOTE})')
    if SUPP == 'ztbfull' and not os.path.isdir(os.path.join(C.ROAD_DATA, 'proc', 'hd', 'ztbus_full', 'trips')):
        reasons.append('proc/hd/ztbus_full/trips is missing')
    if reasons:
        raise EX.Refused('\n'.join(reasons))
    notes.append(f'{len(arms)} arms: {", ".join(sorted(arms))}')
    return arms, notes


# ---------------------------------------------------------------- driver
def build_dev_records():
    if os.path.exists(os.path.join(C.real_dir('dev'), 'BUILT.json')):
        return
    if SUPP == 'ztbfull':
        build_ztb('dev')
    else:
        import build_real
        for f in (build_real.ved, build_real.epa, build_real.hd):
            f('dev')
        json.dump({'split': 'dev'}, open(os.path.join(C.real_dir('dev'), 'BUILT.json'), 'w'))


def rehearse(named=None, workers=6):
    """the supplement's pipeline on dev records (code test; no exam data, no lock)."""
    arms = arms_for()
    if named:
        arms = {k: arms[k] for k in named}
    build_dev_records()
    rid = time.strftime('%Y%m%dT%H%M%S')
    od = EX.outdir('rehearsal', rid)
    os.makedirs(od, exist_ok=True)
    def one(t):
        log = open(os.path.join(od, f'worker_{t}.log'), 'a')
        return t, subprocess.call([sys.executable, '-X', 'utf8', os.path.abspath(__file__), SUPP, '--worker', t,
                                   '--mode', 'rehearsal', '--run-id', rid], stdout=log, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, ['__refs__'] + sorted(arms)))
    print(res, flush=True)
    if any(rc for _, rc in res):
        raise EX.Refused(f'rehearsal workers failed: {[t for t, rc in res if rc]} (logs in {od})')
    fn, out = EX.aggregate('rehearsal', rid, arms)
    if SUPP == 'ztbfull':
        out['ztbus_full'] = ztb_boards(od, rid, arms, split='dev')
        json.dump(out, open(fn, 'w'), indent=1, default=float)
    print('wrote', fn)
    return fn, out


def drive(resume=False, workers=6):
    if resume:
        if not os.path.exists(bake.LOCK):
            raise EX.Refused('nothing to resume')
        lk = json.load(open(bake.LOCK))
        rid = lk['run_id']
        arms, notes = preflight()
    else:
        if os.path.exists(bake.LOCK):
            raise EX.Refused(f'a run is in progress ({bake.LOCK}); use --resume')
        arms, notes = preflight()
        rid = time.strftime('%Y%m%dT%H%M%S')
        json.dump({'run_id': rid, 'supp': SUPP, 'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'arms': {k: r['hash'] for k, r in arms.items()}, 'harness_hash': armlib.harness_hash()[0]},
                  open(bake.LOCK, 'w'), indent=1)
    EX.build_exam_records(rid)
    for n in notes:
        print(n, flush=True)
    od = EX.outdir('exam', rid)
    os.makedirs(od, exist_ok=True)
    env = dict(os.environ, BAKEOFF_EXAM_RUN=rid)
    todo = [t for t in ['__refs__'] + sorted(arms) if not os.path.exists(os.path.join(od, 'preds', t, 'DONE.json'))]
    # conformal wrappers read their base arm's code only (their own predictions), so every target is independent
    def one(t):
        t0 = time.time()
        log = open(os.path.join(od, f'worker_{t}.log'), 'a')
        rc = subprocess.call([sys.executable, '-X', 'utf8', os.path.abspath(__file__), SUPP, '--worker', t,
                              '--run-id', rid], env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f'== {t} exit {rc} in {time.time() - t0:.0f} s', flush=True)
        return t, rc
    order = sorted(todo, key=lambda t: (not t.startswith('trace_'), t))      # slow trace arms first
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, order))
    bad = [t for t, rc in res if rc != 0]
    if bad:
        raise EX.Refused(f'workers failed: {bad}; fix and use --resume')
    bake._unlock(rid)
    try:
        fn, out = EX.aggregate('exam', rid, arms)
    finally:
        bake._relock()
    extra = {'supp': SUPP, 'note': PLATFORM_NOTE, 'main_exam': '20261123T204251'}
    if SUPP == 'ztbfull':
        extra['ztbus_full'] = ztb_boards(od, rid, arms)
    out['supplement'] = extra
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    json.dump({'run_id': rid, 'supp': SUPP, 'finished': out['finished'], 'result': os.path.relpath(fn, C.BAKEOFF).replace(os.sep, '/'),
               'arms': {k: v['hash'] for k, v in out['arms'].items()}}, open(bake.DONE, 'w'), indent=1)
    os.remove(bake.LOCK)
    print('wrote', fn)


def ztb_boards(od, rid, arms, split='exam'):
    """legs (5,847 labelled), missions by the bake-off's rule and by EnergyNet's (legs + layover vs metered kWh)."""
    import realdata as rd
    if split == 'exam':
        bake._unlock(rid)
    try:
        Fr = bake.real_frame(split)
    finally:
        bake._relock()
    T = rd.ztbus_full(table='missions').set_index('name')
    rp = pd.read_parquet(os.path.join(od, 'preds', '__refs__', 'refs.parquet'))
    out = {}
    rows = [(tr, k) for k in sorted(arms) for tr in arms[k]['tracks']] + [('ref', 'dev_median')]
    for tr, k in rows:
        if tr == 'ref':
            p = pd.DataFrame({'pred': rp['dev_median'].values})
            for q in C.QKEYS:
                p[q] = p['pred']
        else:
            p = pd.read_parquet(os.path.join(od, 'preds', k, f'{tr}.parquet'))
        d = Fr.copy()
        for c in ['pred'] + C.QKEYS:
            d[c] = p[c].values
        legs = d[d.set == 'ZTBus_leg']
        ok = np.isfinite(legs.pred) & (legs.pred > 0)
        lu = legs[ok].assign(unit=legs.trip_id, cluster=legs.mission)
        lm = lu.groupby('mission')[['y'] + ['pred'] + C.QKEYS].sum().reset_index()
        lm = lm.assign(unit=lm.mission, cluster=lm.mission)
        z = d[d.set.isin(['ZTBus_leg', 'ZTBus_leg_nolabel', 'ZTBus_layover'])]
        g = z.groupby('mission')[['pred'] + C.QKEYS].sum()
        g = g.join(T[['kWh']])
        g['y'] = g.kWh * 3.6
        n_bad = int((~np.isfinite(z.pred) | (z.pred <= 0)).sum())
        em = g[np.isfinite(g.pred) & (g.pred > 0)].reset_index().rename(columns={'index': 'mission', 'name': 'mission'})
        em = em.assign(unit=em.mission, cluster=em.mission)
        out[f'{k}|{tr}'] = {'legs': S.summarize(lu), 'missions_legsum': S.summarize(lm),
                            'missions_energynet': S.summarize(em), 'n_records_without_prediction': n_bad}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('supp', choices=SUPPS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--run', action='store_true')
    g.add_argument('--resume', action='store_true')
    g.add_argument('--rehearsal', action='store_true')
    g.add_argument('--worker', default=None)
    ap.add_argument('--mode', default='exam')
    ap.add_argument('--arm', action='append', default=None)
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    try:
        if a.worker:
            bake._relock()
            EX.run_worker(a.worker, a.mode, a.run_id)
        elif a.rehearsal:
            rehearse(a.arm, a.workers)
        elif a.check:
            arms, notes = preflight()
            for n in notes:
                print(n)
            print(f'CHECK PASSED: supplement {SUPP} can run. Nothing was read from the exam.')
        elif a.run:
            drive(False, a.workers)
        elif a.resume:
            drive(True, a.workers)
    except EX.Refused as e:
        print('REFUSED:\n' + str(e), flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
