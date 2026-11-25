"""The Vehicle Bake-off exam: run ONCE, after every arm is registered, for all registered arms together.

usage: python3 exam.py --check [--arm ID ...]   preflight only (no exam data): hashes, registry, plan, CPU estimate
       python3 exam.py --run                    THE exam (once): every registered arm on its tracks + every reference row
       python3 exam.py --resume                 continue an interrupted run (same run id; finished jobs are kept)
       python3 exam.py --rehearsal [--arm ID]   the same pipeline on the DEV gate (no exam data, no lock);
                                                writes results/rehearsal_<id>.json (for testing)

Rules it enforces:
  - refuses to start if results/EXAM_DONE.json exists (once only) or a run is in progress (use --resume);
  - refuses if a registered arm's artefact hash changed, if an arm folder under bakeoff/vehicle/arms_*/ (or
    harness/arms_ref/) is not registered (unless listed in registry/excluded.json with a reason), or if a reference arm
    (enet_v21, climatology) is missing;
  - re-hashes each arm right before and right after scoring it;
  - exam records are built (build_real.build('exam')) and read only through bake's guard (lock file
    results/EXAM_RUNNING.json + run id); the guard is closed whenever arm code runs;
  - chooses nothing: every registered arm is scored on every exam record of its tracks.
Exam scope (local data; see HARNESS.md): VED exam trips (4 fleets), EPA exam test cars (fuel / BEV tests), Mersin
exam trips, ICCT exam trucks, and the 78 ZTBus legs of the 8 traced exam missions RoadData v1 holds. EnergyNet's
full-ZTBus rows (5,847 legs, 1,115 missions) are out of scope: those traces are not on this machine.
Outputs: results/exam_<id>.json (boards per track, per set, conditions, paired bootstrap, the EnergyNet reproduction
check against training/exam3.json, compute, hashes), results/exam_<id>/scores.parquet (per unit x set x row x
track), results/exam_<id>/preds/ (per-record predictions), results/exam_<id>/showcase.npz + showcase.json, then
results/EXAM_DONE.json.
"""
import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import bake  # noqa: E402
import score as S  # noqa: E402
import refs as R  # noqa: E402
import armlib  # noqa: E402
import register as REG  # noqa: E402

REQUIRED_REFS = ['enet_v21', 'climatology']
EXCLUDED = os.path.join(C.REGISTRY, 'excluded.json')
ANCHORS = ['enet_v21', 'ephys']
CLASSICAL = {'VED': ['moves_like', 'label'], 'EPA': ['regression'], 'HD': ['dev_median']}
NBOOT_EXAM = 2000


class Refused(Exception):
    pass


def set_family(key):
    return 'VED' if key.startswith('VED') else ('EPA' if key.startswith('EPA') else 'HD')


# ============================================================================ preflight
def discover_arm_dirs():
    pats = [os.path.join(C.BAKEOFF, 'arms_*', '*', 'ratchet.json'), os.path.join(C.BAKEOFF, 'arms_*', '*', '*', 'ratchet.json'),
            os.path.join(C.HARNESS, 'arms_ref', '*', 'ratchet.json')]
    out = set()
    for p in pats:
        for f in glob.glob(p):
            d = os.path.dirname(os.path.abspath(f))
            if os.path.exists(os.path.join(d, 'predict.py')) and '/work/' not in d + '/':
                out.add(d)
    return sorted(out)


def preflight(named=None, mode='exam'):
    reasons, notes = [], []
    if mode == 'exam' and os.path.exists(bake.DONE):
        raise Refused(f'the exam already ran ({json.load(open(bake.DONE)).get("run_id")}); it runs once')
    bake._splits()
    notes.append(f'splits.json sha256 OK ({C.SPLITS_SHA256[:8]})')
    for p in (os.path.join(C.BANK, 'val', 'MANIFEST.json'), os.path.join(C.BANK, 'train', 'MANIFEST.json'),
              os.path.join(C.HARNESS, 'showcase.json'), os.path.join(C.REAL, 'dev', 'BUILT.json')):
        if not os.path.exists(p):
            reasons.append(f'missing {os.path.relpath(p, C.HARNESS)}')
    regs = REG.registered()
    if not regs:
        reasons.append('no registered arms')
    excluded = json.load(open(EXCLUDED)) if os.path.exists(EXCLUDED) else {}
    by_dir = {r['arm_dir']: k for k, r in regs.items()}
    for d in discover_arm_dirs():
        if d not in by_dir and d not in excluded:
            reasons.append(f'unregistered arm folder {os.path.relpath(d, C.BAKEOFF)} (register it, or list it in '
                           f'registry/excluded.json with a reason)')
    for k, r in regs.items():
        if not os.path.isdir(r['arm_dir']):
            reasons.append(f'{k}: arm folder {r["arm_dir"]} is gone')
            continue
        ok, h, changed = REG.verify(r)
        if not ok:
            reasons.append(f'{k}: artefact changed since registration ({r["hash"][:12]} -> {h[:12]}; files {changed})')
    for k in REQUIRED_REFS:
        if k not in regs:
            reasons.append(f'reference arm {k} is not registered')
    arms = regs
    if named:
        arms = {}
        for x in named:
            k = x if x in regs else by_dir.get(os.path.abspath(x))
            if k is None:
                reasons.append(f'--arm {x}: not registered')
            else:
                arms[k] = regs[k]
    if reasons:
        raise Refused('\n'.join(reasons))
    hh = armlib.harness_hash()[0]
    for k, r in arms.items():
        if r.get('harness_hash') != hh:
            notes.append(f'{k}: harness code changed since this arm registered (recorded)')
    est = sum(r['smoke'].get('exam_cpu_estimate_min', 0) * len(r['tracks']) for r in arms.values())
    notes.append(f'{len(arms)} arms: {", ".join(arms)}; estimated predict CPU {est:.0f} min '
                 f'(+ building the exam records ~15 min, references and aggregation ~15 min)')
    return arms, notes


# ============================================================================ data access
class Data:
    def __init__(self, mode, rid):
        self.mode, self.rid = mode, rid
        self.split = 'exam' if mode == 'exam' else 'dev'

    def _g(self, f, *a, **k):
        if self.mode != 'exam':
            return f(*a, **k)
        bake._unlock(self.rid)
        try:
            return f(*a, **k)
        finally:
            bake._relock()

    def frame(self):
        return self._g(bake.real_frame, self.split)

    def batches(self, track):
        """iterate (set name, slice, batch) with the guard open only while the next batch is read."""
        it = None
        while True:
            if self.mode == 'exam':
                bake._unlock(self.rid)
            try:
                if it is None:
                    it = bake._real_batches(self.split, track)
                item = next(it)
            except StopIteration:
                return
            finally:
                if self.mode == 'exam':
                    bake._relock()
            yield item


def outdir(mode, rid):
    return os.path.join(C.RESULTS, f'{"exam" if mode == "exam" else "rehearsal"}_{rid}')


# ============================================================================ worker
def run_worker(target, mode, rid):
    regs = REG.registered()
    if target != '__refs__' and target not in regs:
        raise Refused(f'{target}: not registered')
    od = os.path.join(outdir(mode, rid), 'preds', target)
    os.makedirs(od, exist_ok=True)
    D = Data(mode, rid)
    if target == '__refs__':
        Fr = D.frame()
        P = R.reference_preds(Fr)
        pd.DataFrame(P).to_parquet(os.path.join(od, 'refs.parquet'))
        json.dump({'rows': list(P)}, open(os.path.join(od, 'DONE.json'), 'w'))
        return
    rec = regs[target]
    ok, h, ch = REG.verify(rec)
    if not ok:
        raise Refused(f'{target}: artefact changed since registration: {ch}')
    arm = armlib.Arm(rec['arm_dir'])
    cpu = {}
    nrec = {name: int(n) for name, n in D.frame().recset.value_counts().items()}
    off, acc = {}, 0
    for name in bake.SETS:
        off[name] = acc; acc += nrec.get(name, 0)
    for track in rec['tracks']:
        fin = os.path.join(od, f'{track}.parquet')
        if os.path.exists(fin):
            continue
        part = os.path.join(od, f'{track}.partial.pkl')
        n = acc
        P = np.full(n, np.nan); Q = np.full((n, 5), np.nan); CP = np.full((n, 5), np.nan)
        NC = np.zeros(n, int); ERR = np.array([''] * n, dtype=object); done = set()
        if os.path.exists(part):
            P, Q, CP, NC, ERR, done = pickle.load(open(part, 'rb'))
        c0 = arm.seconds; nb = 0
        for name, sl, b in D.batches(track):
            key = (name, sl.start)
            if key in done:
                continue
            i0 = off[name] + sl.start; i1 = i0 + b['n']
            try:                    # a failing batch is recorded as errors; it never stops the exam
                o = arm.run([armlib.strip_batch(b)])[0]
                p, q, nc, comp = S.check_output(o, b['n'])
                P[i0:i1] = p; Q[i0:i1] = q; NC[i0:i1] = nc
                if comp is not None:
                    CP[i0:i1] = comp
            except Exception as e:  # noqa: BLE001
                ERR[i0:i1] = f'{type(e).__name__}: {e}'[:300]
            done.add(key); nb += 1
            if nb % 20 == 0:
                pickle.dump((P, Q, CP, NC, ERR, done), open(part + '.tmp', 'wb'))
                os.replace(part + '.tmp', part)
                print(f'{target} {track} {len(done)} batches', flush=True)
        df = pd.DataFrame({'pred': P, **{k: Q[:, j] for j, k in enumerate(C.QKEYS)},
                           **{f'c{j}': CP[:, j] for j in range(5)}, 'n_cross': NC, 'error': ERR.astype(str)})
        df.to_parquet(fin + '.tmp.parquet')
        os.replace(fin + '.tmp.parquet', fin)
        if os.path.exists(part):
            os.remove(part)
        cpu[track] = {'predict_cpu_s': round(arm.seconds - c0, 1), 'n_errors': int((ERR != '').sum())}
        print(f'{target} {track} done ({cpu[track]})', flush=True)
    ok, h, ch = REG.verify(rec)
    json.dump({'hash_after': h, 'unchanged': ok, 'changed_files': ch, 'cpu': cpu},
              open(os.path.join(od, 'DONE.json'), 'w'), indent=1)
    if not ok:
        raise Refused(f'{target}: artefact changed DURING the exam: {ch}')


# ============================================================================ aggregation
def _units_for(Fr, preds):
    d = Fr.copy()
    for k in S.OUT_KEYS:
        d[k] = preds[k].values if k in preds else preds['pred'].values
    d = d[np.isfinite(d.pred) & (d.pred > 0)]
    return S.real_units(d)


def reproduction(enet_summary):
    """EnergyNet v2.1 (reference arm, table track) against its published exam (training/exam3.json), rows in scope."""
    pub = json.load(open(C.ENET_EXAM3))['v2']
    out = {}
    for g in ('ICE', 'HEV', 'EV', 'PHEV'):
        s = enet_summary.get(f'VED_{g}', {}); c = enet_summary.get(f'VED_{g}_cars', {}); p = pub['VED'][g]
        if s.get('n'):
            out[f'VED_{g}'] = dict(n=s['n'], n_pub=p['n'], MdAPE=s['MdAPE'], MdAPE_pub=p['trip_MdAPE'], bias=s['bias'],
                                   bias_pub=p['bias'], cars_MdAPE=c.get('MdAPE'), cars_MdAPE_pub=p['veh_MdAPE'])
    for k, p in pub['EPA'].items():
        a, t = k.rsplit('_', 1)
        kk = f"EPA_{'hev' if a.startswith('hev') else a}_{t}"
        s = enet_summary.get(kk, {})
        if s.get('n'):
            out[kk] = dict(n=s['n'], n_pub=p['n'], MdAPE=s['MdAPE'], MdAPE_pub=p['MdAPE'], bias=s['bias'], bias_pub=p.get('bias'))
    for k in ('Mersin_diesel', 'Mersin_cng', 'ICCT'):
        s = enet_summary.get(k, {}); p = pub[k]
        if s.get('n'):
            out[k] = dict(n=s['n'], n_pub=p['n'], MdAPE=s['MdAPE'], MdAPE_pub=p['MdAPE'], bias=s.get('bias'), bias_pub=p.get('bias'))
    for k, v in out.items():
        v['diff_MdAPE'] = v['MdAPE'] - v['MdAPE_pub']
    out['_note'] = ('ZTBus_full rows (5,847 legs, 1,115 missions) are out of scope here; ZTBus_legs / ZTBus_missions '
                    'in the boards are the 78 legs / 8 missions of RoadData v1, a subset with no published v2.1 number')
    return out


def reproduction_refs(RS):
    """the classical reference rows against the baseline numbers EnergyNet published with its exam (exam3.json)."""
    b = json.load(open(C.ENET_EXAM3))['baselines']
    out = {}
    for g, v in b['VED'].items():
        for ref, key in (('label', 'label'), ('moves_like', 'moves_like'), ('class_median', 'class_median')):
            s = RS[ref].get(f'VED_{g}', {})
            if s.get('n'):
                out[f'VED_{g}|{ref}'] = dict(MdAPE=s['MdAPE'], MdAPE_pub=v.get(key), n=s['n'])
    for k, v in b['EPA'].items():
        a, t = k.rsplit('_', 1)
        kk = f"EPA_{'hev' if a.startswith('hev') else a}_{t}"
        s = RS['regression'].get(kk, {})
        if s.get('n') and v.get('regression_all') is not None:
            out[f'{kk}|regression'] = dict(MdAPE=s['MdAPE'], MdAPE_pub=v['regression_all'], n=s['n'],
                                           note='published: all exam vehicles with the regression features')
    for k in ('Mersin_diesel', 'Mersin_cng', 'ICCT'):
        s = RS['dev_median'].get(k, {})
        if s.get('n'):
            out[f'{k}|dev_median'] = dict(MdAPE=s['MdAPE'], MdAPE_pub=b.get(k), n=s['n'])
    return out


def aggregate(mode, rid, arms):
    od = outdir(mode, rid)
    D = Data(mode, rid)
    Fr = D.frame()
    regs = REG.registered()
    rp = pd.read_parquet(os.path.join(od, 'preds', '__refs__', 'refs.parquet'))
    rows = {}                         # (track, row) -> per-record preds frame
    for k in arms:
        for track in regs[k]['tracks']:
            rows[(track, k)] = pd.read_parquet(os.path.join(od, 'preds', k, f'{track}.parquet'))
    out = {'run_id': rid, 'mode': mode, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'harness_hash': armlib.harness_hash()[0], 'splits_sha256': C.SPLITS_SHA256,
           'bank': {s: json.load(open(os.path.join(C.BANK, s, 'MANIFEST.json')))['files'] for s in ('train', 'val')},
           'arms': {}, 'references': R.REF_ROWS, 'boards': {}, 'notes': [
               'MdAPE = median |pred / y - 1| over units; bias = median (pred / y - 1); cov50 / cov90 = share of units '
               'inside q25-q75 / q05-q95; crps_log = 5-quantile CRPS on log energy (point rows: |log error|)',
               'paired bootstrap over clusters (VED vehicles, ZTBus missions, EPA test vehicles, Mersin trips), 2,000 '
               'resamples, against EnergyNet v2.1, E_phys and the set\'s classical baseline(s); win / loss only when '
               'the 95 % interval of the difference excludes 0',
               'VED predictions are scaled by the valid-energy share of the trip (EnergyNet / RoadData label definition)',
               'reference rows label / moves_like / class_median / regression / dev_median are fitted on dev only']}
    for k in arms:
        d = json.load(open(os.path.join(od, 'preds', k, 'DONE.json')))
        out['arms'][k] = {'hash': regs[k]['hash'], 'unchanged_after': d['unchanged'], 'tracks': regs[k]['tracks'],
                          **regs[k]['meta'], 'exam_cpu': d['cpu']}
    units = {}
    for (track, k), p in rows.items():
        err = p['error'].astype(str).str.len() > 0
        units[(track, k)] = (_units_for(Fr, p.where(~err)), int(err.sum()))
    for name in R.REF_ROWS:
        p = pd.DataFrame({'pred': rp[name].values})
        units[('ref', name)] = (_units_for(Fr, p), 0)
    rng = np.random.default_rng(C.SEED)
    score_rows = []
    for track in ('table', 'trace'):
        board_rows = [k for k in arms if track in regs[k]['tracks']]
        anchor_arms = [k for k in ('enet_v21', 'climatology') if k in arms and k not in board_rows]
        tab = {}
        allrows = [(track, k) for k in board_rows] + [('table', k) for k in anchor_arms] + [('ref', r) for r in R.REF_ROWS]
        for key in allrows:
            U, nerr = units[key]
            s = S.real_summary(U)
            s['_n_error_records'] = nerr
            s['_kind'] = 'reference' if (key[0] == 'ref' or regs.get(key[1], {}).get('meta', {}).get('reference')) else 'arm'
            s['_D16'] = S.d16(s) if mode != 'exam' else None
            tab[key[1]] = s
            for sk, u in U.items():
                if sk.startswith('VED_arch'):
                    continue
                sc = S.row_scores(u.y.values, u.pred.values, u[C.QKEYS].values)
                keep = [c for c in ('unit', 'cluster', 'y', 'pred', 'v_kmh', 'T', 'climb_per_km', 'dist_km', 'arch',
                                    'test', 'vehicle_id') if c in u]
                f = u[keep].copy() if keep else pd.DataFrame(index=u.index)
                f = f.assign(**{q: u[q].values for q in C.QKEYS}, ape=sc['ape'], crps_log=sc['crps_log'],
                             set=sk, row=key[1], track=track, row_track=key[0])
                score_rows.append(f)
        paired = {}
        for k in board_rows:
            Uk = units[(track, k)][0]
            paired[k] = {}
            for sk, u in Uk.items():
                if sk.endswith('_cars') or sk.startswith('VED_arch'):
                    continue
                paired[k][sk] = {}
                for anc in ANCHORS + CLASSICAL[set_family(sk)]:
                    if anc == k:
                        continue
                    key = ('ref', anc) if anc in R.REF_ROWS else ('table', anc)
                    if key not in units or sk not in units[key][0] or not len(units[key][0][sk]):
                        continue
                    paired[k][sk][anc] = S.compare(u, units[key][0][sk], rng=rng, nboot=NBOOT_EXAM)
        out['boards'][track] = {'rows': tab, 'paired': paired}
    sc = pd.concat(score_rows, ignore_index=True)
    for c in sc.columns:
        if sc[c].dtype == object:
            sc[c] = sc[c].astype(str)
    sc.to_parquet(os.path.join(od, 'scores.parquet'))
    if ('table', 'enet_v21') in units:
        out['reproduction_enet_v21'] = reproduction(S.real_summary(units[('table', 'enet_v21')][0], False)) \
            if mode == 'exam' else {'note': 'compared with the published exam only in the exam run'}
    if mode == 'exam':
        out['reproduction_refs'] = reproduction_refs({name: S.real_summary(units[('ref', name)][0], False) for name in R.REF_ROWS})
    out['showcase'] = showcase_dump(mode, od, Fr, rows, rp)
    fn = os.path.join(C.RESULTS, f'{"exam" if mode == "exam" else "rehearsal"}_{rid}.json')
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    return fn, out


def showcase_dump(mode, od, Fr, rows, rp):
    """every row's prediction, band and components for the showcase trips, plus each trip's route profile (5 s means
    of speed, grade, elevation, temperature) -> showcase.npz + showcase.json."""
    sc = json.load(open(os.path.join(C.HARNESS, 'showcase.json')))
    trips = [t['trip_id'] for t in sc['trips']] if mode == 'exam' else list(Fr[Fr.recset == 'ved'].trip_id.iloc[[0, 100, 500]])
    split = 'exam' if mode == 'exam' else 'dev'
    D = Data(mode, os.path.basename(od).split('_', 1)[1])
    res, arrays = [], {}
    for tid in trips:
        idx = Fr.index[Fr.trip_id == tid]
        if not len(idx):
            continue
        i = int(idx[0]); r = Fr.loc[i]
        j = int(r.rec)
        tr = D._g(bake.real_trace, split, r.recset, j)
        n5 = len(tr) // 5
        prof = tr[:n5 * 5].reshape(n5, 5, -1).mean(1)
        arrays[f'profile|{tid}'] = prof[:, [0, 1, 2, 3]].astype(np.float32)
        e = {'trip_id': tid, 'set': r.get('set'), 'group': r.get('group'), 'arch': r.get('arch'), 'y_MJ': float(r.y) * 1.0,
             'vr': float(r.vr), 'dist_km': float(r.get('dist_km', np.nan)), 'T': float(r.get('T', np.nan)), 'rows': {}}
        for (track, k), p in rows.items():
            q = p.loc[i]
            e['rows'][f'{k}|{track}'] = {'pred': float(q.pred * r.vr), **{c: float(q[c] * r.vr) for c in C.QKEYS},
                                         'components': [float(q[f'c{j}']) for j in range(5)]}
        for name in R.REF_ROWS:
            v = float(rp[name].iloc[i])
            if np.isfinite(v):
                e['rows'][f'{name}|ref'] = {'pred': v * r.vr}
        res.append(e)
    np.savez_compressed(os.path.join(od, 'showcase.npz'), **arrays)
    json.dump(res, open(os.path.join(od, 'showcase.json'), 'w'), indent=1, default=float)
    return {'file': os.path.relpath(os.path.join(od, 'showcase.json'), C.HARNESS), 'n_trips': len(res),
            'profile': 'showcase.npz profile|<trip>: (n, 4) 5-s means of v m/s, grade, elev rel. m, T_amb C',
            'note': 'pred / quantiles in MJ over the valid seconds (x vr), as scored; components MJ over the whole trip'}


# ============================================================================ driver
def build_exam_records(rid):
    if os.path.exists(os.path.join(C.real_dir('exam'), 'BUILT.json')):
        return
    import build_real
    bake._unlock(rid)
    try:
        t0 = time.time()
        build_real.build('exam')
        print(f'exam records built in {time.time() - t0:.0f} s', flush=True)
    finally:
        bake._relock()


def drive(mode, named=None, resume=False):
    if mode == 'exam':
        if resume:
            if not os.path.exists(bake.LOCK):
                raise Refused('nothing to resume (no results/EXAM_RUNNING.json)')
            lk = json.load(open(bake.LOCK))
            rid = lk['run_id']
            arms, notes = preflight(None, mode)
            if sorted(arms) != sorted(lk['arms']):
                raise Refused(f'registered arms changed since the run started: {sorted(lk["arms"])} vs {sorted(arms)}')
        else:
            if os.path.exists(bake.LOCK):
                raise Refused('an exam run is in progress (results/EXAM_RUNNING.json); use --resume')
            arms, notes = preflight(named, mode)
            if named:
                raise Refused('the exam scores ALL registered arms; --arm is only for --check / --rehearsal')
            rid = time.strftime('%Y%m%dT%H%M%S')
            json.dump({'run_id': rid, 'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
                       'arms': {k: r['hash'] for k, r in arms.items()}, 'harness_hash': armlib.harness_hash()[0]},
                      open(bake.LOCK, 'w'), indent=1)
        build_exam_records(rid)
    else:
        arms, notes = preflight(named, mode)
        rid = time.strftime('%Y%m%dT%H%M%S')
    for n in notes:
        print(n, flush=True)
    os.makedirs(outdir(mode, rid), exist_ok=True)
    env = dict(os.environ)
    if mode == 'exam':
        env['BAKEOFF_EXAM_RUN'] = rid
    for target in ['__refs__'] + sorted(arms):
        if os.path.exists(os.path.join(outdir(mode, rid), 'preds', target, 'DONE.json')):
            continue
        print(f'== {target}', flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), '--worker', target, '--mode', mode,
                              '--run-id', rid], env=env)
        if rc != 0:
            raise Refused(f'worker {target} failed (exit {rc})' + ('; fix and use --resume' if mode == 'exam' else ''))
        print(f'== {target} finished in {time.time() - t0:.0f} s', flush=True)
    if mode == 'exam':
        bake._unlock(rid)
    try:
        fn, out = aggregate(mode, rid, arms)
    finally:
        bake._relock()
    if mode == 'exam':
        json.dump({'run_id': rid, 'finished': out['finished'], 'result': os.path.relpath(fn, C.HARNESS),
                   'arms': {k: v['hash'] for k, v in out['arms'].items()}}, open(bake.DONE, 'w'), indent=1)
        os.remove(bake.LOCK)
    print('wrote', fn)
    for tr, v in out['boards'].items():
        print(tr, {k: {s: round(x['MdAPE'], 3) for s, x in r.items() if s in ('VED_ICE', 'VED_HEV', 'EPA_conv_FTP')
                       and isinstance(x, dict) and x.get('n')} for k, r in v['rows'].items()})


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--run', action='store_true')
    g.add_argument('--resume', action='store_true')
    g.add_argument('--rehearsal', action='store_true')
    g.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--arm', action='append', default=None)
    ap.add_argument('--mode', default='exam', help=argparse.SUPPRESS)
    ap.add_argument('--run-id', default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    try:
        if a.worker:
            if a.mode == 'exam':
                rid = os.environ.get('BAKEOFF_EXAM_RUN')
                bake._relock()          # the worker reopens the guard only around its own reads
                run_worker(a.worker, a.mode, rid or a.run_id)
            else:
                run_worker(a.worker, a.mode, a.run_id)
        elif a.check:
            arms, notes = preflight(a.arm, 'exam')
            for n in notes:
                print(n)
            print('CHECK PASSED: the exam can run (python3 exam.py --run). Nothing was read from the exam.')
        elif a.run:
            drive('exam', a.arm)
        elif a.resume:
            drive('exam', None, resume=True)
        elif a.rehearsal:
            drive('rehearsal', a.arm)
    except Refused as e:
        print('REFUSED:\n' + str(e), flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
