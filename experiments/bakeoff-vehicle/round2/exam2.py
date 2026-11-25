"""Vehicle Bake-off ROUND 2: exam-2. A copy of harness/exam.py (v1.0, unchanged there) with the round-2 changes below.
Run ONCE, by the owner, after every round-2 arm is registered in round2/registry/ (round2/register2.py).

usage: python exam2.py --check [--arm ID ...]   preflight only (no exam data): hashes, registry, plan, CPU estimate
       python exam2.py --run                    exam-2 (once): every round-2 registered arm + every reference row
       python exam2.py --resume                 continue an interrupted run (same run id; finished jobs are kept)
       python exam2.py --rehearsal [--arm ID]   the same pipeline on the DEV gate (no exam data, no lock);
                                                writes round2/results/rehearsal2_<id>.json (for testing)

ROUND-2 CHANGES (everything else is v1.0's code and text below):
  - own registry round2/registry/, results round2/results/, lock EXAM2_RUNNING.json, done file EXAM2_DONE.json. v1.0's
    results/EXAM_DONE.json, harness/registry and v1.0's lock are never written (v1.0's registry records and exam JSON are
    read for the reproduction checks);
  - exam records: harness/build_real.py into round2/work/real/exam, under exam-2's lock (bake's guard);
  - arm folders: round2/arms/<arm>/; reference arms re-run in the same exam and required: EnergyNet v2.1, climatology
    and v1.0's trace_gru, trace_transformer, tab_mlp, tab_gp (registered in round2/registry from their repo folders;
    their hashes must equal v1.0's registry hashes, checked by --check);
  - reproduction: EnergyNet v2.1 vs training/exam3.json (as v1.0) and every v1.0 arm vs its own v1.0 exam numbers
    (results/exam_20261123T204251.json), per set;
  - PREREG.md section 5's comparisons are added under out['prereg'] (pairs in round2/prereg_pairs.json, fixed at
    registration; E16 paired bootstrap; shared-bias set count; cross-family per-unit error correlation);
  - full ZTBus: a marked HOOK, ztbfull_hook(), for the owner's harness/exam_supp_ztbfull.py; 'not run' until wired.

v1.0 text:

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

R2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R2)
import paths as P  # noqa: E402   (round-2 path layer: harness on sys.path, C.REAL = round2/work/real)
import config as C  # noqa: E402
C.RESULTS = P.RESULTS
C.REGISTRY = P.REGISTRY
os.makedirs(C.RESULTS, exist_ok=True)
os.makedirs(C.REGISTRY, exist_ok=True)
P.setup()                         # bake reads round-2 banks (C.BANK = round2/work/banks)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import bake  # noqa: E402
bake.LOCK = os.path.join(C.RESULTS, 'EXAM2_RUNNING.json')
bake.DONE = os.path.join(C.RESULTS, 'EXAM2_DONE.json')
import score as S  # noqa: E402
import refs as R  # noqa: E402
import armlib  # noqa: E402
import register as REG  # noqa: E402
P.patch_registry(REG)          # registry arm_dir (worktree / cloud path) -> this checkout

REQUIRED_REFS = ['enet_v21', 'climatology', 'trace_gru', 'trace_transformer', 'tab_mlp', 'tab_gp']
V1_REFS = {'trace_gru': 'arms_trace/gru', 'trace_transformer': 'arms_trace/transformer', 'tab_mlp': 'arms_table/tab_mlp',
           'tab_gp': 'arms_table/tab_gp', 'enet_v21': 'harness/arms_ref/enet_v21',
           'climatology': 'harness/arms_ref/climatology'}
V1_REGISTRY = os.path.join(P.HARNESS, 'registry')
V1_EXAM = os.path.join(P.BAKEOFF, 'results', 'exam_20261123T204251.json')
EXCLUDED = os.path.join(C.REGISTRY, 'excluded.json')
ANCHORS = ['enet_v21', 'ephys']
CLASSICAL = {'VED': ['moves_like', 'label'], 'EPA': ['regression'], 'HD': ['dev_median']}
NBOOT_EXAM = 2000
SHARED_BIAS_SETS = ['VED_EV', 'VED_PHEV', 'Mersin_diesel', 'Mersin_cng', 'ICCT']
PREREG_PAIRS = os.path.join(R2, 'prereg_pairs.json')   # [[a, b, why], ...], fixed before exam-2 (not in registry/)


class Refused(Exception):
    pass


def set_family(key):
    return 'VED' if key.startswith('VED') else ('EPA' if key.startswith('EPA') else 'HD')


# ============================================================================ full ZTBus hook (the owner wires this)
def ztbfull_hook(mode, rid, arms, od):
    """>>> HOOK: full ZTBus (5,847 exam legs, 1,115 missions; EnergyNet's protocol including layover records). <<<
    The owner's harness/exam_supp_ztbfull.py builds these exam records. Wire it here: build the records under exam-2's
    lock (bake._unlock(rid) ... bake._relock()), run every arm in `arms` on them (one subprocess per arm, as drive()
    does), score legs and missions with score.py's units, and return {'status': 'done', 'rows': {arm_id:
    {'ZTBfull_legs': summary, 'ZTBfull_missions': summary}}, 'paired': ...}. Until then it returns 'not run' and the
    rest of exam-2 runs unchanged. drive() calls it after the main sets, before aggregation; aggregate() stores the
    result under out['ztbus_full']."""
    try:
        import exam_supp_ztbfull  # noqa: F401   (harness/, on sys.path through paths.py)
    except ImportError:
        return {'status': 'not run', 'why': 'harness/exam_supp_ztbfull.py is not on this branch; hook not wired'}
    return {'status': 'not run', 'why': 'harness/exam_supp_ztbfull.py found but not yet wired into exam2.ztbfull_hook'}


# ============================================================================ preflight
def discover_arm_dirs():
    out = set()
    for f in glob.glob(os.path.join(P.ARMS, '*', 'ratchet.json')):
        d = os.path.dirname(os.path.abspath(f))
        if os.path.basename(d) != 'common' and os.path.exists(os.path.join(d, 'predict.py')):
            out.add(d)
    return sorted(out)


def v1_reference_check(regs):
    """the v1.0 arms re-run as references must be byte-identical to v1.0's registered artefacts."""
    reasons = []
    for k in V1_REFS:
        if k not in regs:
            continue
        p = os.path.join(V1_REGISTRY, f'{k}.json')
        if not os.path.exists(p):
            reasons.append(f'{k}: no v1.0 registry record')
            continue
        v1 = json.load(open(p))
        # per-file sha256, path separators normalised (the total hash is over the relpath strings, which use '\' on
        # Windows and '/' on Linux, so only the per-file hashes are comparable across machines)
        a = {f.replace('\\', '/'): h for f, h in v1['files'].items()}
        b = {f.replace('\\', '/'): h for f, h in regs[k]['files'].items()}
        changed = sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))
        if changed:
            reasons.append(f'{k}: artefact differs from its v1.0 registered files: {changed}')
    return reasons


def preflight(named=None, mode='exam'):
    reasons, notes = [], []
    if mode == 'exam' and os.path.exists(bake.DONE):
        raise Refused(f'exam-2 already ran ({json.load(open(bake.DONE)).get("run_id")}); it runs once')
    bake._splits()
    notes.append(f'splits.json sha256 OK ({C.SPLITS_SHA256[:8]})')
    for p in (os.path.join(P.BANKS, 'val', 'MANIFEST.json'), os.path.join(P.BANKS, 'train101', 'MANIFEST.json'),
              os.path.join(C.HARNESS, 'showcase.json'), os.path.join(C.REAL, 'dev', 'BUILT.json')):
        if not os.path.exists(p):
            reasons.append(f'missing {p}')
    for b, s in (('val', 'val'), ('train101', 'train')):
        p = os.path.join(P.BANKS, b, 'MANIFEST.json')
        if os.path.exists(p):
            a = json.load(open(p))['files']; v = json.load(open(os.path.join(C.HARNESS, 'bank', s, 'MANIFEST.json')))['files']
            if a['trace_v.npy']['sha256'] != v['trace_v.npy']['sha256']:
                reasons.append(f'round-2 bank {b}: trace_v.npy differs from harness/bank/{s}/MANIFEST.json')
            diff = [f for f in v if a[f]['sha256'] != v[f]['sha256']]
            notes.append(f'round-2 bank {b}: trace_v equal to the v1.0 manifest; differing (cross-platform floats, '
                         f'PREREG A1 / A2): {diff or "none"}')
    if os.path.exists(os.path.join(P.BAKEOFF, 'results', 'EXAM_DONE.json')):
        notes.append('v1.0 exam is done (results/EXAM_DONE.json, untouched); this is exam-2')
    regs = REG.registered()
    reasons += v1_reference_check(regs)
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
    return os.path.join(C.RESULTS, f'{"exam2" if mode == "exam" else "rehearsal2"}_{rid}')


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
           'bank': {s: json.load(open(os.path.join(P.BANKS, s, 'MANIFEST.json')))['files'] for s in ('train101', 'val')},
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
    # ---- round 2 additions
    out['reproduction_v1_arms'] = reproduction_v1(units) if mode == 'exam' else {
        'note': 'compared with the v1.0 exam numbers only in the exam run'}
    out['prereg'] = prereg_block(units, regs, arms, np.random.default_rng(C.SEED))
    zf = os.path.join(od, 'ztbus_full.json')
    out['ztbus_full'] = json.load(open(zf)) if os.path.exists(zf) else {'status': 'not run'}
    out['showcase'] = showcase_dump(mode, od, Fr, rows, rp)
    fn = os.path.join(C.RESULTS, f'{"exam2" if mode == "exam" else "rehearsal2"}_{rid}.json')
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    return fn, out


# ============================================================================ round 2: reproduction and PREREG section 5
def _track_of(regs, k):
    return 'table' if regs[k]['tracks'] == ['table'] else 'trace'


def reproduction_v1(units):
    """every v1.0 arm re-run in exam-2 against its own v1.0 exam numbers (same records, same code: diff must be 0)."""
    v1 = json.load(open(V1_EXAM))
    out = {}
    for k in ('trace_gru', 'trace_transformer', 'tab_mlp', 'tab_gp', 'enet_v21', 'climatology'):
        for tr in ('table', 'trace'):
            if (tr, k) not in units or k not in v1['boards'].get(tr, {}).get('rows', {}):
                continue
            s = S.real_summary(units[(tr, k)][0], False)
            old = v1['boards'][tr]['rows'][k]
            d = {sk: float(s[sk]['MdAPE'] - old[sk]['MdAPE']) for sk in S.D16 + ['ZTBus_legs', 'ZTBus_missions']
                 if sk in s and isinstance(old.get(sk), dict) and old[sk].get('n')}
            out[f'{k}|{tr}'] = {'max_abs_diff_MdAPE': float(max(abs(x) for x in d.values())) if d else None, 'diff': d}
    return out


def _boot_md(u_a, u_b, rng, nboot):
    """paired cluster bootstrap of MdAPE_a - MdAPE_b on common units -> (point diff, resample array)."""
    a = u_a.set_index('unit'); b = u_b.set_index('unit')
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    ea = np.abs(a.pred.values / a.y.values - 1); eb = np.abs(b.pred.values / b.y.values - 1)
    cl = a['cluster'].values if 'cluster' in a else np.arange(len(a))
    codes, inv = np.unique(cl, return_inverse=True)
    order = np.argsort(inv, kind='stable'); starts = np.searchsorted(inv[order], np.arange(len(codes) + 1))
    groups = [order[starts[i]:starts[i + 1]] for i in range(len(codes))]
    G = len(groups); bm = np.empty(nboot)
    for j in range(nboot):
        ix = np.concatenate([groups[g] for g in rng.integers(0, G, G)])
        bm[j] = np.median(ea[ix]) - np.median(eb[ix])
    return float(np.median(ea) - np.median(eb)), bm


def prereg_block(units, regs, arms, rng):
    """PREREG section 5: E16 paired comparisons of the pre-registered pairs (mean over the 16 sets of per-set paired
    cluster-bootstrap differences, 2,000 resamples), bias moves on the five shared-bias sets, the shared-bias set
    count and the cross-family per-unit log-error correlation among round-2 entries. Chooses nothing."""
    res = {'pairs': {}, 'shared_bias': {}, 'cross_family_corr': {}}
    pairs = json.load(open(PREREG_PAIRS)) if os.path.exists(PREREG_PAIRS) else []
    key = lambda k: (_track_of(regs, k), k) if k in regs else ('ref', k)  # noqa: E731
    for a, b, why in pairs:
        if a not in arms or (b not in arms and b not in R.REF_ROWS):
            res['pairs'][f'{a}|{b}'] = {'why': why, 'skipped': 'not both in this run'}
            continue
        Ua, Ub = units[key(a)][0], units[key(b)][0]
        per, boots = {}, []
        for sk in S.D16:
            if sk in Ua and sk in Ub and len(Ua[sk]) and len(Ub[sk]):
                d, bm = _boot_md(Ua[sk], Ub[sk], rng, NBOOT_EXAM)
                lo, hi = np.percentile(bm, [2.5, 97.5])
                per[sk] = {'d_MdAPE': d, 'ci': [float(lo), float(hi)],
                           'bias_a': float(np.median(Ua[sk].pred / Ua[sk].y - 1)), 'bias_b': float(np.median(Ub[sk].pred / Ub[sk].y - 1))}
                boots.append(bm)
        r = {'why': why, 'per_set': per}
        if len(boots) == len(S.D16):
            e = np.mean(np.stack(boots), 0); lo, hi = np.percentile(e, [2.5, 97.5])
            dE = float(np.mean([per[s]['d_MdAPE'] for s in S.D16]))
            r.update(d_E16=dE, ci_E16=[float(lo), float(hi)], verdict='win' if hi < 0 else ('loss' if lo > 0 else 'tie'))
        res['pairs'][f'{a}|{b}'] = r
    # the round-2 family entries = the cross-family ensemble's members (PREREG section 1); else the round-2 learners
    learners = [k for k in arms if regs[k].get('meta', {}).get('group') == 'round2' and not k.endswith('_conformal')]
    xs = [k for k in learners if regs[k]['meta'].get('family') == 'ensemble' and os.path.exists(
        os.path.join(regs[k]['arm_dir'], 'members.json'))]
    if xs:
        mem = json.load(open(os.path.join(regs[xs[0]]['arm_dir'], 'members.json')))
        ids = [json.load(open(os.path.join(regs[xs[0]]['arm_dir'], os.path.dirname(p), 'ratchet.json')))['arm_id']
               for _, p, _ in mem]
        learners = [k for k in ids if k in arms]
    res['learners'] = learners
    for sk in S.D16:
        bias = {k: float(np.median(units[key(k)][0][sk].pred / units[key(k)][0][sk].y - 1))
                for k in learners if sk in units[key(k)][0] and len(units[key(k)][0][sk])}
        sg = {np.sign(v) for v in bias.values()}
        res['shared_bias'][sk] = {'bias': bias, 'shared_sign_over_3pct': bool(bias) and len(sg) == 1 and
                                  all(abs(v) > 0.03 for v in bias.values())}
    res['n_sets_shared_bias'] = int(sum(v['shared_sign_over_3pct'] for v in res['shared_bias'].values()))
    ent = [k for k in learners if regs[k].get('meta', {}).get('family') not in ('ensemble',)]
    for sk in S.D16:
        E = {}
        for k in ent:
            u = units[key(k)][0].get(sk)
            if u is not None and len(u):
                E[k] = pd.Series(np.log(u.pred.values / u.y.values), index=u.unit.values)
        cs = []
        ks = sorted(E)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                x = pd.concat([E[ks[i]], E[ks[j]]], axis=1, join='inner')
                if len(x) > 2:
                    cs.append(float(np.corrcoef(x.values.T)[0, 1]))
        res['cross_family_corr'][sk] = float(np.median(cs)) if cs else None
    v = [x for x in res['cross_family_corr'].values() if x is not None]
    res['median_cross_family_corr_over_E16'] = float(np.median(v)) if v else None
    return res


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
                raise Refused('nothing to resume (no round2/results/EXAM2_RUNNING.json)')
            lk = json.load(open(bake.LOCK))
            rid = lk['run_id']
            arms, notes = preflight(None, mode)
            if sorted(arms) != sorted(lk['arms']):
                raise Refused(f'registered arms changed since the run started: {sorted(lk["arms"])} vs {sorted(arms)}')
        else:
            if os.path.exists(bake.LOCK):
                raise Refused('an exam-2 run is in progress (round2/results/EXAM2_RUNNING.json); use --resume')
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
    # --jobs N (added 25 Nov 2026, before the exam ran): N workers at once, each target still in its own process and
    # hashed before and after; run time only, scoring unchanged. Slow trace ensembles start first.
    todo = [t for t in ['__refs__'] + sorted(arms)
            if not os.path.exists(os.path.join(outdir(mode, rid), 'preds', t, 'DONE.json'))]
    todo.sort(key=lambda t: (not ('gru' in t or 'transformer' in t or t.startswith('r2_x')), t))

    def one(target):
        t0 = time.time()
        log = open(os.path.join(outdir(mode, rid), f'worker_{target}.log'), 'a')
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), '--worker', target, '--mode', mode,
                              '--run-id', rid], env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f'== {target} exit {rc} in {time.time() - t0:.0f} s', flush=True)
        return target, rc
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, JOBS)) as ex:
        res = list(ex.map(one, todo))
    bad = [t for t, rc in res if rc != 0]
    if bad:
        raise Refused(f'workers failed: {bad}' + ('; fix and use --resume' if mode == 'exam' else ''))
    zf = os.path.join(outdir(mode, rid), 'ztbus_full.json')
    if not os.path.exists(zf):                       # >>> full-ZTBus HOOK (see ztbfull_hook) <<<
        json.dump(ztbfull_hook(mode, rid, arms, outdir(mode, rid)), open(zf, 'w'), indent=1, default=float)
    if mode == 'exam':
        bake._unlock(rid)
    try:
        fn, out = aggregate(mode, rid, arms)
    finally:
        bake._relock()
    if mode == 'exam':
        json.dump({'run_id': rid, 'finished': out['finished'], 'result': os.path.relpath(fn, R2),
                   'arms': {k: v['hash'] for k, v in out['arms'].items()}}, open(bake.DONE, 'w'), indent=1)
        os.remove(bake.LOCK)
    print('wrote', fn)
    for tr, v in out['boards'].items():
        print(tr, {k: {s: round(x['MdAPE'], 3) for s, x in r.items() if s in ('VED_ICE', 'VED_HEV', 'EPA_conv_FTP')
                       and isinstance(x, dict) and x.get('n')} for k, r in v['rows'].items()})


JOBS = 1


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
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    global JOBS
    JOBS = a.jobs
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
            print('CHECK PASSED: exam-2 can run (python exam2.py --run). Nothing was read from the exam.')
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
