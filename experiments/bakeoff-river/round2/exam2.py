"""River Bake-off round 2: the exam for the round-2 arms. Run ONCE, by the owner, after every round-2 arm is registered.

usage: python exam2.py --check                  preflight only (no exam data): hashes, registry, plan, CPU estimate
       python exam2.py --run                    THE exam (once): every round-2 registered arm, main + recent, ungauged
       python exam2.py --resume                 continue an interrupted run (same run id; finished jobs are kept)
       python exam2.py --rehearsal [--arm ID]   the same pipeline on the DEV gate (no exam data, no lock)
       add --jobs N to --run / --resume / --rehearsal to score N arms at once (one process each; default 1)

The logic is harness/exam.py's (copied, not imported, so v1.0's exam stays as it ran), with these differences:
  - round 2's own registry (round2/registry/, register2.py), lock and done files (round2/results/EXAM2_RUNNING.json,
    EXAM2_DONE.json) and outputs (round2/results/exam2_<run id>.json and exam2_<run id>/). bake's guard is pointed at
    EXAM2_RUNNING.json, and its access log at round2/results/. v1.0's lock, done file and registry are not touched;
  - ungauged only (no gauged arm changed in round 2);
  - no reference worker: U-dev, U-k12, GR4J, NWM, climatology and every v1.0 arm row (FlowCast v1.0, the five tied
    arms and their conformal wrappers) are read from v1.0's stored exam scores (results/exam_20261123T074907/
    scores.parquet). NWM is not re-scored (NOAA's daily table is not in this checkout). In --rehearsal the same rows
    come from the dev gate: dev references from results/dev_refs.parquet and the v1.0 arms scored live on dev;
  - comparisons fixed in exam2_plan.json (PREREG section 7): every round-2 arm vs FlowCast v1.0, U-dev, NWM and GR4J,
    plus its own pre-declared comparators (ratchet.json "compare_to").
Rules kept from v1.0: refuses if EXAM2_DONE.json exists or a run is in progress, if a registered arm changed, if an
arm folder under round2/arms/ is not registered (unless listed in round2/registry/excluded.json with a reason);
re-hashes each arm before and after scoring it; exam data only through bake's guard, closed before arm code runs;
chooses nothing: every registered arm is scored on every exam basin with >= 365 observed days.
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

import r2
import config as C  # noqa: E402

C.RESULTS = r2.RESULTS            # bake's access log and anything the harness writes goes to round2/results/
os.makedirs(r2.RESULTS, exist_ok=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import bake  # noqa: E402
import score as S  # noqa: E402
import armlib  # noqa: E402
import register2 as REG2  # noqa: E402

bake.LOCK = REG2.LOCK             # round2/results/EXAM2_RUNNING.json
bake.DONE = REG2.DONE             # round2/results/EXAM2_DONE.json
V1_ID = r2.V1_EXAM_ID
V1_SCORES = os.path.join(r2.V1_RESULTS, f'exam_{V1_ID}', 'scores.parquet')
V1_DONE = os.path.join(r2.V1_RESULTS, 'EXAM_DONE.json')
DEV_REFS = os.path.join(r2.V1_RESULTS, 'dev_refs.parquet')
PLAN = os.path.join(r2.ROUND2, 'exam2_plan.json')
EXCLUDED = os.path.join(REG2.REGISTRY, 'excluded.json')
REF_NAMES = ['udev', 'uk12', 'gr4j', 'nwm', 'clim']
ANCHORS = ['flowcast_v1', 'udev', 'nwm', 'gr4j']
PERIODS_EXAM = ('main', 'recent')


class Refused(Exception):
    pass


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def plan():
    return json.load(open(PLAN))


# ============================================================================ preflight
def discover_arm_dirs():
    out = set()
    for f in glob.glob(os.path.join(r2.ROUND2, 'arms', '*', 'ratchet.json')):
        d = os.path.dirname(os.path.abspath(f))
        if os.path.exists(os.path.join(d, 'predict.py')):
            out.add(os.path.normpath(d))
    return sorted(out)


def preflight(named=None, mode='exam'):
    reasons, notes = [], []
    if mode == 'exam' and os.path.exists(bake.DONE):
        raise Refused(f'exam 2 already ran ({json.load(open(bake.DONE)).get("run_id")}); it runs once')
    sp = bake._splits()
    notes.append(f'splits.json sha256 OK ({C.SPLITS_SHA256[:8]}), {len(sp["camels_us"]["exam"])} exam basins')
    for p in (V1_SCORES, V1_DONE, DEV_REFS, PLAN, os.path.join(r2.HARNESS, 'showcase.json')):
        if not os.path.exists(p):
            reasons.append(f'missing {os.path.relpath(p, r2.BAKEOFF)}')
    if os.path.exists(V1_DONE) and json.load(open(V1_DONE)).get('run_id') != V1_ID:
        reasons.append('v1.0 EXAM_DONE.json names a different run')
    regs = REG2.registered()
    if not regs:
        reasons.append('no registered round-2 arms')
    excluded = json.load(open(EXCLUDED)) if os.path.exists(EXCLUDED) else {}
    excluded = {os.path.normpath(os.path.join(r2.BAKEOFF, k)) for k in excluded}
    by_dir = {os.path.normpath(r['arm_dir']): k for k, r in regs.items()}
    for d in discover_arm_dirs():
        if d not in by_dir and d not in excluded:
            reasons.append(f'unregistered arm folder {os.path.relpath(d, r2.BAKEOFF)} (register it with register2.py, '
                           f'or list it in round2/registry/excluded.json with a reason)')
    for k, r in regs.items():
        if not os.path.isdir(r['arm_dir']):
            reasons.append(f'{k}: arm folder {r["arm_dir_rel"]} is gone')
            continue
        ok, h, changed = REG2.verify(r)
        if not ok:
            reasons.append(f'{k}: artefact changed since registration ({r["hash"][:12]} -> {h[:12]}; files {changed})')
        if r['protocols'] != ['ungauged']:
            reasons.append(f'{k}: round 2 scores ungauged arms only')
    if os.path.exists(PLAN):
        pl = plan()
        v1rows = set(pl['v1_rows'])
        if os.path.exists(V1_SCORES):
            have = set(pd.read_parquet(V1_SCORES, columns=['row']).row.unique())
            miss = sorted((v1rows | set(REF_NAMES)) - have)
            if miss:
                reasons.append(f'v1.0 scores lack rows {miss}')
        for k, r in regs.items():
            for c in (r['meta'].get('compare_to') or []):
                if c not in regs and c not in v1rows and c not in REF_NAMES:
                    reasons.append(f'{k}: compare_to {c} is neither a round-2 arm nor a stored v1.0 row')
    arms = regs
    if named:
        arms = {}
        for x in named:
            if x not in regs:
                reasons.append(f'--arm {x}: not registered')
            else:
                arms[x] = regs[x]
    if reasons:
        raise Refused('\n'.join(reasons))
    hh = armlib.harness_hash()[0]
    for k, r in arms.items():
        if r.get('harness_hash') != hh:
            notes.append(f'{k}: harness code changed since this arm registered (recorded)')
    est = sum(r['smoke'].get('exam_cpu_estimate_min', 0) for r in arms.values())
    notes.append(f'{len(arms)} round-2 arms: {", ".join(arms)}; estimated predict CPU {est:.0f} min '
                 f'(one process; references and v1.0 rows are read, not re-scored)')
    notes.append(f'v1.0 rows reused from results/exam_{V1_ID}/scores.parquet (sha256 {sha(V1_SCORES)[:12]}): '
                 f'{", ".join(plan()["v1_rows"])} + {", ".join(REF_NAMES)}')
    return arms, notes


# ============================================================================ worker (one arm)
class Data:
    def __init__(self, mode, rid):
        self.mode, self.rid = mode, rid
        if mode == 'exam':
            self.periods = PERIODS_EXAM
            self.basins = self._g(bake.exam_basins)
            self.show = [x['basin'] for x in json.load(open(os.path.join(r2.HARNESS, 'showcase.json')))['basins']]
        else:
            import register as REG
            self.periods = ('dev',)
            self.basins = bake.dev_basins()
            self.show = REG.smoke_basins()[:2]

    def _g(self, f, *a, **k):
        if self.mode != 'exam':
            return f(*a, **k)
        bake._unlock(self.rid)
        try:
            return f(*a, **k)
        finally:
            bake._relock()

    def obs(self, bid, per):
        return self._g(bake.exam_obs, bid, per) if self.mode == 'exam' else bake._obs(bid, per)

    def batch(self, bid, per):
        if self.mode == 'exam':
            return self._g(bake.exam_batch, bid, per, 'ungauged', None)
        return bake.dev_batch(bid, 'ungauged', None)


def outdir(mode, rid):
    return os.path.join(r2.RESULTS, f'{"exam2" if mode == "exam" else "rehearsal2"}_{rid}')


def v1_arm_dir(row):
    """v1.0 arm folders (rehearsal only: scored live on dev)."""
    reg = json.load(open(os.path.join(r2.HARNESS, 'registry', f'{row}.json')))
    rel = reg['arm_dir'].split('/bakeoff/river/')[1]
    return os.path.join(r2.BAKEOFF, rel)


def run_worker(target, mode, rid):
    regs = REG2.registered()
    od = os.path.join(outdir(mode, rid), 'rows', target)
    os.makedirs(od, exist_ok=True)
    D = Data(mode, rid)
    rec = None
    if target.startswith('v1__'):
        if mode == 'exam':
            raise Refused('v1.0 rows are read from the stored v1.0 exam, never re-scored')
        row = target[4:]
        arm = armlib.Arm(v1_arm_dir(row))
    else:
        if target not in regs:
            raise Refused(f'{target}: not registered')
        rec = regs[target]
        ok, h, ch = REG2.verify(rec)
        if not ok:
            raise Refused(f'{target}: artefact changed since registration: {ch}')
        arm = armlib.Arm(rec['arm_dir'])
        row = target
    cpu = {}
    for per in D.periods:
        jn = f'{per}_ungauged'
        fin = os.path.join(od, f'{jn}.parquet')
        if os.path.exists(fin):
            continue
        part = os.path.join(od, f'{jn}.partial.pkl')
        rows, show, done = [], {}, set()
        if os.path.exists(part):
            rows, show = pickle.load(open(part, 'rb'))
            done = {r['basin'] for r in rows}
        t0 = time.process_time(); c0 = arm.seconds
        for n, bid in enumerate(D.basins):
            if bid in done:
                continue
            obs = D.obs(bid, per)
            if obs is None or int(obs.notna().sum()) < C.MIN_OBS_DAYS:
                continue
            cls = bake.basin_classes(bid)
            b = D.batch(bid, per)
            try:
                out = arm.run([armlib.strip_batch(b)])[0]
                m, Q, nc = S.check_output(out, b['n_target'])
                r = S.score_real(obs, m, Q, n_cross=nc)
            except Exception as e:  # noqa: BLE001
                r = {'error': f'{type(e).__name__}: {e}'[:300]}
                m = Q = None
            r.update({'basin': bid, 'row': row, 'period': per, 'protocol': 'ungauged', 'lead': 0,
                      'classes': ','.join(cls)})
            rows.append(r)
            if bid in D.show and m is not None:
                show[bid] = np.concatenate([m[:, None], Q], 1).astype(np.float32)
            if (n + 1) % 100 == 0:
                pickle.dump((rows, show), open(part + '.tmp', 'wb'))
                os.replace(part + '.tmp', part)
                print(f'{target} {jn} {n + 1}/{len(D.basins)}', flush=True)
        df = pd.DataFrame(rows)
        df.to_parquet(fin + '.tmp.parquet')
        np.savez_compressed(os.path.join(od, f'{jn}_show.npz'), **show)
        os.replace(fin + '.tmp.parquet', fin)
        if os.path.exists(part):
            os.remove(part)
        cpu[jn] = {'predict_cpu_s': round(arm.seconds - c0, 1), 'process_cpu_s': round(time.process_time() - t0, 1)}
        print(f'{target} {jn} done: {len(df)} rows', flush=True)
    if rec is not None:
        ok, h, ch = REG2.verify(rec)
        json.dump({'hash_after': h, 'unchanged': ok, 'changed_files': ch, 'cpu': cpu},
                  open(os.path.join(od, 'DONE.json'), 'w'), indent=1)
        if not ok:
            raise Refused(f'{target}: artefact changed DURING the exam: {ch}')
    else:
        json.dump({'cpu': cpu}, open(os.path.join(od, 'DONE.json'), 'w'), indent=1)


# ============================================================================ stored rows
def stored_rows(mode):
    """Reference rows and v1.0 arm rows: from v1.0's exam (exam) or the dev gate (rehearsal: references only; the
    v1.0 arms are scored live by workers 'v1__<row>')."""
    if mode == 'exam':
        sc = pd.read_parquet(V1_SCORES)
        keep = sc.row.isin(REF_NAMES + plan()['v1_rows']) & sc.period.isin(PERIODS_EXAM)
        sc = sc[keep & ((sc.protocol != 'gauged'))]
        return sc.assign(source=f'v1.0 exam {V1_ID}')
    d = pd.read_parquet(DEV_REFS)
    d = d[d.method.isin(REF_NAMES)].rename(columns={'method': 'row'})
    d['period'] = 'dev'
    d['protocol'] = d.row.map({'udev': 'ungauged', 'uk12': 'ungauged', 'gr4j': 'gauged-calibration',
                               'nwm': 'operational', 'clim': 'gauged-calibration'})
    d['classes'] = d.basin.map(lambda b: ','.join(bake.basin_classes(b)))
    return d.assign(source='v1.0 dev_refs.parquet')


# ============================================================================ aggregation
def aggregate(mode, rid, arms):
    od = outdir(mode, rid)
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(od, 'rows', '*', '*.parquet')))]
    new = pd.concat(frames, ignore_index=True).assign(source='round 2')
    sc = pd.concat([new, stored_rows(mode)], ignore_index=True)
    sc.to_parquet(os.path.join(od, 'scores.parquet'))
    import waterdata as wd
    attrs = wd.attributes('camels')
    classes = {b: bake.basin_classes(b) for b in sc.basin.unique()}
    in531 = set(attrs.index[attrs.in_531])
    periods = PERIODS_EXAM if mode == 'exam' else ('dev',)
    rng = np.random.default_rng(C.SEED)
    regs = REG2.registered()
    pl = plan()
    out = {'run_id': rid, 'mode': mode, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'what': 'River Bake-off round 2 exam (round2/exam2.py)', 'harness_hash': armlib.harness_hash()[0],
           'splits_sha256': C.SPLITS_SHA256, 'plan': pl, 'v1_exam': V1_ID,
           'v1_scores_sha256': sha(V1_SCORES) if mode == 'exam' else None, 'arms': {}, 'boards': {},
           'notes': ['medians over basins; paired bootstrap over basins of the difference in medians (10,000 '
                     'resamples, default_rng(20261118)); win/loss only when the 95 % interval excludes 0',
                     'reference rows and v1.0 arm rows are v1.0\'s stored exam rows (not re-scored); NWM recent is '
                     'WY2016-2022' if mode == 'exam' else 'rehearsal on the dev gate']}
    for k in arms:
        d = json.load(open(os.path.join(od, 'rows', k, 'DONE.json')))
        out['arms'][k] = {'hash': regs[k]['hash'], 'unchanged_after': d['unchanged'], **regs[k]['meta'],
                          'exam_cpu': d['cpu']}
    v1rows = pl['v1_rows']
    for per in periods:
        rows = list(arms) + v1rows + REF_NAMES
        tab, per_row = {}, {}
        for row in rows:
            g = sc[(sc.row == row) & (sc.period == per)]
            if row in arms or row in v1rows:
                g = g[g.protocol == 'ungauged']
            nerr = int(g['error'].notna().sum()) if 'error' in g else 0
            if nerr:
                g = g[g['error'].isna()]
            if not len(g):
                continue
            g = g.set_index('basin')
            per_row[row] = g
            s = S.summarize(g)
            s['n_errors'] = nerr
            s['nse_531'] = float(g.loc[[b for b in g.index if b in in531], 'nse'].median()) if len(g) else None
            s['by_class'] = S.by_class(g, classes)
            s['kind'] = 'round-2 arm' if row in arms else ('v1.0 arm (stored)' if row in v1rows else 'reference')
            tab[row] = s
        vs = {}
        for row in arms:
            vs[row] = {}
            for anc in ANCHORS + list(regs[row]['meta'].get('compare_to') or []):
                if anc == row or anc not in per_row or anc in vs[row]:
                    continue
                vs[row][anc] = S.compare(per_row[row], per_row[anc], rng=rng, classes=classes)
        out['boards'][f'{per}/ungauged'] = {'rows': tab, 'paired': vs}
    show = {}
    for f in glob.glob(os.path.join(od, 'rows', '*', '*_show.npz')):
        tgt = os.path.basename(os.path.dirname(f))
        jn = os.path.basename(f)[:-9]
        z = np.load(f)
        for k in z.files:
            show[f'{tgt}|{jn}|{k}'] = z[k]
    np.savez_compressed(os.path.join(od, 'showcase.npz'), **show)
    fn = os.path.join(r2.RESULTS, f'{"exam2" if mode == "exam" else "rehearsal2"}_{rid}.json')
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    return fn, out


# ============================================================================ driver
def drive(mode, named=None, resume=False, jobs=1):
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
                raise Refused('exam 2 scores ALL registered round-2 arms; --arm is only for --check / --rehearsal')
            rid = time.strftime('%Y%m%dT%H%M%S')
            json.dump({'run_id': rid, 'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
                       'arms': {k: r['hash'] for k, r in arms.items()}, 'plan_sha256': sha(PLAN),
                       'harness_hash': armlib.harness_hash()[0]}, open(bake.LOCK, 'w'), indent=1)
        workers = sorted(arms)
    else:
        arms, notes = preflight(named, mode)
        rid = time.strftime('%Y%m%dT%H%M%S')
        workers = sorted(arms) + [f'v1__{r}' for r in plan()['v1_rows']]
    for n in notes:
        print(n, flush=True)
    os.makedirs(outdir(mode, rid), exist_ok=True)
    env = dict(os.environ)
    if mode == 'exam':
        env['BAKEOFF_EXAM_RUN'] = rid
    todo = [t for t in workers if not os.path.exists(os.path.join(outdir(mode, rid), 'rows', t, 'DONE.json'))]
    running, failed = [], []
    while todo or running:           # one process per arm, at most `jobs` at once (order does not matter)
        while todo and len(running) < max(1, jobs):
            target = todo.pop(0)
            print(f'== {target}', flush=True)
            p = subprocess.Popen([sys.executable, '-X', 'utf8', os.path.abspath(__file__), '--worker', target,
                                  '--mode', mode, '--run-id', rid], env=env)
            running.append((target, p, time.time()))
        time.sleep(2)
        for item in list(running):
            target, p, t0 = item
            if p.poll() is not None:
                running.remove(item)
                if p.returncode != 0:
                    failed.append(f'{target} (exit {p.returncode})')
                print(f'== {target} finished in {time.time() - t0:.0f} s (exit {p.returncode})', flush=True)
    if failed:
        raise Refused(f'workers failed: {failed}' + ('; fix and use --resume' if mode == 'exam' else ''))
    fn, out = aggregate(mode, rid, arms)
    if mode == 'exam':
        json.dump({'run_id': rid, 'finished': out['finished'], 'result': os.path.relpath(fn, r2.ROUND2),
                   'arms': {k: v['hash'] for k, v in out['arms'].items()}}, open(bake.DONE, 'w'), indent=1)
        os.remove(bake.LOCK)
    print('wrote', fn)
    for b, v in out['boards'].items():
        print(b, {k: round(r['nse'], 3) if r.get('nse') is not None else None for k, r in v['rows'].items()})


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
    ap.add_argument('--jobs', type=int, default=1, help='arm workers in parallel (one thread each; default 1)')
    a = ap.parse_args()
    try:
        if a.worker:
            if a.mode == 'exam':
                bake._relock()
            run_worker(a.worker, a.mode, a.run_id)
        elif a.check:
            arms, notes = preflight(a.arm, 'exam')
            for n in notes:
                print(n)
            print('CHECK PASSED: exam 2 can run (python exam2.py --run). Nothing was read from the exam.')
        elif a.run:
            drive('exam', a.arm, jobs=a.jobs)
        elif a.resume:
            drive('exam', None, resume=True, jobs=a.jobs)
        elif a.rehearsal:
            drive('rehearsal', a.arm, jobs=a.jobs)
    except Refused as e:
        print('REFUSED:\n' + str(e), flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
