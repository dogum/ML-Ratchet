"""The River Bake-off exam: run ONCE, after every arm is registered, for all registered arms together.

usage: python3 exam.py --check [--arm ID ...]   preflight only (no exam data): hashes, registry, plan, CPU estimate
       python3 exam.py --run                    THE exam (once): every registered arm + every reference row
       python3 exam.py --resume                 continue an interrupted run (same run id; finished jobs are kept)
       python3 exam.py --rehearsal [--arm ID]   the same pipeline on the DEV gate (dev basins, dev window); no exam
                                                data, no lock; writes results/rehearsal_<id>/ (for testing)

Rules it enforces:
  - refuses to start if results/EXAM_DONE.json exists (once only) or a run is already in progress (use --resume);
  - refuses if any registered arm's artefact hash differs from its registration, if any arm folder under
    bakeoff/river/arms_*/ or harness/arms_ref/ is not registered (unless listed in registry/excluded.json with a
    reason), if a named --arm is not registered, or if the reference arm flowcast_v1 is missing;
  - re-hashes each arm right before and right after scoring it;
  - exam data is read only through bake's guard (lock file results/EXAM_RUNNING.json + run id), and the guard is
    closed again before any arm code runs;
  - chooses nothing: every registered arm is scored on every exam basin with >= 365 observed days.
Outputs: results/exam_<id>.json (leaderboards, paired bootstrap, classes, compute, hashes),
results/exam_<id>/scores.parquet (per basin x period x protocol x lead x row), results/exam_<id>/showcase.npz
(daily series of every row for the showcase basins), then results/EXAM_DONE.json.
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

REQUIRED_REFS = ['flowcast_v1']
EXCLUDED = os.path.join(C.REGISTRY, 'excluded.json')


class Refused(Exception):
    pass


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
    """Returns (arms: {id: registry record}, notes). Raises Refused with every reason found."""
    reasons, notes = [], []
    if mode == 'exam' and os.path.exists(bake.DONE):
        raise Refused(f'the exam already ran ({json.load(open(bake.DONE)).get("run_id")}); it runs once')
    sp = bake._splits()  # raises on a hash mismatch
    notes.append(f'splits.json sha256 OK ({C.SPLITS_SHA256[:8]}), {len(sp["camels_us"]["exam"])} exam basins')
    for p in (os.path.join(C.BANK, 'MANIFEST.json'), os.path.join(C.HARNESS, 'showcase.json')):
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
            notes.append(f'{k}: harness code changed since this arm registered (scoring is unaffected unless bake/'
                         f'score changed; recorded)')
    est = sum(r['smoke'].get('exam_cpu_estimate_min', 0) for r in arms.values())
    notes.append(f'{len(arms)} arms: {", ".join(arms)}; estimated predict CPU {est:.0f} min (+ references ~10 min)')
    return arms, notes


# ============================================================================ worker (one arm, or the references)
def jobs_for(protocols, periods):
    J = []
    for per in periods:
        if 'ungauged' in protocols:
            J.append((per, 'ungauged', None))
        if 'gauged' in protocols:
            J += [(per, 'gauged', L) for L in C.LEADS]
    return J


def job_name(per, protocol, lead):
    return f'{per}_{protocol}' + ('' if lead is None else f'_L{lead}')


class Data:
    """Exam or rehearsal data access. In exam mode every read goes through bake's guard, which is closed again
    before control returns to arm code."""

    def __init__(self, mode, rid):
        self.mode, self.rid = mode, rid
        if mode == 'exam':
            self.periods = ('main', 'recent')
            self.basins = self._g(bake.exam_basins)
            self.show = [x['basin'] for x in json.load(open(os.path.join(C.HARNESS, 'showcase.json')))['basins']]
        else:
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

    def batch(self, bid, per, protocol, lead):
        if self.mode == 'exam':
            return self._g(bake.exam_batch, bid, per, protocol, lead)
        return bake.dev_batch(bid, protocol, lead)

    def obs_full(self, bid, per):
        import waterdata as wd
        src = C.PERIODS[per][1]
        try:
            return wd.streamflow(bid, src).Q_mm
        except FileNotFoundError:
            return None


def outdir(mode, rid):
    return os.path.join(C.RESULTS, f'{"exam" if mode == "exam" else "rehearsal"}_{rid}')


def run_worker(target, mode, rid):
    """target = arm id or '__refs__'. Resumable per job; checkpoints every 100 basins."""
    regs = REG.registered()
    if target != '__refs__' and target not in regs:
        raise Refused(f'{target}: not registered')
    od = os.path.join(outdir(mode, rid), 'rows', target)
    os.makedirs(od, exist_ok=True)
    D = Data(mode, rid)
    if target == '__refs__':
        arm, rec = None, None
        J = [(per, 'ref', None) for per in D.periods] + [(per, 'persistence', L) for per in D.periods for L in C.LEADS]
    else:
        rec = regs[target]
        ok, h, ch = REG.verify(rec)
        if not ok:
            raise Refused(f'{target}: artefact changed since registration: {ch}')
        arm = armlib.Arm(rec['arm_dir'])
        J = jobs_for(rec['protocols'], D.periods)
    cpu = {}
    for per, protocol, lead in J:
        jn = job_name(per, protocol, lead)
        fin = os.path.join(od, f'{jn}.parquet')
        if os.path.exists(fin):
            continue
        part = os.path.join(od, f'{jn}.partial.pkl')
        rows, show = [], {}
        done = set()
        if os.path.exists(part):
            rows, show = pickle.load(open(part, 'rb'))
            done = {r['basin'] for r in rows}
        t0 = time.process_time(); c0 = arm.seconds if arm else 0.0
        for n, bid in enumerate(D.basins):
            if bid in done:
                continue
            obs = D.obs(bid, per)
            if obs is None or int(obs.notna().sum()) < C.MIN_OBS_DAYS:
                continue
            cls = bake.basin_classes(bid)
            if arm is not None:
                b = D.batch(bid, per, protocol, lead)
                try:                    # a failing basin is recorded as an error row; it never stops the exam
                    out = arm.run([armlib.strip_batch(b)])[0]
                    m, Q, nc = S.check_output(out, b['n_target'])
                    r = S.score_real(obs, m, Q, n_cross=nc)
                except Exception as e:  # noqa: BLE001
                    r = {'error': f'{type(e).__name__}: {e}'[:300]}
                    m = Q = None
                r.update({'basin': bid, 'row': target, 'period': per, 'protocol': protocol, 'lead': lead or 0,
                          'classes': ','.join(cls)})
                rows.append(r)
                if bid in D.show and m is not None:
                    show[bid] = np.concatenate([m[:, None], Q], 1).astype(np.float32)
            elif protocol == 'ref':
                for name in ('udev', 'uk12', 'gr4j', 'nwm', 'clim'):
                    s = D._g(R.reference, name, bid, per) if mode == 'exam' else R.reference(name, bid, per)
                    if s is None:
                        continue
                    r = S.score_real(obs, s.values)
                    r.update({'basin': bid, 'row': name, 'period': per, 'protocol': R.REF_ROWS[name]['protocol'],
                              'lead': 0, 'classes': ','.join(cls)})
                    rows.append(r)
                    if bid in D.show:
                        show[f'{name}|{bid}'] = s.values.astype(np.float32)
                if bid in D.show:
                    show[f'obs|{bid}'] = obs.values.astype(np.float32)
            else:  # persistence at this lead
                full = D._g(D.obs_full, bid, per) if mode == 'exam' else D.obs_full(bid, per)
                s = R.reference('persistence', bid, per, obs_full=full, lead=lead)
                r = S.score_real(obs, s.values)
                r.update({'basin': bid, 'row': 'persistence', 'period': per, 'protocol': 'gauged', 'lead': lead,
                          'classes': ','.join(cls)})
                rows.append(r)
                if bid in D.show:
                    show[f'persistence|{bid}'] = s.values.astype(np.float32)
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
        cpu[jn] = {'predict_cpu_s': round((arm.seconds - c0) if arm else 0.0, 1),
                   'process_cpu_s': round(time.process_time() - t0, 1)}
        print(f'{target} {jn} done: {len(df)} rows', flush=True)
    if arm is not None:
        ok, h, ch = REG.verify(rec)
        json.dump({'hash_after': h, 'unchanged': ok, 'changed_files': ch, 'cpu': cpu},
                  open(os.path.join(od, 'DONE.json'), 'w'), indent=1)
        if not ok:
            raise Refused(f'{target}: artefact changed DURING the exam: {ch}')
    else:
        json.dump({'cpu': cpu}, open(os.path.join(od, 'DONE.json'), 'w'), indent=1)


# ============================================================================ aggregation
def aggregate(mode, rid, arms):
    od = outdir(mode, rid)
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(od, 'rows', '*', '*.parquet')))]
    sc = pd.concat(frames, ignore_index=True)
    sc.to_parquet(os.path.join(od, 'scores.parquet'))
    import waterdata as wd
    attrs = wd.attributes('camels')
    classes = {b: bake.basin_classes(b) for b in sc.basin.unique()}
    in531 = set(attrs.index[attrs.in_531])
    periods = ('main', 'recent') if mode == 'exam' else ('dev',)
    rng = np.random.default_rng(C.SEED)
    regs = REG.registered()
    out = {'run_id': rid, 'mode': mode, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'harness_hash': armlib.harness_hash()[0], 'splits_sha256': C.SPLITS_SHA256,
           'bank': {s: {k: v['sha256'][:16] for k, v in d['files'].items()} for s, d in
                    json.load(open(os.path.join(C.BANK, 'MANIFEST.json')))['splits'].items()},
           'arms': {}, 'references': R.REF_ROWS, 'boards': {},
           'notes': ['medians over basins; paired bootstrap over basins of the difference in medians (10,000 '
                     'resamples, default_rng(20261118)); win/loss only when the 95 % interval excludes 0',
                     'crps = 5-quantile approximation (2 x mean pinball loss), mm/day; crps_rel = crps / mean obs',
                     'point references (U-dev, U-k12, GR4J, NWM, climatology, persistence) have no bands: their crps '
                     'is the MAE and coverage is n/a',
                     'NWM recent exam is WY2016-2022 (NWM v3.0 ends Jan 2023)']}
    for k in arms:
        d = json.load(open(os.path.join(od, 'rows', k, 'DONE.json')))
        out['arms'][k] = {'hash': regs[k]['hash'], 'unchanged_after': d['unchanged'], 'protocols': regs[k]['protocols'],
                          **regs[k]['meta'], 'exam_cpu': d['cpu']}
    for per in periods:
        boards = {'ungauged': (None, [k for k in arms if 'ungauged' in regs[k]['protocols']] +
                               ['udev', 'uk12', 'gr4j', 'nwm', 'clim'])}
        for L in C.LEADS:
            boards[f'gauged_L{L}'] = (L, [k for k in arms if 'gauged' in regs[k]['protocols']] +
                                      ['persistence', 'gr4j', 'nwm'])
        for bname, (L, rows) in boards.items():
            tab, per_row = {}, {}
            for row in rows:
                if row in arms:
                    proto = 'ungauged' if L is None else 'gauged'
                    g = sc[(sc.row == row) & (sc.period == per) & (sc.protocol == proto) & (sc.lead == (L or 0))]
                elif row == 'persistence':
                    g = sc[(sc.row == row) & (sc.period == per) & (sc.lead == L)]
                else:
                    g = sc[(sc.row == row) & (sc.period == per)]
                if 'error' in g:
                    nerr = int(g['error'].notna().sum())
                    g = g[g['error'].isna()] if nerr else g
                else:
                    nerr = 0
                g = g.set_index('basin')
                per_row[row] = g
                s = S.summarize(g)
                s['n_errors'] = nerr
                s['nse_531'] = float(g.loc[[b for b in g.index if b in in531], 'nse'].median()) if len(g) else None
                s['by_class'] = {c: v for c, v in S.by_class(g, classes).items()}
                s['kind'] = 'arm' if row in arms else 'reference'
                s['protocol'] = (('ungauged' if L is None else 'gauged') if row in arms else R.REF_ROWS[row]['protocol'])
                tab[row] = s
            vs = {}
            anchors = (['flowcast_v1', 'udev', 'nwm', 'gr4j'] if L is None else ['flowcast_v1', 'persistence', 'nwm', 'gr4j'])
            for row in rows:
                if row not in arms:
                    continue
                vs[row] = {}
                for anc in anchors:
                    if anc == row or anc not in per_row:
                        continue
                    vs[row][anc] = S.compare(per_row[row], per_row[anc], rng=rng, classes=classes)
            out['boards'][f'{per}/{bname}'] = {'rows': tab, 'paired': vs}
    # showcase
    show = {}
    for f in glob.glob(os.path.join(od, 'rows', '*', '*_show.npz')):
        tgt = os.path.basename(os.path.dirname(f))
        jn = os.path.basename(f)[:-9]
        z = np.load(f)
        for k in z.files:
            key = f'{tgt}|{jn}|{k}' if tgt != '__refs__' else f'{k.split("|")[0]}|{jn}|{k.split("|")[1]}'
            show[key] = z[k]
    np.savez_compressed(os.path.join(od, 'showcase.npz'), **show)
    out['showcase'] = {'file': os.path.relpath(os.path.join(od, 'showcase.npz'), C.HARNESS),
                       'key': '<row>|<period>_<protocol>[_L<lead>]|<basin>: (days, 6) mean,q05..q95 for arms; '
                              '(days,) for references; obs|<period>_ref|<basin> = observed',
                       'basins': json.load(open(os.path.join(C.HARNESS, 'showcase.json')))['basins'] if mode == 'exam' else
                       REG.smoke_basins()[:2]}
    fn = os.path.join(C.RESULTS, f'{"exam" if mode == "exam" else "rehearsal"}_{rid}.json')
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    return fn, out


# ============================================================================ driver
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
    else:
        arms, notes = preflight(named, mode)
        rid = time.strftime('%Y%m%dT%H%M%S') if not resume else sorted(
            glob.glob(os.path.join(C.RESULTS, 'rehearsal_*/')))[-1].rstrip('/').split('rehearsal_')[1]
    for n in notes:
        print(n, flush=True)
    os.makedirs(outdir(mode, rid), exist_ok=True)
    env = dict(os.environ)
    if mode == 'exam':
        env['BAKEOFF_EXAM_RUN'] = rid
    for target in ['__refs__'] + sorted(arms):
        if os.path.exists(os.path.join(outdir(mode, rid), 'rows', target, 'DONE.json')):
            continue
        print(f'== {target}', flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), '--worker', target, '--mode', mode,
                              '--run-id', rid], env=env)
        if rc != 0:
            raise Refused(f'worker {target} failed (exit {rc}); fix and use --resume' if mode == 'exam' else
                          f'worker {target} failed (exit {rc})')
        print(f'== {target} finished in {time.time() - t0:.0f} s', flush=True)
    fn, out = aggregate(mode, rid, arms)
    if mode == 'exam':
        json.dump({'run_id': rid, 'finished': out['finished'], 'result': os.path.relpath(fn, C.HARNESS),
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
    ap.add_argument('--rehearsal-resume', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    try:
        if a.worker:
            if a.mode == 'exam':
                bake._relock()          # the worker reopens the guard only around its own reads
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
            drive('rehearsal', a.arm, resume=a.rehearsal_resume)
    except Refused as e:
        print('REFUSED:\n' + str(e), flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
