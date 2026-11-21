"""Prior Wars harness: one adapter per model, the same protocol for all.

python harness.py <model> tune_dev     G-dev settings by repeated k-fold inside dev (+ P and C-dev on dev, reference)
python harness.py <model> tune_sim     G-sim settings on the model's frozen simulated validation set (no real labels)
python harness.py <model> exam_predict all TabPFN arms on exam INPUTS (needs this model's frozen PREREG section)
python harness.py <model> score        exam scored once (needs frozen PREREG; refuses to run twice)

Everything is resumable: each (arm, target) prediction is saved as it finishes.
"""
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np
import pw_paths as _P
import tp_common as tc
import pwmetrics as M

ROOT = tc.ROOT               # = pw_paths.WORK (the sandbox wrote to ROOT/work/<model>; a rerun writes WORK/<model>)
N_EST = 2                    # pre-registered for every arm (CPU budget)
CV_FOLDS, CV_REPS = 5, 2     # G-dev tuning inside dev
SIMVAL_N = 400               # frozen-sim-val rows used to choose G-sim settings
CTX_GRID = [1024, 2048]      # G-sim context sizes; the larger is kept only if it gains > 0.01 headline skill
LC_REPS = 3
PREDICT_BS = 256
SIMDEV_MAX = 2560            # G-sim+dev context cap: every dev row is kept; the sim part shrinks to fit

# Memory rule (Amendment A1, after the 18:00 shared-memory incident): every TabPFN process runs under a 2 GB
# address-space cap (prlimit --as). Measured: (context + test batch) x (features + 8) <= ~220k cells fits. So
# context rows <= R_max(F) and the test batch fills the rest. Batching does not change predictions (test rows attend
# only to the context; preprocessing is fitted on the context).
CELL_BUDGET = 210_000
NO_2048 = True               # Amendment A3: the 2,048-row G-sim step is dropped everywhere (CPU cost)
SIMDEV_SIM_MAX = 1024        # Amendment A3: G-sim+dev uses at most G-sim's 1,024 simulated rows
MIN_BS = 128


def r_max(F):
    return int(min(SIMDEV_MAX, (CELL_BUDGET // (F + 8) - MIN_BS) // 64 * 64))


def cap_rows(n, F, seed=0):
    """indices of a seeded random subset of at most r_max(F) rows (all rows, in order, when they fit)"""
    R = r_max(F)
    if n <= R:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).permutation(n)[:R])


def A(name):
    return importlib.import_module(f'adapters.{name}')


def wdir(name, *p):
    d = os.path.join(ROOT, name, *p)
    os.makedirs(d if not os.path.splitext(d)[1] else os.path.dirname(d), exist_ok=True)
    return d


def log(name, msg):
    s = time.strftime('%H:%M:%S ') + msg
    print(s, flush=True)
    with open(os.path.join(wdir(name), 'log.txt'), 'a') as f:
        f.write(s + '\n')


def prep(a, X, mode):
    X = np.asarray(X, np.float32)
    if mode == 'norm':
        return np.asarray(a.norm(X), np.float32)
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    return X


def _predict_batched(fn, Xt, bs):
    out, i = [], 0
    while i < len(Xt):
        try:
            out.append(fn(Xt[i:i + bs]))
            i += bs
            continue
        except (RuntimeError, MemoryError) as e:
            if not _is_oom(e) or bs <= 16:
                raise
        _release()                      # outside the except block, so the traceback's tensors are already freed
        bs //= 2
        print(f'  out of memory under the cap: test batch -> {bs}', flush=True)
    return np.concatenate(out)


def _release():
    """return freed heap to the OS: under an address-space cap, glibc's retained heap otherwise blocks later fits"""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def _is_oom(e):
    msg = str(e).lower()
    return isinstance(e, MemoryError) or 'alloc' in msg or 'primitive' in msg


def fit_predict(*args, **kw):
    """Each TabPFN fit runs in a forked child that exits afterwards: the address space a fit leaves behind in a
    long-lived process crept up across fits until later, smaller fits no longer fitted under the cap. Results are
    identical (same inputs, same seeds); the parent stays the only TabPFN process at any time (it waits)."""
    import tempfile
    import traceback
    os.makedirs(ROOT, exist_ok=True)
    fd, fn = tempfile.mkstemp(suffix='.npy', dir=ROOT)
    os.close(fd)
    sys.stdout.flush(); sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        rc = 1
        try:
            np.save(fn, np.asarray(_fit_predict_inproc(*args, **kw)))
            rc = 0
        except BaseException:
            traceback.print_exc()
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(rc)
    _, status = os.waitpid(pid, 0)
    try:
        if os.waitstatus_to_exitcode(status) != 0:
            raise RuntimeError(f'TabPFN fit failed in the child process (status {status}); see the log above')
        return np.load(fn, allow_pickle=False)
    finally:
        os.remove(fn)


def _fit_predict_inproc(kind, Xc, yc, Xt, seed=0, n_est=N_EST, n_classes=None, keep_tail=0):
    """Amendment A4: if even a 16-row test batch does not fit, the context's first len - keep_tail rows (the simulated
    part of G-sim+dev; the whole context elsewhere) are cut to 7/8 by a seeded draw and the fit is retried."""
    tail = np.zeros(len(Xc), bool)
    if keep_tail:
        tail[-keep_tail:] = True
    ok = np.isfinite(yc)
    Xc, yc, tail = Xc[ok], yc[ok], tail[ok]
    F = Xc.shape[1]
    keep = cap_rows(len(Xc), F, seed)
    if len(keep) < len(Xc):
        print(f'  context {len(Xc)} rows > R_max({F}) = {len(keep)}: seeded subsample', flush=True)
        Xc, yc, tail = Xc[keep], yc[keep], tail[keep]
    for attempt in range(6):
        _release()
        try:
            return _fit_predict_once(kind, Xc, yc, Xt, seed, n_est)
        except (RuntimeError, MemoryError) as e:
            if not _is_oom(e) or attempt == 5:
                raise
        head = np.flatnonzero(~tail)
        drop = np.random.default_rng(seed + 100 + attempt).permutation(head)[:len(head) // 8]
        km = np.ones(len(Xc), bool)
        km[drop] = False
        Xc, yc, tail = Xc[km], yc[km], tail[km]
        print(f'  A4: context does not fit at a 16-row batch -> {len(Xc)} rows ({tail.sum()} kept tail rows)', flush=True)


def _fit_predict_once(kind, Xc, yc, Xt, seed, n_est):
    F = Xc.shape[1]
    # first test batch from a more conservative 180k-cell budget (the harness process carries more than the probe)
    # Amendment A3 (was a 180k budget in A2); rows of very narrow tables carry a fixed overhead, hence max(F + 8, 36)
    bs = max(64, CELL_BUDGET // max(F + 8, 36) - len(Xc))
    if kind == 'clf' or kind == 'multi':
        cls = np.unique(yc)
        if len(cls) < 2:
            if kind == 'clf':
                return np.full(len(Xt), float(cls[0]) if len(cls) else 0.0)
            return np.full(len(Xt), cls[0] if len(cls) else 0.0)
        m = tc.classifier(n_estimators=n_est, seed=seed)
        m.fit(Xc, yc.astype(int))
        def f(Xb):
            pr = m.predict_proba(Xb)
            if kind == 'clf':
                j = list(m.classes_).index(1) if 1 in list(m.classes_) else None
                return pr[:, j] if j is not None else np.zeros(len(pr))
            return np.asarray(m.classes_)[pr.argmax(1)].astype(float)
        return _predict_batched(f, Xt, bs)
    if len(yc) < 2:     # e.g. an LC subset of 5 BDG2 buildings with only one known year: constant prediction
        return np.full(len(Xt), float(yc[0]) if len(yc) else np.nan)
    m = tc.regressor(n_estimators=n_est, seed=seed)
    m.fit(Xc, yc)
    return _predict_batched(m.predict, Xt, bs)


def target_scores(a, Y, P, tnames=None):
    if hasattr(a, 'items'):
        out = {}
        for n, (y, p, ms) in a.items(Y, P).items():
            out[n] = {m: M.FUN[m](np.asarray(y, float), np.asarray(p, float)) for m in ms}
        return out
    out = {}
    for t in a.TARGETS:
        n = t['name']
        if tnames and n not in tnames:
            continue
        if n not in P:
            continue
        out[n] = {m: M.FUN[m](np.asarray(Y[n], float), np.asarray(P[n], float)) for m in a.metrics(n)}
    return out


def headline(a, sc):
    if hasattr(a, 'HEADLINE_ITEMS'):
        v = []
        for n in a.HEADLINE_ITEMS:
            if n not in sc:
                return float('nan')
            m = list(sc[n].keys())[0]
            s_ = M.primary_skill(m, sc[n][m])
            v.append(s_ if M.HIGHER[m] else -s_)
        return float(np.mean(v))
    v = []
    for t in a.TARGETS:
        n = t['name']
        if n in sc:
            m = a.metrics(n)[0]
            s = M.primary_skill(m, sc[n][m])
            v.append(s if M.HIGHER[m] else -s)
    return float(np.mean(v)) if v else float('nan')


# ------------------------------------------------------------------ tuning on dev
def tune_dev(name):
    a = A(name)
    from sklearn.model_selection import KFold, StratifiedKFold
    X, Y, units = a.dev()
    groups = getattr(a, 'dev_groups', lambda: None)()
    reps_n = getattr(a, 'CV_REPS', CV_REPS)
    res = {'n_dev': int(len(X)), 'grid': {}, 'rule': 'choose prep with the higher dev-CV headline; raw (TabPFN '
           'default) unless norm wins by > 0.01'}
    P = a.P_predict(X)
    res['P_on_dev'] = target_scores(a, Y, P)
    res['P_on_dev_headline'] = headline(a, res['P_on_dev'])
    log(name, f"P on dev headline {res['P_on_dev_headline']:.4f}")
    modes = getattr(a, 'PREP_MODES', ['raw', 'norm'])
    for mode in modes:
        Xp = prep(a, X, mode)
        oof = {t['name']: np.full((reps_n, len(X)), np.nan) for t in a.TARGETS}
        for rep in range(reps_n):
            for t in a.TARGETS:
                n, kind = t['name'], t['kind']
                y = Y[n]
                ok = np.where(np.isfinite(y))[0]
                if groups is not None:
                    from sklearn.model_selection import GroupKFold
                    gk = GroupKFold(min(CV_FOLDS, len(np.unique(groups[ok]))))
                    perm = np.random.default_rng(rep).permutation(len(ok))
                    sp = [(perm[tr], perm[te]) for tr, te in gk.split(ok[perm], groups=groups[ok][perm])]
                elif kind == 'reg':
                    sp = KFold(CV_FOLDS, shuffle=True, random_state=rep).split(ok)
                else:
                    yy = y[ok].astype(int)
                    nf = max(2, min(CV_FOLDS, np.bincount(yy).min())) if len(np.unique(yy)) > 1 else 2
                    sp = (StratifiedKFold(nf, shuffle=True, random_state=rep).split(ok, yy) if len(np.unique(yy)) > 1
                          else KFold(CV_FOLDS, shuffle=True, random_state=rep).split(ok))
                for tr, te in sp:
                    tr, te = ok[tr], ok[te]
                    oof[n][rep, te] = fit_predict(kind, Xp[tr], y[tr], Xp[te], seed=rep)
        reps = []
        for rep in range(reps_n):
            sc = target_scores(a, Y, dict({k: v[rep] for k, v in oof.items()}, **{k: v for k, v in Y.items() if k.startswith('_')}))
            reps.append({'scores': sc, 'headline': headline(a, sc)})
        hl = float(np.mean([r['headline'] for r in reps]))
        res['grid'][mode] = {'headline': hl, 'reps': reps}
        log(name, f'G-dev CV prep={mode} headline {hl:.4f}')
    if hasattr(a, 'C_dev_cv'):
        res['C_dev_cv'] = a.C_dev_cv(X, Y)
        log(name, f"C-dev CV {res['C_dev_cv'].get('headline')}")
    g = res['grid']
    if 'norm' in g and 'raw' in g:
        res['chosen_prep'] = 'norm' if g['norm']['headline'] > g['raw']['headline'] + 0.01 else 'raw'
    else:
        res['chosen_prep'] = modes[0]
    json.dump(res, open(os.path.join(wdir(name), 'tune_dev.json'), 'w'), indent=1, default=float)
    log(name, f"tune_dev chosen prep {res['chosen_prep']}")


# ------------------------------------------------------------------ tuning on the frozen simulated set
def sim_context(a, Xs, Ys, n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xs))[:n]
    return Xs[idx], {k: v[idx] for k, v in Ys.items()}


def tune_sim(name):
    a = A(name)
    fn = os.path.join(wdir(name), 'tune_sim.json')
    res = json.load(open(fn)) if os.path.exists(fn) else {'grid': {}}
    Xs, Ys, _ = a.sim_pool()
    Xv, Yv, _ = a.sim_val()
    rng = np.random.default_rng(123)
    iv = rng.permutation(len(Xv))[:SIMVAL_N]
    Xv, Yv = Xv[iv], {k: v[iv] for k, v in Yv.items()}
    res.update({'n_pool': int(len(Xs)), 'n_val_used': int(len(Xv)),
                'rule': f'prep chosen at ctx {CTX_GRID[0]} (raw unless norm wins by > 0.01); ctx {CTX_GRID[1]} kept '
                        f'only if it beats ctx {CTX_GRID[0]} by > 0.01 headline skill on the frozen simulated set'})
    P = a.P_predict(Xv)
    res['P_on_simval'] = target_scores(a, Yv, P)
    res['P_on_simval_headline'] = headline(a, res['P_on_simval'])
    modes = getattr(a, 'PREP_MODES', ['raw', 'norm'])

    def run(mode, ctx):
        key = f'{mode}@{ctx}'
        if key in res['grid']:
            return res['grid'][key]['headline']
        Xc, Yc = sim_context(a, Xs, Ys, ctx)
        Xcp, Xvp = prep(a, Xc, mode), prep(a, Xv, mode)
        pr = {}
        for t in a.TARGETS:
            t0 = time.time()
            pr[t['name']] = fit_predict(t['kind'], Xcp, Yc[t['name']], Xvp)
            log(name, f"  sim {key} {t['name']} {time.time() - t0:.0f}s")
        sc = target_scores(a, Yv, pr)
        res['grid'][key] = {'scores': sc, 'headline': headline(a, sc)}
        json.dump(res, open(fn, 'w'), indent=1, default=float)
        log(name, f"G-sim {key} headline {res['grid'][key]['headline']:.4f} (P on simval {res['P_on_simval_headline']:.4f})")
        return res['grid'][key]['headline']

    h = {m: run(m, CTX_GRID[0]) for m in modes}
    if len(modes) > 1:
        best = 'norm' if h['norm'] > h['raw'] + 0.01 else 'raw'
    else:
        best = modes[0]
    res['chosen_prep'] = best
    res['r_max'] = r_max(Xs.shape[1])
    if CTX_GRID[1] <= res['r_max'] and not NO_2048:
        hb = run(best, CTX_GRID[1])
        res['chosen_ctx'] = CTX_GRID[1] if hb > h[best] + 0.01 else CTX_GRID[0]
    else:
        res['ctx_note'] = (f'ctx {CTX_GRID[1]} not tried (Amendment A3: dropped for every model for CPU cost; '
                           f'R_max = {res["r_max"]} rows for {Xs.shape[1]} features)')
        res['chosen_ctx'] = CTX_GRID[0]
    json.dump(res, open(fn, 'w'), indent=1, default=float)
    log(name, f"tune_sim chosen {best}@{res['chosen_ctx']}")


# ------------------------------------------------------------------ prereg gate
def prereg_ok(name):
    fn = _P.PREREG_LOG
    if not os.path.exists(fn):
        return False
    L = json.load(open(fn))
    txt = open(_P.PREREG, encoding='utf-8', newline='').read()
    for e in L:
        if name in e['models']:
            # the frozen prefix must still be exactly what was hashed
            pre = txt[:e['n_chars']]
            return hashlib.sha256(pre.encode()).hexdigest() == e['sha256']
    return False


# ------------------------------------------------------------------ exam predictions (inputs only)
def lc_ks(n):
    ks = [k for k in (5, 10, 20, 40, 80, 160, 320) if k < n]
    return ks


def exam_predict(name):
    assert prereg_ok(name), 'PREREG section for this model is not frozen'
    a = A(name)
    td = json.load(open(os.path.join(wdir(name), 'tune_dev.json')))
    ts = json.load(open(os.path.join(wdir(name), 'tune_sim.json')))
    pd_ = td['chosen_prep']
    ps, ctx = ts['chosen_prep'], ts['chosen_ctx']
    Xe, ue = a.exam_X()
    Xd, Yd, ud = a.dev()
    Xs, Ys, _ = a.sim_pool()
    Xc, Yc = sim_context(a, Xs, Ys, ctx)
    del Xs, Ys
    pdir = wdir(name, 'pred')
    jobs = []
    Xe_d, Xd_d = prep(a, Xe, pd_), prep(a, Xd, pd_)
    Xe_s, Xd_s, Xc_s = prep(a, Xe, ps), prep(a, Xd, ps), prep(a, Xc, ps)
    one = lambda n: np.ones((n, 1), np.float32)
    zero = lambda n: np.zeros((n, 1), np.float32)
    # G-dev context: every dev row if it fits R_max, else a seeded subsample (fit_predict, seed 0)
    # G-sim+dev: total <= R_max(F + 1); every dev row kept if that leaves >= 256 sim rows, else dev subsampled (seed 0)
    R = r_max(Xd_s.shape[1] + 1)
    di = np.arange(len(Xd_s)) if len(Xd_s) + 256 <= R else np.sort(
        np.random.default_rng(0).permutation(len(Xd_s))[:R - 256])
    nsd = int(min(len(Xc_s), SIMDEV_SIM_MAX, R - len(di)))
    Xsd = np.vstack([np.hstack([Xc_s[:nsd], zero(nsd)]), np.hstack([Xd_s[di], one(len(di))])])
    Xe_sd = np.hstack([Xe_s, one(len(Xe_s))])
    log(name, f'contexts: G-dev {min(len(Xd_d), r_max(Xd_d.shape[1]))} of {len(Xd_d)} real rows; G-sim {len(Xc_s)} sim rows; '
        f'G-sim+dev {nsd} sim + {len(di)} real (R_max {R})')
    for t in a.TARGETS:
        n, k = t['name'], t['kind']
        jobs.append(('G-dev', n, k, lambda n=n, k=k: fit_predict(k, Xd_d, Yd[n], Xe_d)))
        jobs.append(('G-sim', n, k, lambda n=n, k=k: fit_predict(k, Xc_s, Yc[n], Xe_s)))
        jobs.append(('G-sim+dev', n, k, lambda n=n, k=k: fit_predict(
            k, Xsd, np.concatenate([Yc[n][:nsd], Yd[n][di]]), Xe_sd, keep_tail=len(di))))
    for kk in getattr(a, 'LC_KS', lc_ks(len(Xd))):
        for r in range(LC_REPS):
            idx = np.random.default_rng(1000 * kk + r).permutation(len(Xd))[:kk]
            for t in a.TARGETS:
                n, k = t['name'], t['kind']
                jobs.append((f'LC_k{kk}_r{r}', n, k, lambda n=n, k=k, idx=idx, r=r: fit_predict(
                    k, Xd_d[idx], Yd[n][idx], Xe_d, seed=r)))
    if hasattr(a, 'C_dev_predict'):
        jobs.append(('C-dev', '_all', '*', lambda: a.C_dev_predict(Xd, Yd, Xe)))
    for arm, n, k, f in jobs:
        fn = os.path.join(pdir, f'{arm}__{n}.npy')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        p = f()
        if isinstance(p, dict):
            np.save(fn, np.array([0]))
            for kk_, v in p.items():
                np.save(os.path.join(pdir, f'{arm}__{kk_}.npy'), np.asarray(v, float))
        else:
            np.save(fn, np.asarray(p, float))
        log(name, f'exam_predict {arm} {n} {time.time() - t0:.0f}s')
    log(name, 'exam_predict done')


# ------------------------------------------------------------------ score once
def boot_idx(units, B, seed=0):
    rng = np.random.default_rng(seed)
    uu, inv = np.unique(units, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uu))]
    for _ in range(B):
        pick = rng.integers(0, len(uu), len(uu))
        yield np.concatenate([groups[i] for i in pick])


def score(name, B=1000):
    assert prereg_ok(name), 'PREREG section for this model is not frozen'
    out_fn = os.path.join(wdir(name), 'exam_scores.json')
    assert not os.path.exists(out_fn), 'exam already scored once'
    a = A(name)
    Ye, ue = a.exam_truth()
    Xe, ue2 = a.exam_X()
    units = getattr(a, 'exam_units', lambda: ue)()
    pdir = wdir(name, 'pred')
    arms = {'P': a.P_predict(Xe)}
    for arm in ['G-dev', 'G-sim', 'G-sim+dev', 'C-dev']:
        d = {}
        for t in a.TARGETS:
            f = os.path.join(pdir, f"{arm}__{t['name']}.npy")
            if os.path.exists(f):
                d[t['name']] = np.load(f)
        if d:
            arms[arm] = d
    base = a.baselines_exam() if hasattr(a, 'baselines_exam') else {}
    allarms = dict(arms, **base)
    res = {'model': name, 'n_exam_rows': int(len(Xe)), 'n_units': int(len(np.unique(units))), 'B': B,
           'arms': {}, 'diff_vs_P': {}, 'lc': {}, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    for arm, P in allarms.items():
        sc = target_scores(a, Ye, P)
        res['arms'][arm] = {'scores': sc, 'headline': headline(a, sc) if all(t['name'] in P for t in a.TARGETS)
                            else None}
    # paired bootstrap
    bs = {arm: [] for arm in allarms}
    for idx in boot_idx(units, B):
        Yb = {k: np.asarray(v)[idx] for k, v in Ye.items()}
        for arm, P in allarms.items():
            sc = target_scores(a, Yb, {k: np.asarray(v)[idx] for k, v in P.items()})
            bs[arm].append((sc, headline(a, sc)))
    for arm in allarms:
        r = res['arms'][arm]
        r['ci'] = {}
        for t, ms in r['scores'].items():
            r['ci'][t] = {m: list(np.nanpercentile([b[0][t][m] for b in bs[arm]], [2.5, 97.5])) for m in ms}
        if r['headline'] is not None:
            r['headline_ci'] = list(np.nanpercentile([b[1] for b in bs[arm]], [2.5, 97.5]))
        if arm != 'P':
            d = {}
            for t, ms in r['scores'].items():
                d[t] = {}
                for m in ms:
                    diffs = np.array([b[0][t][m] - p[0][t][m] for b, p in zip(bs[arm], bs['P'])])
                    d[t][m] = {'diff': r['scores'][t][m] - res['arms']['P']['scores'][t][m],
                               'ci': list(np.nanpercentile(diffs, [2.5, 97.5])),
                               'p_better': float(np.nanmean(diffs > 0) if M.HIGHER[m] else np.nanmean(diffs < 0))}
            if r['headline'] is not None:
                hd = np.array([b[1] - p[1] for b, p in zip(bs[arm], bs['P'])])
                d['_headline'] = {'diff': r['headline'] - res['arms']['P']['headline'],
                                  'ci': list(np.nanpercentile(hd, [2.5, 97.5])), 'p_better': float(np.nanmean(hd > 0))}
            res['diff_vs_P'][arm] = d
    # learning curve (point values; spread over repeats)
    import glob
    ks = sorted({int(os.path.basename(f).split('_k')[1].split('_')[0]) for f in glob.glob(f'{pdir}/LC_k*__*.npy')})
    for kk in ks:
        hs, per = [], []
        for r in range(LC_REPS):
            P = {t['name']: np.load(f"{pdir}/LC_k{kk}_r{r}__{t['name']}.npy") for t in a.TARGETS
                 if os.path.exists(f"{pdir}/LC_k{kk}_r{r}__{t['name']}.npy")}
            sc = target_scores(a, Ye, P)
            hs.append(headline(a, sc))
            per.append(sc)
        res['lc'][kk] = {'headline_mean': float(np.mean(hs)), 'headline_sd': float(np.std(hs)), 'reps': per}
    if 'G-dev' in res['arms']:
        res['lc']['all'] = {'k': int(len(a.dev()[0])), 'headline_mean': res['arms']['G-dev']['headline']}
    # keep every bootstrap replicate (same resamples for all arms) so later contrasts (e.g. C3) are paired too
    bsave = {f'{arm}||_headline': np.array([b[1] if b[1] is not None else np.nan for b in bs[arm]], float)
             for arm in allarms}
    for arm in allarms:
        for t, ms in res['arms'][arm]['scores'].items():
            for m in ms:
                bsave[f'{arm}||{t}||{m}'] = np.array([b[0][t][m] for b in bs[arm]], float)
    np.savez_compressed(os.path.join(wdir(name), 'exam_boot.npz'), **bsave)
    json.dump(res, open(out_fn, 'w'), indent=1, default=float)
    log(name, 'scored: ' + ', '.join(f"{k} {v['headline']:.3f}" for k, v in res['arms'].items()
                                    if v['headline'] is not None))


if __name__ == '__main__':
    name, stage = sys.argv[1], sys.argv[2]
    {'tune_dev': tune_dev, 'tune_sim': tune_sim, 'exam_predict': exam_predict, 'score': score}[stage](name)
