"""Integrity pass, 17 Nov 2026: re-score the v1.0 exam (SEEN) with the released weights, add the fair baselines, and
put a bootstrap interval on every row. Nothing here chooses anything; plan in NOTEBOOK.md (2026-11-17).

    python rescore_seen.py repro               # released weights (numpy) -> dev eval, widen re-fit, exam.py's exam
    python rescore_seen.py twinsim [procs]     # dev-calibrated BuildingSim twins of every dev and exam building
    python rescore_seen.py baselines           # cold / short-history / daily / monthly baselines per exam unit
    python rescore_seen.py boot                # rows, best baseline, 95 % bootstrap intervals, verdicts

Outputs under WORK/exam/ (git-ignored); the summary JSON is copied to training/rescore_seen_r3.json by hand.
Twin knobs ('med'): per knob the value most often chosen on the 52 dev buildings (calib.py -> data/dev_knobs.json); HVAC
for electricity-only buildings from the dev electricity-only buildings; per meter type the dev median a, b.
'use': the same by use type where dev has >= 3 buildings (knobs) or >= 3 fits (a, b), else 'med'.
"""
import sys, os, json, time, collections
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, evaluate as E, lcnet as N

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{C.ROOT}/exam'
SEED = 20261117
P_CP = {'2P': 2, '3PH': 3, '3PC': 3, '4P': 4, '5P': 5}
P_TOWT = 174


# ------------------------------------------------------------------ released weights through the training code
class NPNet:
    def __init__(self, W): self.W = W


def np_nets():
    z = np.load(os.path.join(os.path.dirname(HERE), 'loadcast_weights.npz'))
    return {m: NPNet({k.split('/', 1)[1]: z[k] for k in z.files if k.startswith(m + '/')}) for m in ('cold', 'warm')}


def np_predict(net, X, base, bs=16384):
    mu, ls = N.np_forward(net.W, X, base)
    return mu.astype(np.float32), np.exp(ls).astype(np.float32)


N.predict = np_predict
N.load = lambda path: (np_nets()['cold' if path.endswith('_cold.pt') else 'warm'], {})


def repro():
    import exam as X
    os.makedirs(OUT, exist_ok=True); os.makedirs(f'{C.ROOT}/data', exist_ok=True)
    nets = np_nets(); t0 = time.time()
    dev = D.real_units('dev')
    for u in dev: D.prepare(u, ('full', 'w8', 'w2'))
    dv = E.eval_units(dev, nets, modes=X.MODES)
    dsum = E.summarize(dv, keys=('cold', 'warm', 'warm_w8', 'warm_w2'))
    rj = next(r for r in json.load(open(f'{HERE}/ratchet.json')) if r['run'] == 'r3')['dev']
    rep = dict(dev_score=E.dev_score(dv), dev_score_logged=rj['score'], dev=[], widen=[], exam=[])
    for m, v in rj['by_meter'].items():
        if m == 'all': continue
        for k, s in v.items():
            if k == 'n': continue
            for st in ('cv_med', 'nmbe_med', 'cov_mean'):
                rep['dev'].append((m, k, st, s[st], dsum[m][k][st]))
    W = X.fit_widen(dev, nets)
    Wl = json.load(open(f'{HERE}/widen_r3.json'))
    Wl = Wl.get('widen', Wl)
    rep['widen'] = [(k, Wl.get(k), W[k]) for k in W]
    print('dev + widen', round(time.time() - t0), 's', flush=True)
    json.dump(Wl, open(f'{C.ROOT}/data/widen_r3.json', 'w'))          # the exam uses the released factors
    X.main('r3')
    new = json.load(open(f'{OUT}/exam_r3.json')); old = json.load(open(f'{HERE}/exam_r3.json'))
    for m, T in old['by_meter'].items():
        for name, r in T.items():
            for st, v in r.items():
                rep['exam'].append((m, name, st, v, new['by_meter'].get(m, {}).get(name, {}).get(st)))
    C.jdump(rep, f'{OUT}/repro_r3.json')
    for part in ('dev', 'exam'):
        d = [abs(a - b) for *_, a, b in rep[part] if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
        bad = [r for r in rep[part] if r[-1] is None or r[-2] is None or abs(r[-1] - r[-2]) > 1e-6]
        print(part, len(rep[part]), 'stats; max |diff|', max(d) if d else None, '; >1e-6:', len(bad), bad[:8])
    print('dev score', rep['dev_score'], 'logged', rep['dev_score_logged'])
    print('widen diffs', [r for r in rep['widen'] if r[1] != r[2]])


# ------------------------------------------------------------------ dev-calibrated twins
def twin_knobs():
    kn = json.load(open(f'{C.ROOT}/data/dev_knobs.json'))
    def vote(bs, key):
        c = collections.Counter(json.dumps(kn[b]['best']['knobs'].get(key)) for b in bs)
        return json.loads(c.most_common(1)[0][0])
    allb = list(kn)
    elec_only = [b for b in allb if not (set(kn[b]['meters']) & {'chilledwater', 'steam', 'hotwater', 'gas'})]
    base = {k: vote(allb, k) for k in ('sched', 'oa', 'vav_min', 'sat_reset')}
    sys_e = vote(elec_only, 'sys')
    ab = {}
    for m in C.METERS:
        f = [kn[b]['best']['per'][m] for b in allb if m in kn[b]['best']['per']]
        ab[('*', m)] = (float(np.median([x['a'] for x in f])), float(np.median([x['b'] for x in f])), len(f))
        for use in set(kn[b]['use'] for b in allb):
            fu = [kn[b]['best']['per'][m] for b in allb if kn[b]['use'] == use and m in kn[b]['best']['per']]
            if len(fu) >= 3: ab[(use, m)] = (float(np.median([x['a'] for x in fu])), float(np.median([x['b'] for x in fu])), len(fu))
    by_use = {}
    for use in set(kn[b]['use'] for b in allb):
        bs = [b for b in allb if kn[b]['use'] == use]
        if len(bs) >= 3: by_use[use] = {k: vote(bs, k) for k in ('sched', 'oa', 'vav_min', 'sat_reset')}
    return dict(base=base, sys_elec_only=sys_e, ab={f'{k[0]}|{k[1]}': v for k, v in ab.items()}, by_use=by_use)


def knobs_for(TK, use, meters, variant):
    k = dict(TK['by_use'].get(use, TK['base'])) if variant == 'use' else dict(TK['base'])
    ms = set(meters)
    if not (ms & {'chilledwater', 'steam', 'hotwater', 'gas'}): k['sys'] = TK['sys_elec_only']
    else: k['sys'] = 'auto'
    for m in meters:
        key = f'{use}|{m}' if variant == 'use' and f'{use}|{m}' in TK['ab'] else f'*|{m}'
        if key in TK['ab']: k[f'a_{m}'], k[f'b_{m}'] = TK['ab'][key][:2]
    return k


def _sim_one(args):
    bid, site, use, meters, area, variant, TK = args
    import lcsim as L
    k = knobs_for(TK, use, meters, variant)
    try:
        spec = L.make_spec(C.bd.bdg2_spec(bid), meters, k)
        raw, _ = L.simulate_site(spec, site)
        y = L.post(raw, k, area)
        return bid, variant, {m: y[m].astype(np.float32) for m in meters if m in y}, None
    except Exception as e:
        return bid, variant, {}, repr(e)[:200]


def twinsim(procs=3):
    TK = twin_knobs(); C.jdump(TK, f'{OUT}/twin_knobs.json'); print(json.dumps(TK)[:1500], flush=True)
    jobs = []
    for split in ('dev', 'exam'):
        meta = C.bd.bdg2_meta(split=split).set_index('id')
        q = C.bd.bdg2_quality(split=split)
        q = q[q.meter.isin(C.METERS) & q.q_ok]
        for b in sorted(set(q.building_id)):
            r = meta.loc[b]
            for v in ('med', 'use'):
                jobs.append((b, r['site_id'], r['primary_use'], C.meters_of(r['meters']), float(r['floor_area_m2']), v, TK))
    done = {}
    fn = f'{OUT}/twins.npz'
    if os.path.exists(fn):
        z = np.load(fn); done = {k: z[k] for k in z.files}
    jobs = [j for j in jobs if not any(k.startswith(f'{j[5]}|{j[0]}|') for k in done)]
    print(len(jobs), 'twin simulations to run', flush=True)
    t0 = time.time(); errs = []
    with mp.Pool(procs, maxtasksperchild=50) as p:
        for i, (b, v, Y, err) in enumerate(p.imap_unordered(_sim_one, jobs, chunksize=4)):
            if err: errs.append((b, v, err))
            for m, y in Y.items(): done[f'{v}|{b}|{m}'] = y
            if (i + 1) % 200 == 0:
                print(i + 1, 'sims', round(time.time() - t0), 's', flush=True)
                np.savez(fn + '.tmp.npz', **done); os.replace(fn + '.tmp.npz', fn)
    np.savez(fn + '.tmp.npz', **done); os.replace(fn + '.tmp.npz', fn)
    print('twins done', len(done), 'series', round(time.time() - t0), 's; errors', len(errs), errs[:5], flush=True)


# ------------------------------------------------------------------ baselines per exam unit
def shape_guess(dev):
    """Dev TOWT shape per meter type: median over dev building-meters of a ridge TOWT on 2017 load / its mean."""
    cs = collections.defaultdict(list)
    for u in dev:
        f = D.site_feats(u['site']); t = u['tidx']; y = u['y'][t].astype(float)
        cs[u['meter']].append(C.towt_fit(y / np.nanmean(y), f['T'][t], f['tow'][t], ridge=2.0))
    return {m: np.median(np.array(v), 0) for m, v in cs.items()}


def baselines():
    import exam as X
    t0 = time.time()
    dev = D.real_units('dev')
    for u in dev: D.prepare(u, ('full',))
    dI, dS = X.dumb_cold(dev); SH = shape_guess(dev)
    z = np.load(f'{OUT}/twins.npz')
    ex = D.real_units('exam')
    rows = []
    for u in ex:
        D.prepare(u, ('full', 'w8', 'w2'))
        f = D.site_feats(u['site']); t = u['tidx']; y = u['y'][t].astype(float)
        key = (u['use'], u['meter']) if (u['use'], u['meter']) in dI else ('*', u['meter'])
        P = {'dumb': dI[key] * dS[key][f['tow'][t]],
             'shape': dI[key] * C.towt_predict(SH[u['meter']], f['T'][t], f['tow'][t])}
        for v in ('full', 'w8', 'w2'):
            h = u['H'][v]
            P['towt_lc' + ('' if v == 'full' else '_' + v)] = C.towt_predict(h['coef'], f['T'][t], f['tow'][t]) * h['S']
        sc = u['area'] * C.IREF[u['meter']] / 1000.0
        for v in ('med', 'use'):
            k = f"{v}|{u['bid']}|{u['meter']}"
            P['twin' + ('' if v == 'med' else 'use')] = z[k][t] / sc if k in z.files else np.full(len(t), np.nan)
        r = dict(id=u['id'], building_id=u['bid'], meter=u['meter'], use=u['use'])
        for name, p in P.items():
            cv, nm, _ = C.g14(y, p); r[f'{name}_cv'] = cv; r[f'{name}_nmbe'] = nm
            dd, mm = E.daily_monthly(u, y, p)
            r[f'{name}_daily_cv'], r[f'{name}_daily_nmbe'] = dd; r[f'{name}_monthly_cv'], r[f'{name}_monthly_nmbe'] = mm
        rows.append(r)
    pd.DataFrame(rows).to_csv(f'{OUT}/baselines_r3_seen.csv', index=False)
    print(len(rows), 'exam units', round(time.time() - t0), 's', flush=True)


# ------------------------------------------------------------------ rows, intervals, verdicts
COLD_B = ['dumb', 'twin', 'twinuse', 'shape']
NAMES = {'dumb': 'Median guess per use and meter (dev)', 'twin': 'Dev-calibrated BuildingSim twin (dev median knobs, a, b)',
         'twinuse': 'Dev-calibrated BuildingSim twin by use type', 'shape': 'Dev intensity × dev TOWT shape',
         'towt': 'TOWT fitted on 2016 (BuildingData)', 'g14_changepoint': 'Change-point model fitted on 2016 (BuildingData)',
         'towt_lc': 'Ridge TOWT on the same history (LoadCast\'s anchor)', 'towt_lc_w8': 'Ridge TOWT on the same 8 weeks',
         'towt_lc_w2': 'Ridge TOWT on the same 2 weeks', 'g14_changepoint_daily': 'Change-point model, daily (BuildingData)',
         'g14_changepoint_monthly': 'Change-point model, monthly (BuildingData)'}


def row_defs():
    R = []
    for agg in ('', '_daily', '_monthly'):
        R.append((f'cold{agg}', [b + agg for b in COLD_B]))
    R.append(('warm_w2', ['towt_lc_w2'] + COLD_B)); R.append(('warm_w8', ['towt_lc_w8'] + COLD_B))
    R.append(('warm', ['towt', 'g14_changepoint', 'towt_lc']))
    R.append(('warm_daily', ['g14_changepoint_daily', 'towt_lc_daily']))
    R.append(('warm_monthly', ['g14_changepoint_monthly', 'towt_lc_monthly']))
    return R


def boot(units_csv=None, write=True, quiet=False):
    """units_csv: another exam-units file (27 Nov 2026: rescore_noelec.py); write=False returns the dict."""
    ex = pd.read_csv(units_csv or f'{OUT}/exam_units_r3.csv')
    bl = pd.read_csv(f'{OUT}/baselines_r3_seen.csv')
    d = ex.merge(bl.drop(columns=['use', 'building_id']), on=['id', 'meter'], suffixes=('', '_new'))
    # BuildingData's fitted baselines use n - p in the denominators; put them on LoadCast's p = 0 (fair, and only
    # ever lowers their CV): CV0 = CV sqrt((n - p) / n), NMBE0 = NMBE (n - p) / n.
    bd = pd.read_parquet(f'{C.BLDG}/data/proc/baselines/load_bdg2.parquet')
    bd = bd[bd.split == 'exam'][['building_id', 'meter', 'cp_kind', 'towt_n', 'g14_changepoint_n', 'g14_changepoint_daily_n',
                                 'g14_changepoint_monthly_n']]
    d = d.merge(bd, on=['building_id', 'meter'], how='left', suffixes=('', '_bd'))
    pcp = d['cp_kind'].map(P_CP)
    for col, nn, p in (('towt', 'towt_n', P_TOWT), ('g14_changepoint', 'g14_changepoint_n', pcp),
                       ('g14_changepoint_daily', 'g14_changepoint_daily_n', pcp), ('g14_changepoint_monthly', 'g14_changepoint_monthly_n', pcp)):
        n = d[nn].astype(float); f = (n - p) / n
        d[f'{col}_cv_np'] = d[f'{col}_cv']; d[f'{col}_cv'] = d[f'{col}_cv'] * np.sqrt(f)
        d[f'{col}_nmbe_np'] = d[f'{col}_nmbe']; d[f'{col}_nmbe'] = d[f'{col}_nmbe'] * f
    if write: d.to_csv(f'{OUT}/rescore_units_r3_seen.csv', index=False)
    rng = np.random.default_rng(SEED)
    out = dict(seed=SEED, B=2000, note='SEEN exam (v1.0 took it once). Medians of CV(RMSE) %; BuildingData baselines '
               'converted to p = 0. Interval: 2,000 paired bootstrap resamples of exam building-meters.', rows=[])
    for m in C.METERS:
        g = d[d.meter == m].reset_index(drop=True)
        idx = rng.integers(0, len(g), (2000, len(g)))
        for key, bls in row_defs():
            mc = f'{key}_cv'
            meds = {b: float(g[f'{b}_cv'].median()) for b in bls if f'{b}_cv' in g and g[f'{b}_cv'].notna().any()}
            best = min(meds, key=meds.get)
            ok = (g[mc].notna() & g[f'{best}_cv'].notna()).values
            a = g[mc].values; b = g[f'{best}_cv'].values
            diffs = []
            for ii in idx:
                jj = ii[ok[ii]]
                diffs.append(np.median(a[jj]) - np.median(b[jj]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            pt = float(np.median(a[ok]) - np.median(b[ok]))
            verdict = 'win' if hi < 0 else ('loss' if lo > 0 else 'tie')
            r = dict(meter=m, row=key, n=int(ok.sum()), n_all=int(g[mc].notna().sum()), model=float(g[mc].median()),
                     best=best, baselines=meds, model_paired=float(np.median(a[ok])), best_paired=float(np.median(b[ok])),
                     diff=pt, lo=float(lo), hi=float(hi), verdict=verdict,
                     model_nmbe=float(g[f'{key}_nmbe'].median()),
                     g14_model=float(((g[mc] <= 30) & (g[f'{key}_nmbe'].abs() <= 10)).mean()),
                     g14_best=float(((g[f'{best}_cv'] <= 30) & (g[f'{best}_nmbe'].abs() <= 10)).mean()),
                     best_nmbe=float(g[f'{best}_nmbe'].median()))
            if f'{key}_cov' in g: r['cov90'] = float(g[f'{key}_cov'].mean())
            out['rows'].append(r)
            if not quiet: print(f"{m:12s} {key:13s} n {r['n']:4d} model {r['model']:6.1f} best {best:24s} {meds[best]:6.1f} "
                  f"diff {pt:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] {verdict}", flush=True)
        out.setdefault('nowcast', {})[m] = {k: float(g[f'{k}_cv'].median()) for k in ('persistence_1h', 'persistence_24h', 'same_hour_last_week')}
        out.setdefault('p_conversion', {})[m] = {k: [float(g[f'{k}_cv_np'].median()), float(g[f'{k}_cv'].median())]
                                                 for k in ('towt', 'g14_changepoint', 'g14_changepoint_daily', 'g14_changepoint_monthly')}
    if write: C.jdump(out, f'{OUT}/rescore_seen_r3.json')
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'repro': repro()
    elif a[0] == 'twinsim': twinsim(int(a[1]) if len(a) > 1 else 3)
    elif a[0] == 'baselines': baselines()
    elif a[0] == 'boot': boot()
