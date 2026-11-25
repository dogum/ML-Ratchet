"""Build (and optionally register) a Mondrian conformal wrapper arm around a REGISTERED base arm.

usage (one thread, nice):
    source /home/claude/bakeoff/vehicle/arms_phys/env.sh
    python3 /home/claude/bakeoff/vehicle/arms_phys/conformal/fit_wrapper.py <base_arm_dir> [--register]

Steps (nothing here reads exam data; writes only under arms_phys/conformal/ and, via register.py, harness/registry/):
 1. the base arm must be registered and unchanged (register.py --verify logic);
 2. dev predictions of the base, per declared track: harness evaluate.py <base> --track T --sets dev --no-compare
    --tag base --out conformal/base_preds/<base_id>/ (skipped if already there for the same base hash);
 3. fits the Mondrian shifts per track on ALL dev units (lib/vcf_mondrian.py), plus a grouped 10-fold cross-fit
    (folds of clusters: VED cars, EPA test vehicles, Mersin trips, ICCT tests, ZTBus missions) for an honest coverage
    estimate;
 4. writes <base_id>_conformal/ (predict.py, calib.json, card.md, ratchet.json, work/fit_report.json);
 5. --register: register.py <wrapper> (--replace if it is already registered). A wrapper that fails registration is
    moved to conformal/failed/ so it cannot block the exam's "every arm registered" check.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, '..', '..', 'harness'))
sys.path.insert(0, HARNESS)
sys.path.insert(0, os.path.join(HERE, 'lib'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import config as C  # noqa: E402
import score as S  # noqa: E402
import armlib  # noqa: E402
import register as REG  # noqa: E402
import vcf_mondrian as CF  # noqa: E402

DEV_STRATA = os.path.join(HERE, 'dev_strata.json')
NFOLD = 10
SETS_REPORT = S.D16 + ['ZTBus_legs']


def dev_strata():
    """per dev record (bake.dev_frame order): the stratum read from the inputs (same on both tracks)."""
    if os.path.exists(DEV_STRATA):
        return np.array(json.load(open(DEV_STRATA))['strata'])
    import bake
    Fr = bake.dev_frame()
    st = np.empty(len(Fr), object)
    off = {name: int(Fr.index[Fr.recset == name][0]) for name in bake.SETS}
    for name, s, b in bake._real_batches('dev', 'table'):
        i0 = off[name] + s.start
        st[i0:i0 + b['n']] = CF.strata_of(b)
    json.dump({'strata': st.tolist()}, open(DEV_STRATA, 'w'))
    return st.astype(str)


def unit_table(Fr):
    """record frame (with pred, q..) -> one unit table over the sets with a stratum: set, stratum, unit, cluster, y, Q."""
    U = S.real_units(Fr)
    rows = []
    for k, u in U.items():
        if k not in CF.SET_STRATUM:
            continue
        d = pd.DataFrame({'set': k, 'stratum': CF.SET_STRATUM[k], 'unit': u['unit'].astype(str).values,
                          'cluster': (k.split('_')[0] + ':' + u['cluster'].astype(str)).values, 'y': u['y'].values,
                          'pred': u['pred'].values})
        Q = np.sort(u[CF.QK].values.astype(float), 1)
        for j, q in enumerate(CF.QK):
            d[q] = Q[:, j]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def units_by_stratum(T):
    return {s: (g.y.values, g[CF.QK].values) for s, g in T.groupby('stratum')}


def unit_scores(T, Q):
    s = S.row_scores(T.y.values, T.pred.values, Q)
    return pd.DataFrame({'set': T.set.values, 'stratum': T.stratum.values, 'in50': s['in50'], 'in90': s['in90'],
                         'crps': s['crps_log'], 'w90': s['width90_log']})


def apply_units(T, shifts):
    o = CF.apply({k: T[k].values for k in CF.QK + ['pred']}, T.stratum.values, shifts)
    return np.stack([o[k] for k in CF.QK], 1)


def summarize(sc):
    per = sc.groupby('set').agg(n=('in50', 'size'), cov50=('in50', 'mean'), cov90=('in90', 'mean'),
                                crps_log=('crps', 'mean'), width90_log=('w90', 'median'))
    st = sc.groupby('stratum').agg(n=('in50', 'size'), cov50=('in50', 'mean'), cov90=('in90', 'mean'),
                                   crps_log=('crps', 'mean'))
    d16 = per.loc[[k for k in S.D16 if k in per.index]]
    return {'sets': per.round(4).to_dict('index'), 'strata': st.round(4).to_dict('index'),
            'all_units': {'cov50': float(sc.in50.mean()), 'cov90': float(sc.in90.mean()), 'crps_log': float(sc.crps.mean())},
            'D16_mean': {'cov50': float(d16.cov50.mean()), 'cov90': float(d16.cov90.mean()),
                         'crps_log': float(d16.crps_log.mean()), 'n_sets': int(len(d16))}}


def fit_track(Fr):
    T = unit_table(Fr)
    raw = summarize(unit_scores(T, T[CF.QK].values))
    shifts = CF.fit(units_by_stratum(T))
    ins = summarize(unit_scores(T, apply_units(T, shifts)))
    # grouped K-fold cross-fit (clusters -> folds by a fixed permutation)
    cl = np.unique(T.cluster.values)
    fold_of = dict(zip(np.random.default_rng(C.SEED).permutation(cl), np.arange(len(cl)) % NFOLD))
    f = T.cluster.map(fold_of).values
    Qcv = np.zeros((len(T), 5))
    for k in range(NFOLD):
        te = f == k
        sh = CF.fit(units_by_stratum(T[~te]))
        Qcv[te] = apply_units(T[te], sh)
    cv = summarize(unit_scores(T, Qcv))
    return shifts, {'raw': raw, 'in_sample': ins, 'cross_fit': cv, 'n_units': int(len(T))}


def _reuse_candidates(base, track, released):
    """the arm's own dev-prediction files, released rung first (agents write <arm>/work/eval/<tag>_<track>_dev_preds)."""
    import glob
    c = [os.path.join(base, 'work', 'eval', f'{released}_{track}_dev_preds.parquet')]
    c += sorted(glob.glob(os.path.join(base, 'work', '**', f'*{released}*', '**', f'*_{track}_dev_preds.parquet'),
                          recursive=True))
    c += sorted(glob.glob(os.path.join(base, 'work', '**', f'{released}_{track}_dev_preds.parquet'), recursive=True))
    seen, out = set(), []
    for x in c:
        if os.path.exists(x) and x not in seen:
            seen.add(x); out.append(x)
    return out


def base_preds(base, bid, track, h, released=None, reuse=True):
    """dev predictions of the registered base. First try the arm's own files for its released rung, VERIFIED by
    re-running the arm on the first 24 records of every dev set (verify_preds.py; max relative difference < 1e-6);
    otherwise one full dev pass with evaluate.py. Subprocesses run with cwd=/ (TabPFN's torch stubs have relative
    __file__ names that register.py's audit would resolve inside the bake-off tree)."""
    pdir = os.path.join(HERE, 'base_preds', bid)
    os.makedirs(pdir, exist_ok=True)
    p = os.path.join(pdir, f'base_{track}_dev_preds.parquet')
    stamp = os.path.join(pdir, f'BASE_HASH_{track}')
    cpu = 0.0
    if not (os.path.exists(p) and os.path.exists(stamp) and open(stamp).read().split()[0] == h):
        r0 = os.times()
        src = 'evaluate.py --sets dev'
        for cand in (_reuse_candidates(base, track, released) if reuse and released else []):
            r = subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HERE, 'verify_preds.py'), base, track,
                                cand], capture_output=True, text=True, cwd='/')
            try:
                v = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                v = {'ok': False}
            print(f'reuse check {cand}: {v}', flush=True)
            if v.get('ok'):
                shutil.copy(cand, p); src = f'reused {os.path.relpath(cand, base)} (verified on {v["n_checked"]} records, max rel diff {v["max_rel_diff"]:.1e})'
                break
        else:
            subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HARNESS, 'evaluate.py'), base, '--track',
                            track, '--sets', 'dev', '--no-compare', '--tag', 'base', '--out', pdir], check=True, cwd='/')
        r1 = os.times()
        cpu = (r1.children_user - r0.children_user) + (r1.children_system - r0.children_system)
        open(stamp, 'w').write(h + '\n' + src + '\n')
    Fr = pd.read_parquet(p)
    return Fr, cpu


def fmt_row(k, R):
    a, b, c = R['raw']['sets'].get(k), R['in_sample']['sets'].get(k), R['cross_fit']['sets'].get(k)
    if not a:
        return None
    return (f"| {k} | {a['n']} | {a['cov50']:.2f} / {a['cov90']:.2f} | {b['cov50']:.2f} / {b['cov90']:.2f} | "
            f"{c['cov50']:.2f} / {c['cov90']:.2f} | {a['crps_log']:.3f} | {b['crps_log']:.3f} | {c['crps_log']:.3f} |")


def write_card(wdir, wid, bid, meta, h, tracks, reports, calib, fit_h):
    parts = [f"""# {wid}: {meta.get('label', bid)} + Mondrian conformal bands

**What.** The registered arm `{bid}` (hash `{h[:16]}`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C, `../lib/vcf_mondrian.py`). `pred` and `q50` are the base arm's, so
MdAPE, bias and D16 are identical to `{bid}`; only coverage, CRPS and band width change. Tracks: {', '.join(tracks)}.

**Method.** Strata fixed from the inputs before looking at any residual: powertrain (conventional / hybrid / plug-in /
battery-electric, from `arch`) x regime (heavy duty if class is bus or truck; dynamometer test if the input's
dyno flag is set; otherwise road). Strata with fewer than {CF.MIN_UNITS} dev units fall back to the regime pooled over
powertrains, then the powertrain pooled over regimes, then all units. Scores in log space per band endpoint
(asymmetric CQR): s_hi = log y - log q_hi, s_lo = log q_lo - log y; the shift is the ceil((n+1)(1-a/2))/n quantile of
the stratum's dev units (a = 0.10 for the 90 % band, 0.50 for the 50 % band), every unit weighted equally. Applied as
q_hi' = q_hi e^d_hi, q_lo' = q_lo e^-d_lo, then made monotone. Negative shifts narrow a band: conformal, not widen-only.

**Data.** Fitted on the base arm's dev predictions only (`evaluate.py --sets dev`: VED 16 dev cars, EPA dev families,
Mersin dev week, ICCT dev truck, 5 ZTBus legs; EnergyNet's scoring units). Zero real labels in the base arm's
training; no exam data. Nothing else is learned.

**Dev coverage** (50 % / 90 % band) and mean CRPS (log). "In-sample" = the shifts fitted on all dev units and scored
on the same units (optimistic). "Cross-fit" = grouped 10-fold: shifts refitted without each fold of clusters (VED cars,
EPA vehicles, Mersin trips, ICCT tests, ZTBus missions) and scored on that fold: the honest estimate.
"""]
    for t in tracks:
        R = reports[t]
        rows = [r for r in (fmt_row(k, R) for k in SETS_REPORT) if r]
        m = [R[x]['D16_mean'] for x in ('raw', 'in_sample', 'cross_fit')]
        parts.append(f"""
**Track {t}** ({R['n_units']} dev units). Mean over the 16 D16 sets: cov50 / cov90 raw {m[0]['cov50']:.2f} / {m[0]['cov90']:.2f}
-> in-sample {m[1]['cov50']:.2f} / {m[1]['cov90']:.2f} -> cross-fit {m[2]['cov50']:.2f} / {m[2]['cov90']:.2f}; CRPS
{m[0]['crps_log']:.4f} -> {m[1]['crps_log']:.4f} / {m[2]['crps_log']:.4f}.

| Set | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | In-sample | Cross-fit |
|---|---|---|---|---|---|---|---|
""" + '\n'.join(rows) + """

Coverage per stratum (all dev units of the stratum pooled: the quantity split conformal calibrates; the mean over
sets above weights a 3-unit set like ICCT as much as 2,877 EPA FTP units, so it can move the other way):

| Stratum | units | Raw cov50 / cov90 | In-sample | Cross-fit | Raw CRPS | Cross-fit CRPS |
|---|---|---|---|---|---|---|
""" + '\n'.join(f"| {s} | {a['n']} | {a['cov50']:.2f} / {a['cov90']:.2f} | {R['in_sample']['strata'][s]['cov50']:.2f} / "
                f"{R['in_sample']['strata'][s]['cov90']:.2f} | {R['cross_fit']['strata'][s]['cov50']:.2f} / "
                f"{R['cross_fit']['strata'][s]['cov90']:.2f} | {a['crps_log']:.3f} | {R['cross_fit']['strata'][s]['crps_log']:.3f} |"
                for s, a in R['raw']['strata'].items()) + """

| Stratum | dev units used (source) | 90 % band shift lo / hi | 50 % band shift lo / hi |
|---|---|---|---|
""" + '\n'.join(f"| {s} | {d['n_units']} ({d['source']}) | {d['b90_lo']:+.3f} / {d['b90_hi']:+.3f} | "
                f"{d['b50_lo']:+.3f} / {d['b50_hi']:+.3f} |" for s, d in calib['tracks'][t]['shifts'].items()))
    parts.append(f"""

**Compute.** Fit {fit_h:.3f} CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`{bid}`'s card ({meta.get('compute', {}).get('train_cpu_hours')} CPU-h). Inference = the base arm + microseconds.

**Limits.** Coverage holds on average over units like the dev units, not per car: VED dev is 16 cars (cluster-level
variation is large), EPA dev units are dynamometer tests of other vehicle families, heavy duty is 115 Mersin/ICCT units
(ZTBus legs have only 5 dev units: they use the heavy-duty pool). Exam conditions (more cars, colder or hotter weeks,
other EPA families) can shift coverage. VED car totals are sums of trip predictions; their bands are not recalibrated.
""")
    open(os.path.join(wdir, 'card.md'), 'w').write(''.join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('--register', action='store_true')
    a = ap.parse_args()
    t_cpu0 = time.process_time()
    base = os.path.abspath(a.base)
    meta = armlib.read_meta(base)
    bid = meta['arm_id']
    regs = REG.registered()
    if bid not in regs:
        sys.exit(f'{bid} is not registered: wrap only frozen (registered) arms')
    ok, h, changed = REG.verify(regs[bid])
    if not ok:
        sys.exit(f'{bid} changed since registration: {changed}')
    tracks = list(meta['tracks'])
    wid = f'{bid}_conformal'
    wdir = os.path.join(HERE, wid)
    base_rel = os.path.relpath(base, wdir)
    calib = {'base_id': bid, 'base_hash': h, 'base_rel': base_rel, 'method': 'Mondrian split-conformal (asymmetric CQR '
             'in log space), strata powertrain x road/dyno/hd', 'min_units': CF.MIN_UNITS, 'tracks': {}}
    reports = {}
    child = 0.0
    dev_st = dev_strata()
    for t in tracks:
        Fr, cpu = base_preds(base, bid, t, h, meta.get('released'))
        child += cpu
        shifts, rep = fit_track(Fr)
        calib['tracks'][t] = {'shifts': shifts}
        reports[t] = rep
        m = [rep[x]['D16_mean'] for x in ('raw', 'in_sample', 'cross_fit')]
        print(f"{bid} [{t}] D16-mean cov50/cov90 raw {m[0]['cov50']:.3f}/{m[0]['cov90']:.3f} -> in {m[1]['cov50']:.3f}/"
              f"{m[1]['cov90']:.3f} -> cv {m[2]['cov50']:.3f}/{m[2]['cov90']:.3f}; crps {m[0]['crps_log']:.4f} -> "
              f"{m[1]['crps_log']:.4f} / {m[2]['crps_log']:.4f}", flush=True)
        # check: record-level application reproduces the unit-level in-sample numbers
        o = CF.apply({k: Fr[k].values for k in CF.QK + ['pred']}, dev_st, shifts)
        F2 = Fr.assign(**{k: o[k] for k in CF.QK})
        chk = summarize(unit_scores(unit_table(F2), unit_table(F2)[CF.QK].values))['D16_mean']
        rep['record_level_check'] = chk
        if abs(chk["cov90"] - m[1]["cov90"]) > 2e-3 or abs(chk["cov50"] - m[1]["cov50"]) > 2e-3:
            print('WARNING: record-level and unit-level in-sample coverage differ', chk, m[1], flush=True)
    os.makedirs(os.path.join(wdir, 'work'), exist_ok=True)
    extra = [base_rel, '../lib']
    for x in meta.get('extra_files', []):
        extra.append(os.path.relpath(os.path.normpath(os.path.join(base, x)), wdir))
    json.dump(calib, open(os.path.join(wdir, 'calib.json'), 'w'), indent=1)
    tpl = open(os.path.join(HERE, 'wrapper_predict_template.py')).read().replace('{base_id}', bid)
    open(os.path.join(wdir, 'predict.py'), 'w').write(tpl)
    fit_h = (time.process_time() - t_cpu0 + child) / 3600
    write_card(wdir, wid, bid, meta, h, tracks, reports, calib, fit_h)
    cvs = ', '.join(f"{t} cov50/90 {reports[t]['cross_fit']['D16_mean']['cov50']:.2f}/"
                    f"{reports[t]['cross_fit']['D16_mean']['cov90']:.2f}" for t in tracks)
    rj = {'arm_id': wid, 'label': f"{meta.get('label', bid)} + Mondrian conformal",
          'family': 'conformal', 'family_label': f'Conformal prediction (wrapper around {bid})',
          'group': 'phys', 'tracks': tracks, 'wraps': bid, 'wraps_hash': h,
          'numpy_only': bool(meta['numpy_only']),
          'numpy_only_reason': ('' if meta['numpy_only'] else f"wraps {bid}: {meta.get('numpy_only_reason', '')}"),
          'real_labels_in_training': False,
          'compute': {'train_cpu_hours': round(fit_h, 4),
                      'base_train_cpu_hours': meta.get('compute', {}).get('train_cpu_hours'),
                      'notes': f'conformal fit only (dev predictions of {bid} + shifts); the base arm is not retrained'},
          'feature_engineering': f"none beyond {bid}'s; strata from arch, cls and the dynamometer flag",
          'extra_files': extra, 'margin': 0.0,
          'rule': 'No ladder: one conformal fit on dev residuals (method fixed in advance). Reported raw vs conformal.',
          'rungs': [{'id': 'c0', 'change': 'Mondrian split-conformal bands (asymmetric CQR, log space, strata powertrain '
                     'x road/dyno/hd) fitted on dev residuals', 'decision': 'released',
                     'why': 'the only rung; dev coverage over the D16 sets (grouped 10-fold cross-fit): ' + cvs}],
          'released': 'c0'}
    json.dump(rj, open(os.path.join(wdir, 'ratchet.json'), 'w'), indent=1)
    json.dump({'base_id': bid, 'base_hash': h, 'tracks': reports}, open(os.path.join(wdir, 'work', 'fit_report.json'), 'w'),
              indent=1, default=float)
    print('wrote', wdir, f'fit {fit_h:.4f} CPU-h', flush=True)
    if a.register:
        replace = os.path.exists(os.path.join(C.REGISTRY, f'{wid}.json'))
        r = subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HARNESS, 'register.py'), wdir]
                           + (['--replace'] if replace else []), cwd='/')
        if r.returncode:
            dst = os.path.join(HERE, 'failed', f"{wid}_{time.strftime('%Y%m%dT%H%M%S')}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not replace:
                shutil.move(wdir, dst)
                print(f'!! registration failed: moved {wdir} -> {dst}', flush=True)
            else:
                print('!! re-registration failed; the old registration stands but the folder changed', flush=True)
            sys.exit(r.returncode)


if __name__ == '__main__':
    main()
