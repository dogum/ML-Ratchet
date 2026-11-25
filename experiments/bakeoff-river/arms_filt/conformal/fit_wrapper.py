"""Build (and optionally register) a Mondrian conformal wrapper arm around a REGISTERED base arm.

usage (from anywhere; one thread, nice):
    source /home/claude/bakeoff/river/harness/env.sh
    python3 /home/claude/bakeoff/river/arms_filt/conformal/fit_wrapper.py <base_arm_dir> [--register] [--simval]

Steps:
 1. checks the base arm is registered and unchanged (register.py --verify logic);
 2. dev predictions of the base: harness evaluate.py <base> --sets dev --save-preds  -> base_preds/<base_id>/
    (skipped if already there for the same base hash);
 3. fits the Mondrian shifts per protocol/lead job (lib/cf_mondrian.py) on ALL dev basins, and a leave-one-basin-out
    refit per basin for an honest coverage estimate;
 4. writes <base_id>_conformal/ (predict.py, calib.json, card.md, ratchet.json, fit_report.json);
 5. scores the wrapper with evaluate.py --sets dev (in-sample: the shifts were fitted on these basins);
 6. --register: register.py <wrapper>.
Nothing here reads exam data. Writes only under arms_filt/conformal/.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, '..', '..', 'harness'))
sys.path.insert(0, HARNESS)
sys.path.insert(0, os.path.join(HERE, 'lib'))
import numpy as np  # noqa: E402
import config as C  # noqa: E402
import bake  # noqa: E402
import score as S  # noqa: E402
import armlib  # noqa: E402
import register as REG  # noqa: E402
import cf_mondrian as CF  # noqa: E402

DEV_META = os.path.join(HERE, 'dev_meta.json')


def dev_meta():
    if os.path.exists(DEV_META):
        return json.load(open(DEV_META))
    out = {}
    for bid in bake.dev_basins():
        b = bake.dev_batch(bid, 'ungauged')
        out[bid] = {'classes': b['classes'], 'stratum': CF.stratum(b['classes']), 'c': CF.offset(b['static'])}
    json.dump(out, open(DEV_META, 'w'), indent=1)
    return out


def jobs(protocols):
    J = (['ungauged'] if 'ungauged' in protocols else []) + \
        ([f'gauged_L{L}' for L in C.LEADS] if 'gauged' in protocols else [])
    return J


def basin_scores(P, o=None):
    y = P[:, 0].astype(float)
    m = P[:, 1].astype(float)
    Q = P[:, 2:7].astype(float) if o is None else np.stack([o[k] for k in CF.QK], 1)
    Q = np.sort(Q, 1)
    return S.interval_scores(y, m, Q)


def to_out(P):
    return {'mean': P[:, 1].astype(float), **{k: P[:, 2 + j].astype(float) for j, k in enumerate(CF.QK)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('--register', action='store_true')
    ap.add_argument('--simval', action='store_true', help='also score the wrapper on the frozen sim val set')
    ap.add_argument('--no-eval', action='store_true')
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
    pdir = os.path.join(HERE, 'base_preds', bid)
    J = jobs(meta['protocols'])
    stamp = os.path.join(pdir, 'BASE_HASH')
    have = os.path.exists(stamp) and open(stamp).read().strip() == h and \
        all(os.path.exists(os.path.join(pdir, f'base_dev_{j}_preds.npz')) for j in J)
    child_cpu = 0.0
    if not have:
        os.makedirs(pdir, exist_ok=True)
        r0 = os.times()
        subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HARNESS, 'evaluate.py'), base, '--sets', 'dev',
                        '--save-preds', '--tag', 'base', '--out', pdir], check=True)
        r1 = os.times()
        child_cpu += (r1.children_user - r0.children_user) + (r1.children_system - r0.children_system)
        open(stamp, 'w').write(h)
    DM = dev_meta()
    mt = {b: (DM[b]['stratum'], DM[b]['c']) for b in DM}
    calib = {'base_id': bid, 'base_hash': h, 'method': 'Mondrian split-conformal (asymmetric CQR in log space), '
             'strata snow/arid/humid/other, basins weighted equally', 'c_frac': CF.C_FRAC,
             'min_basins': CF.MIN_BASINS, 'jobs': {}}
    report = {'base_id': bid, 'jobs': {}}
    for j in J:
        Z = np.load(os.path.join(pdir, f'base_dev_{j}_preds.npz'))
        preds = {b: Z[b] for b in Z.files}
        sh = CF.fit_job(preds, mt)
        calib['jobs'][j] = {s: sh.get(s, sh['pooled']) for s in CF.STRATA + ['pooled']}
        rows = []
        for b, P in preds.items():
            raw = basin_scores(P)
            ins = basin_scores(P, CF.apply(to_out(P), calib['jobs'][j][mt[b][0]], mt[b][1]))
            loo = CF.fit_job({k: v for k, v in preds.items() if k != b}, mt)
            out = basin_scores(P, CF.apply(to_out(P), loo.get(mt[b][0], loo['pooled']), mt[b][1]))
            rows.append({'basin': b, 'stratum': mt[b][0],
                         **{f'raw_{k}': raw[k] for k in ('cov50', 'cov90', 'crps_rel', 'width90_rel')},
                         **{f'in_{k}': ins[k] for k in ('cov50', 'cov90', 'crps_rel', 'width90_rel')},
                         **{f'loo_{k}': out[k] for k in ('cov50', 'cov90', 'crps_rel', 'width90_rel')}})
        R = {k: float(np.nanmedian([r[k] for r in rows])) for k in rows[0] if k not in ('basin', 'stratum')}
        R['mean_over_basins'] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]
                                 if k not in ('basin', 'stratum')}
        R['by_stratum_loo_cov90'] = {s: float(np.nanmedian([r['loo_cov90'] for r in rows if r['stratum'] == s]))
                                     for s in CF.STRATA}
        R['by_stratum_raw_cov90'] = {s: float(np.nanmedian([r['raw_cov90'] for r in rows if r['stratum'] == s]))
                                     for s in CF.STRATA}
        report['jobs'][j] = {'median': R, 'basins': rows}
        print(f'{bid} {j}: cov90 raw {R["raw_cov90"]:.3f} -> in-sample {R["in_cov90"]:.3f}, LOBO {R["loo_cov90"]:.3f}; '
              f'cov50 {R["raw_cov50"]:.3f} -> {R["in_cov50"]:.3f} / {R["loo_cov50"]:.3f}; CRPSrel {R["raw_crps_rel"]:.3f} '
              f'-> {R["in_crps_rel"]:.3f} / {R["loo_crps_rel"]:.3f}', flush=True)
    # ------------------------------------------------------------------ the wrapper arm folder
    wid = f'{bid}_conformal'
    wdir = os.path.join(HERE, wid)
    os.makedirs(wdir, exist_ok=True)
    base_rel = os.path.relpath(base, wdir)
    extra = [base_rel, '../lib']
    for x in meta.get('extra_files', []):
        extra.append(os.path.relpath(os.path.normpath(os.path.join(base, x)), wdir))
    json.dump({**calib, 'base_rel': base_rel}, open(os.path.join(wdir, 'calib.json'), 'w'), indent=1)
    tpl = open(os.path.join(HERE, 'wrapper_predict_template.py')).read().replace('{base_id}', bid)
    open(os.path.join(wdir, 'predict.py'), 'w').write(tpl)
    fit_h = (time.process_time() - t_cpu0 + child_cpu) / 3600
    lines = []
    for j in J:
        R = report['jobs'][j]['median']
        lines.append(f"| {j} | {R['raw_cov50']:.2f} / {R['raw_cov90']:.2f} | {R['in_cov50']:.2f} / {R['in_cov90']:.2f} | "
                     f"{R['loo_cov50']:.2f} / {R['loo_cov90']:.2f} | {R['raw_crps_rel']:.3f} | {R['in_crps_rel']:.3f} | "
                     f"{R['loo_crps_rel']:.3f} | {R['mean_over_basins']['raw_cov50']:.2f} / {R['mean_over_basins']['raw_cov90']:.2f} -> "
                     f"{R['mean_over_basins']['loo_cov50']:.2f} / {R['mean_over_basins']['loo_cov90']:.2f} |")
    shifts = []
    for j in J:
        for s in CF.STRATA:
            d = calib['jobs'][j][s]
            shifts.append(f"| {j} | {s} ({d['n_basins']}{', pooled' if d.get('fallback') else ''}) | "
                          f"{d['b90_lo']:+.3f} / {d['b90_hi']:+.3f} | {d['b50_lo']:+.3f} / {d['b50_hi']:+.3f} |")
    card = f"""# {wid}: {meta.get('label', bid)} + Mondrian conformal bands

**What.** The registered arm `{bid}` (hash `{h[:16]}`) run unchanged, with its 5-95 % and 25-75 % bands recalibrated
by a Mondrian split-conformal layer (agent C). The mean and the median (q50) are the base arm's, so NSE, KGE and
FHV are identical to `{bid}`; only coverage, CRPS and band width change. Protocols: {', '.join(meta['protocols'])}.

**Method** (`../lib/cf_mondrian.py`). Strata fixed before looking at any residual: the first of snow / arid / humid
in the basin's class list, else "other" (regulated-only, large-only, unclassed). Scores in log space with offset
c = {CF.C_FRAC} x mean forcing precipitation: upper endpoint log(y+c) - log(q_hi+c), lower endpoint
log(q_lo+c) - log(y+c). Per stratum, protocol/lead and endpoint the shift is the weighted (1 - a/2)
quantile (a = 0.10 or 0.50) of the scores over the stratum's dev basin-days, each basin weighted equally. Strata with fewer
than {CF.MIN_BASINS} dev basins fall back to the pooled shift. Applied as q_hi' = (q_hi+c) e^d_hi - c,
q_lo' = (q_lo+c) e^-d_lo - c, then made monotone. Shifts can be negative (bands narrow): conformal, not widen-only.

**Data.** Fitted on the base arm's daily dev predictions only (52 dev basins, WY2001-2014, Caravan Q;
`evaluate.py --save-preds`). Zero real labels in the base arm's training; no exam data. Nothing else is learned.

**Dev coverage (median over basins; 50 % / 90 % band) and CRPS / mean flow.** "In-sample" is what `evaluate.py`
reports for this wrapper on dev (the shifts were fitted on the same basins, so it is optimistic). "LOBO" refits the
shifts without each basin and scores it: the honest estimate. The fit targets the MEAN coverage over basins
(every basin weighted equally); per-basin coverage is skewed (a few basins are badly covered, most are over-covered),
so the median over basins sits above the nominal 0.50 / 0.90 while the mean is close to it (last column).

| Job | Raw cov50 / cov90 | Conformal in-sample | Conformal LOBO | Raw CRPS/Q | In-sample | LOBO | MEAN over basins: raw -> LOBO |
|---|---|---|---|---|---|---|---|
""" + '\n'.join(lines) + f"""

**Shifts** (log space; lower d_lo / upper d_hi; positive = widen).

| Job | Stratum (dev basins) | 90 % band lo / hi | 50 % band lo / hi |
|---|---|---|---|
""" + '\n'.join(shifts) + f"""

**Compute.** Fit {fit_h:.3f} CPU-h (dev predictions of the base + the fit), one thread. Base arm training: see
`{bid}`'s card ({meta.get('compute', {}).get('train_cpu_hours')} CPU-h). Inference = the base arm + microseconds.

**Limits.** Calibrated on 52 dev basins (8-22 per stratum): coverage holds on average over basins like these, not
for every basin; basin-to-basin variation in coverage stays large (see fit_report.json). Days within a basin are
not exchangeable (persistence of errors), so the guarantee is at the basin level, approximately. Exam periods
(WY1990-99 main, WY2016-23 recent) differ in climate and gauge source (NWIS for recent) from dev.
"""
    open(os.path.join(wdir, 'card.md'), 'w').write(card)
    rj = {'arm_id': wid, 'label': f"{meta.get('label', bid)} + Mondrian conformal",
          'family': 'conformal', 'family_label': 'Conformal prediction (wrapper around ' + bid + ')',
          'group': 'filt', 'protocols': meta['protocols'], 'wraps': bid, 'wraps_hash': h,
          'numpy_only': bool(meta['numpy_only']),
          'numpy_only_reason': ('' if meta['numpy_only'] else f"wraps {bid}: {meta.get('numpy_only_reason', '')}"),
          'compute': {'train_cpu_hours': round(fit_h, 4), 'base_train_cpu_hours': meta.get('compute', {}).get('train_cpu_hours'),
                      'notes': f'conformal fit only (dev predictions of {bid} + shifts); the base arm is not retrained'},
          'feature_engineering': f"none beyond {bid}'s; the conformal offset uses static[11] (mean forcing P) and the "
                                 "class list (strata)",
          'extra_files': extra, 'margin': 0.0,
          'rule': 'No ladder: one conformal fit on dev residuals (method fixed in advance). Reported raw vs conformal.',
          'rungs': [{'id': 'c0', 'change': 'Mondrian split-conformal bands (asymmetric CQR, log space, strata '
                     'snow/arid/humid/other) fitted on dev residuals', 'decision': 'released',
                     'why': 'the only rung; dev coverage (LOBO) ' + ', '.join(
                         f"{j} {report['jobs'][j]['median']['loo_cov90']:.2f}" for j in J)}],
          'released': 'c0'}
    json.dump(rj, open(os.path.join(wdir, 'ratchet.json'), 'w'), indent=1)
    os.makedirs(os.path.join(wdir, 'work'), exist_ok=True)
    json.dump(report, open(os.path.join(wdir, 'work', 'fit_report.json'), 'w'), indent=1, default=float)
    print('wrote', wdir, f'fit {fit_h:.3f} CPU-h')
    if not a.no_eval:
        sets = 'simval,dev' if a.simval else 'dev'
        subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HARNESS, 'evaluate.py'), wdir, '--sets', sets],
                       check=True)
    if a.register:
        subprocess.run([sys.executable, os.path.join(HARNESS, 'register.py'), wdir], check=True)


if __name__ == '__main__':
    main()
