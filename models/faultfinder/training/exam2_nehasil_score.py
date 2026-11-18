"""FaultFinder exam 2 (Nehasil): score v1.0 and v1.1 once, as split_exam2_nehasil.json says. 17 Nov 2026.

    python exam2_nehasil.py build --check     # first: the processed AHU files must reproduce the frozen hashes
    python exam2_nehasil_score.py             # -> training/exam2_nehasil_results.json

v1.0: faultfinder.py and faultfinder_weights.npz from git dfafdcf (T = 4.3), copied to work/exam2/v10/ by
    git show dfafdcf:models/faultfinder/faultfinder.py > work/exam2/v10/faultfinder.py
    git show dfafdcf:models/faultfinder/faultfinder_weights.npz > work/exam2/v10/faultfinder_weights_v10.npz
v1.1: the released faultfinder.py and weights (T = 4.2). Reference days are supplied causally: the same AHU's
processed days with an earlier date, at most 7 days earlier, the most recent 3, chosen by date alone; no label is read.
Baselines (spec): APAR rules (baselines.py run_score, dt 300 s, fault if score > 0.2322, the platform dev threshold),
the majority guess (no fault) and chance (1 / 18) for diagnosis. Interval: 95 % percentile bootstrap over the 4 AHUs,
2,000 resamples, seed 20261117, model metric minus the best baseline's (best on the full exam).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as _P  # noqa: E402
import exam2_common as X  # noqa: E402

sys.path.insert(0, os.path.join(_P.BLDG_CODE, 'data', 'scripts'))
import baselines as BL  # noqa: E402

SPLIT = os.path.join(HERE, 'split_exam2_nehasil.json')
PROC = os.path.join(_P.WORK, 'exam2', 'nehasil', 'proc')
V10 = os.path.join(_P.WORK, 'exam2', 'v10')
OUT = os.path.join(HERE, 'exam2_nehasil_results.json')
APAR_THR = 0.2322
NB, SEED = 2000, 20261117
N_CLASSES = 18
MAX_GAP_DAYS, MAX_REF = 7, 3


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def auc(t, s):
    t, s = np.asarray(t, bool), np.asarray(s, float)
    pos, neg = s[t], s[~t]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def bal_acc(t, f):
    t, f = np.asarray(t, bool), np.asarray(f, bool)
    if t.all() or (~t).all():
        return float('nan')
    return float(0.5 * ((f & t).sum() / t.sum() + (~f & ~t).sum() / (~t).sum()))


def fa_rate(t, f):
    t, f = np.asarray(t, bool), np.asarray(f, bool)
    return float(f[~t].mean()) if (~t).any() else float('nan')


def main():
    S = json.load(open(SPLIT, encoding='utf-8'))
    days = pd.DataFrame(S['exam_days'])
    days['date'] = pd.to_datetime(days.date)
    frames = {}
    for a in S['units']:
        d = pd.read_parquet(os.path.join(PROC, f'{a}.parquet'))
        d['time_local'] = pd.to_datetime(d.time_local)
        d['date'] = d.time_local.dt.normalize()
        # per-unit hash check (the same rule as exam2_nehasil.build)
        k = d.copy()
        k['is_fault'] = (k.label != 'no fault').astype(float)
        h = X.hash_frame(k, X.CONTRACT + ['is_fault'])
        if h != S['units'][a]['sha256']:
            raise SystemExit(f'{a}: processed hash {h} != frozen {S["units"][a]["sha256"]}')
        frames[a] = {day: g[['time_local'] + X.CONTRACT].reset_index(drop=True) for day, g in d.groupby('date')}
    print('per-unit hashes match', flush=True)

    m10 = load_module(os.path.join(V10, 'faultfinder.py'), 'faultfinder_v10')
    ff10 = m10.FaultFinder(os.path.join(V10, 'faultfinder_weights_v10.npz'))
    m11 = load_module(os.path.join(_P.MODEL, 'faultfinder.py'), 'faultfinder_v11')
    ff11 = m11.FaultFinder()
    assert abs(ff10.T - 4.3) < 1e-5 and abs(ff11.T - 4.2) < 1e-5, (ff10.T, ff11.T)

    rows = []
    for r in days.itertuples():
        fr = frames[r.ahu]
        df = fr[r.date]
        earlier = sorted(dd for dd in fr if dd < r.date and (r.date - dd).days <= MAX_GAP_DAYS)[-MAX_REF:]
        o10 = ff10.read_day(df)
        o11 = ff11.read_day(df, ref_days=[fr[dd] for dd in earlier] or None)
        score, _, _, non = BL.run_score(df, 300)
        rows.append(dict(ahu=r.ahu, date=str(r.date.date()), label=r.label, diag_class=r.diag_class,
                         v10_p=o10['p_fault'], v10_d=o10['log_odds'], v10_top=o10['diagnosis'][0][0],
                         v11_p=o11['p_fault'], v11_d=o11['log_odds'], v11_top=o11['diagnosis'][0][0],
                         v11_nref=o11['n_reference_days'], apar=score, apar_on=non))
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(_P.WORK, 'exam2', 'nehasil_scored_days.csv'), index=False)
    t = (E.label == 'fault').to_numpy()
    diag_ok = E.diag_class.isin(['oa_damper_stuck', 'hc_valve_stuck', 'cc_valve_stuck', 'sensor_bias_T_sa',
                                 'sensor_bias_T_ra']).to_numpy() & t
    groups = E.ahu.to_numpy()
    ug = sorted(np.unique(groups))
    gi = [np.where(groups == g)[0] for g in ug]

    def metric(name, idx, who):
        tt = t[idx]
        if name == 'det_auc':
            s = {'v10': E.v10_d, 'v11': E.v11_d, 'apar': E.apar}[who].to_numpy()[idx]
            return auc(tt, s)
        if name == 'diag_top1':
            dm = diag_ok[idx]
            if not dm.any():
                return float('nan')
            if who == 'chance':
                return 1.0 / N_CLASSES
            top = {'v10': E.v10_top, 'v11': E.v11_top}[who].to_numpy()[idx][dm]
            return float((top == E.diag_class.to_numpy()[idx][dm]).mean())
        f = {'v10': E.v10_p.to_numpy() > 0.5, 'v11': E.v11_p.to_numpy() > 0.5, 'apar': E.apar.to_numpy() > APAR_THR,
             'majority': np.zeros(len(E), bool)}[who][idx]
        return bal_acc(tt, f) if name == 'bal_acc' else fa_rate(tt, f)

    SPEC = {'det_auc': (['apar'], 'higher'), 'bal_acc': (['apar', 'majority'], 'higher'),
            'fa_rate': (['apar', 'majority'], 'lower'), 'diag_top1': (['chance'], 'higher')}
    allidx = np.arange(len(E))
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([gi[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(NB)]
    res = dict(exam='FaultFinder exam 2, Nehasil (CTU Prague), scored once on 17 Nov 2026', split_sha256=S['split_sha256'],
               spec=os.path.basename(SPLIT), n_ahus=len(ug), n_days=int(len(E)), n_fault_days=int(t.sum()),
               n_normal_days=int((~t).sum()), n_diagnosable=int(diag_ok.sum()),
               models=dict(v10='FaultFinder v1.0 (git dfafdcf weights and reader, T = 4.3, single day)',
                           v11='FaultFinder v1.1 (released weights, T = 4.2, causal reference days)'),
               apar_threshold=APAR_THR, threshold_p=0.5, interval=dict(resamples=NB, seed=SEED, unit='AHU',
                                                                       level=0.95),
               v11_reference_days=dict(days_with_reference=int((E.v11_nref > 0).sum()),
                                       by_count={int(k): int(v) for k, v in E.v11_nref.value_counts().sort_index().items()}),
               rows={})
    for name, (bases, better) in SPEC.items():
        bvals = {b: metric(name, allidx, b) for b in bases}
        best = max(bvals, key=bvals.get) if better == 'higher' else min(bvals, key=bvals.get)
        row = dict(better=better, baselines=bvals, best_baseline=best,
                   n_days=int(diag_ok.sum()) if name == 'diag_top1' else
                   int((~t).sum()) if name == 'fa_rate' else int(len(E)))
        for who in ('v10', 'v11'):
            v = metric(name, allidx, who)
            diffs = np.array([metric(name, ix, who) - metric(name, ix, best) for ix in draws])
            ok = np.isfinite(diffs)
            lo, hi = (np.percentile(diffs[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan))
            good_below = better == 'lower'
            if lo > 0:
                verdict = 'loss' if good_below else 'win'
            elif hi < 0:
                verdict = 'win' if good_below else 'loss'
            else:
                verdict = 'tie'
            row[who] = dict(value=v, minus_best=v - bvals[best], interval=[float(lo), float(hi)], verdict=verdict,
                            resamples_used=int(ok.sum()))
        # per-AHU values for the record
        row['by_ahu'] = {g: {w: metric(name, gi[i], w) for w in ['v10', 'v11'] + bases} for i, g in enumerate(ug)}
        res['rows'][name] = row
    res['v11_by_reference'] = {k: dict(n=int(m.sum()), n_fault=int(t[m].sum()),
                                       auc=auc(t[m], E.v11_d.to_numpy()[m]),
                                       fa_rate=fa_rate(t[m], E.v11_p.to_numpy()[m] > 0.5))
                               for k, m in (('with_reference', (E.v11_nref > 0).to_numpy()),
                                            ('without_reference', (E.v11_nref == 0).to_numpy()))}
    res['v10_vs_v11_same_days'] = dict(median_abs_logodds_change=float(np.median(np.abs(E.v11_d - E.v10_d))))
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(res, fh, indent=1, default=float)
        fh.write('\n')
    for name, row in res['rows'].items():
        print(name, 'best', row['best_baseline'], {k: round(v, 3) for k, v in row['baselines'].items()},
              'v10', round(row['v10']['value'], 3), [round(x, 3) for x in row['v10']['interval']], row['v10']['verdict'],
              'v11', round(row['v11']['value'], 3), [round(x, 3) for x in row['v11']['interval']], row['v11']['verdict'])
    print('wrote', OUT)


if __name__ == '__main__':
    main()
