"""Import check (added 18 Nov 2026, ML Ratchet import; not part of the original build). Writes nothing.

1) Re-derives what the shipped files allow without the real time series:
   - the observed change of each signature on the 619 exam basins (obs val - obs cal) and the "GR4J vs own record"
     baseline (GR4J val sim - obs cal), from WaterData's shipped signatures.parquet, against training/exam.json;
   - the level rows' own-calibration-value and GR4J-sim baselines;
   - the four demo examples' observed_change.
2) Intervals for every graded exam row (the per-basin network outputs are not shipped, so no paired bootstrap):
   - Spearman rho: Fisher-z 95 % interval (se 1/sqrt(n-3)); a difference of two rhos on the same 619 basins is given
     by Zou's (2007) interval for two INDEPENDENT correlations built from the two Fisher-z intervals. The rhos are
     paired (same basins), so the true interval is narrower: this is conservative;
   - AUC: Hanley & McNeil (1982) normal interval from the AUC and the class counts (approximate);
   - coverage of the error bars: exact binomial (Clopper-Pearson) interval.

python import_check.py        (needs shared/water/data/proc from the repository; no network, no exam time series)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import beta, norm, spearmanr

from _paths import SRC, WATER_PROC, MODEL

EPSQ = 0.01
DELTA_T = ['d_logq', 'd_rr', 'd_bfi', 'd_logq5', 'd_logq95', 'd_hfd']
LEVEL = {'d_logq': 'q_mean', 'd_rr': 'runoff_ratio', 'd_bfi': 'bfi_lh', 'd_logq5': 'q5', 'd_logq95': 'q95',
         'd_hfd': 'hfd_mean'}
LOGT = {'d_logq', 'd_logq5', 'd_logq95'}


def delta(t, cal, val):
    if t in LOGT:
        return np.log(val + EPSQ) - np.log(cal + EPSQ)
    return val - cal


def fisher_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def zou_diff(r1, r2, n1, n2=None):
    """95 % interval of r1 - r2 for independent correlations (Zou 2007), from Fisher-z intervals."""
    n2 = n2 or n1
    l1, u1 = fisher_ci(r1, n1)
    l2, u2 = fisher_ci(r2, n2)
    d = r1 - r2
    return d, d - np.sqrt((r1 - l1) ** 2 + (u2 - r2) ** 2), d + np.sqrt((u1 - r1) ** 2 + (r2 - l2) ** 2)


def hanley_mcneil(a, npos, nneg):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg))
    return float(a - 1.96 * se), float(a + 1.96 * se)


def clopper(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def verdict(lo, hi, higher_better=True):
    if lo > 0:
        return 'win' if higher_better else 'loss'
    if hi < 0:
        return 'loss' if higher_better else 'win'
    return 'tie'


def main():
    ex = json.load(open(f'{SRC}/exam.json'))
    split = json.load(open(f'{WATER_PROC}/splits.json'))
    sig = pd.read_parquet(f'{WATER_PROC}/baselines/signatures.parquet')
    ids = split['camels_us']['exam'] if isinstance(split['camels_us'], dict) and 'exam' in split['camels_us'] else None
    if ids is None:
        import waterdata as wd
        ids = wd.basins('camels', 'exam')
    tab = {k: g.set_index('basin') for k, g in sig.groupby(['period', 'source'])}
    oc, ov, gv = (tab[('cal', 'obs')].reindex(ids), tab[('val', 'obs')].reindex(ids), tab[('val', 'gr4j_sim')].reindex(ids))
    n = len(ids)
    print(f'exam basins: {n}')
    print('\n1) re-derived from signatures.parquet vs exam.json')
    for t in DELTA_T:
        lv = LEVEL[t]
        y = delta(t, oc[lv].values, ov[lv].values)
        g = delta(t, oc[lv].values, gv[lv].values)
        ok = np.isfinite(y) & np.isfinite(g)
        d = ex['delta'][t]
        rho = spearmanr(g[ok], y[ok]).statistic
        print(f'  {t:9s} no-change median|y| {np.nanmedian(np.abs(y)):.4f} (exam.json {d["no change (persistence)"]["mae"]:.4f}); '
              f'GR4J vs own record rho {rho:.4f} / median|err| {np.median(np.abs(g[ok] - y[ok])):.4f} '
              f'(exam.json {d["GR4J (gauged, sim target period minus observed cal)"]["rho"]:.4f} / '
              f'{d["GR4J (gauged, sim target period minus observed cal)"]["mae"]:.4f}); n {ok.sum()}')
    for lv in LEVEL.values():
        y, c, g = ov[lv].values, oc[lv].values, gv[lv].values
        r_c = spearmanr(c, y, nan_policy='omit').statistic
        r_g = spearmanr(g, y, nan_policy='omit').statistic
        L = ex['level'][lv]
        print(f'  level {lv:12s} own-cal rho {r_c:.4f} (exam.json {L["own calibration-period value"]["rho"]:.4f}); '
              f'GR4J sim rho {r_g:.4f} (exam.json {L["GR4J sim (gauged)"]["rho"]:.4f})')
    for f in sorted(os.listdir(f'{MODEL}/examples')):
        if f.endswith('_truth.json'):
            tr = json.load(open(f'{MODEL}/examples/{f}'))
            b = tr['basin']
            worst = max(abs(delta(t, oc.loc[b, LEVEL[t]], ov.loc[b, LEVEL[t]]) - tr['observed_change'][t]) for t in DELTA_T)
            print(f'  example {b}: max |observed_change - re-derived| = {worst:.2e}')

    print('\n2) intervals (95 %)')
    for t in DELTA_T:
        d = ex['delta'][t]
        m = d['BasinDetective']
        print(f'  {t}: BasinDetective rho {m["rho"]:.3f} [{fisher_ci(m["rho"], m["n"])[0]:.3f}, {fisher_ci(m["rho"], m["n"])[1]:.3f}]')
        for name, b in d.items():
            if name == 'BasinDetective' or 'rho' not in b:
                continue
            dd, lo, hi = zou_diff(m['rho'], b['rho'], m['n'], b['n'])
            print(f'     - {name:52s} rho {b["rho"]:.3f}: difference {dd:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict(lo, hi)}')
        for k in ('cov68', 'cov95'):
            kk = round(m[k] * m['n'])
            lo, hi = clopper(kk, m['n'])
            print(f'     {k} {m[k]:.3f} ({kk}/{m["n"]}) [{lo:.3f}, {hi:.3f}]')
    for lv, L in ex['level'].items():
        m = L['BasinDetective']
        for name, b in L.items():
            if name == 'BasinDetective':
                continue
            dd, lo, hi = zou_diff(m['rho'], b['rho'], m['n'], b['n'])
            print(f'  level {lv:12s} rho {m["rho"]:.3f} vs {name:30s} {b["rho"]:.3f}: {dd:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict(lo, hi)}')
    R = ex['regulation']
    for name, r in R.items():
        if 'n_pos' not in r:
            continue
        npos, nneg = r['n_pos'], ex['n_basins'] - r['n_pos']
        for k, v in r.items():
            if k.endswith('_auc'):
                lo, hi = hanley_mcneil(v, npos, nneg)
                print(f'  {name:50s} {k:34s} {v:.3f} [{lo:.3f}, {hi:.3f}]')
    r = R['log10 dor among regulated basins']
    print(f'  log10 storage rho {r["rho"]:.3f} {fisher_ci(r["rho"], r["n"])}; cov68 {r["cov68"]:.3f} '
          f'{clopper(round(r["cov68"] * r["n"]), r["n"])}')
    c = R['confusion_p>0.5']
    print(f'  confusion at p > 0.5: {c}; precision {c["tp"] / (c["tp"] + c["fp"]):.3f} {clopper(c["tp"], c["tp"] + c["fp"])}, '
          f'recall {c["tp"] / (c["tp"] + c["fn"]):.3f} {clopper(c["tp"], c["tp"] + c["fn"])}')
    tw = json.load(open(f'{SRC}/twin_final.json'))['param_recovery']
    for k, v in tw.items():
        ci = fisher_ci(v['rho'], 260) if v['rho'] is not None else None
        print(f'  twins {k:12s} rho {v["rho"]} {ci}; cov68 {v["cov68"]:.3f} {clopper(round(v["cov68"] * 260), 260)}')


if __name__ == '__main__':
    main()
