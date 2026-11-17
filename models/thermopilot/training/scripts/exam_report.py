"""Exam tables: every controller on every exam set (twins of real units, real weather, real tariffs).

usage: exam_report.py TAG     (ThermoPilot results results/<set>/pol_TAG.json) -> results/exam_TAG.json
Savings are summed over episodes (bill = control-dependent energy + demand + gas; fixed charges excluded) with 90 %
bootstrap intervals over units (episodes of the same unit resampled together).
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # training/: tp, _paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT   # data/ and results/ live in WORK (was the sandbox's thermopilot/ folder)
from ratchet import load_base

SETS = ['exam_recs', 'exam_bdg2', 'exam_hue', 'exam_wpuq']
CTRL = ['fixed', 'setback', 'setback_std', 'precool', 'precool_p1', 'mpc_c', 'thermopilot', 'oracle']
METRICS = ['bill', 'J', 'DH', 'DH_home', 'ppd_home', 'pmv_h', 'peak_kW', 'peak_window_kW', 'kWh', 'gas_kWh',
           'onoff_per_day', 'mode_switch_per_day', 'sp_changes_per_day', 'demand_cost', 'fixed']


def table(base, pol, ids, rng):
    units = sorted(set(base[i]['meta']['unit'] for i in ids))
    by_unit = {u: [i for i in ids if base[i]['meta']['unit'] == u] for u in units}
    get = lambda i, c: pol[i] if c == 'thermopilot' else base[i][c]
    out = {}
    for c in CTRL:
        row = {}
        for m in METRICS:
            vals = [get(i, c).get(m, np.nan) for i in ids]
            row[m] = float(np.nanmean(vals)) if m not in ('bill', 'J', 'kWh', 'gas_kWh', 'demand_cost', 'fixed') else float(np.nansum(vals))
        out[c] = row
    # savings vs fixed and vs setback, bootstrap over units
    for c in CTRL:
        for ref in ('fixed', 'setback'):
            def sav(sample_ids, key):
                a = sum(get(i, c)[key] for i in sample_ids); b = sum(base[i][ref][key] for i in sample_ids)
                return 1 - a / b
            pt = sav(ids, 'bill')
            bs = []
            for _ in range(400):
                pick = rng.choice(len(units), len(units))
                sid = [i for p in pick for i in by_unit[units[p]]]
                bs.append(sav(sid, 'bill'))
            out[c][f'bill_saving_vs_{ref}'] = pt
            out[c][f'bill_saving_vs_{ref}_ci90'] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
        Jf = np.array([base[i]['fixed']['J'] for i in ids]); Jo = np.array([base[i]['oracle']['J'] for i in ids])
        Jc = np.array([get(i, c)['J'] for i in ids])
        out[c]['skill'] = float(np.sum((Jf - Jc) / Jf) / max(np.sum((Jf - Jo) / Jf), 1e-9))
        best_rule = np.array([min(base[i][r]['J'] for r in ('fixed', 'setback', 'precool')) for i in ids])
        out[c]['beats_best_rule_frac'] = float(np.mean(Jc < best_rule - 1e-6))
        out[c]['J_vs_best_rule'] = float(1 - Jc.sum() / best_rule.sum())
    return out


def main():
    tag = sys.argv[1]
    rng = np.random.default_rng(0)
    R = {}
    for s in SETS:
        base = load_base(s)
        f = os.path.join(ROOT, 'results', s, f'pol_{tag}.json')
        if not os.path.exists(f) or not base:
            continue
        pol = json.load(open(f))
        ids = [i for i in base if i in pol and 'error' not in pol[i]]
        R[s] = dict(n_episodes=len(ids), n_units=len(set(base[i]['meta']['unit'] for i in ids)), all=table(base, pol, ids, rng))
        # per tariff (RECS / BDG2) and per season
        for key, fn in (('by_tariff', lambda i: base[i]['meta']['tariff']), ('by_month', lambda i: str(base[i]['meta']['month'])),
                        ('by_kind', lambda i: base[i]['meta']['kind'])):
            groups = sorted(set(fn(i) for i in ids))
            R[s][key] = {g: dict(n=len([i for i in ids if fn(i) == g]),
                                 t=table(base, pol, [i for i in ids if fn(i) == g], rng)) for g in groups}
        print(s, R[s]['n_episodes'], 'episodes', R[s]['n_units'], 'units')
        for c in CTRL:
            r = R[s]['all'][c]
            print('  %-12s bill %9.0f  vs fixed %+6.1f%% [%+.1f,%+.1f]  vs setback %+6.1f%%  DH/ep %6.2f  PPD %5.1f  peak %5.2f  onoff/d %5.1f  skill %5.2f  beats-best-rule %.2f' % (
                c, r['bill'], 100 * r['bill_saving_vs_fixed'], 100 * r['bill_saving_vs_fixed_ci90'][0], 100 * r['bill_saving_vs_fixed_ci90'][1],
                100 * r['bill_saving_vs_setback'], r['DH'], r['ppd_home'], r['peak_kW'], r['onoff_per_day'], r['skill'], r['beats_best_rule_frac']))
    json.dump(R, open(os.path.join(ROOT, 'results', f'exam_{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
