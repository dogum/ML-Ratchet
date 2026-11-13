"""Print the NMC811 exam (work/nmc/exam_<rung>.json) as the markdown tables of MODEL_CARD.md.

    python nmc_report.py <exam json>
"""
import sys, json, numpy as np

if __name__ == '__main__':
    E = json.load(open(sys.argv[1])); names = [('model', 'CellSense'), ('median_guess', 'Dev median'), ('no_fade', 'No fade'), ('shift_at_cut', 'Shift at cut'), ('voltage_fit', 'Voltage fit')]
    print('| Reading | Tests | ' + ' | '.join(n for _, n in names) + ' |'); print('|---|---|' + '---|' * len(names))
    for t in E['table']:
        where = 'full curve' if t['depth'] is None else f"stopped at {t['depth']} Ah"; ref = 'with reference' if t['ref'] else 'no reference'
        vals = [t.get(k) for k, _ in names]; best = min(v for v in vals if v is not None)
        cells = [('—' if v is None else (f"**{v:.0f}**" if v == best else f"{v:.0f}")) for v in vals]
        print(f"| {where}, {ref} | {t['n']} | " + ' | '.join(cells) + ' |')
    print('\ncoverage', {k: round(100 * v) for k, v in E['coverage'].items()}, 'factors', E['factors'])
    print('per cell (partial, mAh)', {k: round(v) for k, v in E['per_cell_partial_mAh'].items()}, 'fade corr', round(E['fade_corr'], 3))
    R = E['rows']; full_ref = [r for r in R if r['depth'] is None and r['ref']]
    if full_ref:
        print('mean modes on full curves with reference: lli %.3f lam_ne %.3f lam_pe %.3f' % tuple(np.mean([[r[k] for k in ('lli', 'lam_ne', 'lam_pe')] for r in full_ref], 0)))
