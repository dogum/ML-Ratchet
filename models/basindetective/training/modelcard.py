"""Tables for MODEL_CARD.md and the status doc from ratchet.json, exam.json, twin and sigma JSONs.
python3 modelcard.py <release_model.npz>  -> prints markdown tables (pasted into the docs by finalize step)."""
import common  # noqa: F401
import json
import sys

from common import D, DELTA_T

NICE = {'d_logq': 'mean flow (log ratio)', 'd_rr': 'runoff ratio', 'd_bfi': 'baseflow index (LH)',
        'd_logq5': 'low flow Q5 (log ratio)', 'd_logq95': 'high flow Q95 (log ratio)', 'd_hfd': 'half-flow date (days)'}


def f(v, d=2):
    try:
        return f'{float(v):.{d}f}'
    except Exception:
        return '–'


def tables(model_path):
    R = json.load(open(f'{D}/ratchet.json'))
    ex = json.load(open(f'{D}/exam.json'))
    tw = json.load(open(f'{D}/logs/twin_final.json'))
    sg = json.load(open(model_path.replace('.npz', '_sigma.json')))
    out = []
    out.append('| Rung | Change | Physics | Sim skill (own frozen set) | Dev composite | Decision |\n|---|---|---|---|---|---|')
    for r in R['rungs']:
        out.append(f"| {r['name']} | {r['description']} | {r['physics']} | {f(r['sim_skill'], 3)} | "
                   f"{f(r['dev_composite'], 3)} | {'kept' if r['kept'] else 'rejected'}: {r['reason']} |")
    out.append('\nRe-scores when the physics changed (sim skill / dev composite):\n')
    for rs in R['rescores']:
        out.append(f"- val-{rs['val']}: " + '; '.join(f"{k} {f(v['sim_skill'], 3)} / {f(v['dev_composite'], 3)}"
                                                      for k, v in rs['scores'].items()))
    out.append('\n| Change of | BasinDetective rho | MAE | 68 % cover | 95 % cover | GR4J model-implied rho / MAE | '
               'GR4J vs own record rho / MAE | Budyko-Fu rho / MAE | no-change MAE |\n|---|---|---|---|---|---|---|---|---|')
    G, G2, BU = ('GR4J (gauged, model-implied change)', 'GR4J (gauged, sim target period minus observed cal)',
                 'Budyko-Fu (w fitted on cal)')
    for t in DELTA_T:
        d = ex['delta'][t]
        a = d['BasinDetective']
        bu = d.get(BU)
        dd = 1 if t == 'd_hfd' else 3
        out.append(f"| {NICE[t]} | **{f(a['rho'])}** | {f(a['mae'], dd)} | {a['cov68'] * 100:.0f} % | "
                   f"{a['cov95'] * 100:.0f} % | {f(d[G]['rho'])} / {f(d[G]['mae'], dd)} | {f(d[G2]['rho'])} / "
                   f"{f(d[G2]['mae'], dd)} | {(f(bu['rho']) + ' / ' + f(bu['mae'], dd)) if bu else '–'} | "
                   f"{f(d['no change (persistence)']['mae'], dd)} |")
    out.append('\n| Validation-decade level | BasinDetective rho / R2 | own calibration value rho / R2 | GR4J sim rho / R2 |'
               '\n|---|---|---|---|')
    for lv, d in ex['level'].items():
        a, b, c = d['BasinDetective'], d['own calibration-period value'], d['GR4J sim (gauged)']
        out.append(f"| {lv} | {f(a['rho'], 3)} / {f(a['r2'], 3)} | {f(b['rho'], 3)} / {f(b['r2'], 3)} | "
                   f"{f(c['rho'], 3)} / {f(c['r2'], 3)} |")
    out.append('\n| Regulation truth | positives | BasinDetective AUC | GR4J misfit AUC | seasonal-damping AUC | '
               'precision in top n |\n|---|---|---|---|---|---|')
    for k, v in ex['regulation'].items():
        if 'BasinDetective_auc' in v:
            out.append(f"| {k} | {v['n_pos']} | **{f(v['BasinDetective_auc'])}** | "
                       f"{f(v['GR4J misfit (1-NSE_cal)_auc'])} | {f(v['seasonal damping index_auc'])} | "
                       f"{f(v['BasinDetective_precision_at_n'])} |")
    v = ex['regulation']['log10 dor among regulated basins']
    out.append(f"\nlog10(storage / annual flow) among the {v['n']} exam basins with any upstream reservoir: rho "
               f"{f(v['rho'])}, median |error| {f(v['mae'])} (log10), 68 % coverage {v['cov68'] * 100:.0f} %. "
               f"Confusion at p > 0.5: {ex['regulation']['confusion_p>0.5']}.")
    out.append('\n| Stratum | n | ' + ' | '.join(NICE[t] for t in DELTA_T) + ' |\n|---|---|' + '---|' * 6)
    for k, v in ex['strata'].items():
        out.append(f"| {k} | {v['n']} | " + ' | '.join(f"{f(v[t]['net'])} / {f(v[t]['gr4j'])}" for t in DELTA_T) + ' |')
    out.append('\n| Hidden property (dev twins, 260) | Spearman rho | R2 | median abs err | bias | 68 % cover |'
               '\n|---|---|---|---|---|---|')
    for k, v in tw['param_recovery'].items():
        out.append(f"| {k} | {f(v['rho'])} | {f(v['r2']) if k != 'log_pmult' else '–'} | "
                   f"{f(v['median_abs_err'], 3)} | {f(v['bias'], 3)} | {v['cov68'] * 100:.0f} % |")
    out.append('\nSigma factors (widen-only, dev): ' + ', '.join(
        f"{k} {f(v['factor'])}" for k, v in sg.items() if isinstance(v, dict)) +
        f"; regulation temperature {f(sg['reg_temperature'])}.")
    out.append('\nPlausibility of hidden outputs on exam (no truth): ' + '; '.join(
        f'{k}: {f(v)}' for k, v in ex['params_plausibility'].items()))
    return '\n'.join(out)


if __name__ == '__main__':
    print(tables(sys.argv[1]))
