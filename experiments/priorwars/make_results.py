"""Collect every scored model into results.json and the tables of RESULTS.md (written by hand around them).
python make_results.py   -> WORK/results.json, WORK/results_tables.md (pw_paths.WORK; the shipped results.json in
this folder is the 18-19 Nov 2026 record and is not overwritten)"""
import glob
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
import pw_paths as _P  # noqa
ROOT = _P.WORK          # the sandbox's ROOT/work/<model> is WORK/<model> here
import pwmetrics as M  # noqa

ARMS = ['P', 'G-dev', 'G-sim', 'G-sim+dev', 'C-dev']
LABELS = {'P': 0, 'G-sim': 0}
CONTRASTS = [('C1', 'G-dev', 'P', 'P vs G-dev (simulator, 0 labels, vs generic prior + k dev labels)'),
             ('C2', 'G-sim', 'P', 'P vs G-sim (same simulated data: purpose-built net vs generic in-context learner)'),
             ('C3', 'G-sim+dev', 'G-dev', 'G-sim+dev vs G-dev (does the simulator, as data, add to the real labels?)'),
             ('C4', 'G-sim+dev', 'P', 'G-sim+dev vs P')]
ORDER = ['gridpulse', 'basindetective', 'bldg_recs', 'bldg_bdg2', 'plantdetective', 'energynet']


def verdict(lo, hi):
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def fmt(v, nd=1):
    return '-' if v is None or not np.isfinite(v) else f'{v:.{nd}f}'


def main():
    out = {'experiment': 'Prior Wars (ML Ratchet method experiment C1)', 'rows': [], 'contrasts': [], 'lc': [],
           'models': {}}
    md = []
    names = [n for n in ORDER if os.path.exists(f'{ROOT}/{n}/exam_scores.json')]
    summary = []
    for name in names:
        import importlib
        a = importlib.import_module(f'adapters.{name}')
        R = json.load(open(f'{ROOT}/{name}/exam_scores.json'))
        Bz = np.load(f'{ROOT}/{name}/exam_boot.npz')
        td = json.load(open(f'{ROOT}/{name}/tune_dev.json'))
        ts = json.load(open(f'{ROOT}/{name}/tune_sim.json'))
        import re
        F = int(re.search(r'Feature table: (\d+) columns', open(f'{ROOT}/{name}/prereg_section.md').read()).group(1))
        rmax = int(min(2560, (210_000 // (F + 8) - 128) // 64 * 64))     # harness.r_max (Amendment A1)
        kdev = min(td['n_dev'], rmax)      # labels actually in G-dev's context (EnergyNet: 1,152 of 1,496)
        if 'all' in R['lc']:
            R['lc']['all']['k'] = kdev
        out['models'][name] = {'title': a.TITLE, 'headline': a.HEADLINE, 'n_exam_rows': R['n_exam_rows'],
                               'n_units': R['n_units'], 'unit': getattr(a, 'UNIT', 'exam unit'), 'n_dev': td['n_dev'],
                               'k_in_g_dev_context': kdev, 'n_features': F,
                               'g_dev_prep': td['chosen_prep'], 'g_sim_prep': ts['chosen_prep'],
                               'g_sim_ctx': ts['chosen_ctx'], 'scored': R['time']}
        arms = [k for k in ARMS if k in R['arms']] + [k for k in R['arms'] if k not in ARMS]
        items = list(R['arms']['P']['scores'].keys())
        # ---- rows
        for arm in arms:
            r = R['arms'][arm]
            kind = 'arm' if arm in ARMS else 'card baseline'
            labels = LABELS.get(arm, kdev if arm in ('G-dev', 'G-sim+dev', 'C-dev') else None)
            for it, ms in r['scores'].items():
                for m, v in ms.items():
                    row = {'model': name, 'arm': arm, 'kind': kind, 'real_labels': labels, 'item': it, 'metric': m,
                           'value': v, 'ci95': r['ci'][it][m], 'higher_is_better': M.HIGHER[m]}
                    d = R['diff_vs_P'].get(arm, {}).get(it, {}).get(m)
                    if d:
                        lo, hi = d['ci']
                        if not M.HIGHER[m]:
                            lo, hi = -hi, -lo
                        row.update({'diff_vs_P': d['diff'], 'diff_vs_P_ci95': d['ci'],
                                    'verdict_vs_P': verdict(lo, hi) if np.isfinite(lo) else None})
                    out['rows'].append(row)
            if r['headline'] is not None:
                row = {'model': name, 'arm': arm, 'kind': kind, 'real_labels': labels, 'item': '_headline',
                       'metric': 'headline (higher is better)', 'value': r['headline'], 'ci95': r['headline_ci'],
                       'higher_is_better': True}
                d = R['diff_vs_P'].get(arm, {}).get('_headline')
                if d:
                    row.update({'diff_vs_P': d['diff'], 'diff_vs_P_ci95': d['ci'],
                                'verdict_vs_P': verdict(*d['ci'])})
                out['rows'].append(row)
        # ---- contrasts (paired over the same bootstrap resamples)
        con = {}
        for cid, x, y, desc in CONTRASTS:
            if x not in R['arms'] or y not in R['arms']:
                continue
            dx = Bz[f'{x}||_headline'] - Bz[f'{y}||_headline']
            lo, hi = np.nanpercentile(dx, [2.5, 97.5])
            d = R['arms'][x]['headline'] - R['arms'][y]['headline']
            per = {}
            for it in items:
                m = list(R['arms']['P']['scores'][it].keys())[0]
                kx, ky = f'{x}||{it}||{m}', f'{y}||{it}||{m}'
                if kx in Bz and ky in Bz:
                    dd = Bz[kx] - Bz[ky]
                    if not M.HIGHER[m]:
                        dd = -dd
                    if np.isfinite(dd).sum() > 10:
                        l2, h2 = np.nanpercentile(dd, [2.5, 97.5])
                        v = R['arms'][x]['scores'][it][m] - R['arms'][y]['scores'][it][m]
                        per[it] = {'metric': m, 'diff_skill': float(v if M.HIGHER[m] else -v),
                                   'ci95': [float(l2), float(h2)], 'verdict': verdict(l2, h2)}
            c = {'model': name, 'contrast': cid, 'arm': x, 'vs': y, 'desc': desc, 'headline_diff': float(d),
                 'ci95': [float(lo), float(hi)], 'verdict_for_arm': verdict(lo, hi), 'per_item': per}
            out['contrasts'].append(c)
            con[cid] = c
        # ---- learning curve
        Ph = R['arms']['P']['headline']
        lc = []
        for k, v in R['lc'].items():
            if k == 'all':
                lc.append((int(v['k']), v['headline_mean'], 0.0, True))
            else:
                lc.append((int(k), v['headline_mean'], v['headline_sd'], False))
        lc.sort()
        kstar = None
        for k, hmean, sd, isall in lc:
            if hmean >= Ph:
                kstar = k
                break
        if kstar is None:
            ks = f'> {lc[-1][0]}' if lc else 'n/a'
        elif kstar == 5 and lc[0][0] == 5:
            ks = '<= 5'
        else:
            ks = str(kstar)
        for k, hmean, sd, isall in lc:
            out['lc'].append({'model': name, 'k': k, 'headline_mean': hmean, 'headline_sd_over_3_subsets': sd,
                              'is_full_dev': isall, 'P_headline': Ph})
        out['models'][name]['k_star'] = ks
        summary.append((name, a.TITLE, R, con, ks, kdev))

        # ---- markdown tables
        md.append(f'### {a.TITLE}\n')
        md.append(f'Exam: {R["n_exam_rows"]:,} rows, {R["n_units"]:,} bootstrap units. Dev: {td["n_dev"]:,} rows. '
                  f'Headline: {a.HEADLINE}.\n')
        hdr = '| Arm | Real labels | ' + ' | '.join(f'{it} ({list(R["arms"]["P"]["scores"][it].keys())[0]})'
                                                     for it in items) + ' | Headline [95 % CI] |'
        md.append(hdr)
        md.append('|' + '---|' * (3 + len(items)))
        for arm in arms:
            r = R['arms'][arm]
            labels = LABELS.get(arm, kdev if arm in ('G-dev', 'G-sim+dev', 'C-dev') else '-')
            cells = []
            for it in items:
                m = list(R['arms']['P']['scores'][it].keys())[0]
                v = r['scores'].get(it, {}).get(m)
                cells.append(fmt(v, 3 if m in ('rho', 'auc', 'r2', 'mdae') else 1))
            hl = (f'{r["headline"]:.3f} [{r["headline_ci"][0]:.3f}, {r["headline_ci"][1]:.3f}]'
                  if r['headline'] is not None else '-')
            nm = f'**{arm}**' if arm in ARMS else arm
            md.append(f'| {nm} | {labels} | ' + ' | '.join(cells) + f' | {hl} |')
        md.append('')
        md.append('| Contrast | Headline difference [95 % CI] | Verdict (for the first arm) | Items won / tied / lost |')
        md.append('|---|---|---|---|')
        for cid, c in con.items():
            w = sum(p['verdict'] == 'win' for p in c['per_item'].values())
            t = sum(p['verdict'] == 'tie' for p in c['per_item'].values())
            lo_ = sum(p['verdict'] == 'loss' for p in c['per_item'].values())
            md.append(f'| {cid}: {c["arm"]} - {c["vs"]} | {c["headline_diff"]:+.3f} [{c["ci95"][0]:+.3f}, '
                      f'{c["ci95"][1]:+.3f}] | {c["verdict_for_arm"]} | {w} / {t} / {lo_} |')
        md.append('')
        if lc:
            md.append('| k (real dev labels) | ' + ' | '.join(str(k) for k, *_ in lc) + ' | P (0 labels) |')
            md.append('|---|' + '---|' * (len(lc) + 1))
            md.append('| G-dev headline (mean of 3 subsets) | ' + ' | '.join(
                f'{h:.3f}' + (f' +/- {sd:.3f}' if sd else '') for k, h, sd, _ in lc) + f' | {Ph:.3f} |')
            md.append(f'\nk\\* (fewest labels for G-dev to reach P): **{ks}**.\n')
    # ---- cross-collection
    c1 = [s[3].get('C1') for s in summary if 'C1' in s[3]]
    pw = sum(c['verdict_for_arm'] == 'loss' for c in c1)      # G-dev loses -> P wins
    pl = sum(c['verdict_for_arm'] == 'win' for c in c1)
    pt = sum(c['verdict_for_arm'] == 'tie' for c in c1)
    sentence = ('the domain simulator beats the generic prior' if pw > pl else
                'the generic prior beats the domain simulator' if pl > pw else 'split')
    out['verdict'] = {'C1_P_wins': pw, 'C1_P_losses': pl, 'C1_ties': pt, 'n_models': len(c1), 'sentence': sentence}
    for cid in ('C2', 'C3', 'C4'):
        cc = [s[3][cid] for s in summary if cid in s[3]]
        out['verdict'][cid] = {v: sum(c['verdict_for_arm'] == v for c in cc) for v in ('win', 'tie', 'loss')}
    xs = ['| Model | Dev k | P | G-dev | G-sim | G-sim+dev | C1 (P vs G-dev) | C2 (P vs G-sim) | C3 (G-sim+dev vs G-dev) | k* |',
          '|---|---|---|---|---|---|---|---|---|---|']
    flip = {'win': 'loss', 'loss': 'win', 'tie': 'tie'}
    for name, title, R, con, ks, kdev in summary:
        h = lambda arm: fmt(R['arms'][arm]['headline'], 3) if arm in R['arms'] else '-'  # noqa
        v = lambda cid, f=False: (flip[con[cid]['verdict_for_arm']] if f else con[cid]['verdict_for_arm']) if cid in con else '-'  # noqa
        xs.append(f'| {title} | {kdev} | {h("P")} | {h("G-dev")} | {h("G-sim")} | {h("G-sim+dev")} | P {v("C1", True)} | '
                  f'P {v("C2", True)} | G-sim+dev {v("C3")} | {ks} |')
    # where each arm wins: best headline overall, among 0-label arms and among k-label arms; items where each is best
    ws = ['| Model | Best arm (headline) | Best with 0 real labels | Best with k real labels | Items where each arm is best (P / G-dev / G-sim / G-sim+dev) |',
          '|---|---|---|---|---|']
    out['where_wins'] = {}
    for name, title, R, con, ks, kdev in summary:
        A = {a: R['arms'][a]['headline'] for a in ('P', 'G-dev', 'G-sim', 'G-sim+dev', 'C-dev') if a in R['arms']}
        best = max(A, key=A.get)
        b0 = max(['P', 'G-sim'], key=lambda a: A[a])
        bk = max([a for a in ('G-dev', 'G-sim+dev', 'C-dev') if a in A], key=lambda a: A[a])
        cnt = {a: 0 for a in ('P', 'G-dev', 'G-sim', 'G-sim+dev')}
        for it, ms in R['arms']['P']['scores'].items():
            m = list(ms.keys())[0]
            vals = {a: R['arms'][a]['scores'][it][m] for a in cnt}
            if not all(np.isfinite(v) for v in vals.values()):
                continue
            sgn = 1 if M.HIGHER[m] else -1
            cnt[max(vals, key=lambda a: sgn * vals[a])] += 1
        out['where_wins'][name] = {'best': best, 'best_0_labels': b0, 'best_k_labels': bk, 'items_best': cnt}
        ws.append(f'| {title} | {best} | {b0} | {bk} | {cnt["P"]} / {cnt["G-dev"]} / {cnt["G-sim"]} / {cnt["G-sim+dev"]} |')
    open(f'{ROOT}/results_tables.md', 'w').write('\n'.join(xs) + '\n\n' + '\n'.join(ws) + '\n\n' + '\n'.join(md))
    json.dump(out, open(f'{ROOT}/results.json', 'w'), indent=1, default=float)
    print('\n'.join(xs))
    print(out['verdict'])


if __name__ == '__main__':
    main()
