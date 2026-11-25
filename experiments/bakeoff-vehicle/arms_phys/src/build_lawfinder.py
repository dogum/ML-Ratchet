"""Export the trip-energy law in LawFinder's formats -> arms_phys/lawfinder_road/.

usage: python3 build_lawfinder.py
Reads work/rungs_formula.json (the formula arm's released rung), the ladder caches work/ladder_<tag>_<group>.json and
the bank / dev through the harness. Writes:
  road_law_entry.json    {"road": {...}}: an entry for release/lawfinder_laws.json["laws"] (same keys as the five laws)
                          + "companions" (the hybrid, battery-electric and plug-in laws of the same rung)
  road_ratchet.json      results/road_ratchet.json in LawFinder's ratchet format (rungs = the conventional fronts of the
                          searches in order, decisions re-taken with LawFinder's rule; bars; history; released)
  road_points.json       page data: 1,400 simulated conventional val trips + the VED ICE dev trips (vars + e)
Nothing reads exam data. LawFinder's own pretty.py (read-only) gives the readable text.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'lawfinder_road')
sys.path.insert(0, HERE)
sys.path.insert(0, OUT)
import lawsearch as ls  # noqa: E402
import law_road as L  # noqa: E402

MARGIN, DEV_TOL = 0.01, 1.05


def pretty_text(tree, consts, names):
    sys.path.insert(0, '/home/claude/bb/lawfinder/src')           # read-only import of LawFinder's pretty.py
    import pretty
    sys.path.pop(0)
    try:
        return pretty.text(ls.from_json(tree), np.array(consts, float), names)
    except Exception:  # noqa: BLE001
        return ls.to_str(ls.from_json(tree), np.array(consts, float), names, 4)


def law_eval(law, df):
    d = {k: df[k].values.astype(float) for k in df.columns if k in L.VARS}
    X = np.stack([d[k] for k in law['vars']], 1)
    return ls.evaluate(ls.from_json(law['tree']), X, np.array(law['consts'], float))


def r2log(y, yh):
    ly, lh = np.log(y), np.log(np.where(yh > 0, yh, np.nan))
    ok = np.isfinite(lh)
    if ok.mean() < 0.99:
        return -np.inf
    return float(1 - np.sum((lh[ok] - ly[ok]) ** 2) / np.sum((ly[ok] - ly[ok].mean()) ** 2))


def main():
    R = json.load(open(os.path.join(ROOT, 'work', 'rungs_formula.json')))
    rel = next(r for r in R['rungs'] if r['id'] == R['released'])
    rel_laws = json.load(open(os.path.join(ROOT, 'formula', 'laws.json')))['laws']
    ice = rel_laws['ice']
    va = L.load_bank('v2', 'val')
    dev = L.load_dev()
    # ---------------------------------------------------------------- LawFinder ladder over the conventional fronts
    tags = sorted({json.load(open(os.path.join(ROOT, 'work', 'search', f))).get('tag') for f in
                   os.listdir(os.path.join(ROOT, 'work', 'search')) if f.endswith('_ice.json')})
    rungs = [{'id': 'r0', 'version': 'v2', 'change': 'A constant (the simulator mean)', 'vars': ['p'], 'tree': ['c', 0],
              'consts': [float(np.mean(va.e))], 'complexity': 1, 'expr': f'{np.mean(va.e):.4g}'}]
    for tag in tags:
        lad = json.load(open(os.path.join(ROOT, 'work', f'ladder_{tag}_ice.json')))
        for m in lad['members']:
            rungs.append({'id': f'{tag}-c{m["complexity"]}', 'version': 'v2', 'search': tag,
                          'change': f'search {tag} ({", ".join(m["vars"])}), complexity {m["complexity"]}',
                          **{k: m[k] for k in ('vars', 'tree', 'consts', 'complexity', 'expr')},
                          'dev_bakeoff': m['dev'], 'dev_sets': m['dev_sets'], 'sim_medlog': m['sim_medlog']})
    inc, dec = None, []
    for r in rungs:
        r['js'] = ls.to_js(ls.from_json(r['tree']), np.array(r['consts'], float), r['vars'])
        r['sim'] = r2log(va.e.values, law_eval(r, va))
        r['dev'] = L.dev_error(dev.e.values, law_eval(r, dev))
        r['scores'] = {'v2': {'sim': r['sim'], 'dev': r['dev']}}
        if inc is None:
            d, why = 'kept', 'First rung (the incumbent to beat).'
        elif not np.isfinite(r['sim']) or not np.isfinite(r['dev']):
            d, why = 'rejected', 'Not finite on the frozen set or on dev.'
        elif r['sim'] <= inc['sim'] + MARGIN:
            d, why = 'rejected', f"Frozen-set skill {r['sim']:.3f} does not beat the incumbent's {inc['sim']:.3f} by more than {MARGIN}."
        elif r['dev'] > DEV_TOL * inc['dev']:
            d, why = 'rejected', (f"Frozen-set skill {r['sim']:.3f} beats {inc['sim']:.3f}, but the dev error {r['dev']:.4g} "
                                  f"is more than 5% worse than {inc['dev']:.4g}.")
        else:
            d, why = 'kept', (f"Frozen-set skill {r['sim']:.3f} > {inc['sim']:.3f} + {MARGIN}, dev error {r['dev']:.4g} vs "
                              f"{inc['dev']:.4g} (within 5%).")
        dec.append({'id': r['id'], 'decision': d, 'why': why})
        if d == 'kept':
            inc = r
    lf_released = inc['id']
    # the bake-off's released conventional law (may differ from LawFinder's own walk; both are recorded)
    src = ice.get('source', '')                     # e.g. s2_ice_c18
    bo_id = src.replace('_ice_', '-') if src else None
    rel_r = next((r for r in rungs if r['id'] == bo_id), None)
    # the bake-off's decision as a second history entry (LawFinder's page reads the last one): same decisions, except
    # that the law the bake-off released (and graded on its exam) is marked kept with the bake-off's reason
    bo_hist = []
    if rel_r is not None and rel_r['id'] != lf_released:
        d2 = [dict(x) for x in dec]
        for x in d2:
            if x['id'] == rel_r['id']:
                x['decision'] = 'kept'
                x['why'] = (f"Released by the Vehicle Bake-off's ratchet (rung {R['released']}: sim val skill over all "
                            f"powertrains {rel['table']['simval_skill']:.3f}, dev D16 {rel['table']['dev_D16']:.4f}); the "
                            f"law graded on the bake-off exam. LawFinder's own rule above keeps {lf_released}. "
                            f"(LawFinder rule: {x['why']})")
        bo_hist = [{'frozen_set': 'Vehicle Bake-off rule (sim val skill + D16, all powertrains)', 'decisions': d2,
                    'released': rel_r['id']}]
    # ---------------------------------------------------------------- bars (log space)
    res_v = np.log(law_eval(ice, va) / va.e.values); res_d = np.log(law_eval(ice, dev) / dev.e.values)
    sig = float(np.quantile(np.abs(res_v[np.isfinite(res_v)]), 0.6827))
    k = max(1.0, float(np.quantile(np.abs(res_d[np.isfinite(res_d)]), 0.6827)) / sig)
    bars = {'kind': 'log', 'note': 'sigma is a 68 % half-width in natural log: e x exp(+-sigma)', 'sigma_sim': sig,
            'widen_dev': k, 'sigma': sig * k, 'dev_cov68_raw': float(np.mean(np.abs(res_d) <= sig)),
            'dev_cov68_widened': float(np.mean(np.abs(res_d) <= sig * k))}
    # textbook: road-load with EnergyNet's constants and fitted on dev (LawFinder fits the textbook on dev)
    tb_dev = L.fit_road_load(dev)
    knee = json.load(open(os.path.join(ROOT, 'work', f'ladder_{tags[0]}_ice.json')))
    kn = next(m for m in knee['members'] if m['complexity'] == knee['knee'])
    ratchet = {'bars': bars, 'law': 'road', 'title': L.TITLE, 'vars': L.VARS, 'var_info': L.VAR_INFO, 'target': L.TARGET,
               'margin': MARGIN, 'dev_tol': DEV_TOL,
               'rule': (f'Keep a rung if its frozen-set R^2 of log e beats the incumbent by more than {MARGIN} AND its dev '
                        'error (MdAPE on the VED ICE dev trips) is no more than 5% worse. Rungs = the conventional Pareto '
                        'fronts of the searches in order of search, then complexity. The Vehicle Bake-off released its law '
                        'by its own rule (sim val skill + D16 over all powertrains): bakeoff_released.'),
               'rungs': rungs, 'history': [{'frozen_set': 'val_v2 (bake-off frozen val, conventional trips)',
                                            'decisions': dec, 'released': lf_released}] + bo_hist,
               'released': rel_r['id'] if rel_r else lf_released, 'lawfinder_rule_released': lf_released,
               'bakeoff_released': {'rung': R['released'], 'ice_law': bo_id, 'change': rel['change']}}
    json.dump(ratchet, open(os.path.join(OUT, 'road_ratchet.json'), 'w'), indent=1, default=float)
    # ---------------------------------------------------------------- the lawfinder_laws.json entry
    names = ice['vars']
    entry = {'title': L.TITLE, 'platform': 'RoadSim', 'vars': names, 'var_info': {v: L.VAR_INFO[v] for v in names},
             'target': L.TARGET, 'rung': ratchet['released'], 'expr': ice['expr'],
             'text': pretty_text(ice['tree'], ice['consts'], names), 'tree': ice['tree'], 'consts': ice['consts'],
             'sigma': bars['sigma'], 'sigma_sim': bars['sigma_sim'], 'widen_dev': bars['widen_dev'], 'sigma_kind': 'log',
             'simulator_law': {'rung': f'{tags[0]}-c{kn["complexity"]}', 'text': pretty_text(kn['tree'], kn['consts'], kn['vars']),
                               'tree': kn['tree'], 'consts': kn['consts'], 'vars': kn['vars']},
             'textbook': {'form': 'road_load', 'const': tb_dev, 'const_energynet': list(L.ROAD_LOAD_ENET),
                          'label': 'road-load estimate e = a p + k (m / 1.5)^0.7 t (EnergyNet E_phys, conventional form), '
                                   'a and k fitted on the VED ICE dev trips; EnergyNet\'s simulation constants in const_energynet'},
             'exam': None,
             'exam_note': 'Graded once in the Vehicle Bake-off exam (harness/exam.py) as the formula arm; exam_road.py turns '
                          'that result into this block.',
             'companions': {g: {'vars': l['vars'], 'tree': l['tree'], 'consts': l['consts'], 'expr': l['expr'],
                                'text': pretty_text(l['tree'], l['consts'], l['vars'])}
                            for g, l in rel_laws.items() if g != 'ice'},
             'companions_note': 'hev = the three hybrid systems; bev = battery-electric; plug-ins use the phev law if '
                                'present, else f x E_bev + (1 - f) x E_hev. All laws give e in MJ/km (E = e x d).',
             'bakeoff': {'arm': 'formula', 'rung': R['released'],
                         'simval_skill': rel['table']['simval_skill'], 'simval_MdAPE': rel['table']['simval_MdAPE'],
                         'dev_D16': rel['table']['dev_D16'], 'dev_sets': rel['table']['dev_sets']}}
    json.dump({'road': entry}, open(os.path.join(OUT, 'road_law_entry.json'), 'w'), indent=1, default=float)
    # ---------------------------------------------------------------- page points
    cloud = va.sample(min(1400, len(va)), random_state=0)
    pts = {'cloud': {k: [float(f'{v:.4g}') for v in cloud[k]] for k in L.VARS + ['e']},
           'dev': {k: [float(f'{v:.4g}') for v in dev[k]] for k in L.VARS + ['e']},
           'xdef': 't', 'question': 'How much fuel energy does a car, van, bus or truck use per km on a real trip?'}
    json.dump(pts, open(os.path.join(OUT, 'road_points.json'), 'w'), separators=(',', ':'))
    print('LawFinder walk released', lf_released, '| bake-off ice law', bo_id, '| bars', bars, '| textbook dev', tb_dev)
    print('text:', entry['text'])


if __name__ == '__main__':
    main()
