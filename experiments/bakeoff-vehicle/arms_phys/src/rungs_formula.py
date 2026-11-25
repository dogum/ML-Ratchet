"""Formula-arm rungs: build each candidate law set as an arm folder under formula/work/cand_<id>/, score it with the
harness's evaluate.py (frozen sim val + dev gate, both tracks), and walk the rule in order.

usage: python3 rungs_formula.py <spec.json>
spec: {"rungs": [{"id": "r0", "change": "...", "laws": {"ice": ["s1", 5], "hev": ["s1", 7], "bev": ["s1", 9]}}, ...]}
(laws given as [search tag, complexity] of a front member, or "road_load" for E_phys's own form, not used here).
Writes work/rungs_formula.json (every rung's summary numbers + decisions). Resumable (an evaluated candidate is reused).
Rule (EnergyNet's model-rung rule, the harness's suggestion): keep a rung if its sim val skill beats the incumbent by
more than 0.01 AND its dev D16 (table track) is not worse than the incumbent's by more than 0.01.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
HARNESS = '/home/claude/bakeoff/vehicle/harness'
MARGIN = 0.01


def law_of(tag, g, complexity):
    S = json.load(open(os.path.join(ROOT, 'work', 'search', f'{tag}_{g}.json')))
    for L in S['front']:
        if L['complexity'] == complexity:
            return {**L, 'source': f'{tag}_{g}_c{complexity}'}
    raise KeyError((tag, g, complexity))


def evaluate(cand, track):
    out = os.path.join(cand, 'work', 'eval')
    p = os.path.join(out, f'cand_{track}_summary.json')
    if not os.path.exists(p):
        subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HARNESS, 'evaluate.py'), cand, '--track', track,
                        '--tag', 'cand', '--out', out], check=True)
    return json.load(open(p))


def main(spec_path):
    import build_formula as BF
    spec = json.load(open(spec_path))
    res = {'rule': f'keep if sim val skill > incumbent + {MARGIN} AND dev D16 (table) <= incumbent + {MARGIN}', 'rungs': []}
    inc = None
    for r in spec['rungs']:
        laws = {g: law_of(t, g, c) for g, (t, c) in r['laws'].items()}
        cand = os.path.join(ROOT, 'formula', 'work', f'cand_{r["id"]}')
        if not os.path.exists(os.path.join(cand, 'laws.json')):
            c0 = time.process_time()
            BF.build(cand, laws, r['id'], r['change'])
            r['bands_cpu_s'] = time.process_time() - c0
        sm = {t: evaluate(cand, t) for t in ('table', 'trace')}
        rec = {'id': r['id'], 'change': r['change'], 'laws': {g: laws[g]['source'] for g in laws},
               'exprs': {g: laws[g]['expr'] for g in laws}, 'complexity': {g: laws[g]['complexity'] for g in laws}}
        for t in ('table', 'trace'):
            s = sm[t]
            rec[t] = {'simval_skill': s['simval']['skill'], 'simval_MdAPE': s['simval']['MdAPE'],
                      'simval_cov50': s['simval']['cov50'], 'simval_cov90': s['simval']['cov90'],
                      'simval_crps_log': s['simval']['crps_log'], 'dev_D16': s['dev']['D16'],
                      'dev_sets': {k: round(v['MdAPE'], 4) for k, v in s['dev']['sets'].items() if v.get('n')},
                      'by_arch': s['simval']['by_arch']}
        sk, d16 = rec['table']['simval_skill'], rec['table']['dev_D16']
        if inc is None:
            dec, why = 'base', 'first rung (the incumbent to beat)'
        elif sk <= inc['table']['simval_skill'] + MARGIN:
            dec, why = 'rejected', f"sim val skill {sk:.4f} does not beat the incumbent's {inc['table']['simval_skill']:.4f} by more than {MARGIN}"
        elif d16 > inc['table']['dev_D16'] + MARGIN:
            dec, why = 'rejected', f"sim val skill {sk:.4f} > {inc['table']['simval_skill']:.4f} + {MARGIN}, but dev D16 {d16:.4f} is worse than {inc['table']['dev_D16']:.4f} + {MARGIN}"
        else:
            dec, why = 'kept', f"sim val skill {sk:.4f} > {inc['table']['simval_skill']:.4f} + {MARGIN} and dev D16 {d16:.4f} <= {inc['table']['dev_D16']:.4f} + {MARGIN}"
        rec['decision'], rec['why'] = dec, why
        if dec in ('base', 'kept'):
            inc = rec
        res['rungs'].append(rec)
        print(f"{r['id']:4s} skill {sk:.4f} MdAPE {rec['table']['simval_MdAPE']:.4f} D16 {d16:.4f} "
              f"(trace {rec['trace']['simval_skill']:.4f} / {rec['trace']['dev_D16']:.4f}) {dec}: {why}", flush=True)
    res['released'] = inc['id']
    json.dump(res, open(os.path.join(ROOT, 'work', 'rungs_formula.json'), 'w'), indent=1, default=float)
    print('released', inc['id'])


if __name__ == '__main__':
    main(sys.argv[1])
