"""Exam-3 report (RoadSim v3 pass): physics v2 vs v2.1 vs v3 on the clean exam (checks/exam3_hybrids.json), with the
second-look sets labelled. Computed once, after the v3 freeze commit; nothing is chosen by it.

Clean:       exam-3, EPA test cars no check had simulated (hybrids_epa__exam3__<preset>.json).
Second look: exam-2 EPA (hybrids_epa__exam2__<preset>.json) and VED (ved_twin_exam2__<preset>.json, local), the
             generator check by body class (generator__<preset>.json); seen at the v2.1 report or earlier.
Groups: Toyota / Lexus power-split by engine era (pre-TNGA MY<=2017, TNGA MY>=2018), converter P2, TMED, Honda e:HEV,
Ford power-split, mild, plug-in power-split, plug-in series, plug-in P2. EPA errors on fuel economy (+ = the sim burns
less); VED on fuel or fuel + battery energy (+ = the sim uses more).

usage: python exam3_report.py    -> exam3_report.json and a printed table
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS = ('roadsim-physics-v2', 'roadsim-physics-v2.1', 'roadsim-physics-v3')
CYC = ('FTP', 'HWY', 'US06', 'SC03', 'COLD')
GROUPS = ('toyota_ps:pre-TNGA', 'toyota_ps:TNGA', 'p2_converter', 'p2_tmed', 'p2_clutch', 'honda_series', 'ford_ps',
          'other_ps', 'mild', 'plugin_ps', 'plugin_series', 'plugin_p2')


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return (round(float(np.median(x)), 1), len(x)) if x else (None, 0)


def group(r):
    g = r['group']
    if g == 'toyota_ps':
        y = r.get('model_year')
        if y is None:
            y = int(str(r['name'])[:4])
        return f"toyota_ps:{'pre-TNGA' if y <= 2017 else 'TNGA'}"
    return g


def epa(setname, preset, fresh_key):
    p = os.path.join(HERE, f'hybrids_epa__{setname}__{preset}.json')
    if not os.path.exists(p):
        return None
    E = json.load(open(p))
    out = {'all': {}, 'fresh': {}, 'sha256': E.get('exam_sha256') or E.get('exam2_sha256'), 'n_errors': E.get('n_errors')}
    for tier, keep in (('all', lambda r: True), ('fresh', lambda r: not r.get(fresh_key))):
        for g in GROUPS:
            rr = [r for r in E['vehicles'] if group(r) == g and keep(r)]
            if not rr:
                continue
            o = {'n': len(rr)}
            for c in CYC:
                o[c] = med([r['err_pct'].get(c) for r in rr])
            o['abs_all'] = med([abs(v) for r in rr for v in r['err_pct'].values()])
            out[tier][g] = o
    return out


def ved(preset):
    sys.path.insert(0, HERE)
    import exam2_report
    try:
        return exam2_report.ved(preset)
    except FileNotFoundError:
        return None


def gen(preset):
    p = os.path.join(HERE, f'generator__{preset}.json')
    if not os.path.exists(p):
        return None
    G = json.load(open(p))
    return {c: v['err_pct'] for c, v in (G.get('hev_by_class') or {}).items()} | {'hev_UDDS_hot_median_mpg': G['simulated']['hev']['UDDS_hot']['median']}


def main():
    R = {}
    for p in PRESETS:
        R[p] = dict(exam3=epa('exam3', p, 'family_seen_v2'), exam2=epa('exam2', p, 'family_seen_v2'), ved_exam2=ved(p), generator=gen(p))
    json.dump(dict(about=__doc__.strip(), presets=list(PRESETS), results=R), open(os.path.join(HERE, 'exam3_report.json'), 'w',
                                                                                    encoding='utf-8', newline='\n'), indent=1)
    A, B, C = PRESETS
    for setname, label in (('exam3', 'CLEAN exam-3'), ('exam2', 'SECOND LOOK exam-2')):
        for tier in ('all', 'fresh'):
            print(f'EPA {label} ({tier}), FE %, median (n): v2 -> v2.1 -> v3')
            for g in GROUPS:
                a = (R[A][setname] or {}).get(tier, {}).get(g)
                if not a:
                    continue
                b, c = R[B][setname][tier][g], R[C][setname][tier][g]
                cells = '  '.join(f'{k} {a[k][0]:+.1f}/{b[k][0]:+.1f}/{c[k][0]:+.1f} ({a[k][1]})' for k in CYC if a[k][0] is not None)
                print(f"  {g:20s} n{a['n']:4d}  {cells}  | |err| {a['abs_all'][0]}/{b['abs_all'][0]}/{c['abs_all'][0]}")
    print('SECOND LOOK VED exam-2 (trip median / car median %, engine-on points): v2 -> v2.1 -> v3')
    for arch, a in (R[A]['ved_exam2'] or {}).items():
        b = (R[B]['ved_exam2'] or {}).get(arch); c = (R[C]['ved_exam2'] or {}).get(arch)
        if not (b and c):
            continue
        print(f"  {arch:16s} cars {a['n_cars']:3d}  trip {a['trip_median'][0]}/{b['trip_median'][0]}/{c['trip_median'][0]}"
              f"  car {a['car_median'][0]}/{b['car_median'][0]}/{c['car_median'][0]}  eng-on {a['eng_on_pts'][0]}/{b['eng_on_pts'][0]}/{c['eng_on_pts'][0]}")
    print('SECOND LOOK generator, generated hybrids vs EPA MY2022-24 hybrids by class (hot UDDS vs FTP %, HWY %):')
    for p in PRESETS:
        print(' ', p, R[p]['generator'])


if __name__ == '__main__':
    main()
