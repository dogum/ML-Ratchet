"""RoadSim v3.1 report: physics v3 vs v3.1 on every hybrid set. No clean judge exists for v3.1 (CALIBRATION.md H27-0):
exam-3, exam-2, the EPA held-out vehicles, the D3 exam and the VED exam had all been seen, so every row here is a SECOND
LOOK and judges nothing. Computed once, after the v3.1 freeze commit; nothing is chosen by it.

Sets: exam-3 and exam-2 EPA test cars (hybrids_epa__exam3|exam2__<preset>.json), the generator check by body class
(generator__<preset>.json), VED exam-2 twins (ved_twin_exam2__<preset>.json, local, when present). Groups as
exam3_report.py. EPA errors on fuel economy (+ = the sim burns less).

usage: python v31_report.py    -> v31_report.json and a printed table
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exam3_report as X

PRESETS = ('roadsim-physics-v3', 'roadsim-physics-v3.1')


def main():
    R = {p: dict(exam3=X.epa('exam3', p, 'family_seen_v2'), exam2=X.epa('exam2', p, 'family_seen_v2'),
                 ved_exam2=X.ved(p), generator=X.gen(p)) for p in PRESETS}
    json.dump(dict(about=__doc__.strip(), presets=list(PRESETS), results=R),
              open(os.path.join(HERE, 'v31_report.json'), 'w', encoding='utf-8', newline='\n'), indent=1)
    A, B = PRESETS
    for setname in ('exam3', 'exam2'):
        print(f'EPA {setname} (SECOND LOOK, all records), FE %, median v3 -> v3.1 (n)')
        for g in X.GROUPS:
            a = (R[A][setname] or {}).get('all', {}).get(g)
            b = (R[B][setname] or {}).get('all', {}).get(g)
            if not (a and b):
                continue
            cells = '  '.join(f'{k} {a[k][0]:+.1f} -> {b[k][0]:+.1f} ({a[k][1]})' for k in X.CYC if a[k][0] is not None)
            print(f"  {g:20s} n{a['n']:4d}  {cells}  | |err| {a['abs_all'][0]} -> {b['abs_all'][0]}")
    print('SECOND LOOK VED exam-2 (trip median / car median %):')
    for arch, a in (R[A]['ved_exam2'] or {}).items():
        b = (R[B]['ved_exam2'] or {}).get(arch)
        if b:
            print(f"  {arch:16s} trip {a['trip_median'][0]} -> {b['trip_median'][0]}  car {a['car_median'][0]} -> {b['car_median'][0]}")
    print('SECOND LOOK generator by class:')
    for p in PRESETS:
        print(' ', p, R[p]['generator'])


if __name__ == '__main__':
    main()
