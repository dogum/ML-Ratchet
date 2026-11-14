"""Exam-2 report: physics v2 vs v2.1 on the frozen hybrid exam (checks/exam2_hybrids.json), computed once after the
v2.1 freeze. Reads hybrids_epa__exam2__<preset>.json and ved_twin_exam2__<preset>.json, writes exam2_report.json.

EPA: median error per group and cycle (FE %, + = the sim burns less), also median |error| over every cycle of every
record; and the same for the records of families no check had ever touched ('fresh', family_seen_v2 false).
VED: median trip error and car-median error per architecture (fuel for HEVs, fuel + battery energy for PHEVs; + = the
sim uses more), engine-on gap, and Toyota-type vs other HEVs (fueleconomy.gov candidate models).

usage: python exam2_report.py [A B]   (default roadsim-physics-v2 roadsim-physics-v2.1)
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CYC = ('FTP', 'HWY', 'US06', 'SC03', 'COLD')
GROUPS = ('p2_tmed', 'p2_converter', 'p2_clutch', 'mild', 'toyota_ps', 'ford_ps', 'other_ps', 'honda_series', 'plugin_ps',
          'plugin_series', 'plugin_p2')


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return (round(float(np.median(x)), 1), len(x)) if x else (None, 0)


def epa(preset):
    p = os.path.join(HERE, f'hybrids_epa__exam2__{preset}.json')
    E = json.load(open(p))
    out = {'all': {}, 'fresh': {}}
    for tier, keep in (('all', lambda r: True), ('fresh', lambda r: not r.get('family_seen_v2'))):
        for g in GROUPS:
            rr = [r for r in E['vehicles'] if r['group'] == g and keep(r)]
            if not rr:
                continue
            o = {'n': len(rr)}
            for c in CYC:
                o[c] = med([r['err_pct'].get(c) for r in rr])
            o['abs_all'] = med([abs(v) for r in rr for v in r['err_pct'].values()])
            out[tier][g] = o
    out['sha256'] = E.get('exam2_sha256')
    out['n_errors'] = E.get('n_errors')
    return out


def ved(preset):
    p = os.path.join(HERE, f'ved_twin_exam2__{preset}.json')
    if not os.path.exists(p):
        return None
    V = json.load(open(p))
    tw = V['twins']
    out = {}
    def kind(vid):
        m = str((tw[vid]['info'].get('labels') or {}).get('example_models') or '').upper()
        return 'toyota' if any(k in m for k in ('PRIUS', 'CAMRY', 'LEXUS', 'HIGHLANDER', 'RAV4', 'AVALON', 'CT 200', 'ES 300')) else 'other'
    trips = V['trips']
    for a in sorted({t['arch'] for t in trips}):
        T = [t for t in trips if t['arch'] == a]
        cars = sorted({t['vehicle_id'] for t in T})
        cm = [np.median([t['err_primary'] for t in T if t['vehicle_id'] == c and t.get('err_primary') is not None] or [np.nan]) for c in cars]
        o = dict(n_trips=len(T), n_cars=len(cars), trip_median=med([t.get('err_primary') for t in T]),
                 car_median=med(cm), car_abs_median=med([abs(x) for x in cm]),
                 eng_on_pts=med([t.get('eng_on_pts') for t in T]))
        if a == 'hev_powersplit':
            for k in ('toyota', 'other'):
                TT = [t for t in T if kind(t['vehicle_id']) == k]
                cc = sorted({t['vehicle_id'] for t in TT})
                o[k] = dict(n_cars=len(cc), trip_median=med([t.get('err_primary') for t in TT]),
                            car_median=med([np.median([t['err_primary'] for t in TT if t['vehicle_id'] == c and t.get('err_primary') is not None] or [np.nan]) for c in cc]),
                            eng_on_pts=med([t.get('eng_on_pts') for t in TT]))
        out[a] = o
    return out


def main():
    A, B = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('roadsim-physics-v2', 'roadsim-physics-v2.1')
    R = {A: dict(epa=epa(A), ved=ved(A)), B: dict(epa=epa(B), ved=ved(B))}
    json.dump(dict(about=__doc__.strip(), presets=[A, B], results=R), open(os.path.join(HERE, 'exam2_report.json'), 'w',
                                                                           encoding='utf-8', newline='\n'), indent=1)
    for tier in ('all', 'fresh'):
        print(f'EPA exam-2 ({tier}), FE %, median (n): {A} -> {B}')
        for g in GROUPS:
            a, b = R[A]['epa'][tier].get(g), R[B]['epa'][tier].get(g)
            if not a:
                continue
            cells = '  '.join(f'{c} {a[c][0]:+.1f}->{b[c][0]:+.1f} ({a[c][1]})' for c in CYC if a[c][0] is not None)
            print(f"  {g:14s} n{a['n']:4d}  {cells}  | median |err| {a['abs_all'][0]} -> {b['abs_all'][0]}")
    print(f'VED exam-2 (trip / car median %, + = sim uses more; engine-on points): {A} -> {B}')
    for arch, a in (R[A]['ved'] or {}).items():
        b = R[B]['ved'][arch]
        print(f"  {arch:16s} cars {a['n_cars']:3d} trips {a['n_trips']:5d}  trip {a['trip_median'][0]} -> {b['trip_median'][0]}  car {a['car_median'][0]} -> {b['car_median'][0]}"
              f"  car |err| {a['car_abs_median'][0]} -> {b['car_abs_median'][0]}  eng-on {a['eng_on_pts'][0]} -> {b['eng_on_pts'][0]}")
        for k in ('toyota', 'other'):
            if k in a:
                print(f"     {k:7s} cars {a[k]['n_cars']:3d}  trip {a[k]['trip_median'][0]} -> {b[k]['trip_median'][0]}  car {a[k]['car_median'][0]} -> {b[k]['car_median'][0]}  eng-on {a[k]['eng_on_pts'][0]} -> {b[k]['eng_on_pts'][0]}")


if __name__ == '__main__':
    main()
