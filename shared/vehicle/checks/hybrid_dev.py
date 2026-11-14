"""Dev scoreboard for the hybrid / plug-in calibration (v2.1): every dev source in one run, dev data only.

  EPA     check_hybrids_epa.py --set dev: every EPA dev-family hybrid / plug-in record RoadSim models + the four
          check_cycles dev hybrids (FE %, + = the sim burns less)
  D3      check_d3.py --split dev on the D3 dev hybrids and plug-ins (fuel_corr / energy %, + = the sim uses more)
  VED     check_ved_twin.py on the VED dev HEV / PHEV cars (trip errors %, + = the sim uses more; engine-on points)
Guard metrics (v2 improved them on dev; v2.1 may not make them worse by more than 2 points): EPA dev Accord Hybrid
HWY, D3 Accord Hybrid HWFET, VED Toyota car-median fuel and engine-on gap, P2 FTP median (EPA dev P2s).

usage: python hybrid_dev.py --options roadsim-physics-v2 --tag v2 [--only epa,d3,ved]
       python hybrid_dev.py --options '{"preset": "roadsim-physics-v2", "regen_share": {}}' --tag h11a
Writes roadsim/logs/hybrid_dev/<tag>.json and prints the scoreboard.
"""
import os, sys, json, subprocess, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
LOG = os.path.join(VEH, 'roadsim', 'logs', 'hybrid_dev')
D3_DEV_HYB = ['D3_2021_Hyundai_Sonata_Hybrid', 'D3_2015_Honda_Accord_Hybrid', 'D3_2010_Ford_Fusion_Hybrid', 'D3_2010_Honda_Insight',
              'D3_2010_Toyota_Prius', 'D3_2010_Mercedes_S400h_BlueHybrid', 'D3_2017_Toyota_Prius_Prime', 'D3_2016_Chevrolet_Volt',
              'D3_2013_Ford_Cmax_Energi', 'D3_2013_Ford_Fusion_Energi']
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')


def run(args, log):
    with open(log, 'w', encoding='utf-8') as f:
        r = subprocess.run([sys.executable] + args, cwd=HERE, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f'{args[0]} failed, see {log}')


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return round(float(np.median(x)), 2) if x else None


def collect(tag):
    o = {}
    p = os.path.join(LOG, f'{tag}__epa.json')
    if os.path.exists(p):
        E = json.load(open(p))
        o['epa_by_group'] = {g: {c: v[c]['median'] for c in ('FTP', 'HWY', 'US06', 'SC03', 'COLD') if c in v} | {'n': v['n']}
                             for g, v in E['summary']['by_group'].items()}
        o['epa_check_dev'] = {r['name']: r['err_pct'] for r in E['vehicles'] if r['source'] == 'check_cycles dev vehicle'}
        o['epa_by_make'] = E['summary']['by_group_make']
        p2 = [r['err_pct'].get('FTP') for r in E['vehicles'] if r['group'] in ('p2_tmed', 'p2_clutch', 'p2_converter')]
        o['epa_p2_FTP_median'] = med(p2)
    p = os.path.join(LOG, f'{tag}__d3.json')
    if os.path.exists(p):
        D = json.load(open(p))
        by = {}
        for t in D['tests']:
            k = (t['vehicle'], t['cycle'] if t['cycle'] in ('UDDS', 'HWFET', 'US06', 'SC03') else 'other', t['T_class'])
            by.setdefault(k, []).append(t)
        o['d3_by_vehicle_cycle'] = {f'{a}|{b}|{c}': dict(n=len(v), err=med([t['err'].get('primary_pct') for t in v]),
                                                          eng_on_pts=med([t['err'].get('eng_on_share_pts') for t in v]))
                                    for (a, b, c), v in sorted(by.items())}
        bya = {}
        for t in D['tests']:
            bya.setdefault(t['architecture'], []).append(t)
        o['d3_by_arch'] = {a: dict(n=len(v), err=med([t['err'].get('primary_pct') for t in v]),
                                   eng_on_sim=med([t['sim']['eng_on_share'] for t in v if t['meas'].get('eng_on_share') is not None]),
                                   eng_on_meas=med([t['meas']['eng_on_share'] for t in v if t['meas'].get('eng_on_share') is not None]))
                           for a, v in sorted(bya.items())}
        acc = [t for t in D['tests'] if 'Accord' in t['vehicle'] and t['cycle'] == 'HWFET' and t['T_class'] == 'std']
        o['d3_accord_hwfet'] = med([t['err'].get('primary_pct') for t in acc])
    p = os.path.join(LOG, f'{tag}__ved.json')
    if os.path.exists(p):
        V = json.load(open(p))
        o['ved_by_car'] = {k: dict(arch=c['arch'], err=(c.get('err_primary') or {}).get('median'),
                                   eng_on_pts=(c.get('eng_on_pts') or {}).get('median'))
                           for k, c in V['summary']['by_car'].items()}
        toy = [c for k, c in o['ved_by_car'].items() if c['arch'] == 'hev_powersplit']
        o['ved_toyota_car_median'] = med([c['err'] for c in toy])
        trips = [t for t in V['trips'] if t['arch'] == 'hev_powersplit']
        o['ved_toyota_eng_on_pts'] = med([t.get('eng_on_pts') for t in trips])
    return o


def show(o, ref=None):
    def d(path, v):
        if ref is None:
            return ''
        r = ref
        for k in path:
            r = (r or {}).get(k) if isinstance(r, dict) else None
        if r is None or v is None:
            return ''
        return f' ({v - r:+.1f})'
    print('EPA dev (FE %, + = sim burns less), group medians:')
    for g, v in sorted((o.get('epa_by_group') or {}).items()):
        print(f"  {g:14s} n{v['n']:3d} " + '  '.join(f'{c} {v[c]:+.1f}{d(["epa_by_group", g, c], v[c])}' for c in ('FTP', 'HWY', 'US06', 'SC03', 'COLD') if c in v))
    for n, e in (o.get('epa_check_dev') or {}).items():
        print(f'  check-dev {n[:30]:30s} ' + '  '.join(f'{c} {x:+.1f}{d(["epa_check_dev", n, c], x)}' for c, x in e.items()))
    print(f"  P2 FTP median {o.get('epa_p2_FTP_median')}{d(['epa_p2_FTP_median'], o.get('epa_p2_FTP_median'))}")
    print('D3 dev (energy %, + = sim uses more):')
    for a, v in (o.get('d3_by_arch') or {}).items():
        print(f"  {a:16s} n{v['n']:4d} err {v['err']:+.1f}{d(['d3_by_arch', a, 'err'], v['err'])}  eng-on {v['eng_on_sim']}/{v['eng_on_meas']}")
    print(f"  Accord HWFET {o.get('d3_accord_hwfet')}{d(['d3_accord_hwfet'], o.get('d3_accord_hwfet'))}")
    print('VED dev (trip median %, + = sim uses more):')
    for k, c in (o.get('ved_by_car') or {}).items():
        print(f"  {k:8s} {c['arch']:16s} {c['err']}{d(['ved_by_car', k, 'err'], c['err'])}  eng-on pts {c['eng_on_pts']}")
    print(f"  Toyota car median {o.get('ved_toyota_car_median')}{d(['ved_toyota_car_median'], o.get('ved_toyota_car_median'))}"
          f"  engine-on pts {o.get('ved_toyota_eng_on_pts')}{d(['ved_toyota_eng_on_pts'], o.get('ved_toyota_eng_on_pts'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--options', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--only', default='epa,d3,ved')
    ap.add_argument('--ref', default='v2')
    ap.add_argument('--epa-set', default='dev', choices=['dev', 'devx'])
    a = ap.parse_args()
    os.makedirs(LOG, exist_ok=True)
    only = a.only.split(',')
    t0 = time.time()
    if 'epa' in only:
        run(['check_hybrids_epa.py', '--set', a.epa_set, '--options', a.options, '--out', os.path.join(LOG, f'{a.tag}__epa.json')],
            os.path.join(LOG, f'{a.tag}__epa.log'))
    if 'd3' in only:
        run(['check_d3.py', '--split', 'dev', '--options', a.options, '--vehicles', ','.join(D3_DEV_HYB),
             '--out', os.path.join(LOG, f'{a.tag}__d3.json')], os.path.join(LOG, f'{a.tag}__d3.log'))
    if 'ved' in only:
        run(['check_ved_twin.py', '--options', a.options, '--arch', 'hev_powersplit,hev_parallel_p2,phev_series,phev_powersplit',
             '--out', os.path.join(LOG, f'{a.tag}__ved.json')], os.path.join(LOG, f'{a.tag}__ved.log'))
    o = collect(a.tag)
    o['options'] = a.options; o['wall_s'] = round(time.time() - t0)
    json.dump(o, open(os.path.join(LOG, f'{a.tag}.json'), 'w', encoding='utf-8', newline='\n'), indent=1)
    ref = None
    rp = os.path.join(LOG, f'{a.ref}.json')
    if a.ref != a.tag and os.path.exists(rp):
        ref = json.load(open(rp))
    show(o, ref)
    print(f'{o["wall_s"]} s')


if __name__ == '__main__':
    main()
