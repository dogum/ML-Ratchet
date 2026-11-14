"""Dev-only check: the EPA dev pool (epa_dev_pool.json, RoadData dev families) on the EPA procedures, any options.
Writes epa_pool.json (or epa_pool__<preset>.json). Error = simulated / EPA - 1 on fuel economy (+ = sim burns less).
Medians and IQRs overall and per engine map. Used by the v2 calibration; never contains held-out vehicles.

usage: python check_epa_pool.py [--options roadsim-physics-v2] [--workers 14] [--cycles FTP,HWY,US06,SC03,COLD]
"""
import os, sys, json, time, argparse
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
import _opts
from epa_procedures import ftp, hot_cycle, sc03, test_spec, T_COLD


def one(job):
    v, opt, cycles = job
    _opts.set(opt)
    s = test_spec(v['spec']); e = v['epa']
    row = dict(name=v['name'], family=v['family'], map=s['engine']['bsfc_map'], cls=s['class'],
               trans=s['transmission']['type'], disp=s['engine']['displacement_L'], sim={}, epa={}, err_pct={})
    try:
        if 'FTP' in cycles:
            r = ftp(s)
            row['sim']['FTP'] = r['FE']; row['bags'] = r['bags']
        for cyc, key in (('hwfet', 'HWY'), ('us06', 'US06')):
            if key in cycles and key in e:
                row['sim'][key] = hot_cycle(s, cyc)['FE']
        if 'SC03' in cycles and 'SC03' in e:
            row['sim']['SC03'] = sc03(s)['FE']
        if 'COLD' in cycles and 'COLD' in e:
            row['sim']['COLD'] = ftp(s, T=T_COLD)['FE']
    except Exception as ex:
        row['error'] = f'{type(ex).__name__}: {ex}'[:200]
    for k, x in row['sim'].items():
        if k in e:
            row['epa'][k] = e[k]; row['err_pct'][k] = round(100 * (x / e[k] - 1), 2)
    return row


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    return dict(n=int(len(x)), median=round(float(np.median(x)), 2), p25=round(float(np.percentile(x, 25)), 2),
                p75=round(float(np.percentile(x, 75)), 2), iqr=round(float(np.subtract(*np.percentile(x, [75, 25]))), 2),
                mean_abs=round(float(np.mean(np.abs(x))), 2))


def summarize(rows, cycles=('FTP', 'HWY', 'US06', 'SC03', 'COLD')):
    out = {'all': {c: stats([r['err_pct'].get(c) for r in rows]) for c in cycles}}
    for m in sorted({r['map'] for r in rows}):
        out[m] = {c: stats([r['err_pct'].get(c) for r in rows if r['map'] == m]) for c in cycles}
    # tilt: log(FTP err) - log(HWY err) per vehicle
    t = [100 * (np.log1p(r['err_pct']['FTP'] / 100) - np.log1p(r['err_pct']['HWY'] / 100)) for r in rows if 'FTP' in r['err_pct'] and 'HWY' in r['err_pct']]
    out['all']['tilt_FTP_minus_HWY'] = stats(t)
    return out


def run(options=None, cycles=('FTP', 'HWY', 'US06', 'SC03', 'COLD'), workers=14, vehicles=None):
    P = json.load(open(os.path.join(HERE, 'epa_dev_pool.json')))['vehicles']
    if vehicles is not None:
        P = [v for v in P if v['name'] in vehicles or v['row'] in vehicles]
    with Pool(workers) as pool:
        rows = pool.map(one, [(v, options, tuple(cycles)) for v in P], chunksize=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--cycles', default='FTP,HWY,US06,SC03,COLD')
    _opts.init()
    a = ap.parse_args()
    t0 = time.time()
    cyc = a.cycles.split(',')
    rows = run(_opts.get(), cyc, a.workers)
    summ = summarize(rows, cyc)
    out = dict(check='EPA dev pool (RoadData dev families) on the EPA procedures', options=_opts.get(), options_name=_opts.name(),
               n=len(rows), n_errors=sum('error' in r for r in rows), summary=summ, vehicles=rows, wall_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(HERE, _opts.out('epa_pool.json')), 'w'), indent=1)
    for k, d in summ.items():
        print(k.ljust(20), '  '.join(f"{c} {s['median']:+.1f} [{s['p25']:+.1f},{s['p75']:+.1f}] n{s['n']}" for c, s in d.items() if s))
    print('errors', out['n_errors'], 'wall', out['wall_s'])


if __name__ == '__main__':
    main()
