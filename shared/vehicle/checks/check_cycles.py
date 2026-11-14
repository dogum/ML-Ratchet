"""Check: EPA cycles (FTP-75 / HWFET / US06, plus SC03 and the 20 F cold FTP where tested) for 47 real MY2022-24
test vehicles (22 conventional, 14 HEV, 11 BEV) vs their EPA test-car-list results.  Writes cycles_vs_epa.json.

Error = simulated / EPA - 1 on fuel economy (mpg, or MPGe; for BEVs reported in kWh/100 mi the error is on
1/consumption, i.e. also an 'economy' error).  Calibrated constants were fitted on the dev subset only; the headline
numbers are the held-out vehicles.

usage: python check_cycles.py [--options roadsim-physics-v2] [--split dev|heldout|all]
  --split dev runs only the dev vehicles (what a calibration pass may look at) and writes <out>__dev.json.
  The cold FTP (20 F) runs for every vehicle with an EPA cold result (conventional and hybrid).
"""
import os, sys, json, time, argparse
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
import _opts
from epa_procedures import ftp, hot_cycle, sc03, test_spec, T_COLD
import roadsim

V = json.load(open(os.path.join(HERE, 'epa_vehicles.json')))


def stats(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return None
    return dict(n=int(len(x)), median_pct=round(float(np.median(x)), 2), mean_abs_pct=round(float(np.mean(np.abs(x))), 2),
                p25_pct=round(float(np.percentile(x, 25)), 2), p75_pct=round(float(np.percentile(x, 75)), 2),
                min_pct=round(float(np.min(x)), 2), max_pct=round(float(np.max(x)), 2))


def one(job):
    v, opt = job
    _opts.set(opt)
    worst_closure = 0.0; worst_trace = 0.0
    if True:
        s = test_spec(v['spec']); e = v['epa']; a = s['architecture']
        row = dict(name=v['name'], architecture=a, cls=s['class'], split=v['split'], sim={}, epa={}, err_pct={})
        if a == 'bev':
            for cyc, key in (('udds', 'CD_UDDS'), ('hwfet', 'CD_HWY')):
                r = hot_cycle(s, cyc)
                worst_closure = max(worst_closure, r['closure']); worst_trace = max(worst_trace, r['trace_max'])
                if v['bev_unit'] == 'MPGe':
                    sim = r['MPGe']; ref = e[key]; err = 100 * (sim / ref - 1)
                else:
                    sim = r['kWh_100mi_ac']; ref = e[key]; err = 100 * (ref / sim - 1)
                k = 'UDDS' if cyc == 'udds' else 'HWY'
                row['sim'][k] = round(sim, 2); row['epa'][k] = ref; row['err_pct'][k] = round(err, 2)
            row['unit'] = v['bev_unit']
        else:
            r = ftp(s); worst_closure = max(worst_closure, r['closure']); worst_trace = max(worst_trace, r['trace_max'])
            row['sim']['FTP'] = round(r['FE'], 2); row['epa']['FTP'] = e['FTP']; row['err_pct']['FTP'] = round(100 * (r['FE'] / e['FTP'] - 1), 2)
            if e.get('FE_Bag_1', 0) > 5 and e.get('FE_Bag_3', 0) > 5 and e['FE_Bag_1'] < e['FE_Bag_3']:
                row['bag1_over_bag3'] = dict(sim=round(r['bags'][0] / r['bags'][2], 4), epa=round(e['FE_Bag_1'] / e['FE_Bag_3'], 4))
            for cyc, key in (('hwfet', 'HWY'), ('us06', 'US06')):
                if key not in e:
                    continue
                r = hot_cycle(s, cyc); worst_closure = max(worst_closure, r['closure']); worst_trace = max(worst_trace, r['trace_max'])
                row['sim'][key] = round(r['FE'], 2); row['epa'][key] = e[key]; row['err_pct'][key] = round(100 * (r['FE'] / e[key] - 1), 2)
            if 'SC03' in e:
                r = sc03(s); worst_closure = max(worst_closure, r['closure'])
                row['sim']['SC03'] = round(r['FE'], 2); row['epa']['SC03'] = e['SC03']; row['err_pct']['SC03'] = round(100 * (r['FE'] / e['SC03'] - 1), 2)
                row['sc03_hvac_W'] = round(r['hvac_W'])
                row['sc03_cabin_C'] = dict(start=round(r['cabin_start_C'], 1), t120_300_595=[round(x, 1) for x in r['cabin_C_120_300_595']])
            if 'COLD' in e:
                r = ftp(s, T=T_COLD); worst_closure = max(worst_closure, r['closure'])
                row['sim']['COLD_FTP_20F'] = round(r['FE'], 2); row['epa']['COLD_FTP_20F'] = e['COLD']
                row['err_pct']['COLD_FTP_20F'] = round(100 * (r['FE'] / e['COLD'] - 1), 2)
                for i in (1, 2, 3):
                    if e.get(f'COLD_bag{i}', 0) > 1:
                        row.setdefault('cold_bags', {})[f'bag{i}'] = dict(sim=round(r['bags'][i - 1], 2), epa=e[f'COLD_bag{i}'])
        row['_closure'] = worst_closure; row['_trace'] = worst_trace
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='all', choices=['dev', 'heldout', 'all'])
    ap.add_argument('--workers', type=int, default=14)
    _opts.init()
    a_ = ap.parse_args()
    t0 = time.time()
    VV = [v for v in V if a_.split == 'all' or v['split'] == a_.split]
    with Pool(min(a_.workers, len(VV))) as pool:
        rows = pool.map(one, [(v, _opts.get()) for v in VV], chunksize=1)
    worst_closure = max(r.pop('_closure') for r in rows); worst_trace = max(r.pop('_trace') for r in rows)
    for row in rows:
        print(f"{row['split'][:3]} {row['architecture']:16s} {row['name'][:36]:36s} " + ' '.join(f"{k} {row['err_pct'][k]:+.1f}%" for k in row['err_pct']), flush=True)
    summary = {}
    groups = {'conv': lambda r: r['architecture'] == 'conv', 'hev': lambda r: r['architecture'].startswith('hev'),
              'hev_powersplit': lambda r: r['architecture'] == 'hev_powersplit', 'hev_parallel_p2': lambda r: r['architecture'] == 'hev_parallel_p2',
              'hev_series': lambda r: r['architecture'] == 'hev_series', 'bev': lambda r: r['architecture'] == 'bev'}
    for split in (('heldout', 'dev', 'all') if a_.split == 'all' else (a_.split,)):
        summary[split] = {}
        for g, f in groups.items():
            rr = [r for r in rows if f(r) and (split == 'all' or r['split'] == split)]
            d = {}
            for cyc in ('FTP', 'HWY', 'US06', 'UDDS', 'SC03', 'COLD_FTP_20F'):
                st = stats([r['err_pct'][cyc] for r in rr if cyc in r['err_pct']])
                if st: d[cyc] = st
            allc = [r['err_pct'][c] for r in rr for c in ('FTP', 'HWY', 'US06', 'UDDS') if c in r['err_pct']]
            if allc: d['all_main_cycles'] = stats(allc)
            summary[split][g] = d
    bag = [np.log(r['bag1_over_bag3']['sim'] / r['bag1_over_bag3']['epa']) * 100 for r in rows if 'bag1_over_bag3' in r and r['split'] == 'heldout']
    out = dict(check='EPA cycles vs EPA test-car list (MY2022-2024)', n_vehicles=len(rows),
               n_by_arch={a: sum(r['architecture'] == a for r in rows) for a in sorted(set(r['architecture'] for r in rows))},
               method=__doc__.strip(), procedures=open(os.path.join(HERE, 'epa_procedures.py')).read().split('"""')[1].strip(),
               calibration=json.load(open(os.path.join(HERE, '..', 'roadsim', 'roadsim', 'data', 'calibration.json'))),
               summary=summary, heldout_bag1_over_bag3_logerr_pct=stats(bag),
               ledger_worst_closure_rel=worst_closure, trace_worst_abs_mps=worst_trace, vehicles=rows,
               wall_s=round(time.time() - t0, 1), roadsim_version=roadsim.__version__, physics=roadsim.PHYSICS_VERSION,
               options=_opts.get(), options_name=_opts.name(), split_run=a_.split)
    fn = _opts.out('cycles_vs_epa.json')
    if a_.split != 'all':
        fn = fn.replace('.json', f'__{a_.split}.json')
    json.dump(out, open(os.path.join(HERE, fn), 'w'), indent=1)
    print(json.dumps(summary[a_.split if a_.split != 'all' else 'heldout'], indent=1))
    print('worst closure', worst_closure, 'worst trace', worst_trace, 'wall', out['wall_s'])


if __name__ == '__main__':
    main()
