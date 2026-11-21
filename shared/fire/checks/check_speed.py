"""Speed of the spread solver on one core.

Contract target: a 30 x 30 km landscape at 60-90 m, 24 h of spread, in <= ~1-3 s on one core, plus a batch mode.
Cases (generated mixed-conifer landscapes, dry and windy generated weather so the fire grows large):
  * 500 x 500 cells at 60 m and 334 x 334 at 90 m; methods mtt (16-cell stencil, default), level_set (32), ca (8);
  * crown on, spotting off/on;
  * timings split into prepare (moisture model + fuel bundles + WAF), the arrival-time kernel and the post-processing
    (rasters, hourly perimeters, ledger); the median of 3 repeats after a warm-up call (numba JIT is cached on disk);
  * batch: 40 ignitions on one landscape (simulate_batch: prepare once, kernel per ignition).
"""
import time

import numpy as np

from _common import write
import firesim as fs


THREADS = dict(omp=__import__('os').environ.get('OMP_NUM_THREADS'), numba=__import__('os').environ.get('NUMBA_NUM_THREADS'))


def weather(seed):
    return fs.generate_weather('2021-08-01', hours=24 * 6, climate='west_dry_summer', rng=seed, wind=7.0, rh_min=12.0,
                               p_wet=0.0)


def timed(fn, n=3):
    ts = []
    out = None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


res = dict(threads=THREADS, cases=[])
t = time.perf_counter()
import numba  # noqa: E402
res['numba'] = numba.__version__

for shape, rm in (((500, 500), 60.0), ((334, 334), 90.0)):
    ls = fs.generate_landscape(shape, rm, rng=11, region='mixed_conifer', relief_m=700)
    w = weather(5)
    start, end = '2021-08-03 16:00', '2021-08-04 16:00'
    ign = {'rowcol': [(shape[0] // 2, shape[1] // 2)]}
    # warm-up (JIT from disk cache)
    t0 = time.perf_counter()
    fs.simulate(ls, w, ign, start, end, rng=1)
    warm = time.perf_counter() - t0
    for method in ('mtt', 'level_set', 'ca'):
        for spot in (False, True):
            if method != 'mtt' and spot:
                continue
            tp, pr = timed(lambda: fs.prepare(ls, w, start, end, None, 'nelson'))
            tk, o = timed(lambda: fs.simulate(ls, w, ign, start, end, method=method, spotting=spot, rng=2, prepared=pr,
                                              return_perimeters=False))
            tf, o2 = timed(lambda: fs.simulate(ls, w, ign, start, end, method=method, spotting=spot, rng=2))
            row = dict(shape=list(shape), res_m=rm, km=shape[0] * rm / 1000, hours=24, method=method, spotting=spot,
                       burned_ha=float(o2['area'].iloc[-1]), active_crown_ha=o2['ledger']['area_ha_active_crown'],
                       spot_ignitions=o2['labels']['n_spot_ignitions'], prepare_s=tp,
                       kernel_plus_rasters_s=tk, full_call_s=tf, first_call_s=warm,
                       growth_check_ratio=o2['ledger'].get('growth_check', {}).get('ratio_total'))
            print(row)
            res['cases'].append(row)
    # batch mode
    rng = np.random.default_rng(3)
    burnable = np.argwhere(((ls['fuel'] < 91) | (ls['fuel'] > 99)))
    pick = burnable[rng.choice(len(burnable), 40, replace=False)]
    igns = [{'rowcol': [tuple(int(v) for v in p)]} for p in pick]
    t0 = time.perf_counter()
    out = fs.simulate_batch(ls, w, igns, start, end, rng=4, keep=('arrival_time',))
    tb = time.perf_counter() - t0
    areas = [float(np.isfinite(o['arrival_time']).sum() * rm * rm / 1e4) for o in out]
    row = dict(shape=list(shape), res_m=rm, batch_n=len(igns), batch_total_s=tb, per_fire_s=tb / len(igns),
               per_fire_kernel_s_median=float(np.median([o['wall_s'] for o in out])),
               burned_ha_median=float(np.median(areas)), burned_ha_max=float(np.max(areas)))
    print(row)
    res['cases'].append(row)

# worst case: grass/shrub landscape, strong wind, most of the 30 x 30 km burns in 24 h (cost ~ burned cells)
for shape, rm in (((500, 500), 60.0), ((334, 334), 90.0)):
    ls = fs.generate_landscape(shape, rm, rng=12, region='great_basin', relief_m=300)
    w = fs.generate_weather('2021-08-01', hours=24 * 6, climate='west_dry_summer', rng=6, wind=11.0, rh_min=8.0,
                            p_wet=0.0, wind_dir=270.0)
    start, end = '2021-08-03 16:00', '2021-08-04 16:00'
    ign = {'rowcol': [(shape[0] // 2 + k, 5) for k in range(-shape[0] // 3, shape[0] // 3, 3)]}
    for spot in (False, True):
        tf, o2 = timed(lambda: fs.simulate(ls, w, ign, start, end, spotting=spot, rng=2))
        nb = int(np.isfinite(o2['arrival_time']).sum())
        row = dict(case='worst', shape=list(shape), res_m=rm, hours=24, method='mtt', spotting=spot,
                   burned_ha=float(o2['area'].iloc[-1]), burned_frac=nb / (shape[0] * shape[1]),
                   spot_ignitions=o2['labels']['n_spot_ignitions'], full_call_s=tf,
                   cells_per_s=nb / tf, growth_check_ratio=o2['ledger'].get('growth_check', {}).get('ratio_total'))
        print(row)
        res['cases'].append(row)

main = [c for c in res['cases'] if c.get('method') == 'mtt' and not c.get('spotting') and 'case' not in c]
worst = [c for c in res['cases'] if c.get('case') == 'worst']
res['summary'] = dict(
    target='30 x 30 km at 60-90 m, 24 h, <= ~1-3 s on one core',
    full_call_s={f"{c['res_m']:.0f}m": c['full_call_s'] for c in main},
    kernel_s={f"{c['res_m']:.0f}m": c['kernel_plus_rasters_s'] for c in main},
    prepare_s={f"{c['res_m']:.0f}m": c['prepare_s'] for c in main},
    batch_per_fire_s={f"{c['res_m']:.0f}m": c['per_fire_s'] for c in res['cases'] if 'batch_n' in c},
    worst_full_call_s={f"{c['res_m']:.0f}m spot={c['spotting']}": c['full_call_s'] for c in worst},
    worst_burned_frac={f"{c['res_m']:.0f}m spot={c['spotting']}": c['burned_frac'] for c in worst},
    meets_target=bool(all(c['full_call_s'] <= 3.0 for c in main + worst)),
    total_check_s=time.perf_counter() - t)
print(res['summary'])
write('speed.json', res)
