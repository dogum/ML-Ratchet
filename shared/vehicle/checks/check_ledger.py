"""Check: energy-ledger closure and robustness under stress.

600 runs: 300 generated vehicles (all classes/architectures, their own strategies) x 2 random conditions each:
cycle from the library, ambient -25..+40 C, wind 0-12 m/s from a random direction with the heading changing along the
trip, synthetic grade profile up to +/-8 %, random surface (dry/concrete/wet/gravel/snow), random start SOC and
cold or warm start. Reports the relative closure |sources - sinks| / scale (target < 0.5 %), the worst per-second
power residual, NaN/inf occurrence, and trace-following statistics. Every other check also records its closures.
"""
import os, sys, json, time, warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
from roadsim import simulate, make_trip, cycle
from roadsim.generator import generate_vehicles

LD_CYC = ['udds', 'hwfet', 'us06', 'la92', 'nycc', 'sc03']
HD_CYC = ['hd_transient', 'cbd14', 'gem_55', 'gem_65', 'udds']


def main():
    rng = np.random.default_rng(7)
    V = generate_vehicles(300, seed=17)
    rows = []
    t0 = time.time()
    for v in V:
        hd = v['class'] in ('bus', 'truck_md', 'truck_hd')
        for rep in range(2):
            c = (HD_CYC if hd else LD_CYC)[int(rng.integers(5 if hd else 6))]
            vv = cycle(c); N = len(vv); t = np.arange(N)
            amp = rng.uniform(0, 0.08); per = rng.uniform(60, 600)
            trip = make_trip(vv, T_amb=float(rng.uniform(-25, 40)), grade=amp * np.sin(2 * np.pi * t / per + rng.uniform(0, 6)),
                             wind_speed=float(rng.uniform(0, 12)), wind_dir=float(rng.uniform(0, 360)),
                             heading=(rng.uniform(0, 360) + np.cumsum(rng.normal(0, 2, N))) % 360,
                             surface=int(rng.integers(5)), solar=float(rng.uniform(0, 900)), elev=float(rng.uniform(0, 2000)))
            if c in ('gem_55', 'gem_65'):
                trip['grade'] = trip['grade'] + cycle(c + '_grade')
            warm = rng.random() < 0.5
            init = dict(engine=88.0, driveline=80.0, tire_warm=1.0) if warm else None
            soc = float(rng.uniform(0.3, 0.95)) if v['architecture'] != 'conv' else None
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                o = simulate(v, trip, init_soc=soc, init_temps=init, record=True)
            L = o['ledger']; S = o['summary']
            finite = all(np.isfinite(np.asarray(o[k], float)).all() for k in ('v', 'fuel_rate', 'soc', 'batt_current', 'eng_power'))
            rows.append(dict(id=v['id'], cls=v['class'], arch=v['architecture'], cycle=c, closure=L['closure_rel'],
                             worst_step_W=L['worst_step_W'], finite=bool(finite), trace_rms=S['trace_rms_mps'],
                             trace_frac_over_2mph=S['trace_frac_over_2mph']))
    cl = np.array([r['closure'] for r in rows]); ws = np.array([r['worst_step_W'] for r in rows])
    tr = np.array([r['trace_frac_over_2mph'] for r in rows])
    by_arch = {}
    for a in sorted(set(r['arch'] for r in rows)):
        x = np.array([r['closure'] for r in rows if r['arch'] == a]); t = np.array([r['trace_frac_over_2mph'] for r in rows if r['arch'] == a])
        by_arch[a] = dict(n=len(x), closure_max=float(x.max()), trace_frac_over_2mph_median=round(float(np.median(t)), 4), trace_frac_over_2mph_p95=round(float(np.percentile(t, 95)), 4))
    out = dict(check='energy ledger closure (stress test)', method=__doc__.strip(), n_runs=len(rows),
               closure_rel_max=float(cl.max()), closure_rel_median=float(np.median(cl)), n_over_0p5pct=int((cl > 0.005).sum()),
               worst_step_residual_W_max=float(ws.max()), all_finite=bool(all(r['finite'] for r in rows)),
               trace_frac_seconds_over_2mph=dict(median=round(float(np.median(tr)), 4), p95=round(float(np.percentile(tr, 95)), 4), max=round(float(tr.max()), 4)),
               by_architecture=by_arch, wall_s=round(time.time() - t0, 1), runs=rows)
    json.dump(out, open(os.path.join(HERE, 'ledger.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('runs', 'method')}, indent=1))


if __name__ == '__main__':
    main()
