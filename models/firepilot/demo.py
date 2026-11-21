"""FirePilot demo: plan containment lines on real exam fires (numpy + scipy only, a few seconds).

  python3 demo.py

Each case in examples/ is a real fire from the 48-fire exam replay subset at its first afternoon satellite
detection: the detected burned area, the planner's FORECAST run (arrival time and flame length from FireSim on the
real LANDFIRE landscape and real HRRR weather with a forecast wind error), the line-building cost from the fuel map,
and the crew clock. The planner picks a line plan for two crew budgets. The truth columns are the scored exam
outcomes (FireSim truth world), stored from the one exam run; they are printed for comparison, not recomputed here.
"""
import os
import json
import time
import numpy as np

import firepilot as fp

HERE = os.path.dirname(os.path.abspath(__file__))


def describe(p):
    mode = 'centre-out' if p['mode'] == 0 else 'ends-in (anchor)'
    return (f"{mode}, centre {p['phi_c']:+.0f} deg from the forecast head, half-width {p['omega']:.0f} deg, "
            f"offset head {p['d_h'] / 1000:.1f} km / rear {p['d_f'] / 1000:.1f} km")


def main():
    net = fp.ValueNet(os.path.join(HERE, 'firepilot_weights.npz'))
    cases = sorted(f[:-5] for f in os.listdir(os.path.join(HERE, 'examples')) if f.endswith('.json'))
    for name in cases:
        rec = json.load(open(os.path.join(HERE, 'examples', name + '.json')))
        z = np.load(os.path.join(HERE, 'examples', name + '.npz'))
        print(f"\n== {rec['fire_name']} ({rec['fire_id']}, {rec['period']}): detected {rec['a0_ha']:.0f} ha; "
              f"forecast without action {rec['forecast_none_ha']:.0f} ha after 72 h")
        for b, out in rec['budgets'].items():
            r = dict(rate=float(b), res=rec['res'], head=rec['head'])
            t0 = time.time()
            ctx = fp.Context.from_store(z, r)
            plan, info = fp.plan_value(ctx, net, np.random.default_rng(rec['k']))
            dt = time.time() - t0
            same = all(abs(plan[k] - out['firepilot_plan'][k]) < 1e-6 for k in plan)
            print(f"  budget {b} m/h of line ({float(b) * 12 / 1000:.1f} km per 12 h shift), planned in {dt:.1f} s:")
            print(f"    plan: {describe(plan)}")
            lo, hi = max(0.0, info['pred'] - 1.645 * info['sigma']), min(1.0, info['pred'] + 1.645 * info['sigma'])
            print(f"    predicted share of the no-action growth stopped: {info['pred']:.2f} (90 % range {lo:.2f}-{hi:.2f})"
                  f"   [{'same plan as the exam run' if same else 'differs from the exam plan'}]")
            print('    exam truth (sim), share of growth stopped:  ' + '  '.join(
                f"{k} {v:.2f}" for k, v in out['truth_saved'].items()))


if __name__ == '__main__':
    main()
