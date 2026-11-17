"""ThermoPilot demo: replay three exam episodes (twins of real exam units under their real 2017 weather and real
tariffs) through the numpy controller, check it chooses the same setpoints it chose in the closed-loop BuildingSim
run, and print the month's bill and comfort for every controller.  Runs in a few seconds; needs numpy only.

    python3 demo.py
"""
import json, os, time
import numpy as np
import thermopilot_core as tp

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    W = tp.load(os.path.join(HERE, 'thermopilot_v1.npz'))
    z = np.load(os.path.join(HERE, 'examples', 'bdg2', 'demo_episodes.npz'))   # BDG2-derived weather: CC BY-SA 4.0
    meta = json.loads(str(z['meta']))
    for i, m in enumerate(meta):
        p = f'ep{i}_'
        ctx = {k[len(p) + 4:]: z[k] for k in z.files if k.startswith(p + 'ctx_')}
        ctx['pref_lo'], ctx['pref_hi'] = m['pref_lo'], m['pref_hi']
        card = dict(m['card'], **{k[len(p) + 5:]: z[k] for k in z.files if k.startswith(p + 'card_')})
        pilot = tp.ThermoPilot(W, card, ctx, m['lam'], z[p + 'base_W'])
        T, q, e, fT, fG = (z[p + 'rec_' + k] for k in ('T', 'q', 'e', 'fT', 'fG'))
        same = 0
        for k in range(len(T)):
            sp = pilot.step(k, T[k], q[k], e[k], fT[k], fG[k])
            same += abs(sp['heat_sp'] - z[p + 'rec_h'][k]) < 1e-6 and abs(sp['cool_sp'] - z[p + 'rec_c'][k]) < 1e-6
        print(f"\n{m['id']}  ({m['system']}, weather {m['wsrc']}, tariff {m['tariff']}, month {m['month']})")
        print(f"  numpy replay chose the same setpoints in {same}/{len(T)} hours")
        rows = dict(m['baselines']); rows['thermopilot'] = m['thermopilot']
        for k in ('fixed', 'setback', 'precool', 'mpc_c', 'thermopilot', 'oracle'):
            if k in rows:
                r = rows[k]
                print(f"  {k:12s} bill ${r['bill']:8.2f}   outside comfort band {r['DH']:6.1f} K h   J ${r['J']:8.2f}")
    print(f'\n{time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
