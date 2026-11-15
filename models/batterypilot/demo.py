"""BatteryPilot demo: three real exam node-years (never used in training or for choosing anything).
Runs BatteryPilot (the shipped v1.1 network, batterypilot_weights.npz) and the baselines for a 1 MW / 2 MWh battery and
prints $ per kW-year net of wear. The v1.0 network is batterypilot_weights_v1_0.npz.

    python3 demo.py            (about 10-20 s; numpy + scipy only)
"""
import csv, os, time
import numpy as np
import batterypilot as bp

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    cols = {}
    with open(fn) as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    p = {k: np.array(cols[k], float) for k in ('rt', 'da', 'df', 'T')}
    for k in ('hour', 'day', 'dow', 'doy'):
        p[k] = np.array(cols[k], int)
    p['time_utc'] = cols['time_utc']
    return p


if __name__ == '__main__':
    net = bp.PolicyNet(os.path.join(HERE, 'batterypilot_weights.npz'))
    bat = bp.Battery(P=1.0, duration_h=2.0)
    print(f"{'example':34s} {'perfect':>8s} {'BatteryPilot':>13s} {'DA optimiser':>13s} {'spread':>8s}   $/kW-yr net of wear")
    for fn in sorted(os.listdir(os.path.join(HERE, 'examples'))):
        if not fn.endswith('.csv'):
            continue
        t0 = time.time()
        p = load(os.path.join(HERE, 'examples', fn))
        cl = bat.c_lin(p['T'])
        out = {}
        ch, dis, soc, V = bp.run_policy(net, p, bat, cl)
        out['bp'] = bp.settle(bat, p['rt'], ch, dis, 0.0, p['T'])
        ch, dis = bp.perfect_foresight(p['rt'], bat, cl); out['pf'] = bp.settle(bat, p['rt'], ch, dis, 0.0, p['T'])
        ch, dis = bp.rolling_da_lp(p['da'], p['hour'], p['day'], bat, cl); out['rolling'] = bp.settle(bat, p['rt'], ch, dis, 0.0, p['T'])
        ch, dis = bp.daily_spread(p['da'], p['day'], bat, cl); out['spread'] = bp.settle(bat, p['rt'], ch, dis, 0.0, p['T'])
        k = lambda s: out[s]['net'] / 1000 * 8760 / len(p['rt'])  # noqa
        print(f"{fn[:-4]:34s} {k('pf'):8.1f} {k('bp'):13.1f} {k('rolling'):13.1f} {k('spread'):8.1f}   "
              f"(BP {out['bp']['cycles']:.0f} cycles, {time.time() - t0:.1f}s)")
        # the hour with the highest price: what BatteryPilot offered
        i = int(np.argmax(p['rt']))
        print(f"   peak hour {p['time_utc'][i]} UTC: RT ${p['rt'][i]:.0f}; value of a stored MWh at 0/50/100% full = "
              + ', '.join(f'${v:.0f}' for v in V[i, [0, 2, 4]]) + f"; SOC before {soc[i]:.0%}, after {soc[i + 1]:.0%}")
