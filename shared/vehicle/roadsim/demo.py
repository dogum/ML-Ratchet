"""RoadSim demo: one trip, every architecture, with the energy ledger.

The trip: a 32-minute commute built from EPA cycles (UDDS city part, then HWFET highway), at 5 C with a 6 m/s wind
from the north-west, a 3 % hill on the highway, wet road for the first 10 minutes. The same mid-size car body
(1,650 kg, CdA 0.62 m2) is fitted with each powertrain. Prints the ledger (kJ) per architecture and writes
demo_ledger.png (stacked bars of where the energy went).

    python3 demo.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roadsim import simulate, load_vehicle, make_trip, cycle, list_strategies

ARCHS = ['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series', 'bev']


def commute():
    v = np.concatenate([cycle('udds'), cycle('hwfet')])
    N = len(v); t = np.arange(N)
    grade = np.where((t > 1600) & (t < 1700), 0.03, np.where((t > 1700) & (t < 1800), -0.03, 0.0))
    heading = np.where(t < 1370, (t // 120) * 90 % 360, 45.0)          # city grid, then highway to the north-east
    return make_trip(v, T_amb=5.0, grade=grade, wind_speed=6.0, wind_dir=315.0, heading=heading,
                     surface=np.where(t < 600, 2, 0), elev=250.0)


def main():
    trip = commute()
    print(f"trip: {len(trip['v'])} s, {np.sum(trip['v']) / 1609.344:.1f} mi, 5 C, wind 6 m/s from NW, 3 % hill\n")
    keys = ['fuel', 'batt_stored_change', 'eng_loss', 'mot_loss', 'trans_loss', 'batt_loss', 'aux', 'aero', 'rolling',
            'grade', 'brake', 'kinetic_change', 'residual']
    rows = {}
    for a in ARCHS:
        spec = load_vehicle({'class': 'car', 'architecture': a, 'id': 'demo_' + a})
        t = time.perf_counter()
        o = simulate(spec, trip, init_temps={'engine': 5.0, 'cabin': 5.0, 'battery': 5.0})
        wall = time.perf_counter() - t
        L = o['ledger']; S = o['summary']
        rows[a] = (L, S, o['strategy'], wall)
    print(f"{'kJ':<20s}" + ''.join(f'{a:>16s}' for a in ARCHS))
    for k in keys:
        print(f'{k:<20s}' + ''.join(f'{rows[a][0][k]:16.0f}' for a in ARCHS))
    print(f"{'closure (rel)':<20s}" + ''.join(f'{rows[a][0]["closure_rel"]:16.1e}' for a in ARCHS))
    print(f"{'strategy':<20s}" + ''.join(f'{rows[a][2][:15]:>16s}' for a in ARCHS))
    print(f"{'fuel L/100km':<20s}" + ''.join(f'{rows[a][1].get("L_per_100km", 0):16.2f}' for a in ARCHS))
    print(f"{'battery kWh/100km':<20s}" + ''.join(f'{rows[a][1]["kWh_per_100km_dc"]:16.2f}' for a in ARCHS))
    print(f"{'x real time':<20s}" + ''.join(f'{rows[a][1]["x_realtime"]:16.0f}' for a in ARCHS))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        sinks = [('eng_loss', 'engine losses', '#c0504d'), ('mot_loss', 'motor+inverter', '#f79646'), ('trans_loss', 'transmission', '#9bbb59'),
                 ('batt_loss', 'battery I2R', '#8064a2'), ('aux', 'aux + HVAC', '#4bacc6'), ('aero', 'aero', '#1f497d'),
                 ('rolling', 'rolling', '#7f7f7f'), ('grade', 'net grade', '#948a54'), ('brake', 'friction brakes', '#000000')]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(ARCHS)); bottom = np.zeros(len(ARCHS))
        for k, lab, col in sinks:
            val = np.array([max(rows[a][0][k], 0) / 1e3 for a in ARCHS])
            ax.bar(x, val, bottom=bottom, label=lab, color=col); bottom += val
        ax.set_xticks(x); ax.set_xticklabels([a.replace('_', '\n') for a in ARCHS])
        ax.set_ylabel('energy over the trip (MJ)'); ax.set_title('Where the energy goes: same car body, same trip, seven powertrains')
        ax.legend(fontsize=8, ncol=3); fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_ledger.png')
        fig.savefig(out, dpi=110); print('\nwrote', out)
    except Exception as ex:
        print('no figure:', ex)


if __name__ == '__main__':
    main()
