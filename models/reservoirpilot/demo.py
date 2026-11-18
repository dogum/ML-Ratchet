"""ReservoirPilot demo: operate six real exam dams (a fixed random draw within main-use strata, not picked for looks)
and compare with SOP, the sim-tuned classical rule and the record, using WaterData's cost J. Runs in a few seconds;
numpy + pandas only.

    python demo.py           # prints the table; nothing is written
    python demo.py --plot    # also writes work/demo_reservoirpilot.png (needs matplotlib)

J charges nothing for storage, so the storage columns (mean storage and end - start, as fractions of capacity) are
printed beside it: the pilot buys its low J partly by holding the reservoir lower than SOP does.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reservoirpilot as rp  # noqa: E402


def load_dams():
    dams = json.load(open(os.path.join(HERE, 'examples', 'exam_dams.json'), encoding='utf-8'))
    out = []
    for d in dams:
        dates = pd.date_range(d['start'], periods=len(d['I']), freq='D')
        task = {'I': np.array(d['I']), 'dates': dates, 'Dcurve': np.array(d['Dcurve']), 'Icurve': np.array(d['Icurve']),
                'Qmax': d['Qmax'], 'cap': d['cap'], 'S0': d['S0']}
        task['D'] = task['Dcurve'][dates.day_of_year - 1]
        out.append((d, task))
    return out


def main(plot=False):
    pol = rp.Pilot.load()
    print(f"{'dam':11s} {'use':4s} {'J pilot':>8s} {'J rule':>8s} {'J SOP':>8s} {'J record':>8s} {'J DP*':>8s}"
          f"   mean storage pilot/SOP   end-start pilot/SOP")
    plots = []
    for d, task in load_dams():
        R, S = pol.run(task); Rs, Ss = rp.sop(task); Rt, St = rp.tuned_rule(task)
        cp, cs, ct = rp.cost_components(R, S, task), rp.cost_components(Rs, Ss, task), rp.cost_components(Rt, St, task)
        ch = rp.cost_components(np.array(d['R_obs']), np.array(d['S_obs']), task)
        name = d['cdec_id'] or d['res_id']
        # demo inflows are stored to 6 significant digits, so allow a tiny difference from the exam run
        assert abs(cp['J'] - d['exam_J']['pilot']) <= 1e-4 * max(cp['J'], 1e-3), 'release module disagrees with the exam run'
        print(f"{name:11s} {d['main_use']:4s} {cp['J']:8.4f} {ct['J']:8.4f} {cs['J']:8.4f} {ch['J']:8.4f} {d['exam_J']['dp']:8.4f}"
              f"   {cp['storage_mean_frac']:.2f} / {cs['storage_mean_frac']:.2f}"
              f"              {cp['storage_end_minus_start_frac']:+.2f} / {cs['storage_end_minus_start_frac']:+.2f}")
        plots.append((name, task, S, Ss, St, d))
    print('rule = the classical flood-pool + hedging rule tuned on the simulator (rung c2, the exam\'s fairest baseline).')
    print('* DP = perfect-foresight dynamic programming (knows all future inflow), from WaterData\'s exam baseline.')
    print('Storage is a fraction of capacity; J does not charge for it.')
    print('Exam medians (244 ResOpsUS dams): pilot 0.0154, tuned rule 0.0285, SOP 0.108, rule curve 0.143, '
          'historical 0.183, DP 0.0063.')
    print('Mean storage: pilot 0.38, tuned rule 0.40, SOP 0.59, DP 0.68 of capacity.')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        work = os.environ.get('ML_RATCHET_WORK')
        out = os.path.join(work, 'reservoirpilot') if work else os.path.join(HERE, 'work')
        os.makedirs(out, exist_ok=True)
        fig, ax = plt.subplots(len(plots), 1, figsize=(10, 2.2 * len(plots)), sharex=False)
        for a, (name, task, S, Ss, St, d) in zip(ax, plots):
            dates = task['dates']
            a.plot(dates, np.array(d['S_obs']) / task['cap'], color='#a0703c', lw=0.8, label='record')
            a.plot(dates, Ss / task['cap'], color='#7a858b', lw=1, label='SOP')
            a.plot(dates, St / task['cap'], color='#e0a84c', lw=1, label='tuned rule')
            a.plot(dates, S / task['cap'], color='#a61e4d', lw=1.4, label='ReservoirPilot')
            a.set_ylabel('storage / cap'); a.set_title(name, fontsize=9, loc='left'); a.set_ylim(0, 1.02)
        ax[0].legend(fontsize=8, ncol=4); fig.tight_layout()
        fig.savefig(os.path.join(out, 'demo_reservoirpilot.png'), dpi=110)
        print('wrote', os.path.join(out, 'demo_reservoirpilot.png'))


if __name__ == '__main__':
    main(plot='--plot' in sys.argv[1:])
