"""CellSense on a real exam cell: Severson et al. 2019 cell b3c33, read at seven points through its life."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cellsense
cs = cellsense.CellSense(); d = np.load(os.path.join(HERE, 'examples', 'b3c33.npz'))
print(f"cell b3c33, charged {d['policy']} (0-80% in {float(d['charge_time']):.1f} min), end of life at cycle {int(d['cycle_life'])}\n")
print(' cycle | total capacity (Ah)          | pulse R (mOhm)   | lithium / graphite / LFP lost | life forecast')
print('       | CellSense        measured    | est.   measured  |                               | (cycles)')
for i in range(1, len(d['cycle'])):
    r = cs.diagnose(d['Q'][i], d['T'][i], d['Q'][0], d['T'][0], cycle=int(d['cycle'][i]), charge_time_min=float(d['charge_time']))
    print(f" {int(d['cycle'][i]):5d} | {r['q_total']:.3f} +- {1000 * r['q_total_sig']:3.0f} mAh  {d['measured_q_total'][i]:.3f}       | {r['ir']:5.1f}  {d['measured_ir_mohm'][i]:5.1f}     | "
          f"{100 * r['lli']:4.1f}% / {100 * r['lam_ne']:4.1f}% / {100 * r['lam_pe']:4.1f}%     | {r['life']:5.0f} ({r['life_lo']:.0f}-{r['life_hi']:.0f})")

print('\nThe same readings as partial discharges, stopped after 0.75 Ah (everything below that point thrown away):')
print(' cycle | total capacity (Ah), partial-discharge network | measured')
for i in range(1, len(d['cycle'])):
    q, t = d['Q'][i].copy(), d['T'][i].copy(); cut = q > 0.75; q[cut] = np.nan; t[cut] = np.nan
    r = cs.diagnose(q, t, d['Q'][0], d['T'][0])
    print(f" {int(d['cycle'][i]):5d} | {r['q_total']:.3f} +- {1000 * r['q_total_sig']:3.0f} mAh{'' if r['partial'] else ' (reached 2.0 V: full network)':>25s}           | {d['measured_q_total'][i]:.3f}")
