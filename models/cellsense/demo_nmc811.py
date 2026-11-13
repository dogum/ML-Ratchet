"""CellSense NMC811 on a real exam cell: LG INR21700-M50T cell W9 of Pozzato et al. 2022 (charged at 1C, aged with UDDS
driving discharges), its C/20 capacity tests read in full and as partial discharges stopped after 2.5 Ah."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cellsense
cs = cellsense.CellSense(chemistry='nmc811'); d = np.load(os.path.join(HERE, 'examples', 'm50t_W9.npz'))
Q, ref = d['Q'], d['Q'][0]
print(f"cell {d['cell']} (LG INR21700-M50T, NMC811 / graphite-SiOx), C/20 capacity tests; reference = test {int(d['diag'][0])}\n")
print(' test | capacity to 2.5 V (Ah)                          | lithium / negative / NMC lost')
print('      | full curve      stopped at 2.5 Ah     measured |  (full curve, with reference)')
for i in range(1, len(Q)):
    r = cs.diagnose(Q[i], ref); q = Q[i].copy(); q[q > 2.5] = np.nan; p = cs.diagnose(q, ref)
    print(f" {int(d['diag'][i]):4d} | {r['q_total']:.3f} +- {1000 * r['q_total_sig']:2.0f} mAh  {p['q_total']:.3f} +- {1000 * p['q_total_sig']:3.0f} mAh    {d['q_total'][i]:.3f}  |"
          f"  {100 * r['lli']:4.1f}% / {100 * r['lam_ne']:4.1f}% / {100 * r['lam_pe']:4.1f}%")
print("A full curve's capacity is read off its own end (coulomb counting); the network reads curves that stop early.")
print('\nThe same tests stopped at 2.5 Ah and read without the reference:')
for i in range(1, len(Q)):
    q = Q[i].copy(); q[q > 2.5] = np.nan; p = cs.diagnose(q)
    print(f" {int(d['diag'][i]):4d} | {p['q_total']:.3f} +- {1000 * p['q_total_sig']:3.0f} mAh   measured {d['q_total'][i]:.3f}")
