"""GridPulse demo: six real GB exam events (fixed random draw, seed 7, two per drop-size stratum) and one
illustrative ERCOT summary. Runs in a second or two.   python demo.py"""
import os, numpy as np
from gridpulse import GridPulse
gp = GridPulse(); Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'gb_exam_examples.npz'))
print(f"{'event':20s} {'N':>3s} {'MW lost (1 sigma)':>22s} {'inertia GVA s':>20s} {'NESO':>6s} {'label MW':>9s} {'nadir fcst':>11s} {'real':>6s}")
for i, eid in enumerate(Z['event_id']):
    for N in (5, 60):
        r = gp.read_windows(Z['w'][i][None], N=N, system='GB', demand_MW=Z['demand'][i])
        lab = '' if np.isnan(Z['loss_label'][i]) else f"{Z['loss_label'][i]:.0f}"
        print(f"{eid:20s} {N:3d} {r['loss_MW'][0]:6.0f} ({r['loss_MW_lo'][0]:5.0f}-{r['loss_MW_hi'][0]:5.0f}) "
              f"{r['E_GWs'][0]:5.0f} ({r['E_GWs_lo'][0]:4.0f}-{r['E_GWs_hi'][0]:4.0f}) {Z['E_true'][i]:6.0f} {lab:>9s} "
              f"{r['nadir_Hz'][0] * 1000:7.0f} mHz {Z['nadir'][i] * 1000:5.0f}")
print('\nERCOT summary (illustrative report-style numbers): 0.11 Hz drop, 230 GW s, 45 GW load, 30 % IBR')
print(gp.ercot_summary(0.11, 230e3, 45000, 0.30, mw_loss=600))
