"""Quick look: lead II and V1 of each beat type for a few patients (figs/beats.png) and their numbers."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
import sim, heart as Hm
L = {k: np.array(Hm.DOWER[k]) for k in ('I', 'II', 'aVF', 'V1', 'V5')}
fig, ax = plt.subplots(4, 6, figsize=(20, 11))
for row, seed in enumerate([1, 2, 3, 4]):
    p = sim.Patient(seed, bbb='none'); pl = sim.Patient(seed, bbb='LBBB'); pr = sim.Patient(seed, bbb='RBBB')
    f1, f2 = p.random_vfocus(), p.random_vfocus()
    cases = [('normal', p, p.vmap('normal')), ('LBBB', pl, pl.vmap('normal')), ('RBBB', pr, pr.vmap('normal')),
             ('PVC a', p, p.vmap('pvc', f1, 1.5)), ('PVC b', p, p.vmap('pvc', f2, 1.5)), ('fusion', p, np.fmin(p.vmap('normal'), p.vmap('pvc', f1, 1.5) + 20))]
    for c, (nm, pp, tau) in enumerate(cases):
        H = pp.vdipole(tau, 0.85); a = pp.adipole(pp.amap(pp.sinus)); T = np.zeros((900, 3)); off = 300 if nm[:3] != 'PVC' else 150
        T[off:off + len(H)] += H[:900 - off]
        if nm[:3] != 'PVC': T[off - int(pp.PR):off - int(pp.PR) + len(a)] += a[:900 - off + int(pp.PR)]
        for k, col in (('II', 'C0'), ('V1', 'C3')): ax[row, c].plot(np.arange(900), T @ L[k], col, lw=1, label=k)
        qw = pp.qrs_width(tau); ax[row, c].set_title(f'{nm}  QRS {qw:.0f} ms'); ax[row, c].axhline(0, color='k', lw=0.3)
        qi = T[off:off + int(qw)] @ L['I']; qf = T[off:off + int(qw)] @ L['aVF']; axis = np.degrees(np.arctan2(qf.sum(), qi.sum()))
        print(seed, nm, 'QRS', round(qw), 'axis', round(axis), 'II peak', round((T @ L['II']).max(), 2), 'min', round((T @ L['II']).min(), 2))
ax[0, 0].legend(); plt.tight_layout(); plt.savefig(_P.WORK + '/figs/beats.png', dpi=60)
