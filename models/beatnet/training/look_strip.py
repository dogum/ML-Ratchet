"""Quick look: a few simulated strips with labels (figs/strips.png)."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, time, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sim, strip as S
fig, ax = plt.subplots(6, 1, figsize=(20, 16)); rng = np.random.default_rng(5)
for i in range(6):
    t0 = time.time(); p = sim.Patient(100 + i); P = S.program(rng); V, A = S.events(p, P, rng, 30); sig, on, wd = S.synth(p, P, V, A, rng); x, ann = S.record(sig, on, wd, rng); dt = time.time() - t0
    t = np.arange(x.shape[1]) / 360; ax[i].plot(t, x[0], lw=0.6); lab = [e['label'] for e in V]
    for a, l in zip(ann, lab): ax[i].text(a / 360, x[0].max(), 'NSVF'[l], color='krbg'[l], ha='center')
    ax[i].set_title(f"{P['base']} {P['mode']} {P['pattern']} bbb={p.bbb} hr={P['hr']:.0f}  ({dt:.1f}s)"); ax[i].set_xlim(0, 20)
    print(i, P['base'], P['mode'], np.bincount(lab, minlength=4), round(dt, 2), 's')
plt.tight_layout(); plt.savefig(_P.WORK + '/figs/strips.png', dpi=55)
