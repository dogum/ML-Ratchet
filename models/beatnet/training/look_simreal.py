"""Compare simulated and real (dev) beat windows: random draws per class (figs/simreal.png)."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, _P.SRC); import common as C
bank = sys.argv[1] if len(sys.argv) > 1 else _P.WORK + '/bank/try_000.npz'
z = np.load(_P.WORK + '/data/mitdb_beats.npz'); dev = np.isin(z['rec'], C.dev_records()); s = np.load(bank); rng = np.random.default_rng(0)
fig, ax = plt.subplots(3, 6, figsize=(22, 9)); t = (np.arange(450) - 225) / 180
for c in range(3):
    ir = np.nonzero(dev & (z['y'] == c))[0]; isim = np.nonzero(s['y'] == c)[0]
    for j in range(3):
        k = rng.choice(ir); ax[c, j].plot(t, z['w'][k], 'k', lw=0.8); ax[c, j].set_title(f"real {C.CLASSES[c]} rec {z['rec'][k]}"); ax[c, j].set_ylim(-1.5, 1.5)
        k = rng.choice(isim); ax[c, 3 + j].plot(t, s['w'][k], 'C0', lw=0.8); ax[c, 3 + j].set_title(f"sim {C.CLASSES[c]}"); ax[c, 3 + j].set_ylim(-1.5, 1.5)
plt.tight_layout(); plt.savefig(_P.WORK + '/figs/simreal.png', dpi=50)
for c in range(4):
    a = z['rr'][dev & (z['y'] == c)]; b = s['rr'][s['y'] == c]
    if len(a) and len(b): print(C.CLASSES[c], 'real rr feat median', np.round(np.median(a, 0), 2), ' sim', np.round(np.median(b, 0), 2))
