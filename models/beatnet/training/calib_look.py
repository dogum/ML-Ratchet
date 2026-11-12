"""Dev calibration look: median N-beat window and median |slope| profile, real dev records vs simulated bank (figs/calib.png)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, _P.SRC); import bnet as BN, train as TR, common as C
bank = sys.argv[1] if len(sys.argv) > 1 else BN.VAL
Z = BN.real_bank('mitdb'); _, idx = BN.split_idx('dev'); S = TR.load(bank); t = (np.arange(450) - 225) / 180 * 1000
fig, ax = plt.subplots(2, 4, figsize=(22, 8))
for c in range(3):
    for k, (nm, W, col) in enumerate((('real dev', Z['w'][idx][Z['y'][idx] == c], 'k'), ('sim', S['w'][S['y'] == c], 'C0'))):
        W = W[:4000].astype(np.float32); ax[0, c].plot(t, np.median(W, 0), col, label=nm); ax[1, c].plot(t[1:], np.median(np.abs(np.diff(W, axis=1)), 0), col, label=nm)
        if c == 0:
            for r in C.dev_records():
                m = Z['rec'][idx] == r
                if (m & (Z['y'][idx] == 0)).sum() > 50: ax[0, 3].plot(t, np.median(Z['w'][idx][m & (Z['y'][idx] == 0)].astype(np.float32), 0), lw=0.8, label=str(r))
    ax[0, c].set_title(f'{C.CLASSES[c]} median window'); ax[1, c].set_title(f'{C.CLASSES[c]} median |slope|'); ax[0, c].set_xlim(-400, 700); ax[1, c].set_xlim(-200, 300); ax[0, c].legend()
ax[0, 3].legend(); ax[0, 3].set_xlim(-400, 700)
pp = lambda W: np.median(W[:, 225 - 15:225 + 15].max(1)); print('R peak (norm.) real', pp(Z['w'][idx][Z['y'][idx] == 0].astype(np.float32)), 'sim', pp(S['w'][S['y'] == 0][:4000].astype(np.float32)))
plt.tight_layout(); plt.savefig(_P.WORK + '/figs/calib.png', dpi=50)
