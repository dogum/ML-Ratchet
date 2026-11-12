"""Screen the noise pool: drop dwarfs whose own variability (rotation spots, eclipses, pulsations, strong
systematics) would stand out in 3-283 uHz. Statistics per star: excess = median log power 3-30 uHz minus the
median at 150-280 uHz (white-noise floor); peak = largest narrow peak above a running median. Kept if excess < 1.0 dex
and peak < 1.0 dex. Writes data/noise_screen.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
from scipy.ndimage import median_filter
sys.path.insert(0, _P.SRC); import sq
D = _P.WORK + '/data/'
sp = json.load(open(D + 'split.json')); R = {}
for which in ('noise_train', 'noise_val'):
    for k in sp[which]:
        F1 = sq.prepare(sq.read_npz(f'{D}lc/{k}.npz'))[0]; f = sq.FC
        ex = float(np.median(F1[(f > 3) & (f < 30)]) - np.median(F1[(f > 150) & (f < 280)])); pk = float(np.max(F1 - median_filter(F1, 51)))
        R[str(k)] = dict(which=which, excess=ex, peak=pk, floor=float(np.median(F1[(f > 150) & (f < 280)])), keep=bool(ex < 1.0 and pk < 1.0))
json.dump(R, open(D + 'noise_screen.json', 'w'))
for w in ('noise_train', 'noise_val'):
    v = [r for r in R.values() if r['which'] == w]; print(w, len(v), 'kept', sum(r['keep'] for r in v))
ex = np.array([r['excess'] for r in R.values()]); pk = np.array([r['peak'] for r in R.values()]); print('excess pct', np.percentile(ex, [10, 50, 80, 90, 95]).round(2), 'peak pct', np.percentile(pk, [10, 50, 80, 90, 95]).round(2))
