"""first 2 minutes after P on shallow offshore dev quakes: real, ak135 + slow surface waves + coda (v3), and v3 + ocean/site layers (v4)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, pipeline as PL, scatter as SC, crust as CR, fig_compare as FC
from scipy.signal import butter, sosfiltfilt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = _P.WORK + '/data/'; J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz'); SH = butter(2, [0.03, 0.3], 'band', fs=1.0, output='sos')
ids = [o['i'] for o in json.load(open(D + 'diag_shallow.json')) if o['depth'] <= 40][:6]
fig, ax = plt.subplots(len(ids), 1, figsize=(10, 1.9 * len(ids)))
for a, i in zip(ax, ids):
    m = J['meta'][i]; M, _ = CP.usgs_mt(m['id']); X = CP.synth_for(m, M); rng = np.random.default_rng(i)
    v3 = FC.corrected(m, X); v4, inf = CR.apply(rng, v3, np.radians(m['baz']), m['depth']); R = z['x'][i].astype(float) * 10 ** z['ls'][i]
    t = np.arange(-20, 120); P = PL.PRE
    for k, (y, c, off) in enumerate(((R[0], 'k', 2.2), (v3[0], 'C0', 0), (v4[0], 'C3', -2.2))):
        f = sosfiltfilt(SH, y); seg = f[P - 20:P + 120]; a.plot(t, seg / np.abs(f[P - 5:P + 15]).max() * 0.8 + off, c, lw=.8)
    a.set_title(f"{m['place']} M{m['mw']} {m['depth']} km {m['dist']:.0f}deg   water {inf.get('water_km', 0):.1f} km", fontsize=8); a.set_yticks([])
plt.tight_layout(); plt.savefig(_P.WORK + '/scratch/pwin.png', dpi=75)
