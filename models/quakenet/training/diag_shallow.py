"""Why shallow quakes read too deep: on dev records with a USGS moment tensor, compare real and ak135 synthetic
(1) surface-wave to P amplitude ratio (vertical, 15-60 s band) and (2) P-window shape: energy 5-40 s after P relative to the first 5 s (0.05-0.3 Hz)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, pipeline as PL, simbank as SB
from scipy.signal import butter, sosfiltfilt
D = _P.WORK + '/data/'; J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz'); P0 = PL.PRE
SL = butter(2, [1 / 60, 1 / 15], 'band', fs=1.0, output='sos'); SH = butter(2, [0.05, 0.3], 'band', fs=1.0, output='sos')
def stats(Z, d, zz):
    lp = sosfiltfilt(SL, Z); sp = SB.s_minus_p(min(SB.QS.DEPTHS, key=lambda v: abs(v - zz)), min(max(int(round(d)), 25), 95))
    pa = np.abs(lp[P0:P0 + 60]).max(); tr = int(P0 + sp * 1.0); sw = np.abs(lp[tr:]).max()
    hp = sosfiltfilt(SH, Z); e = lambda a, b: np.mean(hp[P0 + a:P0 + b] ** 2); nz = np.mean(hp[5:P0 - 5] ** 2)
    return np.log10(sw / pa), np.log10(max(e(5, 40) - nz, 1e-30) / max(e(-2, 5) - nz, 1e-30))
out = []
for i, (m, s) in enumerate(zip(J['meta'], J['split'])):
    if s != 'dev': continue
    M, _ = CP.usgs_mt(m['id'])
    if M is None: continue
    X = CP.synth_for(m, M); R = z['x'][i].astype(float) * 10 ** z['ls'][i]
    rs, rp = stats(R[0], m['dist'], m['depth']); ss, sp = stats(X[0], m['dist'], m['depth'])
    out.append(dict(i=i, depth=m['depth'], mw=m['mw'], dist=m['dist'], sw_real=rs, sw_syn=ss, pc_real=rp, pc_syn=sp))
o = {k: np.array([r[k] for r in out]) for k in out[0]}
for nm, k in (('shallow', o['depth'] <= 70), ('deep', o['depth'] > 70)):
    print(nm, k.sum(), 'surface/P log10 real-synth median %.2f (iqr %.2f..%.2f)' % (np.median(o['sw_real'][k] - o['sw_syn'][k]), *np.percentile(o['sw_real'][k] - o['sw_syn'][k], [25, 75])),
          '| P-coda 5-40 s log10 real %.2f synth %.2f' % (np.median(o['pc_real'][k]), np.median(o['pc_syn'][k])))
json.dump(out, open(D + 'diag_shallow.json', 'w'), indent=1)
