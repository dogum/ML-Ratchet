"""Error-bar calibration on dev only: one widening factor per output (numax, dnu), widen-only (factor >= 1), chosen so
that 68.3% of the 18 dev stars x 5 lengths fall within sqrt((f sigma_net)^2 + sigma_Yu^2) of the published value."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import snet as SN

def fit(ckpt):
    net = SN.load_net(_P.WORK + '/ckpt/' + ckpt); _, P, mt = SN.real(net, 'dev'); out = []
    for k, tk, ek in (('numax', 'ln_numax', 'e_numax'), ('dnu', 'ln_dnu', 'e_dnu')):
        e = np.abs(P[tk] - np.log([m[k] for m in mt])); sp = np.array([m[ek] / m[k] for m in mt]); sg = P['sig_' + k]
        f = 1.0
        while np.mean(e < np.sqrt((f * sg) ** 2 + sp ** 2)) < 0.683 and f < 20: f *= 1.02
        out.append(float(f)); print(k, 'raw coverage %.2f' % np.mean(e < np.sqrt(sg ** 2 + sp ** 2)), 'factor %.2f' % f)
    json.dump(dict(ckpt=ckpt, widen=out), open(_P.WORK + '/sigma_calibration.json', 'w')); return out

if __name__ == '__main__': fit(sys.argv[1])
