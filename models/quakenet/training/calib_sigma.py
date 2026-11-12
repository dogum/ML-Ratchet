"""Error bars learned on simulations are too tight on the real Earth. One number per output, set on the dev records only:
the factor that makes 68% of dev errors fall inside the stretched 1-sigma bar (for back-azimuth, a factor on 1/sqrt(kappa))."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import qnet as Q
def errs(P, meta):
    g = lambda k: np.array([m[k] for m in meta], float)
    return dict(mw=np.abs(P['mw'] - g('mw')) / P['mw_sig'], dist=np.abs(P['dist'] - g('dist')) / P['dist_sig'],
                logz=np.abs(P['logz'] - np.log10(np.maximum(g('depth'), 1))) / P['logz_sig'], baz=Q.angdiff(P['baz'], g('baz')) / (57.3 / np.sqrt(np.maximum(P['kappa'], 1e-6))))
if __name__ == '__main__':
    net = Q.load_net(sys.argv[1]); J = json.load(open(_P.WORK + '/data/real_meta.json')); out = {}
    _, idx, P = Q.real(net, 'dev'); e = errs(P, [J['meta'][i] for i in idx]); f = {k: float(np.quantile(v, 0.68)) for k, v in e.items()}
    _, idx, P = Q.real(net, 'exam'); ee = errs(P, [J['meta'][i] for i in idx])
    out = dict(factor=f, dev_n=len(e['mw']), exam_coverage_raw={k: float(np.mean(v < 1)) for k, v in ee.items()}, exam_coverage_cal={k: float(np.mean(v < f[k])) for k, v in ee.items()},
               exam_coverage_cal_2sig={k: float(np.mean(v < 2 * f[k])) for k, v in ee.items()})
    json.dump(out, open(_P.WORK + '/sigma_calibration.json', 'w'), indent=1); print(json.dumps(out, indent=1))
