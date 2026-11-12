"""Error-bar calibration on DEV only (widen-only): for each output, the factor that makes 68% of dev errors fall inside
1 sigma, never below 1. Writes sigma_factor.json."""
import os, sys, json, numpy as np
import evaluate as E

def fit(ck, split='dev', vis='confirmed'):
    net = E.load_net(ck); r, P, ev = E.real(net, split, vis, return_pred=True)
    D = np.array([e['D'] / 1e3 for e in ev]); t0 = np.array([e['t0_rel'] for e in ev]); d = np.array([e['dir'] for e in ev])
    eD = np.abs(np.log10(P['D_km'] / D)) / P['logD_sig']; et = np.abs(P['t0'] - t0) / P['t0_sig']
    ea = np.radians(E.angdiff(P['dir'], d)) * np.sqrt(P['kappa'])
    f = dict(logD=max(1.0, float(np.quantile(eD, 0.68))), t0=max(1.0, float(np.quantile(et, 0.68))), dir=max(1.0, float(np.quantile(ea, 0.68))), loghs=1.0, n_dev=len(ev), ckpt=os.path.basename(ck))
    json.dump(f, open(E.SIGF, 'w'), indent=1); print(f); return f

if __name__ == '__main__': fit(sys.argv[1])
