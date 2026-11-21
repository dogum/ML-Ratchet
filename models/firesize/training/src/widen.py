"""Widen-only uncertainty calibration on dev: one temperature T >= 1 on the network's logits, fitted on the dev 'calib'
half by the log-likelihood of the observed size class; checked on the dev 'gate' half (log loss, 80 % interval coverage).
usage: widen.py <rung>  -> models/final.npz, results/widen.json"""
import sys, json, os
from common import *
import net, supp, train
from scipy.optimize import minimize_scalar


def nll_cov(W, d, X, T):
    W = dict(W, temp=T)
    P = net.forward_np(W, X[list(W['feats'])].values)
    b = np.searchsorted(supp.EDGES, np.log10(d.size_ha.values), side='right')
    c = np.cumsum(P, 1)
    lo = np.argmax(c >= 0.1, 1); hi = np.argmax(c >= 0.9, 1)
    return float(-np.mean(np.log(np.maximum(P[np.arange(len(b)), b], 1e-9)))), float(np.mean((b >= lo) & (b <= hi)))


def main():
    rung = sys.argv[1]
    W = dict(np.load(f'{train.MOD}/{rung}.npz'))
    dc, Xc = train.real_dev('calib')
    dg, Xg = train.real_dev('gate')
    r = minimize_scalar(lambda T: nll_cov(W, dc, Xc, T)[0], bounds=(0.5, 4.0), method='bounded')
    T_fit = float(r.x)
    T = max(1.0, T_fit)                    # widen only
    res = dict(rung=rung, T_fit=T_fit, T=T, calib_before=nll_cov(W, dc, Xc, 1.0), calib_after=nll_cov(W, dc, Xc, T),
               gate_before=nll_cov(W, dg, Xg, 1.0), gate_after=nll_cov(W, dg, Xg, T))
    W['temp'] = T
    np.savez(f'{train.MOD}/final.npz', **W)
    os.makedirs(f'{ROOT}/results', exist_ok=True)
    s, _ = train.score_real(W, dg, Xg)
    res['gate_scores_final'] = s
    json.dump(res, open(f'{ROOT}/results/widen.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'gate_scores_final'}, indent=1), s['escape'], s['size'])


if __name__ == '__main__':
    main()
